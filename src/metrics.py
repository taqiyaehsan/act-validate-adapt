"""
DAG Quality Metrics
=====================

Computes structural comparison metrics between the discovered DAG
and the ground-truth DAG.

Metrics:
  - **Precision**:  fraction of discovered edges that are true.
  - **Recall**:     fraction of true edges that were discovered.
  - **F1**:         harmonic mean of precision and recall.
  - **SHD**:        Structural Hamming Distance — total edge errors
                    (missing + extra + reversed).
  - **Risk/Cost**:  Weighted combination for decision-theoretic evaluation.
                    α · (1 − precision) + β · (1 − recall).

DOMAIN-AGNOSTIC: These metrics are graph-generic.  No domain knobs needed.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple
import numpy as np
from collections import defaultdict
import logging
import math

logger = logging.getLogger(__name__)

@dataclass
class DAGMetrics:
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    risk: float
    cost: float
    shd: int

class MetricsCalculator:
    def __init__(self, ground_truth, alpha=0.5, beta=0.5):
        self.ground_truth = ground_truth
        self.intervention_history = defaultdict(list)
        self.edge_history = defaultdict(list)
        self.alpha = alpha  # Weight for information gain
        self.beta = beta    # Weight for cost vs risk
    
    @staticmethod
    def _normalize_varname(name: str) -> str:
        """Collapse all satisfaction-column variants to a single lowercase token.
        Works on already-lowercased input.  Handles both 'satisfaction' (current
        dataset) and 'overallsatisfaction' (older simulations).
        """
        if name in ('overallsatisfaction', 'overall_satisfaction', 'overall satisfaction'):
            return 'satisfaction'
        return name

    def _norm_edge(self, s, t):
        """Lowercase + variable-name normalisation for one edge tuple."""
        return (self._normalize_varname(str(s).lower()),
                self._normalize_varname(str(t).lower()))

    def calculate_metrics(self, method: str, edges: Set[Tuple[str, str]],
                        method_support: Dict[Tuple[str, str], float],
                        intervention_results: Dict[Tuple[str, str], List[Dict]]) -> DAGMetrics:
        """Calculate metrics for a specific method's DAG"""
        normalized_gt_edges = {self._norm_edge(s, t) for s, t in self.ground_truth.edges}

        # Debug logging
        logger.debug(f"\nProcessing {method} DAG:")
        logger.debug(f"Input edges: {edges}")

        # Normalize edge format
        normalized_edges = {self._norm_edge(s, t) for s, t in edges}
        logger.info(f"Normalized edges: {normalized_edges}")
        
        precision, recall = self._calculate_precision_recall(normalized_edges)
        f1_score = self._calculate_f1_score(precision, recall)

        # Calculate accuracy
        true_pos = len(normalized_edges & normalized_gt_edges)
        true_neg = len(self._get_possible_edges() - (normalized_edges | normalized_gt_edges))
        total_possible = len(self._get_possible_edges())
        accuracy = (true_pos + true_neg) / total_possible if total_possible > 0 else 0.0
        
        cost = self._calculate_method_cost(method, normalized_edges, intervention_results)
        risk = self._calculate_method_risk(method, normalized_edges, method_support, intervention_results)
        shd = self.ground_truth.get_shd(normalized_edges)
        
        logger.info(f"Metrics calculated for {method} - Cost: {cost}, Risk: {risk}, SHD: {shd}")
        
        return DAGMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            risk=risk,
            cost=cost,
            shd=shd
        )
    
    def _get_possible_edges(self):
        """Get all possible valid edges between variables.
        Uses the normalised satisfaction token so comparisons are consistent
        regardless of whether the simulation uses 'Satisfaction' or 'OverallSatisfaction'.
        """
        source_vars = {'temperature', 'humidity', 'airquality'}
        target_vars = {'energyconsumption', 'satisfaction'}   # normalised form
        # source_vars = {'handx', 'handy', 'ballx', 'bally'}
        # target_vars = {'collision'}
        return {(s, t) for s in source_vars for t in target_vars}

    def _calculate_precision_recall(self, dag_edges: Set[Tuple[str, str]]) -> Tuple[float, float]:
        """Calculate precision and recall against ground truth"""
        normalized_gt_edges = {self._norm_edge(s, t) for s, t in self.ground_truth.edges}
        # Re-normalise dag_edges in case caller hasn't applied _norm_edge yet
        dag_edges = {self._norm_edge(s, t) for s, t in dag_edges}
        
        if not dag_edges and not normalized_gt_edges:
            return 1.0, 1.0
        if not dag_edges:
            return 0.0, 0.0
        if not normalized_gt_edges:
            return 0.0, 1.0
            
        true_pos = len(dag_edges & normalized_gt_edges)
        false_pos = len(dag_edges - normalized_gt_edges) # In predicted but not in ground truth
        false_neg = len(normalized_gt_edges - dag_edges) # In ground truth but not predicted
        
        try:
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0.0
            recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0.0
        except ZeroDivisionError:
            logger.error("Division by zero in precision/recall calculation")
            return 0.0, 0.0
            
        return precision, recall

    def _calculate_f1_score(self, precision: float, recall: float) -> float:
        try:
            if precision + recall == 0:
                return 0.0
            return 2 * (precision * recall) / (precision + recall)
        except ZeroDivisionError:
            return 0.0

    def _calculate_method_cost(self, method: str, edges: Set[Tuple[str, str]], 
                         intervention_results: Dict[Tuple[str, str], List[Dict]]) -> float:
        """
        Calculate cost specific to each method based on wrong edges
    
        Cost(m) = sum(edge_cost(e)) / |Em \ E*| if |Em \ E*| > 0
            = 0 otherwise
        """
        if not edges:
            logger.debug(f"No cost edges detected for {method}")
            return 1.0  # Maximum cost for empty edge set
            
        wrong_edges = edges - {self._norm_edge(s, t) for s, t in self.ground_truth.edges}
        if not wrong_edges:
            return 0.0
            
        total_cost = 0.0
        processed_edges = 0
        
        for edge in wrong_edges:
            edge_results = intervention_results.get(edge, [])
            if not edge_results:
                total_cost += 0.5  # Default cost for untested edges
                processed_edges += 1
                continue
                
            edge_cost = 0.0
            for result in edge_results:
                pre_state = result.get('pre_state', {})
                post_state = result.get('post_state', {})
                
                try:
                    # Calculate satisfaction loss
                    sat_pre = float(pre_state.get('Satisfaction', pre_state.get('OverallSatisfaction', 0)))
                    sat_post = float(post_state.get('Satisfaction', post_state.get('OverallSatisfaction', 0)))
                    sat_loss = max(0, (sat_pre - sat_post) / sat_pre if sat_pre > 0 else 0)
                    
                    # Calculate energy increase
                    energy_pre = float(pre_state.get('EnergyConsumption', 0))
                    energy_post = float(post_state.get('EnergyConsumption', 0))
                    energy_increase = max(0, (energy_post - energy_pre) / energy_pre if energy_pre > 0 else 0)
                    
                    # Calculate accuracy loss - if collision_pre changes from true to false
                    # collision_pre = bool(pre_state.get('collision', False))
                    # collision_post = bool(post_state.get('collision', False))
                    # accuracy_loss = 1.0 if collision_pre and not collision_post else 0.0
                    
                    # # Calculate movement cost - change in hand position
                    # hand_x_pre = float(pre_state.get('handX', 0))
                    # hand_x_post = float(post_state.get('handX', 0))
                    # hand_y_pre = float(pre_state.get('handY', 0))
                    # hand_y_post = float(post_state.get('handY', 0))
                    
                    # dist_pre = math.sqrt(hand_x_pre**2 + hand_y_pre**2)
                    # dist_post = math.sqrt(hand_x_post**2 + hand_y_post**2)
                    
                    # # If distance increased, count as movement cost
                    # movement_cost = max(0, (dist_post - dist_pre) / max(dist_pre, 1) if dist_pre > 0 else 0)
                    # Apply alpha and beta weights
                    intervention_cost = self.alpha * sat_loss + self.beta * energy_increase
                    edge_cost += intervention_cost
                    
                except (TypeError, ValueError, ZeroDivisionError) as e:
                    edge_cost += 0.5  # Default cost for failed calculations
                    
            if edge_results:
                edge_cost /= len(edge_results)
            total_cost += edge_cost
            processed_edges += 1
        
        return total_cost / max(processed_edges, 1)

    def _calculate_method_risk(self, method: str, edges: Set[Tuple[str, str]], 
                            method_support: Dict[Tuple[str, str], float],
                            intervention_results: Dict[Tuple[str, str], List[Dict]]) -> float:
        """
        Calculate risk specific to each method based on confidence in wrong edges
    
        Risk(m) = sum(confidence(e) * edge_cost(e)) / |Em \ E*| if |Em \ E*| > 0
            = 0 otherwise
        """
        if not edges:
            logger.debug(f"No risk edges detected for {method}")
            return 1.0  # Maximum risk for empty edge set
            
        wrong_edges = edges - {self._norm_edge(s, t) for s, t in self.ground_truth.edges}
        if not wrong_edges:
            return 0.0
            
        total_risk = 0.0
        processed_edges = 0
        
        for edge in wrong_edges:
            # Get method-specific confidence
            confidence = method_support.get(edge, 0.5)  # Default confidence for missing edges
                
            # Calculate edge cost
            edge_results = intervention_results.get(edge, [])
            if not edge_results:
                total_risk += confidence * 0.5  # Default risk for untested edges
                processed_edges += 1
                continue
                
            edge_cost = 0.0
            for result in edge_results:
                pre_state = result.get('pre_state', {})
                post_state = result.get('post_state', {})
                
                try:
                    sat_pre = float(pre_state.get('Satisfaction', pre_state.get('OverallSatisfaction', 0)))
                    sat_post = float(post_state.get('Satisfaction', post_state.get('OverallSatisfaction', 0)))
                    sat_loss = max(0, (sat_pre - sat_post) / sat_pre if sat_pre > 0 else 0)
                    
                    energy_pre = float(pre_state.get('EnergyConsumption', 0))
                    energy_post = float(post_state.get('EnergyConsumption', 0))
                    energy_increase = max(0, (energy_post - energy_pre) / energy_pre if energy_pre > 0 else 0)
                    
                    # Apply alpha and beta weights
                    intervention_cost = self.alpha * sat_loss + self.beta * energy_increase
                    edge_cost += intervention_cost
                    
                except (TypeError, ValueError, ZeroDivisionError) as e:
                    edge_cost += 0.5  # Default risk for failed calculations
                    
            if edge_results:
                edge_cost /= len(edge_results)
                
            # Risk is confidence * cost, weighted by alpha
            edge_risk = self.alpha * confidence * edge_cost
            total_risk += edge_risk
            processed_edges += 1
        
        return total_risk / max(processed_edges, 1)

    def track_shd(self, method: str, edges: Set[Tuple[str, str]]) -> int:
        """Track SHD for a method"""
        return self.ground_truth.get_shd(edges)