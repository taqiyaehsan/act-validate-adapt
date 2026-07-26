"""
Edge Ranking and Hypothesis Evaluation
========================================

Ranks candidate edges by multi-method agreement and statistical
confidence, then evaluates the resulting DAG against ground truth
(when available).

Components:
  - **EdgeRanker**:          Scores each edge by how many discovery methods
                             (PC, SAM, LLM, VARLiNGAM) proposed it, weighted
                             by per-method confidence.  Returns a sorted list
                             for intervention validation.
  - **HypothesisEvaluator**: Compares the discovered DAG to the ground-truth
                             DAG and computes precision, recall, F1, SHD.

DOMAIN-AGNOSTIC KNOBS:
  - ``alpha``, ``beta``:     Weights for method agreement vs. statistical
                             strength in the ranking score.
  - ``ground_truth``:        Set of true edges.  Defined per-domain in
                             ``ground_truth.py``.  Optional — the pipeline
                             runs without it (ranking still works, just no
                             F1/SHD evaluation).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import logging
from typing import Dict, Set, List, Tuple
from src.metrics import MetricsCalculator

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

class HypothesisEvaluator:
    def __init__(self):
        logger.info("Initializing Hypothesis Evaluator...")
        self.ground_truth = {
            'nodes': ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'Satisfaction'],
            'edges': [
                ('Temperature', 'EnergyConsumption'),
                ('Humidity', 'EnergyConsumption'),
                ('AirQuality', 'EnergyConsumption'),
                ('Temperature', 'Satisfaction'),
                ('Humidity', 'Satisfaction'),
                ('AirQuality', 'Satisfaction')
            ]
        }
        # self.ground_truth = {
        #     'nodes': ['handX', 'handY', 'ballX', 'ballY', 'collision'],
        #     'edges': [
        #         ('handX', 'ballX'),
        #         ('handY', 'ballY'),
        #         ('handX', 'collision'),
        #         ('handY', 'collision'),
        #         ('ballX', 'collision'),
        #         ('ballY', 'collision'),
        #     ]
        # }
        self.ground_truth_edges = self._normalize_edges(self.ground_truth['edges'])
        self.alpha = 0.6  # Information gain weight
        self.beta = 0.2   # Cost weight
        self.gamma = 0.2  # Risk weight
        self.metrics_calculator = MetricsCalculator

    def _normalize_edges(self, edges):
        """Convert all edges to tuples for consistent comparison"""
        return set(tuple(edge) if isinstance(edge, list) else edge for edge in edges)

    def _calculate_relevance(self, hypothesis):
        logger.debug("Calculating relevance for hypothesis...")
        """Alignment with ground truth"""
        correct_edges = self._normalize_edges(hypothesis['edges'])
        ground_truth_edges = self.ground_truth_edges
        return len(correct_edges & ground_truth_edges) / len(ground_truth_edges)

    def _calculate_precision(self, hypothesis):
        """Fraction of correctly identified edges"""
        if not hypothesis['edges']:
            return 0
        correct_edges = self._normalize_edges(hypothesis['edges'])
        ground_truth_edges = self.ground_truth_edges
        return len(correct_edges & ground_truth_edges) / len(correct_edges)

    def _calculate_recall(self, hypothesis):
        """Fraction of ground truth edges found"""
        correct_edges = self._normalize_edges(hypothesis['edges'])
        ground_truth_edges = self.ground_truth_edges
        return len(correct_edges & ground_truth_edges) / len(ground_truth_edges)

    def _calculate_novelty(self, hypothesis, edge_results):
        """Proportion of novel causal insights"""
        if not hypothesis['edges']:
            return 0
        hypothesis_edges = self._normalize_edges(hypothesis['edges'])
        ground_truth_edges = self.ground_truth_edges
        novel_edges = hypothesis_edges - ground_truth_edges
        
        # Convert edge results to dict for easier lookup
        edge_result_dict = {
            tuple(r['edge']) if isinstance(r['edge'], list) else r['edge']: r 
            for r in edge_results if 'edge' in r
        }
        
        valid_novel_edges = [
            edge for edge in novel_edges 
            if edge in edge_result_dict and 
            edge_result_dict[edge].get('confidence_score', 0) > 0.7
        ]
        return len(valid_novel_edges) / len(hypothesis['edges'])

    def _calculate_information_gain(self, edge_results):
        """Average confidence of edge testing results and intervention quality"""
        if not edge_results:
            return 0
        
        confidence_scores = [result.get('confidence_score', 0) for result in edge_results]
        
        # Assess intervention quality
        intervention_scores = []
        for result in edge_results:
            if 'intervention' in result:
                # Check if intervention has expected effects
                if 'expected_effects' in result['intervention']:
                    intervention_scores.append(1.0)
                else:
                    intervention_scores.append(0.5)
        
        # Combine confidence and intervention quality
        if intervention_scores:
            intervention_quality = np.mean(intervention_scores)
            return 0.7 * np.mean(confidence_scores) + 0.3 * intervention_quality
        else:
            return np.mean(confidence_scores)

    def _calculate_cost(self, hypothesis):
        """Cost based on number of interventions needed"""
        return len(hypothesis['edges']) / len(self.ground_truth['edges'])

    def _calculate_risk(self, hypothesis):
        """Risk based on potential incorrect edges"""
        incorrect_edges = set(hypothesis['edges']) - set(self.ground_truth['edges'])
        return len(incorrect_edges) / len(hypothesis['edges']) if hypothesis['edges'] else 1
    
    def _calculate_regret(self, hypothesis, edge_results):
        """Cumulative regret calculation"""
        optimal_utility = self._calculate_utility(self.ground_truth, edge_results)
        current_utility = self._calculate_utility(hypothesis, edge_results)
        return optimal_utility - current_utility

    def calculate_metrics(self, hypothesis, edge_results):
        metrics = {}
        for method, result in hypothesis.items():
            edges = set(map(tuple, result['edges']))
            method_support = self.edge_ranker.edge_confidence if hasattr(self, 'edge_ranker') else {}
            
            metrics[method] = self.metrics_calculator.calculate_metrics(
                method,
                edges,
                method_support,
                edge_results
            )
        return metrics

    def _calculate_utility(self, hypothesis, edge_results):
        metrics = self.calculate_metrics(hypothesis, edge_results)
        # Higher utility = higher performance, lower cost/risk
        return metrics.f1_score - (0.4 * metrics.cost + 0.6 * metrics.risk)

    def rank_hypotheses(self, results):
       return dict(sorted(
           results.items(),
           key=lambda x: x[1]['metrics']['utility'],
           reverse=True
       ))

class HypothesisVisualizer:
    def plot_metrics(self, results_dict, best_method=None):
        try:
            logger.info("Creating visualization...")
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

            # Get best method
            best_method = max(results_dict.items(), 
                            key=lambda x: x[1]['metrics']['utility'])[0]
            
            # Update title to include best method
            fig.suptitle(f'Causal Analysis Results (Best Method: {best_method})', fontsize=16)
            
            self._plot_core_metrics(results_dict, ax1)
            self._plot_regret(results_dict, ax2)
            self._plot_utility_components(results_dict, ax3)
            self._plot_best_dag(results_dict, ax4, best_method)
            
            # Adjust spacing between plots
            plt.subplots_adjust(hspace=0.4, wspace=0.4)
            # Add tight_layout with more space at top for title
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            return fig
            
        except Exception as e:
            logger.error(f"Error in plot_metrics: {str(e)}")
            raise

    def _plot_core_metrics(self, results_dict, ax):
        try:
            # Filter out 'tester' and get only methods with complete metrics
            # methods = [m for m in results_dict.keys() 
            #           if m != 'tester' and 'metrics' in results_dict[m]]

            methods = [m for m in results_dict.keys() if 'metrics' in results_dict[m]]
            
            if not methods:
                ax.text(0.5, 0.5, 'No core metrics available', 
                       ha='center', va='center')
                return
                
            metrics = ['relevance', 'precision', 'recall']
            data = []
            
            for metric in metrics:
                metric_values = []
                for m in methods:
                    try:
                        value = results_dict[m]['metrics'].get(metric, 0)
                        metric_values.append(value)
                    except (KeyError, TypeError):
                        logger.info(f"Warning: Missing {metric} for {m}")
                        metric_values.append(0)
                data.append(metric_values)
            
            x = np.arange(len(methods))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                ax.bar(x + i*width, data[i], width, label=metric)
            
            ax.set_xticks(x + width)
            ax.set_xticklabels(methods, rotation=45)
            ax.legend()
            ax.set_title('Core Metrics')
            
        except Exception as e:
            logger.info(f"Error in _plot_core_metrics: {str(e)}")
            ax.text(0.5, 0.5, f'Error plotting core metrics: {str(e)}', 
                   ha='center', va='center')

    def _plot_regret(self, results_dict, ax):
        try:
            # Include all methods for regret
            methods = list(results_dict.keys())
            regrets = []
            
            for m in methods:
                try:
                    regret = results_dict[m]['metrics'].get('regret', 0)
                    regrets.append(regret)
                except (KeyError, TypeError):
                    logger.info(f"Warning: Missing regret for {m}")
                    regrets.append(0)
            
            # Create x-axis positions and set ticks properly
            x_pos = np.arange(len(methods))
            ax.plot(x_pos, regrets, marker='o')
            ax.set_xticks(x_pos)  # Set the tick positions
            ax.set_xticklabels(methods, rotation=45)  # Then set the labels
            ax.set_title('Cumulative Regret')
            
        except Exception as e:
            logger.info(f"Error in _plot_regret: {str(e)}")
            ax.text(0.5, 0.5, f'Error plotting regret: {str(e)}', 
                   ha='center', va='center')

    def _plot_utility_components(self, results_dict, ax):
        try:
            methods = list(results_dict.keys())
            components = []
            
            for m in methods:
                try:
                    method_components = [
                        results_dict[m]['metrics'].get('information_gain', 0),
                        results_dict[m]['metrics'].get('cost', 0),
                        results_dict[m]['metrics'].get('risk', 0)
                    ]
                    components.append(method_components)
                except (KeyError, TypeError):
                    logger.info(f"Warning: Missing utility components for {m}")
                    components.append([0, 0, 0])
            
            components = np.array(components)
            
            # Separate bars for each component
            x = np.arange(len(methods))
            width = 0.25
            
            # Create bar handles for legend
            bar1 = ax.bar(x - width, components[:,0], width, label='Info Gain')
            bar2 = ax.bar(x, components[:,1], width, label='Cost')
            bar3 = ax.bar(x + width, components[:,2], width, label='Risk')
            
            ax.set_xticks(x)
            ax.set_xticklabels(methods, rotation=45)
            ax.set_title('Utility Components')
            
            # Add total utility line with handle
            total_utility = [results_dict[m]['metrics'].get('utility', 0) for m in methods]
            ax2 = ax.twinx()
            line1 = ax2.plot(x, total_utility, 'r-', label='Total Utility')[0]
            line2 = ax2.axhline(y=0.4, color='r', linestyle='--', label='Optimal Utility')
            ax2.set_ylabel('Total Utility')
            
            # Combine legends with all handles
            lines = [bar1, bar2, bar3, line1, line2]
            labels = [l.get_label() for l in lines]
            ax.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.15, 1))

        except Exception as e:
            logger.info(f"Error in _plot_utility_components: {str(e)}")
            ax.text(0.5, 0.5, f'Error plotting utility components: {str(e)}', 
                ha='center', va='center')

    def _plot_best_dag(self, results_dict, ax, best_method):
        try:
            valid_results = {k: v for k, v in results_dict.items() 
                        if 'metrics' in v and 'utility' in v['metrics']}
            
            if not valid_results:
                ax.text(0.5, 0.5, 'No valid DAGs available', ha='center', va='center')
                return
                
            best_hypothesis = max(valid_results.items(),
                key=lambda x: x[1]['metrics']['utility'])[1]['hypothesis']
            
            G = nx.DiGraph()
            G.add_edges_from(best_hypothesis['edges'])
            pos = nx.spring_layout(G)
            nx.draw(G, pos, ax=ax, with_labels=True, 
                node_color='lightblue',
                node_size=1000, 
                arrowsize=20,
                font_size=8,
                font_weight='bold')
            ax.set_title(f'Best Performing DAG ({best_method})')
            
        except Exception as e:
            logger.error(f"Error in _plot_best_dag: {str(e)}")
            ax.text(0.5, 0.5, f'Error plotting best DAG: {str(e)}', 
                ha='center', va='center')
            
class EdgeRanker:
    def __init__(self, hypotheses, validated_edges=None):
        self.hypotheses = {}
        for name, result in hypotheses.items():
            if isinstance(result, dict):
                if 'hypothesis' in result:
                    self.hypotheses[name] = result['hypothesis']
                elif 'edges' in result:
                    self.hypotheses[name] = result
            else:
                self.hypotheses[name] = {'edges': []}
                
        self.edge_confidence = {}
        self.tested_edges = set()
        self.discarded_edges = set()
        self.validated_edges = validated_edges or set()
        self._normalize_hypotheses()
        self.compute_edge_confidence()

        # self.variable_mapping = {
        #     # handX variations
        #     'handx': 'handX',
        #     'handX': 'handX',
        #     'HANDX': 'handX',
        #     'hand_x': 'handX',
        #     'hand_X': 'handX',
            
        #     # handY variations
        #     'handy': 'handY',
        #     'handY': 'handY',
        #     'HANDY': 'handY',
        #     'hand_y': 'handY',
        #     'hand_Y': 'handY',
            
        #     # ballX variations
        #     'ballx': 'ballX',
        #     'ballX': 'ballX',
        #     'BALLX': 'ballX',
        #     'ball_x': 'ballX',
        #     'ball_X': 'ballX',
            
        #     # ballY variations
        #     'bally': 'ballY',
        #     'ballY': 'ballY',
        #     'BALLY': 'ballY',
        #     'ball_y': 'ballY',
        #     'ball_Y': 'ballY',
            
        #     # collision variations
        #     'collision': 'collision',
        #     'Collision': 'collision',
        #     'COLLISION': 'collision',
        # }
        self.variable_mapping = {
            # Temperature variations
            'temperature': 'Temperature',
            'Temperature': 'Temperature',
            'TEMPERATURE': 'Temperature',
            
            # Humidity variations
            'humidity': 'Humidity',
            'Humidity': 'Humidity',
            'HUMIDITY': 'Humidity',
            
            # Air Quality variations
            'airquality': 'AirQuality',
            'air quality': 'AirQuality',
            'air_quality': 'AirQuality',
            'AirQuality': 'AirQuality',
            'Airquality': 'AirQuality',
            'AIRQUALITY': 'AirQuality',
            'AIR_QUALITY': 'AirQuality',
            'airQuality': 'AirQuality',
            
            # HVAC Setpoint variations
            'hvacsetpoint': 'HVACSetpoint',
            'hvac setpoint': 'HVACSetpoint',
            'hvac_setpoint': 'HVACSetpoint',
            'HVACSetpoint': 'HVACSetpoint',
            'Hvacsetpoint': 'HVACSetpoint',
            'HVACSETPOINT': 'HVACSetpoint',
            'HVAC_SETPOINT': 'HVACSetpoint',
            'hvacSetpoint': 'HVACSetpoint',
            
            # Lighting Level variations
            'lightinglevel': 'LightingLevel',
            'lighting level': 'LightingLevel',
            'lighting_level': 'LightingLevel',
            'LightingLevel': 'LightingLevel',
            'Lightinglevel': 'LightingLevel',
            'LIGHTINGLEVEL': 'LightingLevel',
            'LIGHTING_LEVEL': 'LightingLevel',
            'lightingLevel': 'LightingLevel',
            
            # Occupant Count variations
            'occupantcount': 'OccupantCount',
            'occupant count': 'OccupantCount',
            'occupant_count': 'OccupantCount',
            'OccupantCount': 'OccupantCount',
            'Occupantcount': 'OccupantCount',
            'OCCUPANTCOUNT': 'OccupantCount',
            'OCCUPANT_COUNT': 'OccupantCount',
            'occupantCount': 'OccupantCount',
            'occupancy': 'OccupantCount',
            'Occupancy': 'OccupantCount',
            
            # Energy Consumption variations
            'energyconsumption': 'EnergyConsumption',
            'energy consumption': 'EnergyConsumption',
            'energy_consumption': 'EnergyConsumption',
            'EnergyConsumption': 'EnergyConsumption',
            'Energyconsumption': 'EnergyConsumption',
            'ENERGYCONSUMPTION': 'EnergyConsumption',
            'ENERGY_CONSUMPTION': 'EnergyConsumption',
            'energyConsumption': 'EnergyConsumption',
            
            # Overall Satisfaction variations (all map to 'Satisfaction')
            'overallsatisfaction': 'Satisfaction',
            'overall satisfaction': 'Satisfaction',
            'overall_satisfaction': 'Satisfaction',
            'OverallSatisfaction': 'Satisfaction',
            'Overallsatisfaction': 'Satisfaction',
            'OVERALLSATISFACTION': 'Satisfaction',
            'OVERALL_SATISFACTION': 'Satisfaction',
            'overallSatisfaction': 'Satisfaction',
            'satisfaction': 'Satisfaction',
            'Satisfaction': 'Satisfaction',
            'SATISFACTION': 'Satisfaction',
            
            # PMV variations  
            'pmv': 'PMV',
            'PMV': 'PMV',
            
            # Window Open variations
            'windowopen': 'WindowOpen',
            'window open': 'WindowOpen',
            'WindowOpen': 'WindowOpen',
            'WINDOWOPEN': 'WindowOpen',

            # OutdoorTemperature variations
            'outdoortemperature': 'OutdoorTemperature',
            'outdoor temperature': 'OutdoorTemperature',
            'outdoor_temperature': 'OutdoorTemperature',
            'OutdoorTemperature': 'OutdoorTemperature',
            'Outdoortemperature': 'OutdoorTemperature',
            'OUTDOORTEMPERATURE': 'OutdoorTemperature',
            'OUTDOOR_TEMPERATURE': 'OutdoorTemperature',
            'outdoorTemp': 'OutdoorTemperature',
            'outdoortemp': 'OutdoorTemperature',

            # Thermal Comfort variations
            'thermalcomfort': 'ThermalComfort',
            'thermal comfort': 'ThermalComfort',
            'thermal_comfort': 'ThermalComfort',
            'ThermalComfort': 'ThermalComfort',
            'Thermalcomfort': 'ThermalComfort',
            'THERMALCOMFORT': 'ThermalComfort',
            'THERMAL_COMFORT': 'ThermalComfort',
            'thermalComfort': 'ThermalComfort',
            
            # Visual Comfort variations
            'visualcomfort': 'VisualComfort',
            'visual comfort': 'VisualComfort',
            'visual_comfort': 'VisualComfort',
            'VisualComfort': 'VisualComfort',
            'Visualcomfort': 'VisualComfort',
            'VISUALCOMFORT': 'VisualComfort',
            'VISUAL_COMFORT': 'VisualComfort',
            'visualComfort': 'VisualComfort',
            
            # Air Quality Index variations
            'airqualityindex': 'AirQualityIndex',
            'air quality index': 'AirQualityIndex',
            'air_quality_index': 'AirQualityIndex',
            'AirQualityIndex': 'AirQualityIndex',
            'Airqualityindex': 'AirQualityIndex',
            'AIRQUALITYINDEX': 'AirQualityIndex',
            'AIR_QUALITY_INDEX': 'AirQualityIndex',
            'airQualityIndex': 'AirQualityIndex',
            
            # HVAC Power variations
            'hvacpower': 'HVACPower',
            'hvac power': 'HVACPower',
            'hvac_power': 'HVACPower',
            'HVACPower': 'HVACPower',
            'Hvacpower': 'HVACPower',
            'HVACPOWER': 'HVACPower',
            'HVAC_POWER': 'HVACPower',
            'hvacPower': 'HVACPower',
            
            # Lighting Power variations
            'lightingpower': 'LightingPower',
            'lighting power': 'LightingPower',
            'lighting_power': 'LightingPower',
            'LightingPower': 'LightingPower',
            'Lightingpower': 'LightingPower',
            'LIGHTINGPOWER': 'LightingPower',
            'LIGHTING_POWER': 'LightingPower',
            'lightingPower': 'LightingPower'
        }
        
    def _normalize_edge(self, edge):
        """Normalize edge format to tuple of lowercase strings"""
        if isinstance(edge, (list, tuple)):
            return tuple(str(x).strip().lower() for x in edge)
        return edge
        
    def _normalize_hypotheses(self):
        """Normalize all edges in hypotheses"""
        for name, hypothesis in self.hypotheses.items():
            if 'edges' in hypothesis:
                hypothesis['edges'] = [
                    self._normalize_edge(edge) 
                    for edge in hypothesis['edges'] 
                    if edge
                ]
    
    def get_union_edges(self):
        """Get all unique edges from all hypotheses"""
        union_edges = set()
        for hyp in self.hypotheses.values():
            edges = hyp.get('edges', [])
            normalized_edges = {self._normalize_edge(edge) for edge in edges if edge}
            union_edges.update(normalized_edges)
        
        logger.info(f"{len(union_edges)} Union Edges identified: {union_edges}")
        return union_edges
        
    def get_common_edges(self):
        """Get edges that appear in all hypotheses"""
        edge_sets = []
        for hyp in self.hypotheses.values():
            edges = hyp.get('edges', [])
            normalized_edges = {self._normalize_edge(edge) for edge in edges if edge}
            if normalized_edges:
                edge_sets.append(normalized_edges)
                
        if edge_sets:
            common = set.intersection(*edge_sets)
            logger.info(f"\nCommon edges: {common}")
            return common
        return set()
        
    def get_disagreeable_edges(self):
        """Get edges that don't appear in all hypotheses"""
        union = self.get_union_edges()
        common = self.get_common_edges()
        return union - common
        
    def compute_edge_confidence(self):
        """Compute confidence scores based on method agreement and intervention validation"""
        union_edges = self.get_union_edges()
        
        for edge in union_edges:
            normalized_edge = self._normalize_edge(edge)
            
            # For validated edges, set high confidence
            if normalized_edge in self.validated_edges:
                self.edge_confidence[normalized_edge] = 1.0
                continue
                
            # Calculate method agreement
            method_count = sum(1 for method in self.hypotheses.values() 
                             if normalized_edge in {self._normalize_edge(e) for e in method.get('edges', [])})
            method_agreement = method_count / len(self.hypotheses)
            
            self.edge_confidence[normalized_edge] = method_agreement
            
    # def get_ranked_edges(self):
    #     """Return edges ranked from lowest to highest confidence"""
    #     union_edges = self.get_union_edges()
        
    #     # Filter out validated edges and edges with 'collision' as source
    #     to_rank = [e for e in union_edges 
    #             if e not in self.validated_edges 
    #             and e[0].lower() != 'collision']
        
    #     return sorted(
    #         to_rank,
    #         key=lambda e: self.edge_confidence.get(self._normalize_edge(e), 0)
    #     )
    def get_ranking_confidence(self, edge):
        """Method agreement only - for ranking purposes"""
        method_count = sum(1 for method in self.hypotheses.values() 
                        if edge in method.get('edges', []))
        return method_count / len(self.hypotheses)

    # def get_ranked_edges(self):
    #     """Return edges ranked by method agreement only"""
    #     return sorted(
    #         self.get_union_edges(),
    #         key=lambda e: self.get_ranking_confidence(e)
    #     )
    def get_ranked_edges(self):
        """Return edges ranked from LOWEST to HIGHEST confidence (disagreeable edges first)"""
        union_edges = self.get_union_edges()
        
        # Filter out validated edges and non-intervenable sources
        non_intervenable = {'windowopen', 'outdoortemperature', 'satisfaction', 'pmv'}
        # NOTE: hvacpower/lightingpower are ACTUATORS, not outputs — keep them
        # as valid sources so they can be tested via intervention
        output_vars = {'energyconsumption', 'overallsatisfaction', 'satisfaction',
                    'meter_reading', 'thermalcomfort', 'visualcomfort',
                    'airqualityindex'}
        
        to_rank = [
            e for e in union_edges 
            if e not in self.validated_edges 
            and e[0].lower() not in non_intervenable
            and e[0].lower() not in output_vars
        ]
        
        # Sort by confidence ASCENDING (lowest first = most disagreeable first)
        ranked = sorted(to_rank, key=lambda e: self.edge_confidence.get(e, 0.0))
        
        logger.info(f"\n=== Edge Ranking (Lowest Confidence First) ===")
        for i, edge in enumerate(ranked[:10]):  # Show top 10
            conf = self.edge_confidence.get(edge, 0.0)
            logger.info(f"  {i+1}. {edge}: confidence={conf:.3f}")
        
        return ranked