"""
Causal World Model (CWM) — Section III.A of PolicyGRID
========================================================

Maintains a probabilistic causal graph with per-edge confidence scores,
competing graph hypotheses, and predictive error monitoring.

Core components:
  - **EdgeConfidence**: Per-edge Bayesian confidence C_{i,j} updated via
    Ψ(prediction_error, correlation) with exponential smoothing (α=0.1).
  - **GraphHypothesis**: Competing graph structures (H_0: window closed,
    H_1: window open) with associated predictive models and posterior weights.
  - **PredictiveModel**: Ridge regression one-step-ahead predictor for each
    variable, conditioned on a specific graph hypothesis's parent set.
  - **Bayesian regime filter**: Binary H_0/H_1 monitoring with forgetting
    factor λ that prevents posterior lock-in at absorbing states.
    Update rule: P(H_k) ∝ [(1−λ)·P(H_k) + λ·0.5] · p(y_t | H_k)

DOMAIN-AGNOSTIC KNOBS:
  - ``forget_factor``:           λ for prior leak (default 0.15).  Increase for
                                 faster regime recovery; decrease for stability.
  - ``SIGMA_MAX``:               Cap on likelihood precision σ = 1/MAE (default 30).
  - ``detection_threshold``:     P(H_k) > threshold → switch active hypothesis.
  - ``variables``:               List of observed variable names.  The CWM
                                 builds parent sets from the DAG over these.
  - ``temperature_var``:         Which variable to use for likelihood evaluation.
                                 Default 'temperature'; change for non-HVAC domains.

Paper reference: Section III.A, Equations (3)-(7).
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import deque
import networkx as nx
from regime_aware_validation import WindowStatePredictor, RegimeAwareValidator

logger = logging.getLogger(__name__)

@dataclass
class EdgeConfidence:
    """Confidence score for a causal edge"""
    source: str
    target: str
    confidence: float = 0.5  # Initial neutral confidence
    prediction_errors: deque = field(default_factory=lambda: deque(maxlen=50))
    correlations: deque = field(default_factory=lambda: deque(maxlen=50))
    last_updated: int = 0
    
    def update(self, prediction_error: float, correlation: float, alpha: float = 0.1):
        """Update confidence based on new evidence"""
        old_confidence = self.confidence
        self.prediction_errors.append(prediction_error)
        self.correlations.append(correlation)
        
        # Ψ function: converts low errors + high correlation → high confidence
        normalized_error = max(0, 1 - abs(prediction_error))  # Lower error = higher score
        normalized_corr = (correlation + 1) / 2  # Map [-1,1] to [0,1]
        
        # Weighted combination
        evidence_score = 0.6 * normalized_error + 0.4 * normalized_corr
        
        # Exponential smoothing
        self.confidence = (1 - alpha) * self.confidence + alpha * evidence_score
        self.confidence = np.clip(self.confidence, 0.0, 1.0)

        logger.debug(f"  [{self.source}→{self.target}] Confidence: {old_confidence:.3f} → {self.confidence:.3f} "
          f"(error={prediction_error:.3f}, corr={correlation:.3f})")


@dataclass
class GraphHypothesis:
    """A hypothesis about the causal graph structure"""
    name: str
    graph: nx.DiGraph
    predictive_model: 'PredictiveModel'
    weight: float = 1.0  # Posterior probability
    prediction_history: List[float] = field(default_factory=list)
    
    def compute_likelihood(self, observed_data: Dict[str, float]) -> float:
        """Compute p(y | G) - likelihood of observed data under this hypothesis"""
        predictions = self.predictive_model.predict(observed_data)
        
        # Gaussian likelihood based on prediction errors
        log_likelihood = 0
        for var, observed in observed_data.items():
            if var in predictions:
                predicted = predictions[var]
                error = observed - predicted
                # Assume unit variance for simplicity (could be learned)
                log_likelihood += -0.5 * error**2
        
        return np.exp(log_likelihood)


class PredictiveModel:
    """
    Structural Equation Model (SEM) for next-state prediction
    X̂_{t+1} = f̂_G(X_t, A_t) = SEM_θ(X_t, A_t; G)
    
    Learns linear coefficients for each edge to minimize prediction error.
    """
    
    def __init__(self, graph: nx.DiGraph, variables: List[str], quadratic: bool = False):
        self.graph = graph
        self.variables = variables
        self.quadratic = quadratic
        self.parameters = {}  # θ parameters for each node
        self._initialize_parameters()
        self.coefficients = {}  # Learned edge weights from observations
        self._parent_cwm = None

    def _initialize_parameters(self):
        """Initialize linear model parameters: node = intercept + Σ(weight_i * parent_i)"""
        for node in self.graph.nodes():
            parents = list(self.graph.predecessors(node))
            self.parameters[node] = {
                'intercept': 0.0,
                'weights': {},  # Empty - only populated by interventions
                'learned_count': 0
            }
        
    # def predict(self, state: Dict[str, float]) -> Dict[str, float]:
    #     """
    #     Predict next state: X̂_{t+1} = f̂_G(X_t, A_t)
        
    #     Uses learned coefficients from observations, falls back to parameters.
        
    #     Args:
    #         state: Current values of all variables
            
    #     Returns:
    #         Predicted next-state values
    #     """
    #     # Map windowOpen to W1
    #     state_mapped = state.copy()
    #     if 'windowOpen' in state:
    #         state_mapped['W1'] = state['windowOpen']
        
    #     predicted = {}
        
    #     for target in self.variables:
    #         parents = list(self.graph.predecessors(target)) if target in self.graph.nodes() else []
            
    #         if not parents:
    #             # No causal parents - predict constant (mean from history or 0)
    #             predicted[target] = 0.0
    #             continue
            
    #         # MUST compute from parents - don't use observed value
    #         prediction = self.parameters[target]['intercept']
    #         all_parents_present = True
            
    #         for parent in parents:
    #             if parent not in state_mapped:
    #                 all_parents_present = False
    #                 break
    #             coef = self.parameters[target]['weights'].get(parent, 0.1)
    #             prediction += state_mapped[parent] * coef
            
    #         if all_parents_present:
    #             predicted[target] = prediction
    #             logger.debug(f"Predicted {target}={prediction:.2f} from parents {parents}")
    #         else:
    #             predicted[target] = 0.0  # Missing data - predict 0, not observed value
        
    #     # return predicted
    #     return {var: state.get(var, 0.5) for var in self.variables}
    
    def predict(self, state: Dict[str, float]) -> Dict[str, float]:
        """Compute predictions from graph structure (not observed values)."""
        predicted = {}
        targets = self.graph.nodes() if self.graph is not None else self.variables
        for target in targets:
            parents = list(self.graph.predecessors(target)) if target in self.graph.nodes() else []

            if not parents:
                predicted[target] = self.parameters[target].get('intercept', 0.0)
                continue

            prediction = self.parameters[target].get('intercept', 0.0)

            for parent in parents:
                if parent in state:
                    coef = self.parameters[target]['weights'].get(parent, 0.0)
                    prediction += state[parent] * coef
                    # Quadratic term: β_quad × x²
                    if self.quadratic:
                        quad_coef = self.parameters[target].get('quad_weights', {}).get(parent, 0.0)
                        prediction += quad_coef * state[parent] ** 2

            predicted[target] = prediction

        return predicted

    def update(self, pre_state: Dict[str, float], post_state: Dict[str, float]):
        """
        Online learning: update coefficients from single observation.
        Simple gradient descent on prediction error.
        """
        for target in self.variables:
            parents = list(self.graph.predecessors(target))
            
            if not parents or target not in post_state:
                continue
            
            for parent in parents:
                if parent not in pre_state:
                    continue
                
                edge = (parent, target)
                current_coef = self.coefficients.get(edge, 1.0)
                
                parent_val = pre_state[parent]
                target_val = post_state[target]
                
                if parent_val != 0:
                    # Gradient descent: coef += α * error * parent
                    error = target_val - (current_coef * parent_val)
                    self.coefficients[edge] = current_coef + 0.01 * error * parent_val
    
    def update_parameters(self, data: pd.DataFrame, learning_rate: float = 0.01):
        """
        Batch parameter learning using gradient descent.
        Minimizes MSE on recent data window.
        """
        for node in self.graph.nodes():
            if node not in data.columns:
                continue
            
            parents = list(self.graph.predecessors(node))
            if not parents:
                continue
            
            for idx in range(len(data) - 1):
                current = {var: data.iloc[idx][var] for var in data.columns}
                next_true = data.iloc[idx + 1][node]
                next_pred = self.predict(current)[node]
                
                error = next_pred - next_true
                
                # Update weights via gradient
                for parent in parents:
                    if parent in data.columns:
                        gradient = error * data.iloc[idx][parent]
                        self.parameters[node]['weights'][parent] -= learning_rate * gradient
                
                # Update intercept
                self.parameters[node]['intercept'] -= learning_rate * error

    def fit_temporal(self, temporal_df: pd.DataFrame):
        """Train on (t, t+1) sequences using linear or quadratic regression.

        Args:
            temporal_df: DataFrame with columns ``var`` (time t) and
                ``var_next`` (time t+1) for each variable.
        """
        from sklearn.linear_model import LinearRegression

        mode = "quadratic" if self.quadratic else "linear"
        logger.info(f"fit_temporal ({mode}) called with {len(temporal_df)} rows, columns: {list(temporal_df.columns)}")

        for target in self.variables:
            logger.info(f"Processing target: {target}")
            if target not in temporal_df.columns:
                logger.warning(f"  {target} not in temporal_df columns")
                continue
            if target not in self.parameters:
                logger.info(f"  {target} not in graph nodes, skipping")
                continue

            parents = list(self.graph.predecessors(target)) if target in self.graph.nodes() else []
            logger.info(f"  Parents: {parents}")

            # X: parent values at time t, y: target value at t+1
            X_cols = [p for p in parents if p in temporal_df.columns]
            logger.info(f"  X_cols (available parents): {X_cols}")

            if not X_cols:
                logger.warning(f"  No predictors for {target}")
                continue

            y_col = f'{target}_next'
            if y_col not in temporal_df.columns:
                logger.warning(f"  {y_col} not in temporal_df columns")
                continue

            X_lin = temporal_df[X_cols].values
            y = temporal_df[y_col].values

            if len(X_lin) != len(y):
                X_lin = X_lin[:-1]

            if self.quadratic:
                parent_X = temporal_df[X_cols].values
                if len(parent_X) != len(y):
                    parent_X = parent_X[:-1]
                X = np.hstack([X_lin, parent_X ** 2])
            else:
                X = X_lin

            model = LinearRegression()
            model.fit(X, y)

            self.parameters[target]['intercept'] = model.intercept_
            for i, parent in enumerate(X_cols):
                self.parameters[target]['weights'][parent] = model.coef_[i]

            if self.quadratic:
                n_lin = len(X_cols)
                self.parameters[target]['quad_weights'] = {}
                for i, parent in enumerate(X_cols):
                    self.parameters[target]['quad_weights'][parent] = model.coef_[n_lin + i]

            logger.debug(f"  {target}: intercept={model.intercept_:.3f}, "
                        f"coefs={dict(zip(X_cols, model.coef_[:len(X_cols)]))}")


class CausalWorldModel:
    """
    Enhanced Causal World Model with:
    - Confidence scoring on edges (C_{i,j})
    - Predictive model (f̂_G)
    - Hypothesis maintenance (H_0, H_1, ...)
    - Uncertainty quantification (U_t)
    """
    
    def __init__(self, initial_graph: nx.DiGraph, variables: List[str], method_dags: Dict = None):
        self.graph = initial_graph
        self.variables = variables
        self.hypothesis_bank = HypothesisBank()
        self.edge_confidences = {}
        
        # Initialize edge confidence scores with method agreement
        for edge in initial_graph.edges():
            if method_dags:
                # Count how many methods agree on this edge
                agreement_count = sum(1 for dag in method_dags.values() 
                                    if edge in self._extract_edges_from_dag(dag))
                initial_confidence = agreement_count / len(method_dags)
            else:
                initial_confidence = 0.5  # Fallback
            
            self.edge_confidences[edge] = EdgeConfidence(
                source=edge[0],
                target=edge[1],
                confidence=initial_confidence
            )
        
        # Edge confidence scores
        # self.edge_confidences: Dict[Tuple[str, str], EdgeConfidence] = {}
        # self._initialize_edge_confidences()
        
        # Primary predictive model
        self.predictive_model = PredictiveModel(self.graph, self.variables)
        
        # Hypothesis set for structural alternatives
        self.hypotheses: List[GraphHypothesis] = []
        self.active_hypothesis_idx = 0
        
        # Uncertainty tracking
        self.uncertainty_history = []
        self.prediction_error_buffer = deque(maxlen=100)
        
        # Data buffer for online learning
        self.data_buffer = pd.DataFrame(columns=variables)
        
        self.intervention_count = 0
        self.min_interventions_for_training = 20

        logger.info(f"CWM initialized with {len(self.graph.edges())} edges")

        # Regime-aware validation
        try:
            self.window_predictor = WindowStatePredictor()
            self.window_predictor.load('window_predictor_model.pkl')
            self.regime_validator = RegimeAwareValidator(self.window_predictor)
            logger.info("Loaded RF window predictor")
        except:
            self.regime_validator = None
            logger.warning("RF predictor not found, skipping regime inference")
    
    def _extract_edges_from_dag(self, dag):
        """Extract edge tuples from DAG dict"""
        edges = dag.get('edges', [])
        return [tuple(e) if not isinstance(e, dict) else e.get('edge', e) 
                for e in edges]
    
    def _initialize_edge_confidences(self):
        """Initialize confidence trackers for all edges"""
        for source, target in self.graph.edges():
            self.edge_confidences[(source, target)] = EdgeConfidence(
                source=source,
                target=target,
                confidence=0.7  # Start with moderate confidence
            )
    
    def predict_next_state(self, current_state: Dict[str, float]) -> Dict[str, float]:
        """
        Main prediction interface: X̂_{t+1} = f̂_G(X_t, A_t)
        Uses the best hypothesis's predictive model
        """
        if not self.hypotheses:
            # Fallback: use main graph
            return self._predict_from_graph(current_state, self.graph)
        
        # Use highest-weight hypothesis
        best_hyp = max(self.hypotheses, key=lambda h: h.weight)
        return best_hyp.predictive_model.predict(current_state)

    def _predict_from_graph(self, current_state: Dict[str, float], graph: nx.DiGraph) -> Dict[str, float]:
        """Helper for prediction from arbitrary graph"""
        predicted = {}
        
        for target in self.variables:
            if target not in graph.nodes():
                predicted[target] = 0.0
                continue
            
            parents = list(graph.predecessors(target))
            if not parents:
                predicted[target] = 0.0
                continue
            
            prediction = sum(
                current_state.get(parent, 0) * self.edge_confidences.get((parent, target), EdgeConfidence(parent, target, 0.5)).confidence
                for parent in parents
            )
            predicted[target] = prediction
        
        return predicted

    def compute_prediction_error(self, predicted: Dict[str, float], 
                                observed: Dict[str, float]) -> float:
        """MSE between predicted and observed states"""
        errors = []
        for var in self.variables:
            if var in predicted and var in observed:
                errors.append((predicted[var] - observed[var]) ** 2)
        
        return np.sqrt(np.mean(errors)) if errors else float('inf')
        
    def update_confidence(self, 
                         observed_state: Dict[str, float],
                         predicted_state: Dict[str, float],
                         alpha: float = 0.1):
        """
        Update edge confidences based on prediction accuracy
        
        C_{i,j}^{(t+1)} = (1-α)C_{i,j}^{(t)} + α Ψ(-|E_j^{(t)}|, ρ_{i,j}^{(t)})
        """
        # Compute prediction errors
        prediction_errors = {}
        for var in self.variables:
            if var in observed_state and var in predicted_state:
                error = observed_state[var] - predicted_state[var]
                prediction_errors[var] = error
                self.prediction_error_buffer.append(abs(error))
        
        # Update each edge confidence
        for (source, target), edge_conf in self.edge_confidences.items():
            if target in prediction_errors:
                # Compute recent correlation between source and target
                if len(self.data_buffer) > 10:
                    recent_data = self.data_buffer.tail(50)
                    if source in recent_data.columns and target in recent_data.columns:
                        correlation = recent_data[source].corr(recent_data[target])
                        if np.isnan(correlation):
                            correlation = 0.0
                    else:
                        correlation = 0.0
                else:
                    correlation = 0.0
                
                # Update confidence
                edge_conf.update(
                    prediction_error=prediction_errors[target],
                    correlation=correlation,
                    alpha=alpha
                )
    
    def compute_uncertainty(self) -> float:
        """
        Compute global causal uncertainty: U_t = 1 - min_{X_i→X_j ∈ G}(C_{i,j})
        
        Returns:
            Uncertainty score in [0, 1] where 0 = certain, 1 = very uncertain
        """
        if not self.edge_confidences:
            return 1.0
        
        min_confidence = min(ec.confidence for ec in self.edge_confidences.values())
        uncertainty = 1.0 - min_confidence
        
        # ADD THIS:
        weakest_edge = min(self.edge_confidences.items(), 
                        key=lambda x: x[1].confidence)
        logger.info(f"\n  U_t = {uncertainty:.3f} (weakest: {weakest_edge[0][0]}→{weakest_edge[0][1]} "
            f"with C={weakest_edge[1].confidence:.3f})")
        
        self.uncertainty_history.append(uncertainty)
        return uncertainty

    def add_hypothesis(self, name: str, alternative_graph: nx.DiGraph):
        """
        Add an alternative structural hypothesis (e.g., H_0: normal, H_1: window open)
        """
        hyp = GraphHypothesis(
            name=name,
            graph=alternative_graph.copy(),
            predictive_model=PredictiveModel(alternative_graph, self.variables),
            weight=1.0 / (len(self.hypotheses) + 1)
        )

        # Rebalance existing hypotheses
        for h in self.hypotheses:
            h.weight = 1.0 / (len(self.hypotheses) + 1)
        
        self.hypotheses.append(hyp)
        logger.info(f"Added hypothesis '{name}' with {len(alternative_graph.edges())} edges")
    
    # def update_hypothesis_weights(self, observed_data: Dict[str, float]):
    #     """
    #     Bayesian update: w_{t+1}^{(k)} ∝ p(y_t | f̂_G^{(k)}) w_t^{(k)}
    #     Enhanced with regime-aware likelihood for H0/H1
    #     """
    #     if not self.hypotheses:
    #         return
        
    #     observed_data = {k.lower(): v for k, v in observed_data.items()}
        
    #     # RF regime inference
    #     regime_info = None
    #     if self.regime_validator:
    #         try:
    #             regime_info = self.regime_validator.validate_intervention(observed_data)
    #             logger.info(f"RF: {regime_info['inferred_regime']} (conf={regime_info['confidence']:.2f})")
    #         except Exception as e:
    #             logger.debug(f"RF failed: {e}")
        
    #     # Compute likelihoods
    #     likelihoods = []
    #     for hyp in self.hypotheses:
    #         predicted = hyp.predictive_model.predict(observed_data)
    #         error = self.compute_prediction_error(predicted, observed_data)
    #         likelihood = np.exp(-error)
            
    #         # RF boost for H0/H1
    #         if regime_info and ('H0' in hyp.name or 'H1' in hyp.name):
    #             regime = regime_info['inferred_regime']
    #             conf = regime_info['confidence']
                
    #             regime_match = 1.5 if (('H0' in hyp.name and regime == 'closed') or 
    #                                 ('H1' in hyp.name and regime == 'open')) else 0.7
    #             boost = (1 - conf) + conf * regime_match
    #             likelihood *= boost
    #             logger.debug(f"  {hyp.name}: regime_boost={boost:.3f}")
            
    #         likelihoods.append(likelihood)
    #         logger.info(f"  {hyp.name}: error={error:.3f}, likelihood={likelihood:.4f}")
        
    #     # Bayesian update
    #     total = sum(likelihoods)
    #     if total > 0:
    #         logger.info("\n  Hypothesis weights before update:")
    #         for hyp in self.hypotheses:
    #             logger.info(f"    {hyp.name}: {hyp.weight:.4f}")
            
    #         logger.debug(f"\n  Computing normalized weights (total={total:.4f}):")
    #         for i, hyp in enumerate(self.hypotheses):
    #             old = hyp.weight
    #             hyp.weight = likelihoods[i] / total
    #             logger.debug(f"    {hyp.name}: {likelihoods[i]:.4f}/{total:.4f} = {hyp.weight:.4f}")
    #             if abs(old - hyp.weight) > 0.01:
    #                 logger.info(f"    {hyp.name}: {old:.4f} → {hyp.weight:.4f}")
    
    def update_hypothesis_weights(self, observed_data: Dict[str, float]):
        if not self.hypotheses:
            return
        
        observed_data = {k.lower(): v for k, v in observed_data.items()}
        
        # RF regime inference (DISABLED - classifier needs retraining)
        regime_info = None
        if self.regime_validator:
            try:
                regime_info = self.regime_validator.validate_intervention(observed_data)
                logger.warning(f"RF: {regime_info['inferred_regime']} (conf={regime_info['confidence']:.2f}) - BOOSTS DISABLED")
            except Exception as e:
                logger.debug(f"RF failed: {e}")
        
        # Compute likelihoods with temperature weighting
        likelihoods = []
        for hyp in self.hypotheses:
            predicted = hyp.predictive_model.predict(observed_data)

            # All models judged on temperature prediction (the regime signal)
            temp_error = abs(predicted.get('temperature', 0) - observed_data.get('temperature', 0))
            weighted_error = temp_error

            # σ is estimated from H0 closed-regime training residuals in
            # pipeline_cwm.py and stored on this CWM instance.  Using a
            # data-driven σ avoids hardcoding the likelihood bandwidth.
            sigma = getattr(self, 'likelihood_sigma', 1.0)
            likelihood = np.exp(-sigma * weighted_error)
            likelihoods.append(likelihood)
            logger.debug(f"  {hyp.name}: temp_err={temp_error:.3f}, likelihood={likelihood:.4f}")

        # BAYESIAN UPDATE with forgetting factor:
        #   weight = (1-λ) × Bayes_posterior + λ × uniform_prior
        # λ = 0.05 means 5% of each weight leaks back toward 1/N every step.
        # This prevents saturation lock-in when the regime switches back.
        FORGET = getattr(self, 'forget_factor', 0.05)
        N = len(self.hypotheses)
        uniform = 1.0 / N

        unnormalized_posteriors = []
        for i, hyp in enumerate(self.hypotheses):
            unnorm = likelihoods[i] * hyp.weight  # Prior × Likelihood
            unnormalized_posteriors.append(unnorm)

        total = sum(unnormalized_posteriors)
        if total > 0:
            logger.debug("\n  Hypothesis weights before update:")
            for hyp in self.hypotheses:
                logger.debug(f"    {hyp.name}: {hyp.weight:.4f}")

            for i, hyp in enumerate(self.hypotheses):
                old = hyp.weight
                bayes_post = unnormalized_posteriors[i] / total
                hyp.weight = (1.0 - FORGET) * bayes_post + FORGET * uniform
                if abs(old - hyp.weight) > 0.01:
                    logger.debug(f"    {hyp.name}: {old:.4f} → {hyp.weight:.4f}")
        
        # Calibration tracking — per-hypothesis weights only.
        # timestamps and actual_window_state are owned by monitor_regime_changes.
        if hasattr(self, '_parent_pipeline') and self._parent_pipeline:
            tracker = self._parent_pipeline.calibration_tracker
            for hyp in self.hypotheses:
                key = f'P_{hyp.name}'  # e.g. P_H0_WindowClosed, P_H1_WindowOpen
                if key not in tracker:
                    tracker[key] = []
                tracker[key].append(hyp.weight)
        
        # Check switch
        h0 = next((h for h in self.hypotheses if 'H0' in h.name), None)
        h1 = next((h for h in self.hypotheses if 'H1' in h.name), None)
        if h0 and h1:
            if h1.weight > 0.8:
                logger.warning(f"*** HYPOTHESIS SWITCH TO H1 ***")
            elif h0.weight > 0.8:
                logger.warning(f"*** HYPOTHESIS SWITCH TO H0 ***")

        # Check for dual failure - query hypothesis bank
        FAILURE_THRESHOLD = 0.1
        all_failing = all(h.weight < FAILURE_THRESHOLD for h in self.hypotheses)
        
        if all_failing:
            logger.error("⚠️ ALL HYPOTHESES FAILING - Querying hypothesis bank")
            
            best_hyp = max(self.hypotheses, key=lambda h: h.weight)
            predicted = best_hyp.predictive_model.predict(observed_data)
            
            residuals = {var: abs(observed_data.get(var, 0) - predicted.get(var, 0)) 
                        for var in observed_data if var in predicted}
            
            high_res_vars = sorted(residuals.items(), key=lambda x: x[1], reverse=True)[:3]
            high_res_var_names = [v[0] for v in high_res_vars]
            
            candidate_edges = []
            for var in high_res_var_names:
                candidates = self.hypothesis_bank.query_edges_for_variable(var)
                candidate_edges.extend(candidates)
            
            hypothesis_weights = {h.name: h.weight for h in self.hypotheses}
            self.hypothesis_bank.flag_diagnostic_event(
                timestamp=len(self.hypothesis_bank.flagged_diagnostics),
                hypothesis_weights=hypothesis_weights,
                high_residual_vars=high_res_var_names,
                candidate_edges=candidate_edges
            )
    
    def update_from_observation(self, 
                                current_state: Dict[str, float],
                                next_state: Dict[str, float]):
        """
        Complete update cycle:
        1. Make prediction
        2. Compare with observation
        3. Update confidences
        4. Update hypothesis weights
        5. Update model parameters
        """
        # 1. Predict
        predicted = self.predict_next_state(current_state)
        
        # 2. Store data
        data_row = {**current_state, **{f"{k}_next": v for k, v in next_state.items()}}
        self.data_buffer = pd.concat([
            self.data_buffer,
            pd.DataFrame([data_row])
        ], ignore_index=True)
        
        # 3. Update edge confidences
        self.update_confidence(next_state, predicted)
        
        # 4. Update hypothesis weights
        if self.hypotheses:
            self.update_hypothesis_weights(next_state)
        
        # 5. Update model parameters (if enough data)
        if len(self.data_buffer) >= 20:
            recent_data = self.data_buffer.tail(100)
            self.predictive_model.update_parameters(recent_data)
            
            # Also update hypothesis models
            for hyp in self.hypotheses:
                hyp.predictive_model.update_parameters(recent_data)
    
    def get_low_confidence_edges(self, threshold: float = 0.5) -> List[Tuple[str, str, float]]:
        """
        Identify edges that need validation through intervention
        
        Returns:
            List of (source, target, confidence) tuples sorted by confidence (lowest first)
        """
        low_conf_edges = []
        for (source, target), edge_conf in self.edge_confidences.items():
            if edge_conf.confidence < threshold:
                low_conf_edges.append((source, target, edge_conf.confidence))
        
        # Sort by confidence (lowest first)
        low_conf_edges.sort(key=lambda x: x[2])
        return low_conf_edges
    
    def get_confidence_matrix(self) -> pd.DataFrame:
        """Get confidence scores as a matrix for visualization"""
        matrix = pd.DataFrame(0.0, index=self.variables, columns=self.variables)
        
        for (source, target), edge_conf in self.edge_confidences.items():
            if source in matrix.index and target in matrix.columns:
                matrix.loc[source, target] = edge_conf.confidence
        
        return matrix
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get current CWM state for monitoring/debugging"""
        return {
            'num_edges': len(self.graph.edges()),
            'num_hypotheses': len(self.hypotheses),
            'active_hypothesis': self.hypotheses[self.active_hypothesis_idx].name if self.hypotheses else 'None',
            'uncertainty': self.compute_uncertainty(),
            'mean_confidence': np.mean([ec.confidence for ec in self.edge_confidences.values()]) if self.edge_confidences else 0.0,
            'recent_prediction_error': np.mean(list(self.prediction_error_buffer)) if self.prediction_error_buffer else 0.0,
            'data_buffer_size': len(self.data_buffer)
        }


# ============================================================================
# Example Usage and Integration
# ============================================================================

def example_usage():
    """Demonstrate CWM usage"""
    
    # 1. Initialize with a causal graph
    G = nx.DiGraph()
    G.add_edges_from([
        ('Temperature', 'Satisfaction'),
        ('Humidity', 'Satisfaction'),
        ('AirQuality', 'Satisfaction'),
        ('Temperature', 'EnergyConsumption'),
        ('OutdoorTemp', 'Temperature')
    ])
    
    variables = ['Temperature', 'Humidity', 'AirQuality', 'OutdoorTemp', 
                'Satisfaction', 'EnergyConsumption']
    
    cwm = CausalWorldModel(G, variables)
    
    # 2. Add alternative hypothesis (e.g., window open scenario)
    G_window_open = G.copy()
    G_window_open.add_edge('OutdoorTemp', 'Temperature')  # Stronger coupling
    cwm.add_hypothesis('H1_WindowOpen', G_window_open)
    
    # 3. Simulate observations and updates
    for t in range(100):
        # Current state
        current = {
            'Temperature': 22 + np.random.randn(),
            'Humidity': 50 + np.random.randn(),
            'AirQuality': 75 + np.random.randn(),
            'OutdoorTemp': 15 + np.random.randn(),
            'Satisfaction': 70 + np.random.randn(),
            'EnergyConsumption': 30 + np.random.randn()
        }
        
        # Simulate next state (with some dynamics)
        next_state = {k: v + np.random.randn() * 0.5 for k, v in current.items()}
        
        # Update CWM
        cwm.update_from_observation(current, next_state)
        
        if t % 20 == 0:
            summary = cwm.get_state_summary()
            logger.info(f"\nTime {t}: {summary}")
    
    # 4. Identify edges needing intervention
    low_conf = cwm.get_low_confidence_edges(threshold=0.6)
    logger.info(f"\nLow confidence edges: {low_conf}")
    
    # 5. Get confidence matrix
    conf_matrix = cwm.get_confidence_matrix()
    logger.info(f"\nConfidence Matrix:\n{conf_matrix}")

class HypothesisBank:
    """Store rejected edges for potential resurrection during monitoring failures"""
    
    def __init__(self):
        self.rejected_edges = []
        self.flagged_diagnostics = []
    
    def add_rejected_edge(self, source, target, rejection_reason, intervention_data=None):
        """Add edge that failed validation"""
        self.rejected_edges.append({
            'edge': (source, target),
            'rejection_reason': rejection_reason,
            'intervention_data': intervention_data,
            'timestamp': len(self.rejected_edges)
        })
        logger.info(f"📦 Banked: {source}→{target} ({rejection_reason})")
    
    def query_edges_for_variable(self, variable):
        """Find banked edges that could explain poor fit for variable"""
        candidates = [e for e in self.rejected_edges 
                     if e['edge'][1] == variable]
        return candidates
    
    def flag_diagnostic_event(self, timestamp, hypothesis_weights, high_residual_vars, candidate_edges):
        """Log monitoring failure for manual review"""
        import json
        
        self.flagged_diagnostics.append({
            'timestamp': timestamp,
            'hypothesis_weights': hypothesis_weights,
            'high_residual_variables': high_residual_vars,
            'candidate_edges': [e['edge'] for e in candidate_edges]
        })
        
        with open('flagged_diagnostics.json', 'w') as f:
            json.dump(self.flagged_diagnostics, f, indent=2)
        
        logger.warning(f"   DIAGNOSTIC FLAGGED: All hypotheses failing")
        logger.warning(f"   Weights: {hypothesis_weights}")
        logger.warning(f"   High residuals: {high_residual_vars}")
        logger.warning(f"   Candidate edges: {[e['edge'] for e in candidate_edges]}")
        logger.warning(f"   Saved to flagged_diagnostics.json")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_usage()