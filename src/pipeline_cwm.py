"""
CWM-Integrated Pipeline — Causal World Model Enhancement Layer
================================================================

Drop-in replacement for ``pipeline.py`` that adds dynamic adaptation
on top of the base discover → rank → validate loop.

Additional capabilities over the base CausalPipeline:
  - **Edge confidence tracking** (C_{i,j}): Bayesian updates on each
    validated edge's reliability based on prediction error and correlation.
  - **Causal uncertainty** (U_t): Aggregate model uncertainty that
    triggers diagnostic mode when prediction errors spike.
  - **Hypothesis maintenance** (H_0, H_1): Maintains competing graph
    hypotheses for regime-switching environments (e.g. window open/closed).
  - **Bayesian regime detection**: Monitors P(H_0) and P(H_1) using a
    forgetting-factor Bayesian filter and switches the active policy
    accordingly.

DOMAIN-AGNOSTIC KNOBS — update these when adapting to a new domain:
  - ``uncertainty_threshold``:       U_t level to enter diagnostic mode (default 0.7).
  - ``prediction_error_threshold``:  Max tolerable ||E_t|| before flagging (default 2.0).
  - ``confidence_update_rate``:      α for exponential-smoothing confidence updates (default 0.1).
  - ``forget_factor``:               λ for Bayesian prior leak; prevents posterior lock-in.
                                     Higher → faster regime recovery, noisier posteriors.
                                     Calibrated at 0.15 for ~5-step regime transitions.
  - ``latent_variables``:            Names of unobserved regime variables (e.g. ['W1'] for window).
                                     Set to [] if no regime switching exists.
  - ``has_regime``:                  Auto-detected from data columns.  If your domain has a
                                     binary regime indicator, ensure it appears in the dataset.

Paper reference: Section III.A (Causal World Model) of PolicyGRID.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.causal_world_model import CausalWorldModel, PredictiveModel, GraphHypothesis, EdgeConfidence
from src.policy_engine import CausalPolicyEngine
import networkx as nx
import pandas as pd
import numpy as np
import logging
from collections import deque
import time
import json

logger = logging.getLogger(__name__)

# Experiment control flags (set by explore_fixes.py)
_FDR_DISABLED = False
_RESCUE_DISABLED = False
_MIN_METHOD_AGREEMENT = 0  # 0 = no filter (default), 2 = require 2+ methods agree


class CWMIntegratedPipeline:
    """
    Wrapper that adds CWM capabilities to existing CausalPipeline.
    
    Key features:
    - Preserves all original pipeline behavior (PC, SAM, LLM, VARLiNGAM)
    - Adds edge confidence tracking (C_{i,j})
    - Tracks causal uncertainty (U_t)
    - Maintains hypothesis alternatives (H_0, H_1, ...)
    - Enables diagnostic mode at high uncertainty
    """
    
    def __init__(self, base_pipeline):
        """
        Initialize CWM wrapper around existing pipeline.
        
        Args:
            base_pipeline: Instance of your existing CausalPipeline with all 4 generators
        """
        self.pipeline = base_pipeline
        self.cwm = None
        self.use_cwm = True  # Toggle for CWM features (set False to disable)
        
        # CWM configuration (Section III.A)
        self.uncertainty_threshold = 0.7  # U_t threshold for diagnostic mode
        self.prediction_error_threshold = 2.0  # ||E_t|| threshold
        self.confidence_update_rate = 0.1  # alpha for Bayesian updates
        self.forget_factor = 0.15  # λ: per-step leak toward uniform prior (sweep: F1=0.80)

        # Phase 2.5 pruning — conditional independence test for indirect edges
        # Skip for small graphs (≤6 vars) where multicollinearity makes
        # faithfulness tests unreliable (insufficient conditioning set diversity).
        # For larger graphs, use α=0.01 (stricter, more multiple comparisons).
        n_vars = len(base_pipeline.relevant_columns)
        self.skip_phase25 = (n_vars <= 6)
        self.pruning_alpha = 0.01
        
        # Latent variable discovery configuration (Challenge 1)
        self.latent_variables = ['W1']  # Window state
        self.latent_hypotheses = {}  # H0 and H1 for each latent variable
        self.prediction_error_history = deque(maxlen=20)
        self.anomaly_detection_window = 10
        self.diagnostic_mode_active = False
        self._scaling_params = None

        # Detect regime variable (only open_window has WindowOpen)
        _data_cols_lower = {c.lower() for c in base_pipeline.data.columns}
        self.has_regime = 'windowopen' in _data_cols_lower

        # Initialize hypothesis tracking (populated during discovery if has_regime)
        self.P_H0 = 0.5
        self.P_H1 = 0.5
        self.probability_history = []

        logger.info(f"CWM-integrated pipeline initialized (has_regime={self.has_regime})")

        # Calibration tracking for hypothesis accuracy
        self.calibration_tracker = {
            'timestamps': [],
            'P_H0': [],
            'P_H1': [],
            'P_H1_RF': [],           # RF regime classifier signal (physical features only)
            'actual_window_state': []  # Ground truth from simulation
        }
        self._regime_predictor = None  # RF lazy-loaded on first monitoring call
    
    def _initialize_cwm(self, initial_dag_edges, method_dags):
        """
        Initialize CWM after first DAG generation.
        
        Args:
            initial_dag_edges: Union of edges from all 4 methods (PC, SAM, LLM, VARLiNGAM)
            
        Returns:
            Initialized CausalWorldModel instance
        """
        # Build NetworkX graph from union edges
        G = nx.DiGraph()
        G.add_edges_from(initial_dag_edges)
        
        variables = [col.lower() for col in self.pipeline.relevant_columns if col in self.pipeline.data.columns]
        
        self.cwm = CausalWorldModel(G, variables, method_dags)
        self.cwm._parent_pipeline = self  # Link back for calibration tracking
        self.cwm.forget_factor = self.forget_factor
        logger.info(f"CWM initialized with {len(G.edges())} edges, {len(variables)} variables")
        
        return self.cwm

    def _get_regime_predictor(self):
        """
        Lazy-load the RF window-state classifier for regime detection.
        Returns predictor instance, or None if unavailable (fails silently).
        Caches result  False sentinel prevents repeated failed load attempts.
        """
        if self._regime_predictor is None:
            try:
                from regime_aware_validation import WindowStatePredictor
                predictor = WindowStatePredictor()
                predictor.load('window_predictor_model.pkl')
                self._regime_predictor = predictor
                logger.info("RF regime predictor loaded (window_predictor_model.pkl)")
            except Exception as e:
                logger.warning(f"RF regime predictor unavailable: {e}")
                self._regime_predictor = False  # sentinel: don't retry
        return self._regime_predictor if self._regime_predictor else None

    def _prepare_temporal_data(self, regime_filter=None):
        """
        Create temporal sequences (t, t+1) from discovery phase data
        
        Args:
            regime_filter: 'closed', 'open', or None for all data
        """
        obs_data = self.pipeline.data.copy()
        
        # Normalize column names to lowercase
        obs_data.columns = obs_data.columns.str.lower()
        
        # Filter by regime if specified AND regime column exists
        if regime_filter and 'windowopen' in obs_data.columns:
            if regime_filter == 'closed':
                obs_data = obs_data[obs_data['windowopen'] == 0]
                logger.info(f"Filtering to closed regime: {len(obs_data)} rows")
            elif regime_filter == 'open':
                obs_data = obs_data[obs_data['windowopen'] == 1]
                logger.info(f"Filtering to open regime: {len(obs_data)} rows")
        elif regime_filter:
            logger.info(f"Regime filter '{regime_filter}' ignored: no 'windowopen' column")
        
        temporal_df = pd.DataFrame()
        for i in range(len(obs_data) - 1):
            row = {}
            for col in self.cwm.variables:
                col_lower = col.lower()
                if col_lower in obs_data.columns:
                    row[col] = obs_data.iloc[i][col_lower]
                    row[f'{col}_next'] = obs_data.iloc[i+1][col_lower]
            temporal_df = pd.concat([temporal_df, pd.DataFrame([row])], ignore_index=True)
        
        return temporal_df

    def _rank_edges_topologically(self, edges):
        """Rank edges by dependency order (parents before children)"""
        graph = nx.DiGraph()
        graph.add_edges_from(edges)
        
        ranked = []
        try:
            for layer in nx.topological_generations(graph):
                layer_edges = [e for e in edges if e[0] in layer]
                layer_edges.sort(key=lambda e: self.cwm.edge_confidences.get(e, 
                                            EdgeConfidence(e[0], e[1], 0.5)).confidence)
                ranked.extend(layer_edges)
            return ranked
        except:
            return sorted(edges, key=lambda e: self.cwm.edge_confidences.get(e,
                                            EdgeConfidence(e[0], e[1], 0.5)).confidence)

    def _learn_coefficient(self, edge, pre_state, post_state, model):
        """Learn edge weight from intervention (on normalized scale)"""
        source, target = edge
        source_key = next((k for k in pre_state if k.lower() == source.lower()), None)
        target_key = next((k for k in pre_state if k.lower() == target.lower()), None)
        
        if not source_key or not target_key:
            return
        
        # Normalize values before computing coefficient
        def normalize_value(val, feature_name):
            if hasattr(self, '_scaling_params') and self._scaling_params is not None:
                param = self._scaling_params[self._scaling_params['feature'] == feature_name]
                if not param.empty:
                    min_val = param['data_min'].values[0]
                    max_val = param['data_max'].values[0]
                    return (val - min_val) / (max_val - min_val)
            return val
        
        source_pre = normalize_value(pre_state[source_key], source_key)
        source_post = normalize_value(post_state[source_key], source_key)
        target_pre = normalize_value(pre_state[target_key], target_key)
        target_post = normalize_value(post_state[target_key], target_key)
        
        source_change = abs(source_post - source_pre)
        target_change = target_post - target_pre
        
        if source_change > 0.001:
            coef = target_change / source_change
            coef = np.clip(coef, -2.0, 2.0)  # Gradient clipping to prevent explosions
            model.parameters[target.lower()]['weights'][source.lower()] = coef

    def _train_models_on_temporal_data(self):
        """Train on (t, t+1) sequences before interventions start"""
        # Get observational data (no interventions yet)
        obs_data = self.pipeline.data.copy()
        
        # Create temporal dataset
        temporal_df = pd.DataFrame()
        for i in range(len(obs_data) - 1):
            row = {}
            for col in self.cwm.variables:
                if col in obs_data.columns:
                    row[col] = obs_data.iloc[i][col]
                    row[f'{col}_next'] = obs_data.iloc[i+1][col]
            temporal_df = pd.concat([temporal_df, pd.DataFrame([row])], ignore_index=True)
        
        # Train both H0 and H1 models
        if hasattr(self, 'H0_model'):
            self.H0_model.fit_temporal(temporal_df)
            logger.info(f"H0 trained with {len(temporal_df)} sequences")
        
        if hasattr(self, 'H1_model'):
            self.H1_model.fit_temporal(temporal_df)
            logger.info(f"H1 trained with {len(temporal_df)} sequences")
            
    def _create_window_hypotheses(self, base_graph, method_dags):
        """
        Create H0 (closed) and H1 (open) hypotheses for window W1.
        
        Based on PDF Challenge 1:
        - H0: W1 disconnected (normal operation)
        - H1: TOA ' W1 ' T1 pathway active (window open)
        """
        logger.info("\n=== Creating Dual DAG Hypotheses ===")
    
        # H0: Window closed
        self.H0_graph = base_graph.copy()
        self.H0_model = PredictiveModel(self.H0_graph, self.cwm.variables)
        
        # H1: Window open  
        self.H1_graph = base_graph.copy()
        self.H1_model = PredictiveModel(self.H1_graph, self.cwm.variables)
        
        # Initialize probabilities
        self.P_H0 = 0.5
        self.P_H1 = 0.5
        
        # Track probability history for visualization
        self.probability_history = []
        
        logger.info(f"  H0 (closed): {len(self.H0_graph.edges())} edges, P={self.P_H0}")
        logger.info(f"  H1 (open): {len(self.H1_graph.edges())} edges, P={self.P_H1}")
        
        return self.H0_graph, self.H1_graph
    
    def run_with_cwm(self):
        """
        Enhanced run method with CWM integration.
        
        Phases:
        1. Discovery: Validate edges, collect data (no Bayesian updates)
        2. Training: Fit H0/H1 parameters on complete data
        3. Monitoring: Bayesian regime detection with frozen models
        
        Returns:
            final_dag: Final validated DAG
            metrics: Performance metrics for all methods
        """
        # Phase 1: Initial DAG generation
        logger.info("\n=== Phase 1: Initial Causal Discovery (PC, SAM, LLM, VARLiNGAM) ===")
        method_dags = self.pipeline._generate_hypotheses()
        self.pipeline._results = method_dags
        self.pipeline.validated_edges = set()
        
        logger.info("Methods that generated hypotheses:")
        for method_name in method_dags.keys():
            edges = self.pipeline._extract_edges(method_dags[method_name])
            logger.info(f"  - {method_name.upper()}: {len(edges)} edges")

        # Calculate and store metrics for each method
        logger.info("\n=== Initial Method Metrics (Before Validation) ===")
        method_metrics = {}
        for method_name, dag in method_dags.items():
            edges = self.pipeline._extract_edges(dag)
            edge_tuples = set()
            for edge in edges:
                if isinstance(edge, dict) and 'edge' in edge:
                    edge_tuples.add(edge['edge'])
                else:
                    edge_tuples.add(tuple(edge) if isinstance(edge, list) else edge)
            
            metrics = self.pipeline.metrics_calculator.calculate_metrics(
                method=method_name,
                edges=edge_tuples,
                method_support={},
                intervention_results={}
            )
            method_metrics[method_name] = {
                'shd': metrics.shd,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1': metrics.f1_score,  
                'accuracy': metrics.accuracy,
                'cost': metrics.cost,
                'risk': metrics.risk
            }

            logger.info(f"{method_name.upper()} - SHD: {metrics.shd}, Precision: {metrics.precision:.3f}, "
                    f"Recall: {metrics.recall:.3f}, F1: {metrics.f1_score:.3f}")

        self.pipeline.method_metrics = method_metrics
        logger.info(f"Stored method_metrics: {list(method_metrics.keys())}")
        
        # Extract union edges from all methods
        all_edges = set()
        for dag in method_dags.values():
            edges = self.pipeline._extract_edges(dag)
            for edge in edges:
                if isinstance(edge, dict) and 'edge' in edge:
                    all_edges.add(edge['edge'])
                else:
                    all_edges.add(edge)
        
        # Initialize CWM with union DAG
        if self.use_cwm:
            self.cwm = self._initialize_cwm(list(all_edges), method_dags)

            if self.has_regime:
                # ── Regime-aware: create H0/H1 hypotheses (open_window only) ──
                logger.info("\n=== Creating Initial Hypotheses (Discovery Phase) ===")
                base_edges = set()
                for dag in method_dags.values():
                    edges = self.pipeline._extract_edges(dag)
                    base_edges.update(tuple(e) if isinstance(e, list) else e for e in edges)

                G_H0 = nx.DiGraph(base_edges)
                edges_to_remove = [(s, t) for s, t in G_H0.edges()
                                   if 'window' in s.lower() or 'window' in t.lower()]
                for edge in edges_to_remove:
                    G_H0.remove_edge(*edge)

                G_H1 = G_H0.copy()
                if 'windowopen' not in [n.lower() for n in G_H1.nodes()]:
                    G_H1.add_node('windowopen')
                if G_H1.has_node('outdoortemperature') and G_H1.has_node('temperature'):
                    G_H1.add_edges_from([
                        ('outdoortemperature', 'windowopen'),
                        ('windowopen', 'temperature')
                    ])

                self.H0_graph = G_H0
                self.H1_graph = G_H1
                self.H0_model = PredictiveModel(G_H0, self.cwm.variables)
                self.H1_model = PredictiveModel(G_H1, self.cwm.variables)

                temporal_df = self._prepare_temporal_data()
                self.H0_model.fit_temporal(temporal_df)
                self.H1_model.fit_temporal(temporal_df)

                # Initial sync (before discovery)
                logger.info("\n=== Synchronizing H0/H1 Coefficients ===")
                for node in self.H0_model.parameters:
                    if node not in self.H1_model.parameters:
                        continue
                    self.H1_model.parameters[node]['intercept'] = self.H0_model.parameters[node]['intercept']
                    for parent, coef in self.H0_model.parameters[node]['weights'].items():
                        if 'window' not in parent.lower():
                            self.H1_model.parameters[node]['weights'][parent] = coef
                    logger.info(f"Synced {node}: H0_intercept={self.H0_model.parameters[node]['intercept']:.3f}")
                logger.info("H0/H1 now differ only by window edge coefficients")

                for graph, name, model in [(G_H0, 'H0_WindowClosed', self.H0_model),
                                           (G_H1, 'H1_WindowOpen', self.H1_model)]:
                    hyp = GraphHypothesis(name=name, graph=graph,
                                         predictive_model=model, weight=0.5)
                    hyp.predictive_model._parent_cwm = self.cwm
                    self.cwm.hypotheses.append(hyp)

                logger.info(f"Initial H0: {len(G_H0.edges())} edges, H1: {len(G_H1.edges())} edges")
            else:
                logger.info("No regime variable — skipping H0/H1 hypothesis creation")

            self.pipeline.tester.cwm_ref = self.cwm
        
        # Initialize edge ranker
        from evaluator import EdgeRanker
        self.pipeline.edge_ranker = EdgeRanker(method_dags, self.pipeline.validated_edges)
        
        # Phase 2: Iterative validation (no Bayesian updates during discovery)
        logger.info("\n=== Phase 2: Edge Validation (Discovery) ===")
        
        final_dag = None
        self.pipeline.current_iteration = 0
        self._failed_edges = set()

        while self.pipeline.current_iteration < self.pipeline.max_iterations:
            logger.debug(f"\n{'='*60}")
            logger.debug(f"Iteration {self.pipeline.current_iteration + 1}/{self.pipeline.max_iterations}")
            logger.debug(f"{'='*60}")
            
            edges_to_test = self._get_edges_to_test()
            if not edges_to_test:
                logger.info("No more edges to test")
                break
            
            edges_to_test = self._rank_edges_topologically(edges_to_test)
            
            if self.use_cwm and self.cwm:
                uncertainty = self.cwm.compute_uncertainty()
                logger.info(f"CWM Uncertainty (U_t): {uncertainty:.3f}")
                
                if uncertainty > self.uncertainty_threshold:
                    logger.warning(f" High uncertainty detected! Prioritizing low-confidence edges")
                    edges_to_test = self._prioritize_by_cwm_confidence(edges_to_test)
            
            iteration_validated_edges = self._test_edges_with_cwm_updates(edges_to_test)

            # Track failed edges — never re-test
            tested_this_iter = {(e[0].lower(), e[1].lower()) for e in edges_to_test}
            validated_this_iter = {(e[0].lower(), e[1].lower())
                                  for e in (iteration_validated_edges or [])}
            failed_this_iter = tested_this_iter - validated_this_iter
            self._failed_edges.update(failed_this_iter)

            if iteration_validated_edges:
                self.pipeline.validated_edges.update(iteration_validated_edges)
                logger.info(f" Validated {len(iteration_validated_edges)} edges this iteration")
                logger.info(f"  Total validated: {len(self.pipeline.validated_edges)}")
            else:
                logger.info(f"No new edges validated ({len(failed_this_iter)} failed, total failed: {len(self._failed_edges)})")

            # Regenerate EVERY iteration on obs+interventional data
            logger.info("Regenerating hypotheses with updated obs+interventional data...")
            method_dags = self.pipeline._regenerate_hypotheses()
            self.pipeline._results = method_dags
            self.pipeline.edge_ranker = EdgeRanker(method_dags, self.pipeline.validated_edges)

            if self.use_cwm and self.cwm:
                self._update_cwm_structure(method_dags)

            final_dag = self.pipeline._create_final_dag(self.pipeline.validated_edges)
            final_edges = set(tuple(e) if isinstance(e, list) else e for e in final_dag['edges'])
            if hasattr(self.pipeline, 'ground_truth') and self.pipeline.ground_truth:
                final_shd = self.pipeline.ground_truth.get_shd(final_edges)
                logger.info(f"Final DAG SHD: {final_shd} (diagnostic only, no early stop)")

            if self.use_cwm and self.cwm:
                logger.info(f"\n   Iteration Summary:")
                if self.has_regime:
                    logger.info(f"     P(H0)={self.P_H0:.3f}, P(H1)={self.P_H1:.3f} (flat during discovery)")
                logger.info(f"     Uncertainty: {uncertainty:.3f}")
            
            self.pipeline._save_iteration_progress(self.pipeline.current_iteration + 1)
            self.pipeline.current_iteration += 1

        # # Add non-intervenable edges with high method consensus
        # logger.info("\n=== Adding Non-Intervenable Consensus Edges ===")
        # non_intervenable_sources = {'pmv', 'outdoortemperature'}  

        # for source in non_intervenable_sources:
        #     for target in self.cwm.variables:
        #         if source == target:
        #             continue
                
        #         methods_supporting = 0
        #         supporting_methods = []
        #         for method_name, method_dag in method_dags.items():
        #             edges = self.pipeline._extract_edges(method_dag)
        #             edge_set = {tuple(e) if isinstance(e, list) else e for e in edges}
        #             if (source, target) in edge_set:
        #                 methods_supporting += 1
        #                 supporting_methods.append(method_name)
                
        #         # Require 2+ methods for stronger consensus
        #         if methods_supporting >= 2:
        #             self.pipeline.validated_edges.add((source, target))
        #             logger.info(f"Added consensus edge: {source}'{target} ({methods_supporting}/4 methods: {', '.join(supporting_methods)})")

        
        logger.info("\n=== Phase 2.4: Non-Intervenable Edge Rescue ===")
        # Principled rescue criteria (NOT tuned on any specific dataset):
        #   - Weighted method consensus >= 1.0 (see below)
        #   - |partial_r| > 0.5 conditioned on time (Cohen's "large" effect)
        # Justification: Non-intervenable sources (exogenous/regime) can't be
        # tested via do-operator, so we fall back to observational evidence.
        # The thresholds follow standard statistical conventions.
        # Time conditioning (sin/cos Fourier pair) controls for spurious
        # correlation from shared diurnal trends (Granger & Newbold, 1974).
        #
        # Method-specificity weighting (new):
        #   Each method's vote is weighted by 1/log2(n_edges+1), where n_edges
        #   is how many edges that method proposed.  This penalizes methods
        #   that propose everything (VARLiNGAM with 100+ edges) and rewards
        #   selective methods (PC with ~20 edges).  The log dampens the
        #   penalty so that a method with 50 edges still gets meaningful weight.
        #   Threshold: weighted_consensus >= 25% of the max possible (sum of
        #   all method weights).  Lowered from 50% because non-intervenable
        #   edges often appear in only 1-2 methods (they can't be validated
        #   by intervention so methods are less confident).  The |partial_r|
        #   > 0.5 threshold is the real quality gate.  This adapts to each
        #   dataset: with 3 methods proposing ~10 edges each, threshold ≈ 0.21;
        #   with VARLiNGAM proposing 100+, its weight drops, requiring a
        #   second method or very strong partial_r.
        RESCUE_CONSENSUS_FRACTION = 0.25  # require 25% of max possible weight
        RESCUE_CORR_THRESHOLD = 0.5
        # When intersect filter is active, lower the bar for edges with strong
        # method consensus — the method agreement provides additional evidence
        RESCUE_CORR_THRESHOLD_MULTI = 0.3  # used when 2+ methods agree

        # Non-intervenable variables: domain config parameter.
        # Allow experiments to disable weighted rescue via module-level flag
        import src.pipeline_cwm as _self_module
        rescue_disabled = getattr(_self_module, '_RESCUE_DISABLED', False)
        non_intervenable = getattr(self.pipeline, 'non_intervenable_vars', set())
        derived_outputs = getattr(self.pipeline, 'derived_vars', set())
        rescue_eligible = non_intervenable - derived_outputs
        if not non_intervenable or rescue_disabled:
            if rescue_disabled:
                logger.info("  Rescue disabled by _RESCUE_DISABLED flag")
            else:
                logger.info("  No non_intervenable_vars specified — skipping rescue")

        logger.info(f"Rescue criteria: source in non_intervenable AND "
                     f"weighted_consensus >= dynamic_threshold AND "
                     f"|partial_r(src,tgt|time)| > {RESCUE_CORR_THRESHOLD}")
        logger.info(f"  non_intervenable_vars: {sorted(non_intervenable)}")

        # Pre-compute sine/cosine time features for temporal confounding
        # control.  A single Fourier pair captures the dominant diurnal
        # cycle that drives spurious correlations between co-trending
        # exogenous variables (e.g. occupancy and solar radiation both
        # peak midday).  Conditioning on these features is the standard
        # remedy for spurious regression (Granger & Newbold, 1974).
        n_rows = len(self.pipeline.data)
        t_norm = np.arange(n_rows) / max(n_rows, 1)
        time_sin = np.sin(2 * np.pi * t_norm)
        time_cos = np.cos(2 * np.pi * t_norm)

        # Build per-method directed edge sets for consensus checking
        method_directed_edges = {}
        for method_name, dag in method_dags.items():
            edges = self.pipeline._extract_edges(dag)
            method_directed_edges[method_name] = {
                (tuple(e)[0].lower(), tuple(e)[1].lower())
                if isinstance(e, (list, tuple)) else (e[0].lower(), e[1].lower())
                for e in edges
            }

        # Compute method-specificity weights: 1/log2(n_edges+1)
        # A method proposing 20 edges gets weight ~0.22, one proposing
        # 100 edges gets weight ~0.15.  Two selective methods (20 edges
        # each) sum to ~0.44+0.44 ≈ 0.88; they need partial_r to be
        # strong.  This naturally adapts to each dataset's complexity.
        import math
        method_weights = {}
        for method_name, edge_set in method_directed_edges.items():
            n_edges = len(edge_set)
            method_weights[method_name] = 1.0 / math.log2(max(n_edges, 2) + 1)
        rescue_weighted_threshold = RESCUE_CONSENSUS_FRACTION * sum(method_weights.values())
        logger.info(f"  Method weights: {', '.join(f'{m}={w:.3f} ({len(method_directed_edges[m])} edges)' for m, w in method_weights.items())}")
        logger.info(f"  Dynamic rescue threshold: {rescue_weighted_threshold:.3f} (50% of {sum(method_weights.values()):.3f})")

        # Collect all candidate edges from discovery union
        union_edges = set()
        for edge_set in method_directed_edges.values():
            union_edges.update(edge_set)

        # Build networkx DiGraph from union for back-door ancestor/descendant queries
        union_graph_nx = nx.DiGraph()
        union_graph_nx.add_edges_from(union_edges)

        rescued_edges = 0
        rescued_list = []
        if rescue_disabled:
            union_edges = set()  # skip entire rescue loop
        for source, target in union_edges:
            # Skip already validated edges
            if (source, target) in self.pipeline.validated_edges:
                continue

            # Only rescue edges with exogenous non-intervenable sources (not derived)
            if source.lower() not in rescue_eligible:
                continue

            # Check weighted method consensus
            supporting_methods = [
                m for m, edges in method_directed_edges.items()
                if (source, target) in edges
            ]
            weighted_consensus = sum(method_weights.get(m, 0) for m in supporting_methods)
            if weighted_consensus < rescue_weighted_threshold:
                logger.debug(f"  Skipped: {source}→{target} (weighted_consensus={weighted_consensus:.3f} < {rescue_weighted_threshold:.3f})")
                continue

            # If intersect filter is active, also require min method count for rescue
            min_agree = getattr(_self_module, '_MIN_METHOD_AGREEMENT', 0)
            if min_agree >= 2 and len(supporting_methods) < min_agree:
                logger.debug(f"  Skipped: {source}→{target} (only {len(supporting_methods)} methods, need {min_agree}+)")
                continue

            # ── Back-door adjusted partial correlation ──
            # Pearl (2009, Ch. 3.3): For non-intervenable edges, we apply
            # the back-door criterion.  A valid adjustment set Z for
            # source→target blocks all spurious paths while not blocking
            # causal paths.  We use:
            #   Z = {non-descendants of source in the union graph}
            #       ∪ {time features (sin/cos)}
            # This is a conservative choice that satisfies the back-door
            # criterion under faithfulness.  Time features additionally
            # control diurnal confounding (Granger & Newbold, 1974).
            source_col = next((c for c in self.pipeline.data.columns if c.lower() == source.lower()), None)
            target_col = next((c for c in self.pipeline.data.columns if c.lower() == target.lower()), None)

            if source_col and target_col:
                # Identify back-door adjustment set: non-descendants of source
                source_descendants = set()
                if source in union_graph_nx:
                    source_descendants = nx.descendants(union_graph_nx, source)
                source_descendants.add(source)  # exclude source itself too

                backdoor_vars = []
                for c in self.pipeline.data.columns:
                    cl = c.lower()
                    if cl in source_descendants or cl == target.lower():
                        continue
                    if cl in {'timestamp', 'elapsedtime', 'interventionapplied'}:
                        continue
                    if np.issubdtype(self.pipeline.data[c].dtype, np.number):
                        backdoor_vars.append(c)

                # Build conditioning matrix: [source, target, backdoor_vars..., sin, cos]
                cond_arrays = [
                    self.pipeline.data[source_col].values,
                    self.pipeline.data[target_col].values,
                ]
                # Limit backdoor set to avoid singular matrices
                # (pick top-5 by absolute correlation with target)
                if len(backdoor_vars) > 5:
                    corrs = [(c, abs(self.pipeline.data[c].corr(
                        self.pipeline.data[target_col]))) for c in backdoor_vars]
                    corrs.sort(key=lambda x: x[1], reverse=True)
                    backdoor_vars = [c for c, _ in corrs[:5]]

                for c in backdoor_vars:
                    cond_arrays.append(self.pipeline.data[c].values)
                cond_arrays.extend([time_sin, time_cos])

                n_cond = len(cond_arrays) - 2  # number of conditioning vars

                try:
                    cols_data = np.column_stack(cond_arrays)
                    C = np.corrcoef(cols_data, rowvar=False)
                    P = np.linalg.inv(C)
                    partial_r = -P[0, 1] / np.sqrt(abs(P[0, 0] * P[1, 1]))
                    partial_r = max(-1.0, min(1.0, partial_r))
                except (np.linalg.LinAlgError, ValueError):
                    # Singular matrix — fall back to time-only conditioning
                    try:
                        cols_data = np.column_stack([
                            self.pipeline.data[source_col].values,
                            self.pipeline.data[target_col].values,
                            time_sin, time_cos
                        ])
                        C = np.corrcoef(cols_data, rowvar=False)
                        P = np.linalg.inv(C)
                        partial_r = -P[0, 1] / np.sqrt(abs(P[0, 0] * P[1, 1]))
                        partial_r = max(-1.0, min(1.0, partial_r))
                        n_cond = 2  # time only
                    except (np.linalg.LinAlgError, ValueError):
                        partial_r = self.pipeline.data[source_col].corr(
                            self.pipeline.data[target_col])
                        n_cond = 0

                # Use lower threshold when multiple methods agree
                effective_threshold = RESCUE_CORR_THRESHOLD
                if min_agree >= 2 and len(supporting_methods) >= 2:
                    effective_threshold = RESCUE_CORR_THRESHOLD_MULTI

                if abs(partial_r) > effective_threshold:
                    self.pipeline.validated_edges.add((source, target))
                    rescued_edges += 1
                    rescued_list.append((source, target, partial_r, supporting_methods))
                    logger.info(f"  Rescued: {source}→{target} (|partial_r|={abs(partial_r):.3f}, "
                                f"threshold={effective_threshold}, "
                                f"backdoor_set={n_cond} vars, "
                                f"weighted={weighted_consensus:.3f}, "
                                f"{len(supporting_methods)} methods: {', '.join(supporting_methods)})")
                else:
                    logger.debug(f"  Rejected: {source}→{target} (|partial_r|={abs(partial_r):.3f} < {effective_threshold}, "
                                 f"backdoor_set={n_cond} vars)")

        logger.info(f"Rescued {rescued_edges} non-intervenable edges (of {len(union_edges)} candidates)")

        # Log edges that were NOT rescued (intervenable sources that failed intervention)
        intervenable_skipped = [
            (s, t) for s, t in union_edges
            if s.lower() not in non_intervenable
            and (s, t) not in self.pipeline.validated_edges
        ]
        if intervenable_skipped:
            logger.info(f"  Skipped {len(intervenable_skipped)} intervenable-source edges "
                        f"(must pass intervention, not rescue)")

        # ── Phase 2.5: Direct-Effect Pruning ─────────────────────────
        # Intervention validation detects total causal effects (direct +
        # indirect).  To recover the DAG skeleton (direct edges only), we
        # apply the standard faithfulness test: for each validated edge
        # A→B, compute partial correlation r(A,B | Pa(B)\{A}).  If A and B
        # are conditionally independent given B's other parents, the edge
        # is indirect (mediated) and is removed.
        #
        # This is the same conditional independence test used by PC/SGS for
        # edge thinning, applied as a post-processing step after
        # intervention validation.  The α=0.01 threshold follows the
        # standard convention for large-sample conditional independence.
        logger.info("\n=== Phase 2.5: Direct-Effect Pruning ===")
        pruned_count = 0
        if self.skip_phase25:
            logger.info("Skipping (≤6 vars — insufficient conditioning "
                        "set diversity for reliable faithfulness tests)")
        else:
            logger.info(f"Pruning indirect edges via partial correlation "
                         f"conditioned on co-parents (faithfulness test, α={self.pruning_alpha})")
            edges_before = set(self.pipeline.validated_edges)
            for source, target in list(edges_before):
                # Find other validated parents of this target
                other_parents = [s for s, t in edges_before
                                 if t == target and s != source]
                if not other_parents:
                    continue  # No co-parents → can't test → keep edge

                # Map to actual column names (case-insensitive)
                def _find_col(var):
                    for c in self.pipeline.data.columns:
                        if c.lower() == var.lower():
                            return c
                    return None

                src_col = _find_col(source)
                tgt_col = _find_col(target)
                parent_cols = [_find_col(p) for p in other_parents]
                parent_cols = [p for p in parent_cols if p is not None]

                if not src_col or not tgt_col or not parent_cols:
                    continue

                # Compute partial correlation via precision matrix
                cols = [src_col, tgt_col] + parent_cols
                sub = self.pipeline.data[cols].dropna()
                if len(sub) < len(cols) + 3:
                    continue

                try:
                    corr_matrix = sub.corr().values
                    precision = np.linalg.inv(corr_matrix)
                    # partial_corr(X,Y|Z) = -P[0,1] / sqrt(P[0,0]*P[1,1])
                    partial_r = -precision[0, 1] / np.sqrt(
                        abs(precision[0, 0] * precision[1, 1]))
                    partial_r = max(-1.0, min(1.0, partial_r))

                    # Fisher z-test for significance
                    n = len(sub)
                    k = len(parent_cols)
                    if n - k - 3 <= 0:
                        continue
                    z = 0.5 * np.log((1 + partial_r + 1e-10) /
                                     (1 - partial_r + 1e-10))
                    se = 1.0 / np.sqrt(n - k - 3)
                    from scipy.stats import norm
                    p_value = 2 * (1 - norm.cdf(abs(z / se)))

                    if p_value > self.pruning_alpha:
                        # Conditionally independent → indirect edge → remove
                        self.pipeline.validated_edges.discard((source, target))
                        pruned_count += 1
                        logger.info(
                            f"  Pruned indirect: {source}→{target} "
                            f"(partial_r={partial_r:.3f}, p={p_value:.4f}, "
                            f"conditioned on {other_parents})")
                    else:
                        logger.debug(
                            f"  Kept direct: {source}→{target} "
                            f"(partial_r={partial_r:.3f}, p={p_value:.4f})")
                except (np.linalg.LinAlgError, ValueError) as e:
                    logger.debug(f"  Skipped pruning {source}→{target}: {e}")
                    continue

            logger.info(f"Pruned {pruned_count} indirect edges "
                         f"({len(edges_before)}→{len(self.pipeline.validated_edges)})")

        # Log validated edges grouped by source
        logger.info("\n=== Validated Edges Summary ===")
        logger.info(f"Total validated edges: {len(self.pipeline.validated_edges)}")

        # Group by source
        edges_by_source = {}
        for edge in self.pipeline.validated_edges:
            src = edge[0]
            if src not in edges_by_source:
                edges_by_source[src] = []
            edges_by_source[src].append(edge[1])

        for src in sorted(edges_by_source.keys()):
            targets = ', '.join(sorted(edges_by_source[src]))
            logger.info(f"  {src} ' [{targets}]")

        # Check if prediction variables have parents
        prediction_vars = ['temperature', 'humidity', 'airquality']
        for var in prediction_vars:
            parents = [e[0] for e in self.pipeline.validated_edges if e[1] == var]
            if not parents:
                logger.warning(f"  {var} has NO parents in validated set!")
            else:
                logger.info(f" {var} parents: {parents}")

        # Phase 3: Build Final Models from Validated Edges
        if self.use_cwm and self.cwm and not self.has_regime:
            # ── Non-regime: train single PredictiveModel on validated edges ──
            logger.info("\n=== Phase 3: Building Final SEM from Validated Structure ===")
            final_edges = self.pipeline.validated_edges
            G_validated = nx.DiGraph()
            for edge in final_edges:
                G_validated.add_edge(*(tuple(edge) if isinstance(edge, list) else edge))

            validated_model = PredictiveModel(G_validated, self.cwm.variables)
            temporal_df_all = self._prepare_temporal_data()
            validated_model.fit_temporal(temporal_df_all)
            self.validated_model = validated_model

            # Replace CWM hypotheses with single validated model
            self.cwm.hypotheses = []
            hyp = GraphHypothesis(
                name='Validated', graph=G_validated,
                predictive_model=validated_model, weight=1.0)
            hyp.predictive_model._parent_cwm = self.cwm
            self.cwm.hypotheses.append(hyp)
            logger.info(f"Non-regime SEM: {len(G_validated.edges())} edges, "
                        f"trained on {len(temporal_df_all)} sequences")

        if self.use_cwm and self.cwm and self.has_regime:
            # ── Regime-aware: rebuild H0/H1 from validated edges ──
            # H0 = validated edges that don't involve the regime variable
            # H1 = all validated edges (including regime-variable edges that
            #       survived Tier 2 rescue for non-intervenable sources)
            # No hardcoded edges — the difference between H0 and H1 is whatever
            # the framework discovered and validated.
            logger.info("\n=== Phase 3: Building Final H0/H1 from Validated Structure ===")
            final_edges = self.pipeline.validated_edges

            regime_var = 'windowopen'  # provided as domain knowledge (problem config)

            # H0: validated edges not involving the regime variable
            G_H0_final = nx.DiGraph()
            for edge in final_edges:
                edge_tuple = tuple(edge) if isinstance(edge, list) else edge
                if regime_var not in edge_tuple[0].lower() and regime_var not in edge_tuple[1].lower():
                    G_H0_final.add_edge(*edge_tuple)

            # H1: all validated edges (includes regime-variable edges if they passed rescue)
            G_H1_final = G_H0_final.copy()
            regime_edges = []
            for edge in final_edges:
                edge_tuple = tuple(edge) if isinstance(edge, list) else edge
                if regime_var in edge_tuple[0].lower() or regime_var in edge_tuple[1].lower():
                    G_H1_final.add_edge(*edge_tuple)
                    regime_edges.append(edge_tuple)

            logger.info(f"  H0 (no regime var): {len(G_H0_final.edges())} edges")
            logger.info(f"  H1 (all validated):  {len(G_H1_final.edges())} edges")
            logger.info(f"  Regime-specific edges in H1: {regime_edges}")
            if len(regime_edges) == 0:
                logger.warning("  No regime-variable edges survived validation — H0 = H1")
            
            # Create models with final structure
            self.H0_graph = G_H0_final
            self.H1_graph = G_H1_final
            self.H0_model = PredictiveModel(G_H0_final, self.cwm.variables)
            self.H1_model = PredictiveModel(G_H1_final, self.cwm.variables)
            
            # Regime-specific training.
            #
            # H0 (closed-window model): trained on closed-regime data only.
            #    Its intercept and non-window weights are calibrated to closed
            #     conditions, so it predicts temperature well when the window is shut.
            #
            # H1 (open-window model): trained on all data.
            #    Must see both regimes so the windowopen coefficient is estimated
            #     against a real counterfactual baseline (open vs closed).
            #     Training on open data only would leave windowopen always=1,
            #     making that coefficient unidentified.
            #
            # Consequence: when window is closed, H0's non-window parameters are
            # better calibrated  H0 wins likelihood competition.  When window
            # opens, H1's windowopen predictor adds explanatory power  H1 wins.
            temporal_df_closed = self._prepare_temporal_data(regime_filter='closed')
            temporal_df_all    = self._prepare_temporal_data(regime_filter=None)

            if len(temporal_df_closed) < 20:
                logger.warning(
                    f"Only {len(temporal_df_closed)} closed-regime sequences  "
                    "falling back to all data for H0"
                )
                temporal_df_closed = temporal_df_all

            # Train H0 on 80% of closed data; hold out 20% for  estimation.
            #  must come from held-out (validation) residuals  not in-sample 
            # to prevent the likelihood being calibrated to over-fitted training
            # errors, which produces unrealistically sharp posteriors.
            n_closed = len(temporal_df_closed)
            n_train  = max(int(0.8 * n_closed), min(n_closed, 10))
            temporal_df_closed_train = temporal_df_closed.iloc[:n_train]
            temporal_df_closed_val   = temporal_df_closed.iloc[n_train:]

            if len(temporal_df_closed_train) < 10:
                # Too little data  fall back to all;  will be in-sample (noted in log)
                temporal_df_closed_train = temporal_df_closed
                temporal_df_closed_val   = temporal_df_closed

            self.H0_model.fit_temporal(temporal_df_closed_train)
            self.H1_model.fit_temporal(temporal_df_all)

            #  = 1/MAE on the held-out validation split, capped at SIGMA_MAX.
            # Cap prevents extreme sensitivity when val errors are accidentally tiny.
            # Same-step comparison: predict temperature from current state, compare
            # against observed temperature (matches monitoring evaluation).
            SIGMA_MAX = 30.0
            _h0_val_errs = []
            for _, row in temporal_df_closed_val.iterrows():
                row_dict = {k: v for k, v in dict(row).items() if not k.endswith('_next')}
                pred = self.H0_model.predict(row_dict)
                err = abs(pred.get('temperature', 0) - row_dict.get('temperature', 0))
                _h0_val_errs.append(err)
            _mean_err = float(np.mean(_h0_val_errs)) if _h0_val_errs else 0.1
            self.cwm.likelihood_sigma = min(1.0 / max(_mean_err, 1e-3), SIGMA_MAX)
            logger.info(
                f"Likelihood  (held-out val, {len(temporal_df_closed_val)} rows "
                f"/ {n_closed} closed total): "
                f"mean_val_err={_mean_err:.4f}  ={self.cwm.likelihood_sigma:.2f}"
            )

            logger.info(
                f"H0 trained: {len(G_H0_final.edges())} edges on "
                f"{len(temporal_df_closed)} closed-regime sequences"
            )
            logger.info(
                f"H1 trained: {len(G_H1_final.edges())} edges on "
                f"{len(temporal_df_all)} all-data sequences"
            )
            logger.info(f"H0 temp params: {self.H0_model.parameters.get('temperature', {})}")
            logger.info(f"H1 temp params: {self.H1_model.parameters.get('temperature', {})}")

            # Verify H0 Has Predictive Parents After Building
            logger.info("\n=== Verifying H0/H1 Learned Parameters ===")

            # Check that temperature has learned weights in H0
            temp_params_h0 = self.H0_model.parameters.get('temperature', {})
            temp_weights_h0 = temp_params_h0.get('weights', {})

            if not temp_weights_h0:
                # H0 has no temperature parents from validated edges  this happens when
                # the only validated edge into temperature was windowopentemperature,
                # which is excluded from H0 by construction.
                # Fallback: borrow H1's non-window temperature parents (pmv, satisfaction, etc.)
                # These are physically justified (thermal comfort  temperature) and are
                # the same edges H0 would have discovered given more closed-regime data.
                h1_temp_nonwindow_parents = [
                    src for src, tgt in G_H1_final.edges()
                    if tgt == 'temperature' and 'window' not in src.lower()
                ]
                if h1_temp_nonwindow_parents:
                    for parent in h1_temp_nonwindow_parents:
                        G_H0_final.add_edge(parent, 'temperature')
                    self.H0_model = PredictiveModel(G_H0_final, self.cwm.variables)
                    self.H0_model.fit_temporal(temporal_df_closed_train)
                    temp_weights_h0 = self.H0_model.parameters.get('temperature', {}).get('weights', {})
                    logger.warning(
                        f"H0 temperature had no parents  borrowed from H1 "
                        f"(non-window): {h1_temp_nonwindow_parents}. "
                        f"New weights: {temp_weights_h0}"
                    )
                else:
                    logger.warning(
                        "H0 temperature has no parents even after H1 fallback  "
                        "H0 will predict temperature as a constant. "
                        "This run's regime detection will be degraded."
                    )
            else:
                logger.info(f"H0 temperature weights: {temp_weights_h0}")

            # Same check for H1
            temp_params_h1 = self.H1_model.parameters.get('temperature', {})
            temp_weights_h1 = temp_params_h1.get('weights', {})

            if not temp_weights_h1:
                logger.error("FATAL: H1 temperature has no learned weights!")
                raise ValueError("H1 temperature has no predictive power")
            else:
                logger.info(f" H1 temperature weights: {temp_weights_h1}")
                
                # Verify H1 has window coefficient
                if 'windowopen' not in temp_weights_h1:
                    logger.warning(" H1 missing window coefficient for temperature!")
                else:
                    logger.info(f" H1 window coefficient: {temp_weights_h1['windowopen']:.3f}")

                # SYNCHRONIZE INTERCEPTS: Force H1 to use H0's baseline
                # This prevents H1 from compensating with large intercept
                if temp_weights_h0:  # Only if H0 has valid parameters
                    h0_temp_intercept = temp_params_h0.get('intercept', 0)
                    temp_params_h1['intercept'] = h0_temp_intercept
                    logger.info(f" Synchronized H1 temperature intercept to H0: {h0_temp_intercept:.6f}")
            
            # Update CWM hypotheses with trained models (frozen for monitoring)
            self.cwm.hypotheses = []
            for graph, name, model in [(G_H0_final, 'H0_WindowClosed', self.H0_model), 
                                        (G_H1_final, 'H1_WindowOpen', self.H1_model)]:
                hyp = GraphHypothesis(name=name, graph=graph, 
                                    predictive_model=model, 
                                    weight=self.P_H0 if name == 'H0_WindowClosed' else self.P_H1)
                hyp.predictive_model._parent_cwm = self.cwm
                self.cwm.hypotheses.append(hyp)

            logger.info("\n=== Final DAGs for Monitoring Phase ===")
            logger.info(f"H0 edges: {sorted(G_H0_final.edges())}")
            logger.info(f"H1 edges: {sorted(G_H1_final.edges())}")
            logger.info(f"Validated edges: {len(final_edges)}")
            logger.info(f"P(H0)={self.P_H0:.3f}, P(H1)={self.P_H1:.3f}")

        # Phase 4: Add Method DAGs as Alternative Hypotheses
        logger.info("\n=== Phase 4: Adding Method DAGs to Hypothesis Space ===")
        
        if self.use_cwm and self.cwm:
            # Get method DAGs from discovery
            method_dags = self.pipeline.get_method_dags()
            
            for method_name, dag in method_dags.items():
                # Create predictive model for this DAG
                model = PredictiveModel(dag, self.cwm.variables)
                
                # Train on all data (regime-agnostic)
                temporal_df_all = self._prepare_temporal_data(regime_filter=None)
                model.fit_temporal(temporal_df_all)
                
                # Add as hypothesis
                hyp = GraphHypothesis(
                    name=f"{method_name.upper()}_graph",
                    graph=dag,
                    predictive_model=model,
                    weight=1.0  # Will renormalize below
                )
                hyp.predictive_model._parent_cwm = self.cwm
                self.cwm.hypotheses.append(hyp)
                
                logger.info(f"Added {method_name.upper()}: {len(dag.edges())} edges, trained on {len(temporal_df_all)} sequences")
            
            # Renormalize all weights to sum to 1.0
            total_weight = sum(h.weight for h in self.cwm.hypotheses)
            for h in self.cwm.hypotheses:
                h.weight /= total_weight
            
            logger.info(f"\n=== Monitoring {len(self.cwm.hypotheses)} Hypotheses ===")
            for h in self.cwm.hypotheses:
                logger.info(f"  {h.name}: {len(h.graph.edges())} edges, P={h.weight:.3f}")
            
        # Phase 3: Training on complete data
        # logger.info("\n=== Phase 3: Training H0/H1 on Complete Data ===")

        # if self.use_cwm and self.cwm:
        #     temporal_df = self._prepare_temporal_data()
            
        #     # Train both models normally
        #     self.H0_model.fit_temporal(temporal_df)
        #     self.H1_model.fit_temporal(temporal_df)
            
        #     logger.info(f"H0 trained on {len(temporal_df)} sequences")
        #     logger.info(f"H1 trained on {len(temporal_df)} sequences")

        #     logger.info(f"POST-TRAINING H0 graph: {sorted(self.H0_model.graph.edges())}")
        #     logger.info(f"POST-TRAINING H1 graph: {sorted(self.H1_model.graph.edges())}")

        #     logger.info(f"H0 temp params: {self.H0_model.parameters.get('temperature', {})}")
        #     logger.info(f"H1 temp params: {self.H1_model.parameters.get('temperature', {})}")
            
        # Rebuild H0/H1 from validated edges only
        # if self.use_cwm and self.cwm:
        #     logger.info("\n=== Rebuilding H0/H1 from Validated Edges ===")
            
        #     final_edges = self.pipeline.validated_edges
            
        #     # H0: Validated edges without window
        #     G_H0_final = nx.DiGraph()
        #     for edge in final_edges:
        #         edge_tuple = tuple(edge) if isinstance(edge, list) else edge
        #         if 'window' not in edge_tuple[0].lower() and 'window' not in edge_tuple[1].lower():
        #             G_H0_final.add_edge(*edge_tuple)
            
        #     # H1: H0 + window pathway
        #     G_H1_final = G_H0_final.copy()

        #     for node in ['windowopen', 'outdoortemperature', 'temperature']:
        #         if node not in G_H1_final.nodes():
        #             G_H1_final.add_node(node)

        #     G_H1_final.add_edges_from([
        #         ('outdoortemperature', 'windowopen'),
        #         ('windowopen', 'temperature')
        #     ])
            
        #     # Rebuild models with validated structure
        #     self.H0_graph = G_H0_final
        #     self.H1_graph = G_H1_final
        #     self.H0_model = PredictiveModel(G_H0_final, self.cwm.variables)
        #     self.H1_model = PredictiveModel(G_H1_final, self.cwm.variables)

        #     # Fit on validated structure
        #     self.H0_model.fit_temporal(temporal_df)
        #     self.H1_model.fit_temporal(temporal_df)
        #     logger.info("Retrained models on validated structure")

        #     logger.info(f"POST-REBUILD H0 graph: {sorted(self.H0_model.graph.edges())}")
        #     logger.info(f"POST-REBUILD H1 graph: {sorted(self.H1_model.graph.edges())}")
        #     logger.info(f"POST-REBUILD H0 temp params: {self.H0_model.parameters.get('temperature', {})}")
        #     logger.info(f"POST-REBUILD H1 temp params: {self.H1_model.parameters.get('temperature', {})}")

        #     logger.info(f"Final H0: {len(G_H0_final.edges())} edges")
        #     logger.info(f"Final H1: {len(G_H1_final.edges())} edges")
        #     logger.info(f"Window edges in H1: {[e for e in G_H1_final.edges() if 'window' in str(e)]}")
            
        #     # Update CWM hypotheses
        #     self.cwm.hypotheses = []
        #     for graph, name, model in [(G_H0_final, 'H0_WindowClosed', self.H0_model), 
        #                                 (G_H1_final, 'H1_WindowOpen', self.H1_model)]:
        #         hyp = GraphHypothesis(name=name, graph=graph, 
        #                             predictive_model=model, 
        #                             weight=self.P_H0 if name == 'H0_WindowClosed' else self.P_H1)
        #         hyp.predictive_model._parent_cwm = self.cwm
        #         self.cwm.hypotheses.append(hyp)

        #     logger.info("\n=== DAGs for Monitoring Phase ===")
        #     logger.info(f"H0 edges: {sorted(G_H0_final.edges())}")
        #     logger.info(f"H1 edges: {sorted(G_H1_final.edges())}")
        #     logger.info(f"Final DAG edges: {sorted(final_edges)}")
        #     logger.info(f"P(H0)={self.P_H0:.3f}, P(H1)={self.P_H1:.3f}")
        
        # Finalize
        if final_dag is None:
            final_dag = self.pipeline._create_final_dag(self.pipeline.validated_edges)
        
        final_metrics = self.pipeline._calculate_final_metrics(final_dag, method_dags)

        if hasattr(self.pipeline, 'method_metrics'):
            final_metrics['method_metrics'] = self.pipeline.method_metrics
        
        if self.use_cwm and self.cwm:
            cwm_summary = self.cwm.get_state_summary()
            logger.info(f"\n=== CWM Final State ===")
            if self.has_regime:
                logger.info(f"  P(H0)={self.P_H0:.3f}, P(H1)={self.P_H1:.3f}")
            for key, val in cwm_summary.items():
                logger.info(f"  {key}: {val}")
        
        self.pipeline.final_dag = final_dag
        return final_dag, final_metrics
    
    def _get_edges_to_test(self):
        """
        Get filtered and ranked edges to test.
        Uses edge ranker which considers all 4 methods (PC, SAM, LLM, VARLiNGAM).

        Returns:
            List of edges to test, filtered and ranked
        """
        edges = self.pipeline.edge_ranker.get_ranked_edges()

        # Intersect-then-confirm: only test edges with N+ method agreement
        import src.pipeline_cwm as _self_module
        min_agree = getattr(_self_module, '_MIN_METHOD_AGREEMENT', 0)
        if min_agree >= 2:
            edge_method_counts = {}
            for method_name, hyp in self.pipeline.edge_ranker.hypotheses.items():
                method_edges = hyp.get('edges', [])
                for e in method_edges:
                    norm_e = self.pipeline.edge_ranker._normalize_edge(e)
                    edge_method_counts[norm_e] = edge_method_counts.get(norm_e, 0) + 1
            before = len(edges)
            edges = [e for e in edges
                     if edge_method_counts.get(
                         self.pipeline.edge_ranker._normalize_edge(e), 0) >= min_agree]
            logger.info(f"  Intersect filter: {before} → {len(edges)} edges "
                       f"(require {min_agree}+ methods)")

        # Define output variables based on dataset type
        if self.pipeline.dataset_type == 'ashrae':
            output_vars = {'energyconsumption', 'meter_reading'}
        else:
            # NOTE: hvacpower/lightingpower are actuators, not outputs — keep them testable
            output_vars = {'energyconsumption', 'overallsatisfaction', 'satisfaction',
                          'meter_reading', 'thermalcomfort', 'visualcomfort',
                          'airqualityindex'}

        # Edge direction correction:
        # - output→actuator: reverse to actuator→output (methods often
        #   get the direction wrong between controlled and derived vars)
        # - output→output: remove (no causal mechanism between derived vars)
        actuator_vars = set(getattr(self.pipeline, 'actuator_vars', set()))
        corrected = []
        for e in edges:
            src, tgt = e[0].lower(), e[1].lower()
            if src in output_vars and tgt in output_vars:
                logger.debug(f"  Dropped output→output: {src}→{tgt}")
                continue
            if src in output_vars and tgt in actuator_vars:
                # Reverse: test actuator→output instead
                rev_src = next((c for c in self.pipeline.data.columns if c.lower() == tgt), tgt)
                rev_tgt = next((c for c in self.pipeline.data.columns if c.lower() == src), src)
                corrected.append((rev_src, rev_tgt))
                logger.info(f"  Reversed output→actuator: {src}→{tgt} to {tgt}→{src}")
            else:
                corrected.append(e)
        edges = corrected

        # Filter out remaining output-source edges, already validated, and failed
        failed = getattr(self, '_failed_edges', set())
        validated_norm = {(e[0].lower(), e[1].lower()) for e in self.pipeline.validated_edges}
        edges = [e for e in edges
                if e[0].lower() not in output_vars
                and (e[0].lower(), e[1].lower()) not in validated_norm
                and (e[0].lower(), e[1].lower()) not in failed]

        # Actuator proactive testing: actuators (directly controllable variables)
        # are often missed by observational discovery methods due to controller
        # confounding.  Since we have a simulator, we can test actuator→X edges
        # directly via intervention — this is the key advantage of PolicyGRID.
        #
        # Which variables are actuators is DOMAIN KNOWLEDGE (user-specified),
        # analogous to knowing which knobs you can turn in a real system.
        # No correlation pre-filter — the paired intervention design correctly
        # rejects non-causal edges, so we test all actuator→X candidates.
        actuator_vars = set(getattr(self.pipeline, 'actuator_vars', set()))
        if not actuator_vars:
            logger.info("  No actuator_vars specified — skipping proactive testing")
        dataset_cols = {c.lower() for c in self.pipeline.data.columns}
        active_actuators = actuator_vars & dataset_cols
        existing = set((e[0].lower(), e[1].lower()) for e in edges) | \
                   set((e[0].lower(), e[1].lower()) for e in self.pipeline.validated_edges) | \
                   failed
        actuator_edges = []
        for act in sorted(active_actuators):
            act_col = next((c for c in self.pipeline.data.columns if c.lower() == act), act)
            for col in self.pipeline.data.columns:
                if col.lower() == act:
                    continue
                if (act, col.lower()) not in existing:
                    actuator_edges.append((act_col, col))
                    logger.info(f"  Actuator proactive: {act_col}→{col}")
        if actuator_edges:
            logger.info(f"  Added {len(actuator_edges)} actuator proactive edges")
        edges = edges + actuator_edges

        return edges
    
    def _prioritize_by_cwm_confidence(self, edges):
        """
        Reorder edges based on CWM confidence scores.
        When uncertainty is high, test low-confidence edges first.
        
        Args:
            edges: List of edges to test
            
        Returns:
            Reordered edges with low-confidence edges first
        """
        if not self.cwm:
            return edges
        
        # Get low-confidence edges from CWM (C_{i,j} < 0.6)
        low_conf = self.cwm.get_low_confidence_edges(threshold=0.6)
        low_conf_set = set((src, tgt) for src, tgt, _ in low_conf)
        
        # Prioritize edges that appear in low-confidence set
        priority_edges = [e for e in edges if e in low_conf_set]
        other_edges = [e for e in edges if e not in low_conf_set]
        
        reordered = priority_edges + other_edges
        
        if priority_edges:
            logger.info(f"CWM prioritized {len(priority_edges)} low-confidence edges:")
            logger.info(f"\n   Edge Prioritization:")
            logger.info(f"     High priority (low confidence): {len(priority_edges)}")
            logger.info(f"     Normal priority: {len(other_edges)}")
            for edge in priority_edges[:3]:  # Show top 3
                conf = next((c for s, t, c in low_conf if (s, t) == edge), 0)
                logger.info(f"  {edge[0]} ' {edge[1]}: confidence={conf:.3f}")
        
        return reordered
    
    def _test_edges_with_cwm_updates(self, edges_to_test):
        """
        Test edges via simulator intervention + regression effect measurement.

        Two-stage validation per edge (A→B):
          1. **Intervention** — run the simulator with A forced to a value,
             observe pre/post states (standard PolicyGRID do-operator).
          2. **Effect measurement** — compute partial_corr(A, B | Z) on the
             accumulated observational + interventional data to quantify the
             direct causal effect, controlling for confounders.

        After all edges are tested, applies Benjamini-Hochberg FDR correction
        (Benjamini & Hochberg, 1995) to control the false discovery rate across
        the batch of tested edges.  This automatically adapts to the number of
        candidates: stricter when 119 edges are tested (smart_building_rich),
        lenient when only 20 are tested (open_window).

        Args:
            edges_to_test: List of edges to validate

        Returns:
            Set of validated edges
        """
        validated = set()

        # Define output variables for filtering
        if self.pipeline.dataset_type == 'ashrae':
            output_vars = {'energyconsumption', 'meter_reading'}
        elif self.pipeline.dataset_type == 'open_window':
            output_vars = {'satisfaction', 'energyconsumption'}
        else:
            output_vars = {'energyconsumption', 'overallsatisfaction', 'satisfaction',
                          'meter_reading', 'thermalcomfort', 'visualcomfort',
                          'airqualityindex'}

        # Non-intervenable variables: domain config parameter
        non_intervenable = getattr(self.pipeline, 'non_intervenable_vars', set())

        # Build set of valid column names (case-insensitive) from dataset
        dataset_cols_lower = {c.lower() for c in self.pipeline.data.columns}
        mapped_cols = set()
        if hasattr(self.pipeline, 'column_mapping'):
            for k, v in self.pipeline.column_mapping.items():
                mapped_cols.add(k.lower())
                mapped_cols.add(v.lower())
        valid_names = dataset_cols_lower | mapped_cols

        # ── Collect all edge test results for batch FDR correction ──
        edge_test_results = []  # list of (edge, is_valid, results, regression_effect)

        for edge in edges_to_test:
            logger.info(f"Attempting to test edge: {edge}")
            # Skip edges with output variables or non-intervenable sources
            if edge[0].lower() in output_vars or edge[0].lower() in non_intervenable:
                logger.info(f"Skipped: {edge[0]} in exclusion list")
                continue

            # Skip edges whose variable names don't exist in the dataset
            if edge[0].lower() not in valid_names or edge[1].lower() not in valid_names:
                logger.info(f"Skipped edge {edge}: variable(s) not in dataset columns")
                continue

            # Intervention design: three complementary strategies per edge
            #  1. Deterministic MAX — maximum signal, no domain knowledge needed
            #  2. Paired counterfactual — do(max) vs do(0), signed direction
            #  3. LLM-designed — domain-aware intermediate operating point
            source_lower = edge[0].lower()
            actuator_vars = getattr(self.pipeline, 'actuator_vars', set())

            mapped_src = self.pipeline.tester.variable_mapping.get(
                source_lower, edge[0])
            var_range = self.pipeline.tester.variable_ranges.get(
                mapped_src, {'min': 0, 'max': 100})

            # Primary: deterministic MAX (used for initial screening)
            intervention = {
                'variables': [{
                    'variable': mapped_src,
                    'action': 'set',
                    'value': var_range['max']
                }],
                'expected_effects': {},
                'strategy_type': 'deterministic_max'
            }
            logger.info(f"Deterministic intervention: {mapped_src}={var_range['max']}")

            # Additional strategies for confirmatory phase
            additional_interventions = []

            # Paired counterfactual: do(0) — paired with the max from primary
            counterfactual = {
                'variables': [{
                    'variable': mapped_src,
                    'action': 'set',
                    'value': var_range['min']
                }],
                'expected_effects': {},
                'strategy_type': 'paired_counterfactual_min'
            }
            additional_interventions.append(counterfactual)

            # LLM-designed: domain-aware intermediate value
            if source_lower in actuator_vars:
                try:
                    llm_intervention = self.pipeline.tester._design_llm_intervention(edge)
                    if llm_intervention:
                        llm_intervention['strategy_type'] = 'llm_designed'
                        additional_interventions.append(llm_intervention)
                except Exception as e:
                    logger.warning(f"LLM intervention design failed for {edge}: {e}")
                    # Fallback: midpoint intervention
                    midpoint = {
                        'variables': [{
                            'variable': mapped_src,
                            'action': 'set',
                            'value': (var_range['min'] + var_range['max']) / 2
                        }],
                        'expected_effects': {},
                        'strategy_type': 'midpoint_fallback'
                    }
                    additional_interventions.append(midpoint)
            else:
                # Non-actuator source: use LLM as primary, no deterministic fallback
                try:
                    intervention = self.pipeline.tester._design_llm_intervention(edge)
                    if intervention:
                        intervention['strategy_type'] = 'llm_designed'
                except Exception as e:
                    logger.warning(f"LLM intervention design failed for {edge}: {e}")
                additional_interventions = []  # no additional strategies

            if intervention:
                # ── Stage 1: Simulator intervention (3 initial + 7 confirmatory) ──
                is_valid, results = self.pipeline.tester.test_edge_iteratively(
                    edge, intervention, additional_interventions=additional_interventions or None)
                logger.info(f"Results keys: {results.keys() if results else 'None'}")

                # ── Stage 2: Regression effect measurement ──
                regression_effect = self._compute_regression_effect(edge)
                if results:
                    results['partial_r'] = regression_effect.get('partial_r', 0)
                    results['regression_p'] = regression_effect.get('p_value', 1)
                    results['method'] = 'intervention+regression'

                # Update CWM with observation (Section III.A)
                if self.use_cwm and self.cwm and results:
                    pre_state = results.get('preInterventionData', {})
                    post_state = results.get('postInterventionData', {})
                    self._update_cwm_from_intervention(edge, pre_state, post_state,
                                                    is_valid, results)

                edge_test_results.append((edge, is_valid, results, regression_effect))

        # ── Benjamini-Hochberg FDR correction ──
        # Controls the expected proportion of false discoveries among all
        # accepted edges.  Automatically adapts to the number of candidates:
        # testing 119 edges requires stronger evidence per edge than testing 20.
        # Reference: Benjamini & Hochberg (1995), JRSS-B.
        #
        # Only applied when n_tested >= FDR_MIN_EDGES.  With fewer edges,
        # the multiple-testing burden is low and the original per-edge
        # threshold (p < 0.05) is adequate.  This prevents FDR from being
        # overly aggressive on small candidate sets (e.g. open_window with
        # 5-10 intervenable edges).
        FDR_ALPHA = 0.10  # target FDR = 10%
        FDR_MIN_EDGES = 20  # only apply FDR when batch is large enough
        n_tested = len(edge_test_results)
        # Allow experiments to disable FDR via module-level flag
        import src.pipeline_cwm as _self_module
        fdr_force_off = getattr(_self_module, '_FDR_DISABLED', False)
        apply_fdr = (not fdr_force_off) and (n_tested >= FDR_MIN_EDGES)

        if n_tested > 0:
            if apply_fdr:
                # Collect p-values
                p_values = []
                for edge, is_valid, results, reg_eff in edge_test_results:
                    p = results.get('p_value', 1.0) if results else 1.0
                    p_values.append(p)

                # Sort p-values for BH procedure
                import numpy as np
                indexed_pvals = sorted(enumerate(p_values), key=lambda x: x[1])

                # Find the largest k where p_(k) <= (k/m) * alpha
                max_reject_rank = 0
                for rank_idx, (orig_idx, p) in enumerate(indexed_pvals):
                    rank = rank_idx + 1
                    if p <= (rank / n_tested) * FDR_ALPHA:
                        max_reject_rank = rank

                # All edges with rank <= max_reject_rank are accepted
                accepted_indices = set()
                for rank_idx, (orig_idx, p) in enumerate(indexed_pvals):
                    if rank_idx + 1 <= max_reject_rank:
                        accepted_indices.add(orig_idx)

                logger.info(f"\n  FDR correction: {n_tested} edges tested, "
                           f"{max_reject_rank} pass BH(alpha={FDR_ALPHA})")
            else:
                # Small batch — skip FDR, accept all that passed per-edge test
                accepted_indices = set(range(n_tested))
                logger.info(f"\n  FDR skipped: only {n_tested} edges tested "
                           f"(< {FDR_MIN_EDGES}), using per-edge threshold")

            for idx, (edge, is_valid, results, regression_effect) in enumerate(edge_test_results):
                fdr_accepted = idx in accepted_indices
                p = results.get('p_value', 1.0) if results else 1.0

                if is_valid and fdr_accepted:
                    # Accepted by both per-edge t-test AND FDR correction
                    validated.add(edge)
                    self.pipeline._update_training_data_with_interventions(edge)
                    self.pipeline._save_intermediate_data(edge)
                    logger.info(f"  Edge validated: {edge[0]}→{edge[1]} "
                              f"(p={p:.4f}, {'FDR-pass' if apply_fdr else 'no-FDR'}, "
                              f"|partial_r|={abs(regression_effect.get('partial_r', 0)):.3f})")
                elif is_valid and not fdr_accepted:
                    # Would have passed per-edge test but REJECTED by FDR.
                    # Rescue if strong observational evidence backs it up:
                    #   - |partial_r| > 0.3 (moderate obs correlation)
                    #   - intervention p < 0.15 (marginally significant)
                    # This is a Bayesian-flavored approach: strong prior
                    # (observational evidence) lowers the bar for the
                    # interventional test.  Real-world applicable since
                    # all systems have historical sensor data.
                    obs_partial_r = abs(regression_effect.get('partial_r', 0))
                    if obs_partial_r > 0.3 and p < 0.15:
                        validated.add(edge)
                        self.pipeline._update_training_data_with_interventions(edge)
                        self.pipeline._save_intermediate_data(edge)
                        logger.info(f"  Edge FDR-rescued (obs+intv): {edge[0]}→{edge[1]} "
                                  f"(p={p:.4f}, |partial_r|={obs_partial_r:.3f})")
                    else:
                        logger.info(f"  Edge FDR-rejected: {edge[0]}→{edge[1]} "
                                  f"(p={p:.4f}, |partial_r|={obs_partial_r:.3f}, "
                                  f"rank too high for BH threshold)")
                    if not (obs_partial_r > 0.3 and p < 0.15):
                        if self.use_cwm and self.cwm and hasattr(self.cwm, 'hypothesis_bank'):
                            self.cwm.hypothesis_bank.add_rejected_edge(
                                edge[0], edge[1],
                                rejection_reason="Failed FDR correction",
                                intervention_data=results
                            )
                else:
                    # Failed per-edge test already
                    if self.use_cwm and self.cwm and hasattr(self.cwm, 'hypothesis_bank'):
                        self.cwm.hypothesis_bank.add_rejected_edge(
                            edge[0], edge[1],
                            rejection_reason="Failed intervention validation",
                            intervention_data=results
                        )

        return validated

    def _compute_regression_effect(self, edge):
        """
        Compute partial correlation of source→target controlling for all
        other observed variables.  Used as effect measurement alongside
        simulator intervention.

        Returns:
            dict with 'partial_r' and 'p_value'
        """
        from scipy.stats import norm as sp_norm

        source, target = edge
        obs_data = self.pipeline.data
        src_col = next((c for c in obs_data.columns
                        if c.lower() == source.lower()), None)
        tgt_col = next((c for c in obs_data.columns
                        if c.lower() == target.lower()), None)
        if not src_col or not tgt_col:
            return {'partial_r': 0, 'p_value': 1.0}

        covariate_cols = [c for c in obs_data.columns
                          if c != src_col and c != tgt_col
                          and c.lower() not in {'timestamp', 'elapsedtime',
                                                 'interventionapplied'}]
        cols = [src_col, tgt_col] + covariate_cols
        sub = obs_data[cols].dropna()
        n = len(sub)
        k = len(covariate_cols)

        if n < k + 5:
            return {'partial_r': 0, 'p_value': 1.0}

        try:
            C = sub.corr().values
            P = np.linalg.inv(C)
            partial_r = -P[0, 1] / np.sqrt(abs(P[0, 0] * P[1, 1]))
            partial_r = max(-1.0, min(1.0, partial_r))
        except np.linalg.LinAlgError:
            partial_r = obs_data[src_col].corr(obs_data[tgt_col])
            if partial_r is None or np.isnan(partial_r):
                partial_r = 0.0

        if n - k - 3 > 0:
            z = 0.5 * np.log((1 + partial_r + 1e-10) /
                             (1 - partial_r + 1e-10))
            se = 1.0 / np.sqrt(n - k - 3)
            p_value = 2 * (1 - sp_norm.cdf(abs(z / se)))
        else:
            p_value = 1.0

        return {'partial_r': partial_r, 'p_value': p_value}
    
    def _get_current_state(self):
        """
        Get current system state for CWM updates.
        
        Returns:
            Dictionary of current variable values
        """
        # Use most recent row from data
        if len(self.pipeline.data) > 0:
            latest = self.pipeline.data.iloc[-1]
            state = {col: latest[col] for col in self.pipeline.relevant_columns 
                    if col in latest.index}
            return state
        return {}
    
    def _detect_anomaly(self, current_error: float) -> bool:
        """
        Detect sustained prediction error indicating structural change.
        Based on PDF: Et = T1_obs - T1_pred
        """
        self.prediction_error_history.append(abs(current_error))
        
        if len(self.prediction_error_history) < self.anomaly_detection_window:
            return False
        
        # Check for sustained high error
        recent_errors = list(self.prediction_error_history)[-self.anomaly_detection_window:]
        mean_error = np.mean(recent_errors)
        
        is_anomaly = mean_error > self.prediction_error_threshold
        
        if is_anomaly:
            logger.warning(f" ANOMALY DETECTED: Mean error={mean_error:.3f} > {self.prediction_error_threshold}")
        
        return is_anomaly
    
    def _design_diagnostic_intervention(self) -> dict:
        """
        Design intervention to maximize information gain between H0 and H1.
        Based on PDF: int(Xt) = arg max E[H(pt) - H(pt+1|y)] - Jpen
        
        For window detection: heating pulse to distinguish thermal coupling strength.
        """
        logger.info("\n DESIGNING DIAGNOSTIC INTERVENTION")
        logger.info("  Goal: Distinguish H0 (window closed) vs H1 (window open)")
        
        # Heating pulse intervention
        intervention = {
            'variable': 'A1',
            'action': 'increase',
            'value': 0.3,  # 30% increase in heating
            'reasoning': 'Heat pulse to test TOA. T1 coupling strength (window open shows faster decay)'
        }
        
        logger.info(f"  Intervention: Increase {intervention['variable']} by {intervention['value']}")
        logger.info(f"  Expected H0 response: Slow temperature change (normal insulation)")
        logger.info(f"  Expected H1 response: Fast temperature decay (window open)")
        
        return intervention
    
    def _check_hypothesis_switch(self) -> bool:
        """
        Check if H1 (window open) has sufficient evidence to become active.
        Based on PDF: Switch when p(H1) > threshold (0.7)
        """
        if not self.cwm or 'W1' not in self.latent_hypotheses:
            return False
        
        # Find H0 and H1 weights
        h0_weight = None
        h1_weight = None
        
        for hyp in self.cwm.hypotheses:
            if hyp.name == "H0_WindowClosed":
                h0_weight = hyp.weight
            elif hyp.name == "H1_WindowOpen":
                h1_weight = hyp.weight
        
        if h0_weight is None or h1_weight is None:
            return False
        
        logger.info(f"\n Hypothesis Posterior: H0={h0_weight:.3f}, H1={h1_weight:.3f}")
        
        # Switch if H1 dominant
        if h1_weight > self.confidence_threshold and self.latent_hypotheses['W1']['active'] == 'H0':
            logger.info(f" SWITCHING TO H1_WindowOpen (p={h1_weight:.3f} > {self.confidence_threshold})")
            self.latent_hypotheses['W1']['active'] = 'H1'
            
            # Add W1 edges to validated set
            self.pipeline.validated_edges.add(('TOA', 'W1'))
            self.pipeline.validated_edges.add(('W1', 'T1'))
            logger.info("  Added edges: TOA'W1, W1'T1")
            
            return True
        
        return False

    def _update_cwm_from_intervention(self, edge, pre_state, post_state, is_valid, results):
        """
        Update CWM based on intervention results - DISCOVERY PHASE ONLY
        
        During discovery: Structure incomplete, parameters untrained
        ' Skip Bayesian updates (defer to Monitoring Phase post-training)
        
        Implements: p(H_k^{(t+1)} | D_{1:t})  p(y_t | H_k) p(H_k^{(t)} | D_{1:t-1})
        But only after Training Phase completes.
        
        Args:
            edge: Edge (i,j) being validated
            pre_state: X_t (state before intervention)
            post_state: X_{t+1} (state after intervention)
            is_valid: Whether edge validated via statistical test
            results: Intervention results with effect size
        """
        # Discovery phase: maintain p(H0) = p(H1) = 0.5 (no inference)
        logger.debug(f"Discovery: edge {edge}, maintaining uniform priors")
        
        # Track window state W_t for visualization
        window_open = post_state.get('WindowOpen', post_state.get('windowopen', False))
        logger.debug(f"W_t = {'1' if window_open else '0'}")
        
        # Store flat probabilities (Bayesian updates happen post-training)
        self.probability_history.append({
            'time': len(self.probability_history) * 2.0,
            'P_H0': 0.5,  # p(H0 | D) held constant during discovery
            'P_H1': 0.5,  # p(H1 | D) held constant during discovery
            'actual_window': window_open
        })
        
        # DEFERRED TO MONITORING PHASE:
        # Bayesian update: p(H_k | y_t)  exp(-||_t - y_t||/2)  p(H_k)
        # where _t = f_G_k(X_t, _k) from trained models
        #
        # Discovery phase has:
        # - Incomplete G_k (edges still being validated)
        # - Untrained _k (parameters fit on initial data only)
        # ' Predictions _t unreliable, posteriors meaningless
        #
        # Proper sequence:
        # 1. Discovery: validate edges, collect D = {(X_t, X_{t+1})}
        # 2. Training: fit _k via MLE on complete D
        # 3. Monitoring: Bayesian inference with frozen (G_k, _k)
                
    def _update_cwm_structure(self, method_dags):
        """
        Update CWM graph structure after DAG regeneration.
        
        Updates hypothesis weights for all 4 methods: p(H_k | D_{1:t})
        
        Args:
            method_dags: Regenerated DAGs from all methods
        """
        if not self.cwm:
            return
        
        # Update hypothesis weights based on new DAGs from all methods
        for method_name, dag in method_dags.items():
            hyp_name = f"H_{method_name}"
            
            # Find matching hypothesis
            matching_hyp = None
            for hyp in self.cwm.hypotheses:
                if hyp.name == hyp_name:
                    matching_hyp = hyp
                    break
            
            if matching_hyp:
                # Update its graph structure
                edges = self.pipeline._extract_edges(dag)
                
                # Handle VARLiNGAM edges with lag info
                edge_tuples = []
                for edge in edges:
                    if isinstance(edge, dict) and 'edge' in edge:
                        edge_tuples.append(edge['edge'])
                    else:
                        edge_tuples.append(edge)
                
                G_new = nx.DiGraph()
                G_new.add_edges_from(edge_tuples)
                matching_hyp.graph = G_new
                matching_hyp.predictive_model = PredictiveModel(G_new, self.cwm.variables)
                logger.debug(f"Updated CWM hypothesis {hyp_name} with {len(edge_tuples)} edges")
        
        logger.debug("CWM hypothesis structures updated for all methods")

    def monitor_regime_changes(self, duration=1800, sample_interval=5):
        """Continuous monitoring with scheduled window transitions"""
        import subprocess
        import tempfile
        import json
        
        logger.info(f"\n=== MONITORING MODE ({duration/60:.0f} minutes) ===")
        
        # Load scaling params
        if not hasattr(self, '_scaling_params') or self._scaling_params is None:
            try:
                scaling_path = 'data_regen/challenge1_data_10k_processed_scaling.csv'
                self._scaling_params = pd.read_csv(scaling_path)
                logger.info(f"Loaded scaling params from {scaling_path}")
            except:
                logger.warning("No scaling params found, using raw values")
                self._scaling_params = None
        
        observations = []


        # ── Policy engine: regime-aware instances ──────────────────────────────
        # Three engines, each with a different edge set, selected per-step based on
        # the current regime posterior (P_H1 / P_H0 thresholds).
        _pe_validated = _pe_h0 = _pe_h1 = None
        try:
            _dataset = self.pipeline.data
            _validated = getattr(self.pipeline, 'validated_edges', set())
            _h0_edges  = set(self.H0_model.graph.edges()) if hasattr(self, 'H0_model') and self.H0_model else set()
            _h1_edges  = set(self.H1_model.graph.edges()) if hasattr(self, 'H1_model') and self.H1_model else set()
            _pe_validated = CausalPolicyEngine({'validated_edges': _validated}, _dataset)
            _pe_h0        = CausalPolicyEngine({'validated_edges': _h0_edges},  _dataset)
            _pe_h1        = CausalPolicyEngine({'validated_edges': _h1_edges},  _dataset)
            logger.info(f"Policy engines initialised — validated:{len(_validated)} "
                        f"H0:{len(_h0_edges)} H1:{len(_h1_edges)} edges")
        except Exception as _pe_init_err:
            logger.warning(f"Policy engine init failed (monitoring continues): {_pe_init_err}")
        # ── End policy engine init ─────────────────────────────────────────────

        # Create continuous simulation
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(f'''
            const {{ SmartRoom }} = require('{self.pipeline.tester.smart_room_path}');
            
            const room = new SmartRoom();
            
            room._initialize().then(() => {{
                room.start();
                
                const interval = setInterval(() => {{
                    room.updateDependentVariables().then(() => {{
                        const state = room.getState();
                        console.log('STATE:' + JSON.stringify(state));
                    }}).catch(err => {{
                        console.error('Update error:', err);
                    }});
                }}, {sample_interval * 1000});
                
                setTimeout(() => {{
                    clearInterval(interval);
                    room.stop();
                    room.cleanup().then(() => process.exit(0));
                }}, {duration * 1000});
            }});
            ''')
            temp_path = f.name
        
        try:
            process = subprocess.Popen(
                ['node', temp_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            start_time = time.time()
            temp_history      = []    # rolling buffer for temp_drift (RF feature)
            _prev_h1_dominant = None  # tracks dominant hypothesis for regime-switch events

            for line in iter(process.stdout.readline, ''):
                if 'STATE:' in line:
                    try:
                        state_json = line.split('STATE:', 1)[1]
                        state = json.loads(state_json)

                        # Normalize
                        state_norm = {}
                        for k, v in state.items():
                            # Drop non-numeric / non-boolean fields (e.g., timestamp)
                            if not isinstance(v, (int, float, np.number, bool)):
                                continue

                            k_lower = k.lower()

                            # Handle booleans explicitly
                            if isinstance(v, bool):
                                state_norm[k_lower] = float(v)
                                continue

                            # Scale if params exist
                            if self._scaling_params is not None:
                                param = self._scaling_params[
                                    self._scaling_params['feature'].str.lower() == k_lower
                                ]
                                if not param.empty:
                                    min_val = param['data_min'].values[0]
                                    max_val = param['data_max'].values[0]
                                    state_norm[k_lower] = (v - min_val) / (max_val - min_val)
                                else:
                                    state_norm[k_lower] = v
                            else:
                                state_norm[k_lower] = v

                        logger.debug(f"STATE: {state_norm}")

                        # Observed temperature + rolling temp_drift (for RF)
                        T_obs = state_norm.get('temperature', 0)
                        temp_history.append(T_obs)
                        # 5-step rolling mean of first-differences  matches training feature
                        if len(temp_history) >= 2:
                            recent_diffs = [temp_history[i] - temp_history[i-1]
                                            for i in range(max(1, len(temp_history)-5),
                                                           len(temp_history))]
                            _temp_drift = sum(recent_diffs) / len(recent_diffs)
                        else:
                            _temp_drift = 0.0

                        window_state = state_norm.get('windowopen', False)

                        if len(observations) == 0:  # Log once
                            logger.info(f"MONITORING H0 graph: {sorted(self.H0_model.graph.edges())}")
                            logger.info(f"MONITORING H1 graph: {sorted(self.H1_model.graph.edges())}")
                            logger.info(f"MONITORING H0 temp params: {self.H0_model.parameters.get('temperature', {})}")
                            logger.info(f"MONITORING H1 temp params: {self.H1_model.parameters.get('temperature', {})}")

                        # H0: predicts from non-window parents only
                        # H1: predicts from same parents + windowopen edge
                        # H1 assumes windowopen=1 (its hypothesis is "window
                        # is open").  Neither model observes the actual window
                        # state — the Bayesian update determines which
                        # assumption matches the observed temperature better.
                        state_no_window = {k: v for k, v in state_norm.items()
                                           if k != 'windowopen'}
                        state_h1 = dict(state_no_window)
                        state_h1['windowopen'] = 1.0

                        pred_H0 = self.H0_model.predict(state_no_window)
                        pred_H1 = self.H1_model.predict(state_h1)
                        T_pred_H0 = pred_H0.get('temperature', 0)
                        T_pred_H1 = pred_H1.get('temperature', 0)
                        err_H0 = abs(T_obs - T_pred_H0)
                        err_H1 = abs(T_obs - T_pred_H1)

                        logger.debug(f"t={time.time()-start_time:.2f}s W={int(window_state)} "
                                     f"T_obs={T_obs:.3f} T_H0={T_pred_H0:.3f} T_H1={T_pred_H1:.3f} "
                                     f"err_H0={err_H0:.3f} err_H1={err_H1:.3f}")

                        # ── Binary H0/H1 regime posterior ─────────────────
                        # H0 and H1 compete directly, isolated from the
                        # multi-method pool.  Same sigma and forgetting
                        # factor as the full pool for consistent scale.
                        sigma     = getattr(self.cwm, 'likelihood_sigma', 1.0)
                        bin_lh_H0 = np.exp(-sigma * err_H0)
                        bin_lh_H1 = np.exp(-sigma * err_H1)
                        unnorm_H0 = bin_lh_H0 * self.P_H0
                        unnorm_H1 = bin_lh_H1 * self.P_H1
                        bin_total = unnorm_H0 + unnorm_H1
                        if bin_total > 0:
                            lam = self.forget_factor
                            self.P_H0 = (1 - lam) * (unnorm_H0 / bin_total) + lam * 0.5
                            self.P_H1 = (1 - lam) * (unnorm_H1 / bin_total) + lam * 0.5

                        # Full CWM pool (appendix: multi-method structural comparison)
                        # Runs in parallel; does NOT overwrite self.P_H0/P_H1.
                        # windowopen masked: same level playing field.
                        self.cwm.update_hypothesis_weights(state_no_window)

                        # ── Regime-aware policy action ─────────────────────────
                        _policy_fields = {}
                        if _pe_validated is not None:
                            try:
                                # Select engine by current regime posterior
                                if self.P_H1 > 0.8 and _pe_h1 is not None:
                                    _active_pe    = _pe_h1
                                    _active_regime = 'H1_WindowOpen'
                                elif self.P_H0 > 0.8 and _pe_h0 is not None:
                                    _active_pe    = _pe_h0
                                    _active_regime = 'H0_WindowClosed'
                                else:
                                    _active_pe    = _pe_validated
                                    _active_regime = 'uncertain'

                                # Normalised state for policy engine
                                _ps = {
                                    'Temperature':       state_norm.get('temperature', 0.5),
                                    'Humidity':          state_norm.get('humidity', 0.5),
                                    'AirQuality':        state_norm.get('airquality', 0.5),
                                    'EnergyConsumption': state_norm.get('energyconsumption', 0.0),
                                    'Satisfaction':      state_norm.get('satisfaction', 0.0),
                                }
                                _pr = _active_pe.optimize_policy(
                                    objectives={
                                        'energyconsumption': {'target': 0.3, 'weight': 0.5},
                                        'satisfaction':      {'target': 0.7, 'weight': 0.5},
                                    },
                                    constraints={
                                        'Temperature':  (0.0, 1.0),
                                        'Humidity':     (0.0, 1.0),
                                        'AirQuality':   (0.0, 1.0),
                                        'comfort_limit': 0.8,
                                    },
                                    current_state=_ps,
                                )
                                _sat_key = _active_pe._satisfaction_col
                                _policy_fields = {
                                    'policy_active_graph':       _active_regime,
                                    'policy_energy_pred':        round(float(_pr.predicted_outcomes.get('EnergyConsumption', float('nan'))), 4),
                                    'policy_satisfaction_pred':  round(float(_pr.predicted_outcomes.get(_sat_key, float('nan'))), 4),
                                    'policy_cost':               round(float(_pr.cost), 4),
                                    'policy_confidence':         round(float(_pr.confidence), 4),
                                }
                            except Exception as _pe_err:
                                logger.debug(f"Policy step skipped: {_pe_err}")
                        # ── End regime-aware policy ────────────────────────────

                        obs_record = {
                            'time': (time.time() - start_time) / 60.0,
                            'actual_window': state_norm.get('windowopen', False)
                        }
                        obs_record.update(_policy_fields)

                        # Binary regime posteriors  primary signal for D1/A2 experiments
                        obs_record['P_H0_binary'] = round(self.P_H0, 6)
                        obs_record['P_H1_binary'] = round(self.P_H1, 6)

                        # Full CWM pool weights  structural comparison (appendix)
                        if self.cwm and self.cwm.hypotheses:
                            for hyp in self.cwm.hypotheses:
                                obs_record[f'P_{hyp.name}'] = hyp.weight

                        # RF regime classification (A2 detection signal)
                        # Physical-state features only  no outcome variables.
                        # state_norm keys are all-lowercase; RF expects camelCase.
                        rf_predictor = self._get_regime_predictor()
                        if rf_predictor:
                            state_rf = {
                                'temperature':        state_norm.get('temperature', 0),
                                'humidity':           state_norm.get('humidity', 0),
                                'airQuality':         state_norm.get('airquality', 0),
                                'outdoorTemperature': state_norm.get('outdoortemperature', 0),
                                'temp_drift':         _temp_drift,
                            }
                            try:
                                rf_result    = rf_predictor.predict_window_state(state_rf)
                                prob_open_rf = rf_result['prob_open']
                            except Exception as e:
                                logger.warning(f"RF prediction failed: {e}")
                                prob_open_rf = self.P_H1
                        else:
                            prob_open_rf = self.P_H1

                        obs_record['P_H1_RF'] = prob_open_rf

                        #  Prediction errors (regime-detection evidence, D1/A2) 
                        obs_record['T_obs']     = round(float(T_obs), 6)
                        obs_record['T_pred_H0'] = round(float(T_pred_H0), 6)
                        obs_record['T_pred_H1'] = round(float(T_pred_H1), 6)
                        obs_record['err_H0']    = round(float(err_H0), 6)
                        obs_record['err_H1']    = round(float(err_H1), 6)
                        obs_record['sigma']     = round(
                            float(getattr(self.cwm, 'likelihood_sigma', 1.0)), 4)

                        #  Regime-switch event markers 
                        _h1_dominant_now = int(self.P_H1 > self.P_H0)
                        obs_record['h1_dominant'] = _h1_dominant_now
                        if _prev_h1_dominant is None:
                            obs_record['regime_switch'] = 0
                        else:
                            obs_record['regime_switch'] = int(
                                _h1_dominant_now != _prev_h1_dominant)
                        _prev_h1_dominant = _h1_dominant_now

                        observations.append(obs_record)

                        # Populate calibration tracker (feeds compute_calibration_metrics)
                        self.calibration_tracker['timestamps'].append(obs_record['time'])
                        self.calibration_tracker['P_H1_RF'].append(prob_open_rf)
                        self.calibration_tracker['P_H0'].append(self.P_H0)
                        self.calibration_tracker['P_H1'].append(self.P_H1)
                        self.calibration_tracker['actual_window_state'].append(
                            int(state_norm.get('windowopen', 0))
                        )

                    except json.JSONDecodeError as e:
                        logger.error(f"JSON error: {e}")
                        continue
                    except KeyError as e:
                        logger.error(f"Model input mismatch: {e}")
                        raise
            
            process.wait()
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        logger.info(f"Monitoring complete: {len(observations)} observations")
        return pd.DataFrame(observations)

# Import required classes
try:
    from evaluator import EdgeRanker
except ImportError:
    # Fallback if running from different directory
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from evaluator import EdgeRanker


def create_cwm_pipeline(csv_data, api_key, smart_room_path=None, **kwargs):
    """
    Factory function to create CWM-integrated pipeline.
    
    This is a drop-in replacement for CausalPipeline with CWM capabilities.
    All original functionality is preserved (PC, SAM, LLM, VARLiNGAM).
    
    Usage:
        # Instead of:
        #   pipeline = CausalPipeline(data, api_key, smart_room_path)
        #   final_dag, metrics = pipeline.run()
        
        # Use:
        pipeline = create_cwm_pipeline(data, api_key, smart_room_path)
        final_dag, metrics = pipeline.run_with_cwm()
    
    Args:
        csv_data: Input dataset
        api_key: OpenAI API key
        smart_room_path: Path to simulation (optional)
        **kwargs: Additional arguments passed to CausalPipeline
        
    Returns:
        CWMIntegratedPipeline instance
    """
    from src.pipeline import CausalPipeline
    
    # Create base pipeline (with all 4 generators: PC, SAM, LLM, VARLiNGAM)
    base_pipeline = CausalPipeline(csv_data, api_key, smart_room_path, **kwargs)
    
    # Wrap with CWM integration
    cwm_pipeline = CWMIntegratedPipeline(base_pipeline)
    
    logger.info("CWM pipeline created with all 4 causal discovery methods")
    
    return cwm_pipeline

def plot_calibration_analysis(observations_df, output_path):
    """Plot hypothesis posteriors with ground truth shading"""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    time_minutes = observations_df['time'] / 60.0 if observations_df['time'].max() > 100 else observations_df['time']
    
    # Shade background by ground truth
    current_state = observations_df['actual_window'].iloc[0]
    start_idx = 0
    
    for i in range(1, len(observations_df)):
        if observations_df['actual_window'].iloc[i] != current_state:
            color = 'lightcoral' if current_state else 'lightblue'
            ax.axvspan(time_minutes.iloc[start_idx], time_minutes.iloc[i], 
                      color=color, alpha=0.3, zorder=0)
            ax.axvline(time_minutes.iloc[i], color='gray', linestyle='--', 
                      linewidth=1.5, zorder=1)
            start_idx = i
            current_state = observations_df['actual_window'].iloc[i]
    
    # Final region
    color = 'lightcoral' if current_state else 'lightblue'
    ax.axvspan(time_minutes.iloc[start_idx], time_minutes.iloc[-1], 
              color=color, alpha=0.3, zorder=0)
    
    # Visual hierarchy:
    #   PRIMARY    binary H0/H1 posteriors (main paper signal, thick solid)
    #   BASELINE   RF classifier (discriminative comparison, thick dashed)
    #   SECONDARY  full CWM pool weights (appendix, thin/transparent)
    STYLE = {
        # name            color           lw    ls      alpha  zorder  label_override
        'H0_binary':     ('royalblue',    2.8,  '-',    1.0,   4,  'P(H0)  window closed (CWM binary)'),
        'H1_binary':     ('crimson',      2.8,  '-',    1.0,   4,  'P(H1)  window open  (CWM binary)'),
        'H1_RF':         ('black',        2.2,  '--',   1.0,   3,  'P(H1_RF)  RF classifier (baseline)'),
        # Full pool  de-emphasised
        'H0_WindowClosed':('royalblue',   1.0,  ':',    0.4,   2,  None),
        'H1_WindowOpen':  ('crimson',     1.0,  ':',    0.4,   2,  None),
        'PC_graph':       ('green',       1.0,  '-',    0.35,  2,  None),
        'SAM_graph':      ('purple',      1.0,  '-',    0.35,  2,  None),
        'VARLINGAM_graph':('darkorange',  1.0,  '-',    0.45,  2,  None),
        'LLM_graph':      ('saddlebrown', 1.0,  '-',    0.35,  2,  None),
    }

    for col in observations_df.columns:
        if not col.startswith('P_'):
            continue
        hyp_name = col[2:]
        style = STYLE.get(hyp_name)
        if style:
            clr, lw, ls, alpha, zo, lbl = style
            label = lbl if lbl else f'P({hyp_name})'
        else:
            clr, lw, ls, alpha, zo, label = 'gray', 1.0, '-', 0.3, 1, f'P({hyp_name})'
        ax.plot(time_minutes, observations_df[col],
                color=clr, linewidth=lw, linestyle=ls, alpha=alpha,
                label=label, zorder=zo)
    
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Probability', fontsize=12)
    ax.set_ylim(0, 1.0)
    ax.set_title('Calibration Analysis - Hypothesis Posteriors Over Time', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2)
    
    # Legend
    blue_patch = mpatches.Patch(color='lightblue', alpha=0.3, label='Window Closed (Ground Truth)')
    red_patch = mpatches.Patch(color='lightcoral', alpha=0.3, label='Window Open (Ground Truth)')
    handles, labels = ax.get_legend_handles_labels()
    handles = [blue_patch, red_patch] + handles
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


# Example usage for testing
if __name__ == "__main__":
    import pandas as pd
    
    # Example: Load your data
    data = pd.read_csv("building_data.csv")
    
    # Create CWM-integrated pipeline (all 4 methods: PC, SAM, LLM, VARLiNGAM)
    pipeline = create_cwm_pipeline(
        csv_data=data,
        api_key="your-api-key",
        smart_room_path="/path/to/smart_room.js",
        max_iterations=5,
        alpha=0.5,
        beta=0.5
    )
    
    # Run with CWM (replaces pipeline.run())
    final_dag, metrics = pipeline.run_with_cwm()
    
    logger.info(f"\nFinal validated edges: {pipeline.pipeline.validated_edges}")
    logger.info(f"Metrics: {metrics}")
    
    # Access CWM state
    if pipeline.cwm:
        logger.info(f"\nCWM Uncertainty: {pipeline.cwm.compute_uncertainty():.3f}")
        logger.info(f"Edge confidences:")
        for (src, tgt), conf in list(pipeline.cwm.edge_confidences.items())[:5]:
            logger.info(f"  {src} ' {tgt}: {conf.confidence:.3f}")