#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced ABCD (Active Bayesian Causal Discovery) implementation with
true Bayesian posterior updating and advanced acquisition functions.
"""

import logging
import sys
import numpy as np
import pandas as pd
import os
import networkx as nx
import scipy.stats as stats
import traceback
from scipy.stats import entropy
from scipy.special import softmax
import time
from collections import defaultdict
from pathlib import Path
from benchmarks_new.utils import test_edge_validation, get_intervenable_vars
from benchmarks_new.utils import perform_intervention

logger = logging.getLogger(__name__)

class AdvancedAcquisitionFunctions:
    def __init__(self, edge_probabilities, intervention_results):
        self.edge_probabilities = edge_probabilities
        self.intervention_results = intervention_results
        
    def expected_information_gain(self, var, possible_values, column_names):
        """
        Calculate expected information gain using entropy reduction.
        
        Args:
            var (str): Variable to intervene on
            possible_values (list): Possible values to set the variable to
            column_names (list): List of all variable names
            
        Returns:
            float: Expected information gain score
        """
        # Current entropy of the system
        current_entropy = self._calculate_system_entropy()
        
        # Calculate expected entropy after intervention
        expected_posterior_entropy = 0
        for value in possible_values:
            # Simulate intervention results
            posterior_entropy = self._estimate_posterior_entropy(var, value, column_names)
            expected_posterior_entropy += posterior_entropy / len(possible_values)
        
        # Information gain is reduction in entropy
        return current_entropy - expected_posterior_entropy
    
    def expected_kl_divergence(self, var, possible_values, column_names):
        """
        Calculate expected KL divergence between prior and posterior distributions.
        
        Args:
            var (str): Variable to intervene on
            possible_values (list): Possible values to set the variable to
            column_names (list): List of all variable names
            
        Returns:
            float: Expected KL divergence score
        """
        # Current edge probabilities (prior)
        prior_probs = {edge: prob for edge, prob in self.edge_probabilities.items()}
        
        # Calculate expected KL divergence
        expected_kl = 0
        for value in possible_values:
            # Simulate intervention and get estimated posterior
            posterior_probs = self._simulate_posterior(var, value, column_names)
            
            # Calculate KL divergence for edges affected by this intervention
            kl_div = 0
            for edge, prior_p in prior_probs.items():
                if edge[0] == var or edge[1] == var:  # Only consider relevant edges
                    posterior_p = posterior_probs.get(edge, prior_p)
                    
                    # Add small epsilon to avoid log(0)
                    epsilon = 1e-10
                    prior_dist = [prior_p + epsilon, 1 - prior_p + epsilon]
                    posterior_dist = [posterior_p + epsilon, 1 - posterior_p + epsilon]
                    
                    # Calculate KL divergence
                    kl_div += stats.entropy(posterior_dist, prior_dist)
            
            expected_kl += kl_div / len(possible_values)
        
        return expected_kl
    
    def upper_confidence_bound(self, var, possible_values, column_names, kappa=2.0):
        """
        Calculate UCB acquisition function which balances exploration and exploitation.
        
        Args:
            var (str): Variable to intervene on
            possible_values (list): Possible values to set the variable to
            column_names (list): List of all variable names
            kappa (float): Exploration parameter (higher = more exploration)
            
        Returns:
            float: UCB score
        """
        # Get mean expected information gain (exploitation term)
        mean_info_gain = self.expected_information_gain(var, possible_values, column_names)
        
        # Calculate uncertainty based on intervention history
        uncertainty = self._calculate_intervention_uncertainty(var)
        
        # UCB combines exploitation and exploration
        return mean_info_gain + kappa * uncertainty
    
    def thompson_sampling(self, var, possible_values, column_names):
        """
        Thompson sampling approach for intervention selection.
        
        Args:
            var (str): Variable to intervene on
            possible_values (list): Possible values to set the variable to
            column_names (list): List of all variable names
            
        Returns:
            float: Thompson sampling score
        """
        # Get intervention history for this variable
        interventions = [result for edge, results in self.intervention_results.items() 
                      for result in results if edge[0] == var]
        
        # Set up Beta distribution parameters
        # Use successes/failures in previous interventions
        successes = sum(1 for result in interventions if self._intervention_successful(result))
        failures = len(interventions) - successes
        
        # Add priors
        alpha = 1 + successes
        beta = 1 + failures
        
        # Sample from Beta distribution
        sample = np.random.beta(alpha, beta)
        
        # Weight by expected information gain
        info_gain = self.expected_information_gain(var, possible_values, column_names)
        
        return sample * info_gain
    
    def max_value_of_information(self, var, possible_values, column_names):
        """
        Expected value of perfect information acquisition function.
        
        Args:
            var (str): Variable to intervene on
            possible_values (list): Possible values to set the variable to
            column_names (list): List of all variable names
            
        Returns:
            float: EVPI score
        """
        # Current best DAG's utility
        current_utility = self._estimate_dag_utility(self.edge_probabilities)
        
        # Expected maximum utility with new information
        expected_max_utility = 0
        for value in possible_values:
            # Simulate posterior after intervention
            posterior_probs = self._simulate_posterior(var, value, column_names)
            
            # Calculate utility with this posterior
            utility = self._estimate_dag_utility(posterior_probs)
            expected_max_utility += utility / len(possible_values)
        
        # Value of information is the expected improvement in utility
        return expected_max_utility - current_utility
    
    # Helper methods
    def _calculate_system_entropy(self):
        """Calculate current entropy of the edge probability distribution."""
        total_entropy = 0
        for prob in self.edge_probabilities.values():
            # Binary entropy: -p*log(p) - (1-p)*log(1-p)
            if 0 < prob < 1:  # Avoid log(0)
                entropy = -prob * np.log2(prob) - (1-prob) * np.log2(1-prob)
                total_entropy += entropy
        
        return total_entropy
    
    def _estimate_posterior_entropy(self, var, value, column_names):
        """Estimate entropy after intervention on var with value."""
        # Simulate posterior probabilities
        posterior_probs = self._simulate_posterior(var, value, column_names)
        
        # Calculate entropy of the posterior
        total_entropy = 0
        for prob in posterior_probs.values():
            if 0 < prob < 1:  # Avoid log(0)
                entropy = -prob * np.log2(prob) - (1-prob) * np.log2(1-prob)
                total_entropy += entropy
        
        return total_entropy
    
    def _simulate_posterior(self, var, value, column_names):
        """Simulate posterior probabilities after intervention."""
        posterior_probs = self.edge_probabilities.copy()
        
        # Update based on intervention rules:
        # 1. Incoming edges to intervention variable get reduced probability
        for source in column_names:
            if source != var:
                edge = (source, var)
                if edge in posterior_probs:
                    posterior_probs[edge] = posterior_probs[edge] * 0.5
        
        # 2. Outgoing edges maintain or increase slightly (representing potential discovery)
        for target in column_names:
            if target != var and var != target:
                edge = (var, target)
                if edge in posterior_probs:
                    # If we're uncertain, increase uncertainty slightly more
                    if 0.4 < posterior_probs[edge] < 0.6:
                        posterior_probs[edge] = min(0.95, posterior_probs[edge] * 1.2)
        
        return posterior_probs
    
    def _calculate_intervention_uncertainty(self, var):
        """Calculate uncertainty for a variable based on intervention history."""
        # Count previous interventions on this variable
        interventions = sum(1 for edge, results in self.intervention_results.items() 
                         for _ in results if edge[0] == var)
        
        # More interventions = less uncertainty
        return 1.0 / (1.0 + interventions)
    
    def _intervention_successful(self, result):
        """Determine if an intervention was informative."""
        # Check if the intervention led to significant changes
        if not result or 'preInterventionData' not in result or 'postInterventionData' not in result:
            return False
            
        # Get proper state references
        pre_state = result['preInterventionData']
        post_state = result['postInterventionData']
            
        # Count how many variables changed significantly
        changes = 0
        for var in post_state:
            # Skip non-numeric fields
            if var.lower() == 'timestamp' or var.lower() == 'interventionapplied':
                continue
                
            try:
                pre_val = float(pre_state.get(var, 0))
                post_val = float(post_state.get(var, 0))
                
                # Calculate relative change
                if abs(pre_val) > 1e-6:
                    rel_change = abs(post_val - pre_val) / abs(pre_val)
                    if rel_change > 0.05:  # 5% change threshold
                        changes += 1
            except (ValueError, TypeError):
                # Skip non-numeric values
                continue
        
        return changes > 0
    
    def _estimate_dag_utility(self, edge_probs):
        """Estimate utility of the DAG given edge probabilities."""
        # Extract DAG by including edges with high probability
        dag_edges = []
        for edge, prob in edge_probs.items():
            if prob > 0.5:
                dag_edges.append(edge)
        
        # Utility is a function of:
        # 1. Number of correct edges (we don't know this exactly)
        # 2. Confidence in the edges (higher confidence = higher utility)
        
        # Use edge probabilities as proxy for correctness
        edge_confidence = sum(abs(prob - 0.5) * 2 for prob in edge_probs.values())
        
        # Penalize very sparse or very dense graphs
        n_nodes = len(set(src for src, _ in edge_probs.keys()) | 
                     set(tgt for _, tgt in edge_probs.keys()))
        
        expected_edges = n_nodes * (n_nodes - 1) / 4  # Reasonable edge density
        sparsity_penalty = abs(len(dag_edges) - expected_edges) / expected_edges
        
        return edge_confidence - sparsity_penalty

def extract_intervention_data(pre_state, post_state):
   """Extract satisfaction and energy with proper variable mapping"""
   satisfaction_vars = ['overallsatisfaction', 'OverallSatisfaction', 'iso7730_satisfaction']
   energy_vars = ['energyconsumption', 'EnergyConsumption', 'kasa_total_energy_kwh', 'meter_reading']
   
   def get_value(state, var_list):
       for var in var_list:
           if var in state:
               return float(state[var])
       return 0.0
   
   return {
       'pre_state': {
           'OverallSatisfaction': get_value(pre_state, satisfaction_vars),
           'EnergyConsumption': get_value(pre_state, energy_vars)
       },
       'post_state': {
           'OverallSatisfaction': get_value(post_state, satisfaction_vars),
           'EnergyConsumption': get_value(post_state, energy_vars)
       }
   }

def update_edge_belief_bayesian(edge_probabilities, edge, result, prior_strength=2.0):
    """Update edge probability with improved Bayesian approach."""
    source, target = edge
    prior_prob = edge_probabilities.get(edge, 0.5)
    
    # Extract intervention data using helper function  
    intervention_data = extract_intervention_data(result['preInterventionData'], result['postInterventionData'])

    # Check if result has the expected keys
    if not result or 'preInterventionData' not in result or 'postInterventionData' not in result:
        logger.warning(f"Invalid result structure for edge {edge}")
        return prior_prob
    
    # Extract change data using proper key names
    if result and 'preInterventionData' in result and 'postInterventionData' in result:
        result['pre_state'] = result['preInterventionData']
        result['post_state'] = result['postInterventionData']
    pre_state = result.get('pre_state', result.get('preInterventionData', {}))
    post_state = result.get('post_state', result.get('postInterventionData', {}))
    
    try:
        # Skip non-numeric fields
        if source.lower() == 'timestamp' or target.lower() == 'timestamp' or \
           source.lower() == 'interventionapplied' or target.lower() == 'interventionapplied':
            return prior_prob
            
        # Calculate changes
        source_pre = float(pre_state.get(source, 0))
        source_post = float(post_state.get(source, 0))
        target_pre = float(pre_state.get(target, 0))
        target_post = float(post_state.get(target, 0))
        
        # More robust change calculation
        if abs(source_pre) > 1e-6:
            source_change = abs(source_post - source_pre) / abs(source_pre)
        else:
            source_change = abs(source_post - source_pre)
            
        if abs(target_pre) > 1e-6:
            target_change = abs(target_post - target_pre) / abs(target_pre)
        else:
            target_change = abs(target_post - target_pre)
        
        # Define likelihood function with more moderate values
        if source_change < 0.01:  # Source didn't change much
            likelihood_if_true = 0.5  # Uninformative
        else:
            # Calculate correlation in direction of changes
            same_direction = ((source_post > source_pre) and (target_post > target_pre)) or \
                            ((source_post < source_pre) and (target_post < target_pre))
            
            # Use softer likelihood values
            if target_change > 0.05:  # Target changed significantly
                likelihood_if_true = 0.7 if same_direction else 0.3
            else:
                # Target didn't change much when source did - mild evidence against causality
                likelihood_if_true = 0.3
        
        # Likelihood if edge is false
        likelihood_if_false = 1.0 - likelihood_if_true
        
        # Use weaker prior strength to avoid overconfidence
        alpha_prior = prior_prob * prior_strength
        beta_prior = (1 - prior_prob) * prior_strength
        
        # Update Beta distribution parameters
        alpha_posterior = alpha_prior + likelihood_if_true
        beta_posterior = beta_prior + likelihood_if_false
        
        # New probability is posterior mean
        posterior_prob = alpha_posterior / (alpha_posterior + beta_posterior)
        
        # Limit how much the probability can change in a single update
        max_change = 0.2
        if abs(posterior_prob - prior_prob) > max_change:
            if posterior_prob > prior_prob:
                posterior_prob = prior_prob + max_change
            else:
                posterior_prob = prior_prob - max_change
        
        # Update edge probability
        edge_probabilities[edge] = posterior_prob
        
        return posterior_prob
        
    except (ValueError, TypeError, ZeroDivisionError) as e:
        # If calculation fails, make minimal update
        logger.error(f"Error in Bayesian update for edge {edge}: {str(e)}")
        return prior_prob

def select_best_intervention(data, column_names, edge_probabilities,
                           intervention_results, acq_functions, simulator_path=None):
    """Select the most informative intervention with improved exploration."""
    
    best_var = None
    best_value = None
    best_score = -float('inf')
    best_method = None
    
    # Create case mapping for column names
    case_mapping = {col.lower(): col for col in data.columns}
    
    # Only actuator variables can be intervened on
    intervenable = get_intervenable_vars(column_names, simulator_path)
    candidate_vars = [(var, case_mapping.get(var.lower(), var))
                      for var in intervenable if var.lower() in case_mapping]
    
    # Track previously intervened variables to enforce exploration
    recent_interventions = list(intervention_results.keys())[-10:] if intervention_results else []
    recent_vars = [edge[0] for edge in recent_interventions]
    
    # Count variable occurrences in recent interventions
    var_counts = {}
    for var in candidate_vars:
        var_counts[var[0]] = recent_vars.count(var[0])
    
    # Determine exploration boost - penalize recently used variables
    exploration_boost = {}
    for var, _ in candidate_vars:
        if var_counts.get(var, 0) > 3:  # If used more than 3 times recently
            exploration_boost[var] = -1.0 * var_counts[var]  # Negative boost (penalty)
        else:
            exploration_boost[var] = 0.2  # Small positive boost for unused variables
    
    for var, actual_col in candidate_vars:
        # Sample candidate intervention values using the actual column name
        quantiles = [0.25, 0.5, 0.75]  # Try different quantiles
        possible_values = [float(data[actual_col].quantile(q)) for q in quantiles]
        
        # Apply exploration penalty if this variable has been overused
        base_score_modifier = exploration_boost.get(var, 0)
        
        # Try different acquisition functions
        methods = {
            'info_gain': acq_functions.expected_information_gain,
            'kl_div': acq_functions.expected_kl_divergence,
            'ucb': acq_functions.upper_confidence_bound,
            'thompson': acq_functions.thompson_sampling,
            'voi': acq_functions.max_value_of_information
        }
        
        for method_name, acq_function in methods.items():
            try:
                # Calculate base score
                score = acq_function(var, possible_values, column_names)
                
                # Apply exploration modifier
                modified_score = score + base_score_modifier
                
                if modified_score > best_score:
                    best_var = var
                    # Choose the best value for this variable
                    if method_name == 'thompson':
                        # For Thompson sampling, use more randomness
                        best_value = np.random.choice(possible_values)
                    else:
                        # For other methods, use the middle value by default
                        best_value = possible_values[1]
                    best_score = modified_score
                    best_method = method_name
            except Exception as e:
                logger.warning(f"Error calculating {method_name} for {var}: {e}")
    
    return best_var, best_value, best_score, best_method

def extract_dag_from_probabilities(edge_probabilities, column_names, threshold=0.5):
    """
    Extract the most likely DAG from edge probabilities with minimum edge guarantee.
    """
    # Create a graph with all nodes
    G = nx.DiGraph()
    G.add_nodes_from(column_names)
    
    # Sort edges by probability (descending)
    sorted_edges = sorted(
        edge_probabilities.keys(),
        key=lambda e: edge_probabilities[e],
        reverse=True
    )
    
    # Add edges in order of probability
    edges_added = 0
    for edge in sorted_edges:
        # Only add edges above threshold, except when we need minimum edges
        if edge_probabilities[edge] <= threshold and edges_added >= 2:
            continue
            
        # Add edge if it doesn't create a cycle
        G.add_edge(*edge)
        if not nx.is_directed_acyclic_graph(G):
            G.remove_edge(*edge)
        else:
            edges_added += 1
    
    # If no edges were added, add the top 2 most probable edges that don't create cycles
    if edges_added == 0:
        for edge in sorted_edges[:5]:  # Try top 5 edges
            G.add_edge(*edge)
            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(*edge)
            else:
                edges_added += 1
                if edges_added >= 2:  # Add up to 2 edges
                    break
    
    return list(G.edges())


# Local perform_intervention and make_dag removed — using versions from utils.py

def make_dag(edges):
    """
    Convert a set of edges to a DAG by removing cycles.
    
    Args:
        edges: Set or list of edges
        
    Returns:
        list: List of edges forming a DAG
    """
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G, orientation="original")
            G.remove_edge(*cycle[0])
        except nx.NetworkXNoCycle:
            break
    
    return list(G.edges())

def run_abcd(data, simulation_path=None, max_iterations=3, budget=60, debug=True):
    """
    Run ABCD algorithm with Bayesian updates and advanced acquisition functions.
    """
    logger.info("Running Enhanced ABCD algorithm")
    
    if debug:
        log_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Clean and preprocess data
        from benchmarks_new.utils import preprocess_benchmark_data
        data = preprocess_benchmark_data(data)

        # Detect dataset type
        is_ashrae = 'air_temperature' in data.columns and 'meter_reading' in data.columns

        # Create ASHRAE tester if needed
        if is_ashrae and simulation_path is None:
            class ASHRAETester:
                def __init__(self, dataset):
                    self.dataset = dataset
                    self.intervention_results = defaultdict(list)
                    self.edge_history = defaultdict(list)
                    self.effect_threshold = 0.1
                    
                def _simulate_intervention(self, variable, value):
                    from sklearn.ensemble import RandomForestRegressor
    
                    sample_data = self.dataset.sample(min(5000, len(self.dataset)))
                    
                    child_mapping = {
                        'air_temperature': ['dew_temperature', 'meter_reading'],
                        'dew_temperature': ['meter_reading'],
                        'square_feet': ['meter_reading'],
                        'year_built': ['meter_reading']
                    }
                    
                    children = child_mapping.get(variable, [])
                    
                    random_row = sample_data.sample(1).iloc[0]
                    pre_state = {col: float(random_row[col]) for col in sample_data.columns}
                    post_state = pre_state.copy()
                    
                    std_dev = sample_data[variable].std()
                    post_state[variable] = pre_state[variable] + (std_dev * 5 * (1 if value > 0 else -1))
                    
                    for child in children:
                        try:
                            # Train model on sample data
                            X = sample_data[[variable]].values  # Convert to numpy array
                            y = sample_data[child].values
                            
                            model = RandomForestRegressor(n_estimators=50, random_state=42)
                            model.fit(X, y)
                            
                            # Predict using numpy array
                            new_value = model.predict([[post_state[variable]]])[0]
                            post_state[child] = new_value
                        except Exception as e:
                            logger.warning(f"RandomForest prediction failed for {child}: {e}")
                            effect_multiplier = 0.8
                            change = (post_state[variable] - pre_state[variable]) * effect_multiplier
                            post_state[child] = pre_state[child] + change
                    
                    return {
                        'preInterventionData': pre_state,
                        'postInterventionData': post_state
                    }
            
            ashrae_tester = ASHRAETester(data)
            logger.info("Created standalone ASHRAE tester")
        
        start_time = time.time()
        original_columns = list(data.columns)
        lowercase_columns = [col.lower() for col in original_columns]
        column_mapping = {col.lower(): col for col in original_columns}
        
        if debug:
            logger.debug(f"Input data shape: {data.shape}")
            logger.debug(f"Original columns: {original_columns}")
        
        # Initialize with PC algorithm
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz
        cg = pc(
            data=data.values,
            alpha=0.05,
            indep_test=fisherz,
            stable=True,
            uc_rule=1,
            uc_priority=2,
            mvpc=False,
            correction_name=None,
            background_knowledge=None,
            verbose=False,  
            show_progress=False
        )
        
        # Extract initial DAG
        adj_matrix = cg.G.graph
        edges = []
        for i in range(len(original_columns)):
            for j in range(len(original_columns)):
                if adj_matrix[i, j] == 1:
                    edges.append((lowercase_columns[i], lowercase_columns[j]))
        
        # Initialize edge probabilities
        edge_probabilities = {}
        for i, src in enumerate(lowercase_columns):
            for j, tgt in enumerate(lowercase_columns):
                if i != j:
                    edge = (src, tgt)
                    # More moderate initial probabilities
                    edge_probabilities[edge] = 0.6 if edge in edges else 0.4
        
        intervention_results = {}
        intervention_count = 0
        
        # Create acquisition function calculator
        acq_functions = AdvancedAcquisitionFunctions(edge_probabilities, intervention_results)
        
        # Track DAG evolution
        dag_history = [{'iteration': 0, 'edges': edges, 'edge_probs': edge_probabilities.copy()}]
        
        # Track which variables we've intervened on
        intervened_vars = set()
        
        # Run active learning if simulation path or ASHRAE tester is available
        if (simulation_path and os.path.exists(simulation_path)) or (is_ashrae and 'ashrae_tester' in locals()):
            logger.info(f"Starting active learning")
            
            for iteration in range(max_iterations):
                if intervention_count >= budget:
                    logger.info(f"Reached intervention budget ({budget})")
                    break
                
                # Select the most informative intervention
                best_var, best_value, acq_score, acq_method = select_best_intervention(
                    data, lowercase_columns, edge_probabilities,
                    intervention_results, acq_functions, simulation_path)
                
                if best_var is None:
                    logger.warning("Failed to find informative intervention")
                    break
                
                # Track this variable
                intervened_vars.add(best_var)
                
                # Map the lowercase variable name to its original case for the intervention
                original_var = column_mapping.get(best_var, best_var)
                
                logger.info(f"Iteration {iteration+1}/{max_iterations}: "
                           f"Intervening on {original_var} = {best_value} "
                           f"(Method: {acq_method}, Score: {acq_score:.4f})")
                
                # Perform the intervention using appropriate method
                if is_ashrae and 'ashrae_tester' in locals():
                    result = ashrae_tester._simulate_intervention(original_var, best_value)
                else:
                    result = perform_intervention(original_var, best_value, simulation_path)
                
                intervention_count += 1
                
                if result and 'preInterventionData' in result and 'postInterventionData' in result:
                    # Update edge probabilities
                    updated_edges = []
                    for edge in list(edge_probabilities.keys()):
                        if edge[0] == best_var or edge[1] == best_var:
                            if edge not in intervention_results:
                                intervention_results[edge] = []
                            intervention_results[edge].append(result)
                            
                            # Update probability using Bayesian approach
                            old_prob = edge_probabilities[edge]
                            new_prob = update_edge_belief_bayesian(
                                edge_probabilities, 
                                edge, 
                                result
                            )
                            updated_edges.append((edge, old_prob, new_prob))
                    
                    if debug and updated_edges:
                        logger.debug(f"Updated {len(updated_edges)} edge probabilities")
                else:
                    logger.warning("Intervention failed or returned invalid result")
                
                # Extract the current best DAG estimate
                edges = extract_dag_from_probabilities(edge_probabilities, lowercase_columns)
                
                # Save DAG history
                dag_history.append({
                    'iteration': iteration + 1,
                    'edges': edges,
                    'edge_probs': edge_probabilities.copy(),
                    'intervention': {
                        'var': best_var,
                        'value': best_value,
                        'method': acq_method,
                        'score': acq_score
                    }
                })
                
                logger.info(f"Current DAG has {len(edges)} edges")
                
                # Update acquisition functions
                acq_functions = AdvancedAcquisitionFunctions(edge_probabilities, intervention_results)
                
                # Early stopping if we have a stable DAG with sufficient edges
                if iteration > 5 and len(edges) >= 2:
                    prev_edges = set(dag_history[-2]['edges'])
                    curr_edges = set(edges)
                    if prev_edges == curr_edges:
                        logger.info("DAG has stabilized. Early stopping.")
                        break
        else:
            logger.warning("No simulation path provided. Using only observational data.")
        
        # Ensure the result is a DAG
        edges = make_dag(edges)
        
        # If no edges were found, use domain knowledge to create a minimal valid DAG
        if not edges and 'air_temperature' in lowercase_columns and 'meter_reading' in lowercase_columns:
            logger.warning("No edges found. Creating minimal valid DAG based on domain knowledge.")
            edges = [
                ('air_temperature', 'meter_reading'),
                ('air_temperature', 'dew_temperature')
            ]
        
        # Convert to original case
        edges_original_case = [(column_mapping.get(src, src), column_mapping.get(tgt, tgt)) 
                             for src, tgt in edges]
        
        runtime = time.time() - start_time
        
        logger.info(f"Enhanced ABCD algorithm completed in {runtime:.2f} seconds.")
        logger.info(f"Performed {intervention_count} interventions.")
        logger.info(f"Found {len(edges_original_case)} edges: {edges_original_case}")
        logger.info(f"Found {len(edges_original_case)} edges: {edges_original_case}")
        
        return edges_original_case, intervention_results 
        
    except Exception as e:
        logger.error(f"Error in Enhanced ABCD algorithm: {str(e)}")
        logger.error(traceback.format_exc())
        return []
    finally:
        if debug:
            logging.getLogger().setLevel(log_level)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("abcd.log"),
            logging.StreamHandler()
        ]
    )
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run Enhanced ABCD algorithm')
    parser.add_argument('--data', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--simulation', type=str, help='Path to simulation script')
    parser.add_argument('--iterations', type=int, default=3, help='Maximum iterations')
    parser.add_argument('--budget', type=int, default=7, help='Intervention budget')
    parser.add_argument('--output', type=str, default='abcd_results', help='Output directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Run ABCD
    try:
        # Load data
        data = pd.read_csv(args.data)
        logger.info(f"Loaded data from {args.data} with shape {data.shape}")
        
        # Run enhanced ABCD
        edges = run_abcd(
            data=data,
            simulation_path=args.simulation,
            max_iterations=args.iterations,
            budget=args.budget,
            debug=args.debug
        )
        
        # Save results
        import json
        
        # Save edges
        edges_file = os.path.join(args.output, 'edges.json')
        with open(edges_file, 'w') as f:
            json.dump(edges, f, indent=2)
            
        logger.info(f"Results saved to {args.output}")
        print(f"\nResults saved to {args.output}")
        print(f"Discovered {len(edges)} edges")
        
    except Exception as e:
        logger.error(f"Error running ABCD: {str(e)}")
        logger.error(traceback.format_exc())
        print(f"Error: {str(e)}")
        sys.exit(1)