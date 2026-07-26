#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
IID (Informative Intervention Design) implementation for comparison.
"""
import os
import logging
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import entropy
from scipy.special import softmax
from benchmarks.utils import perform_intervention, make_dag
from benchmarks.utils import test_edge_validation
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

logger = logging.getLogger(__name__)

def run_iid(data, simulation_path=None, max_iterations=5, beta=1.0):
    """Run basic IID algorithm with PC initialization for both datasets."""
    logger.info("Running IID algorithm")
    
    try:
        # Clean data
        if 'Timestamp' in data.columns: data = data.drop(columns=['Timestamp'])
        if 'weight' in data.columns: data = data.drop(columns=['weight'])
            
        column_names = data.columns.tolist()
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Variables: {column_names}")
        
        # Detect dataset type
        is_ashrae = 'air_temperature' in data.columns and 'meter_reading' in data.columns
        
        # Create ASHRAE tester if needed
        if is_ashrae and simulation_path is None:
            class ASHRAETester:
                def __init__(self, dataset):
                    self.dataset = dataset
                    
                def _simulate_intervention(self, variable, value):
                    sample_data = self.dataset.sample(min(5000, len(self.dataset)))
                    random_row = sample_data.sample(1).iloc[0]
                    pre_state = {col: float(random_row[col]) for col in sample_data.columns}
                    post_state = pre_state.copy()
                    
                    # Apply intervention
                    std_dev = sample_data[variable].std()
                    post_state[variable] = pre_state[variable] + (std_dev * 2 * (1 if value > 0 else -1))
                    
                    # Simple linear effects
                    if variable == 'air_temperature' and 'meter_reading' in post_state:
                        post_state['meter_reading'] *= 1.2
                    if variable == 'square_feet' and 'meter_reading' in post_state:
                        post_state['meter_reading'] *= 1.1
                    
                    return {
                        'preInterventionData': pre_state,
                        'postInterventionData': post_state
                    }
            
            ashrae_tester = ASHRAETester(data)
            logger.info("Created standalone ASHRAE tester")
        
        # Initialize with PC algorithm
        logger.info("Initializing with PC algorithm")
        cg = pc(
            data=data.values,
            alpha=0.2,
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
        initial_edges = []
        for i in range(len(column_names)):
            for j in range(len(column_names)):
                if adj_matrix[i, j] == 1:
                    initial_edges.append((column_names[i], column_names[j]))
        
        # Initialize current DAG
        G = nx.DiGraph()
        G.add_edges_from(initial_edges)
        
        # Initialize edge probabilities uniformly
        edge_probabilities = {}
        for i, src in enumerate(column_names):
            for j, tgt in enumerate(column_names):
                if i != j:
                    edge = (src, tgt)
                    edge_probabilities[edge] = 0.5  # Uniform initial probabilities
        
        intervention_results = {}

        # Run interventions if possible
        if (simulation_path and os.path.exists(simulation_path)) or (is_ashrae and 'ashrae_tester' in locals()):
            logger.info("Starting interventions")
            
            intervention_history = []
            
            for iteration in range(max_iterations):
                # Filter out likely invalid intervention targets
                avoid_vars = []
                if is_ashrae:
                    avoid_vars = ['meter_reading']
                else:
                    avoid_vars = ['EnergyConsumption', 'OverallSatisfaction', 'InterventionApplied']
                
                # Select any valid variable that hasn't been intervened on recently
                intervention_candidates = [v for v in column_names 
                                        if v not in avoid_vars 
                                        and v not in (intervention_history[-3:] if intervention_history else [])]
                
                if not intervention_candidates:
                    intervention_candidates = [v for v in column_names if v not in avoid_vars]
                
                if not intervention_candidates:
                    logger.warning("No valid intervention candidates")
                    break
                
                # Select random variable
                intervention_var = np.random.choice(intervention_candidates)
                intervention_value = data[intervention_var].quantile(0.5)  # Median value
                
                intervention_history.append(intervention_var)
                logger.info(f"Iteration {iteration+1}/{max_iterations}: Intervening on {intervention_var}")
                
                # Perform intervention
                if is_ashrae and 'ashrae_tester' in locals():
                    result = ashrae_tester._simulate_intervention(intervention_var, intervention_value)
                else:
                    result = perform_intervention(intervention_var, intervention_value, simulation_path)
                
                if result and 'postInterventionData' in result and 'preInterventionData' in result:
                    # Store intervention data for MetricsCalculator
                    for target_var in column_names:
                        if target_var != intervention_var:
                            edge = (intervention_var, target_var)
                            intervention_data = extract_intervention_data(
                                result['preInterventionData'], 
                                result['postInterventionData']
                            )
                            if edge not in intervention_results:
                                intervention_results[edge] = []
                            intervention_results[edge].append(intervention_data)
               
                    # Basic belief update
                    for var in column_names:
                        if var == intervention_var or var not in result['preInterventionData'] or var not in result['postInterventionData']:
                            continue
                        
                        try:
                            pre_val = float(result['preInterventionData'][var])
                            post_val = float(result['postInterventionData'][var])
                            
                            # If variable changed after intervention, update edge probability
                            if abs(post_val - pre_val) > 0.05 * abs(pre_val):
                                edge = (intervention_var, var)
                                if edge in edge_probabilities:
                                    edge_probabilities[edge] += 0.1  # Simple increment
                                    edge_probabilities[edge] = min(1.0, edge_probabilities[edge])
                        except:
                            pass
                else:
                    logger.warning("Intervention failed")
                
                # Extract current DAG
                G = extract_dag_from_probabilities(edge_probabilities, column_names)
        else:
            logger.warning("No simulation path provided. Using only observational data.")
        
        # Extract final edges
        edges = list(G.edges())
        edges = make_dag(edges)
        
        logger.info(f"IID algorithm completed. Found {len(edges)} edges")
        logger.info(f"Edges: {edges}")
        
        return edges, intervention_results
        
    except Exception as e:
        logger.error(f"Error in IID algorithm: {str(e)}")
        return []
    
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

def select_intervention(data, edge_probabilities, column_names, intervention_history, beta=1.0):
    """Select the most informative intervention with exploration."""
    best_var = None
    best_value = None
    best_gain = -float('inf')
    
    # Hardcoded sinks for different datasets
    likely_sinks = []
    
    # Smart room dataset
    if 'Temperature' in column_names and 'EnergyConsumption' in column_names:
        likely_sinks = ['EnergyConsumption', 'OverallSatisfaction', 'InterventionApplied']
    
    # ASHRAE dataset
    elif 'air_temperature' in column_names and 'meter_reading' in column_names:
        likely_sinks = ['meter_reading']
    
    # Convert to lowercase for case-insensitive comparison
    likely_sinks_lower = [s.lower() for s in likely_sinks]
    
    # Track recent interventions for exploration
    recent_count = {}
    for var in column_names:
        recent_count[var] = intervention_history[-10:].count(var) if intervention_history else 0
    
    # For each variable, compute expected information gain
    for var in column_names:
        # Skip sinks or recently used variables
        if var.lower() in likely_sinks_lower or recent_count[var] > 2:
            continue
            
        # Add exploration bonus
        exploration_bonus = beta * (1.0 - min(1.0, recent_count[var] / 3.0))
        
        try:
            q1 = data[var].quantile(0.25)
            q3 = data[var].quantile(0.75)
            
            for value in [q1, q3]:
                # Add noise to information gain
                info_gain = compute_information_gain(var, value, edge_probabilities, column_names)
                info_gain = info_gain * 0.7 + np.random.uniform(0, 0.3)
                
                adjusted_gain = info_gain + exploration_bonus
                
                if adjusted_gain > best_gain:
                    best_var = var
                    best_value = value
                    best_gain = adjusted_gain
        except:
            continue
    
    # Random backup if nothing selected
    if best_var is None:
        candidates = [v for v in column_names if v.lower() not in likely_sinks_lower and recent_count[v] < 2]
        if candidates:
            best_var = np.random.choice(candidates)
            best_value = data[best_var].median()
            best_gain = 0.0
    
    return best_var, best_value, best_gain

def compute_information_gain(var, value, edge_probabilities, column_names):
    """
    Compute expected information gain for an intervention.
    
    Args:
        var (str): Variable to intervene on
        value (float): Value to set the variable to
        edge_probabilities (dict): Current edge probabilities
        column_names (list): List of variable names
        
    Returns:
        float: Expected information gain
    """
    # Identify edges that would be affected by this intervention
    outgoing_edges = [(src, tgt) for src, tgt in edge_probabilities.keys() if src == var]
    incoming_edges = [(src, tgt) for src, tgt in edge_probabilities.keys() if tgt == var]
    
    # Intervention on var d-separates it from its parents
    # So incoming edges would become less likely
    incoming_entropy = sum(
        binary_entropy(edge_probabilities[edge])
        for edge in incoming_edges
        if 0 < edge_probabilities[edge] < 1
    )
    
    # Outgoing edges would become more certain
    outgoing_entropy = sum(
        binary_entropy(edge_probabilities[edge])
        for edge in outgoing_edges
        if 0 < edge_probabilities[edge] < 1
    )
    
    # Weight by current probabilities to get expected information gain
    info_gain = incoming_entropy + outgoing_entropy
    
    return info_gain

def binary_entropy(p):
    """Compute binary entropy: -p*log(p) - (1-p)*log(1-p)."""
    if p <= 0 or p >= 1:
        return 0
    return -p * np.log2(p) - (1-p) * np.log2(1-p)

def update_beliefs_adaptive(edge_probabilities, intervention_var, pre_state, post_state, data, column_names):
    """Update beliefs with adaptive thresholds based on data."""
    # Calculate typical variation for each variable
    std_devs = {col: data[col].std() for col in column_names if col in data.columns}
    
    # Decrease probability of incoming edges (intervention breaks these)
    incoming_edges = [(src, tgt) for src, tgt in edge_probabilities.keys() if tgt == intervention_var]
    for edge in incoming_edges:
        # More moderate decrease
        edge_probabilities[edge] = max(0.1, edge_probabilities[edge] * 0.8)
    
    # Check which variables changed after intervention
    for var in column_names:
        if var == intervention_var or var not in pre_state or var not in post_state:
            continue
        
        try:
            pre_val = float(pre_state[var])
            post_val = float(post_state[var])
            var_std = std_devs.get(var, abs(pre_val) * 0.1)  # Fallback if std unknown
            
            # Adaptive threshold based on variable's standard deviation
            threshold = max(0.05, min(0.3, var_std / max(abs(pre_val), 0.001)))
            
            # If variable changed significantly, adjust edge probability
            if abs(post_val - pre_val) > threshold * abs(pre_val):
                edge = (intervention_var, var)
                if edge in edge_probabilities:
                    # Directly access the current probability value
                    current = edge_probabilities[edge]
                    # Larger changes result in larger probability updates
                    change_ratio = min(3.0, abs(post_val - pre_val) / (threshold * abs(pre_val)))
                    # Update probability with diminishing returns
                    edge_probabilities[edge] = min(0.9, current + (0.9 - current) * 0.2 * change_ratio)
        except (ValueError, TypeError, ZeroDivisionError):
            pass

def extract_dag_from_probabilities(edge_probabilities, column_names, threshold=0.45):
    """Extract the most likely DAG from edge probabilities."""
    G = nx.DiGraph()
    G.add_nodes_from(column_names)
    
    # Sort edges by probability
    sorted_edges = sorted(
        edge_probabilities.keys(),
        key=lambda e: edge_probabilities[e],
        reverse=True
    )
    
    # Add edges with probability above threshold
    for edge in sorted_edges:
        if edge_probabilities[edge] <= threshold:
            continue
            
        # Add edge if it doesn't create a cycle
        G.add_edge(*edge)
        if not nx.is_directed_acyclic_graph(G):
            G.remove_edge(*edge)
    
    return G

def make_dag(edges):
    """Convert edges to a DAG by removing cycles."""
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycles = list(nx.simple_cycles(G))
            if cycles:
                cycle = cycles[0]
                if len(cycle) >= 2:
                    G.remove_edge(cycle[0], cycle[1])
            else:
                break
        except nx.NetworkXNoCycle:
            break
    
    return list(G.edges())

if __name__ == "__main__":
    # Small test for debugging
    logging.basicConfig(level=logging.INFO)
    
    # Generate random data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 3)  # 3 variables
    X[:, 2] = 0.8 * X[:, 0] + 0.2 * X[:, 1] + 0.1 * np.random.randn(n_samples)  # X2 = f(X0, X1)
    
    df = pd.DataFrame(X, columns=['X0', 'X1', 'X2'])
    
    try:
        edges = run_iid(df)  # No simulation path for this test
        print(f"Discovered edges: {edges}")
    except ImportError:
        print("Skipping IID test due to missing dependencies")