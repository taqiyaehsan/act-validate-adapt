#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GIES (Greedy Interventional Equivalence Search) implementation using juangamella/gies.
Implementation based on the actual package API structure.
"""

import logging
import numpy as np
import pandas as pd
import networkx as nx
from benchmarks_new.utils import perform_intervention, make_dag, get_intervenable_vars

logger = logging.getLogger(__name__)

def run_gies(data, simulation_path=None, max_intervention_sets=10):
    """
    Run the GIES algorithm with interventions on the provided data using juangamella/gies.
    
    Args:
        data (pd.DataFrame): Input data for causal discovery
        simulation_path (str, optional): Path to simulation script for intervention
        max_intervention_sets (int, optional): Maximum number of intervention sets
        
    Returns:
        list: List of edges in the discovered causal graph
    """
    logger.info("Running GIES algorithm with interventions (juangamella/gies)")
    
    try:
        # First check if gies is installed
        try:
            import gies
            from gies.scores import GaussIntL0Pen
            logger.info(f"Found gies package: {gies.__file__}")
        except ImportError as e:
            logger.error(f"Error importing from juangamella/gies package: {e}")
            logger.error("Make sure you installed with: pip install git+https://github.com/juangamella/gies.git")
            raise ImportError(f"Error importing from juangamella/gies package: {e}")
        
        # Preprocess: standardize + handle near-singular correlation matrices
        from benchmarks_new.utils import preprocess_benchmark_data
        data = preprocess_benchmark_data(data)

        column_names = list(data.columns)
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Variables: {column_names}")

        # Convert data to numpy array
        X = data.values
        n_samples, n_vars = X.shape
        
        # Prepare dataset list and interventions list
        datasets = [X]  # Start with observational data
        interventions = [[]]  # No intervention for observational data
        intervention_results = {} 
        
        # If simulation path is provided, collect intervention data
        if simulation_path:
            logger.info(f"Starting interventions with {simulation_path}")
            
            # Only actuator variables can be intervened on in the simulator
            intervenable_names = get_intervenable_vars(column_names, simulation_path)
            interventional_vars = [i for i, col in enumerate(column_names)
                                   if col in intervenable_names]
            intervention_vars = interventional_vars
            logger.info(f"Intervenable variables: {[column_names[i] for i in intervention_vars]}")
            
            # Perform interventions for different variables
            intervention_count = 0
            
            for var_idx in intervention_vars:
                var_name = column_names[var_idx]
                
                # Define intervention values
                low_val = data[var_name].quantile(0.25)
                high_val = data[var_name].quantile(0.75)
                
                for value in [low_val, high_val]:
                    intervention_data = []
                    
                    for _ in range(3):
                        result = perform_intervention(var_name, value, simulation_path)
                        if result and 'postInterventionData' in result and 'preInterventionData' in result:
                            # Store intervention data for MetricsCalculator
                            for target_col in column_names:
                                if target_col != var_name:
                                    edge = (var_name, target_col)
                                    intervention_data_formatted = extract_intervention_data(
                                        result['preInterventionData'], 
                                        result['postInterventionData']
                                    )
                                    
                                    if edge not in intervention_results:
                                        intervention_results[edge] = []
                                    intervention_results[edge].append(intervention_data_formatted)
                            
                            # Convert for GIES algorithm
                            row = []
                            for col in column_names:
                                if col in result['postInterventionData']:
                                    try:
                                        val = float(result['postInterventionData'][col])
                                        row.append(val)
                                    except (ValueError, TypeError):
                                        row.append(data[col].mean())
                                else:
                                    row.append(data[col].mean())
                            
                            intervention_data.append(row)
                            intervention_count += 1
                            if intervention_count >= max_intervention_sets * 6:
                                break
                    
                    # Add dataset for GIES
                    if intervention_data:
                        datasets.append(np.array(intervention_data))
                        interventions.append([var_idx])
                
                if intervention_count >= max_intervention_sets * 6:
                    break
            
            logger.info(f"Created {len(datasets)-1} intervention datasets")
        
        # Create the score using the GaussIntL0Pen class
        score = GaussIntL0Pen(datasets, interventions)
        
        # Run GIES using the fit function
        estimate, score_value = gies.fit(score)
        
        logger.info(f"GIES fit completed with score: {score_value}")
        
        # Convert CPDAG adjacency matrix to edge list
        # For directed edges (estimate[i,j]!=0 but estimate[j,i]==0): add i→j
        # For undirected edges (both non-zero): orient using regression R²
        from sklearn.linear_model import LinearRegression
        edges = []
        seen = set()
        for i in range(n_vars):
            for j in range(n_vars):
                if i == j or (i, j) in seen:
                    continue
                if estimate[i, j] != 0 and estimate[j, i] == 0:
                    # Directed: i→j
                    edges.append((column_names[i], column_names[j]))
                    seen.add((i, j))
                elif estimate[i, j] != 0 and estimate[j, i] != 0:
                    # Undirected: orient by regression R²
                    seen.add((i, j))
                    seen.add((j, i))
                    xi = data[[column_names[i]]].values
                    xj = data[[column_names[j]]].values
                    r2_ij = LinearRegression().fit(xi, xj.ravel()).score(xi, xj.ravel())
                    r2_ji = LinearRegression().fit(xj, xi.ravel()).score(xj, xi.ravel())
                    if r2_ij >= r2_ji:
                        edges.append((column_names[i], column_names[j]))
                    else:
                        edges.append((column_names[j], column_names[i]))
        
        logger.info(f"GIES algorithm completed. Found {len(edges)} edges")
        logger.info(f"Edges: {edges}")
        
        return edges
        
    except Exception as e:
        logger.error(f"Error in GIES algorithm: {str(e)}")
        raise

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
        edges = run_gies(df)  # No simulation path for this test
        print(f"Discovered edges: {edges}")
    except ImportError:
        print("Skipping GIES test due to missing dependencies")