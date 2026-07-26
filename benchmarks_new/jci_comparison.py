#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
JCI (Joint Causal Inference) implementation for comparison.
This implementation uses a simplified approach with PC algorithm and custom handling
of observational and interventional data.
"""

import logging
import numpy as np
import pandas as pd
import tempfile
import os
from collections import defaultdict
import networkx as nx
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

from benchmarks_new.utils import perform_intervention, make_dag, get_intervenable_vars
from benchmarks_new.utils import test_edge_validation

logger = logging.getLogger(__name__)

def run_jci(data, simulation_path=None, max_interventions=10):
    """
    Run the JCI algorithm with support for both ASHRAE and smart_room datasets.
    """
    logger.info("Running JCI algorithm")
    
    try:
        # Clean and preprocess data
        from benchmarks_new.utils import preprocess_benchmark_data
        data = preprocess_benchmark_data(data)
            
        column_names = list(data.columns)
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Variables: {column_names}")
        
        # Detect dataset type
        is_ashrae = 'air_temperature' in data.columns and 'meter_reading' in data.columns
        
        # Initialize with PC algorithm
        try:
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
            initial_edges = []
            for i in range(len(column_names)):
                for j in range(len(column_names)):
                    if adj_matrix[i, j] == 1:
                        initial_edges.append((column_names[i], column_names[j]))
            
            G = nx.DiGraph()
            G.add_edges_from(initial_edges)
            logger.info(f"PC algorithm found {len(initial_edges)} initial edges")
            
        except Exception as e:
            logger.warning(f"PC algorithm failed: {str(e)}")
            G = nx.DiGraph()
            G.add_nodes_from(column_names)
            initial_edges = []
        
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
        
        # Determine which variables can be intervened on
        if is_ashrae:
            interventional_vars = [col for col in column_names if col.lower() != 'meter_reading']
        else:
            interventional_vars = get_intervenable_vars(column_names, simulation_path)
        
        # Choose variables to intervene on
        np.random.seed(42)
        if len(interventional_vars) > max_interventions:
            intervention_vars = np.random.choice(interventional_vars, size=max_interventions, replace=False)
        else:
            intervention_vars = interventional_vars
            
        logger.info(f"Selected variables for intervention: {intervention_vars}")
        
        intervention_results = {} 

        # Run interventions if we have a way to do them
        from benchmarks_new.utils import _raw_benchmark_data
        if (simulation_path and os.path.exists(simulation_path)) or (is_ashrae and 'ashrae_tester' in locals()) or (_raw_benchmark_data is not None):
            logger.info("Collecting intervention data")
            
            intervention_datasets = {}
            intervention_count = 0
            
            for var in intervention_vars:
                try:
                    low_val = data[var].quantile(0.25)
                    high_val = data[var].quantile(0.75)
                    
                    intervention_values = [low_val, high_val]
                    intervention_datasets[var] = []
                    
                    # Perform interventions
                    for value in intervention_values:
                        for _ in range(3):  # 3 repetitions
                            if intervention_count >= max_interventions: break
                                
                            # Use appropriate intervention method
                            if is_ashrae and 'ashrae_tester' in locals():
                                result = ashrae_tester._simulate_intervention(var, value)
                            else:
                                result = perform_intervention(var, value, simulation_path)
                                
                            intervention_count += 1
                            
                            if result and 'postInterventionData' in result and 'preInterventionData' in result:
                                # Store intervention data for MetricsCalculator
                                for target_var in column_names:
                                    if target_var != var:
                                        edge = (var, target_var)
                                        intervention_data = extract_intervention_data(
                                            result['preInterventionData'], 
                                            result['postInterventionData']
                                        )
                                        if edge not in intervention_results:
                                            intervention_results[edge] = []
                                        intervention_results[edge].append(intervention_data)
                                
                                # Store for JCI processing
                                post_data = pd.Series(result['postInterventionData'])
                                missing_cols = set(column_names) - set(post_data.index)
                                for col in missing_cols:
                                    post_data[col] = np.nan
                                intervention_datasets[var].append(post_data)
                            
                        if intervention_count >= max_interventions: break
                    
                    logger.info(f"Collected {len(intervention_datasets[var])} intervention samples for {var}")
                    
                except Exception as e:
                    logger.warning(f"Error with {var}: {str(e)}")
                    continue
            
            # Process intervention data
            causal_effects = []
            
            for intervened_var in intervention_vars:
                if not intervention_datasets.get(intervened_var): continue
                    
                intervention_df = pd.DataFrame(intervention_datasets[intervened_var])
                if intervention_df.empty: continue
                
                for target_var in column_names:
                    if target_var == intervened_var or target_var not in intervention_df.columns:
                        continue
                        
                    try:
                        orig_mean = data[target_var].mean()
                        orig_std = data[target_var].std()
                        int_mean = intervention_df[target_var].mean()
                        
                        if orig_std > 0:
                            effect_size = abs(int_mean - orig_mean) / orig_std
                            
                            if effect_size > 0.2:
                                causal_effects.append((intervened_var, target_var, effect_size))
                    except Exception as e:
                        logger.warning(f"Effect calculation error: {str(e)}")
            
            # Sort and add edges
            causal_effects.sort(key=lambda x: x[2], reverse=True)
            logger.info(f"Identified {len(causal_effects)} causal effects")
            
            added_edges = 0
            for src, tgt, effect in causal_effects:
                G.add_edge(src, tgt, weight=effect)
                if not nx.is_directed_acyclic_graph(G):
                    G.remove_edge(src, tgt)
                else:
                    added_edges += 1
            
            logger.info(f"Added {added_edges} edges based on intervention data")
        else:
            logger.warning("No simulation path or tester. Using only observational data.")
        
        # Extract final edges
        edges = list(G.edges())
        edges = make_dag(edges)
        
        logger.info(f"JCI algorithm completed. Found {len(edges)} edges")
        logger.info(f"Edges: {edges}")
        
        return edges, intervention_results
        
    except Exception as e:
        logger.error(f"Error in JCI algorithm: {str(e)}")
        logger.error("Returning empty DAG as fallback")
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
        edges = run_jci(df)  # No simulation path for this test
        print(f"Discovered edges: {edges}")
    except Exception as e:
        print(f"JCI test failed: {str(e)}")