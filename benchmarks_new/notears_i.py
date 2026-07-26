#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified NOTEARS with Intervention (NOTEARS-I) implementation for comparison.
This version doesn't rely on causal-learn's NOTEARS implementation.
"""
import os
import logging
import numpy as np
import pandas as pd
import networkx as nx
from benchmarks_new.utils import perform_intervention, make_dag, get_intervenable_vars
from benchmarks_new.utils import test_edge_validation

logger = logging.getLogger(__name__)

def run_notears_i(data, simulation_path=None, max_interventions=10, threshold=0.2):
    """Run simplified NOTEARS-I algorithm with support for different datasets."""
    logger.info("Running simplified NOTEARS-I algorithm")
    
    try:
        # Clean and preprocess data
        from benchmarks_new.utils import preprocess_benchmark_data
        data = preprocess_benchmark_data(data)
            
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
                    from sklearn.ensemble import RandomForestRegressor    
                    sample_data = self.dataset.sample(min(5000, len(self.dataset)))
                    
                    # Define potential connections based on correlations, not ground truth
                    corr_matrix = sample_data.corr().abs()
                    potential_targets = [col for col in sample_data.columns 
                                       if corr_matrix.loc[variable, col] > threshold
                                       and col != variable]
                    
                    random_row = sample_data.sample(1).iloc[0]
                    pre_state = {col: float(random_row[col]) for col in sample_data.columns}
                    post_state = pre_state.copy()
                    
                    std_dev = sample_data[variable].std()
                    post_state[variable] = pre_state[variable] + (std_dev * 2 * (1 if value > 0 else -1))
                    
                    # Train models to predict effects without predefined relationships
                    for target in potential_targets:
                        try:
                            X = sample_data[[variable]].values
                            y = sample_data[target].values
                            
                            model = RandomForestRegressor(n_estimators=50, random_state=42)
                            model.fit(X, y)
                            
                            new_value = model.predict([[post_state[variable]]])[0]
                            post_state[target] = new_value
                        except Exception as e:
                            logger.warning(f"RandomForest prediction failed for {target}: {e}")
                            # No fallback with predefined effects
                            continue
                    
                    return {
                        'preInterventionData': pre_state,
                        'postInterventionData': post_state
                    }
           
            ashrae_tester = ASHRAETester(data)
            logger.info("Created standalone ASHRAE tester")
        
        # Initial causal discovery based on correlation
        X = data.values
        corr_matrix = np.abs(np.corrcoef(X.T))
        np.fill_diagonal(corr_matrix, 0)
        
        W_est = np.zeros_like(corr_matrix)
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                if i != j and corr_matrix[i, j] > threshold:
                    W_est[i, j] = corr_matrix[i, j]
        
        edges = []
        for i in range(len(column_names)):
            for j in range(len(column_names)):
                if W_est[i, j] != 0:
                    edges.append((column_names[i], column_names[j]))
        
        G = nx.DiGraph()
        G.add_edges_from(edges)
        G = nx.DiGraph(make_dag(list(G.edges())))
        
        intervention_results = {}

        # Perform interventions if possible
        if (simulation_path and os.path.exists(simulation_path)) or (is_ashrae and 'ashrae_tester' in locals()):
            logger.info("Starting interventions")
            
            # Only actuator variables can be intervened on
            intervenable = get_intervenable_vars(column_names, simulation_path)
            potential_vars = [(col, 1.0) for col in intervenable]
            
            # Sort by correlation (highest first)
            potential_vars.sort(key=lambda x: x[1], reverse=True)
            vars_to_test = [v[0] for v in potential_vars[:min(5, len(potential_vars))]]
            
            intervention_count = 0
            intervention_data = []
            
            for var in vars_to_test:
                if intervention_count >= max_interventions:
                    break
                    
                # Define quantile values for testing
                try:
                    low_val = data[var].quantile(0.25)
                    high_val = data[var].quantile(0.75)
                except Exception as e:
                    logger.warning(f"Error calculating quantiles for {var}: {e}")
                    continue
                
                # Perform interventions with different values
                for value in [low_val, high_val]:
                    if is_ashrae and 'ashrae_tester' in locals():
                        result = ashrae_tester._simulate_intervention(var, value)
                    else:
                        result = perform_intervention(var, value, simulation_path)
                    
                    if result and 'postInterventionData' in result and 'preInterventionData' in result:
                        # Store intervention data for MetricsCalculator
                        for target in column_names:
                            if target != var and target.lower() not in {'timestamp', 'interventionapplied', 'weight'}:
                                edge = (var, target)
                                intervention_data_formatted = extract_intervention_data(
                                    result['preInterventionData'], 
                                    result['postInterventionData']
                                )
                                if edge not in intervention_results:
                                    intervention_results[edge] = []
                                intervention_results[edge].append(intervention_data_formatted)
                        
                        # Update edge beliefs for NOTEARS-I
                        update_edge_belief(G, var, target, 
                                        result['preInterventionData'], 
                                        result['postInterventionData'])
                            
                        intervention_count += 1
                        if intervention_count >= max_interventions:
                            break
            
            logger.info(f"Performed {intervention_count} interventions")
        else:
            logger.warning("No simulation path provided. Using only observational data.")
        
        # Extract edges from final graph
        edges = list(G.edges())
        
        # Remove metadata-related edges
        filtered_edges = []
        for src, tgt in edges:
            # Skip edges involving metadata fields
            if (src.lower() in {'timestamp', 'interventionapplied', 'weight'} or 
                tgt.lower() in {'timestamp', 'interventionapplied', 'weight'}):
                continue
            filtered_edges.append((src, tgt))
            
        # Ensure it's a DAG
        filtered_edges = make_dag(filtered_edges)
        
        logger.info(f"Simplified NOTEARS-I algorithm completed. Found {len(filtered_edges)} edges")
        logger.info(f"Edges: {filtered_edges}")
        
        return filtered_edges, intervention_results
        
    except Exception as e:
        logger.error(f"Error in simplified NOTEARS-I algorithm: {str(e)}")
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

def fix_ashrae_edge_directions(edges):
    """Fix edge directions specifically for ASHRAE dataset."""
    fixed_edges = []
    
    for src, tgt in edges:
        # Check if edge direction needs to be flipped
        if src.lower() == 'meter_reading' and tgt.lower() in {
            'air_temperature', 'dew_temperature', 
            'square_feet', 'year_built'
        }:
            # Flip the edge direction
            fixed_edges.append((tgt, src))
        else:
            fixed_edges.append((src, tgt))
    
    # Make sure we still have a DAG
    return make_dag(fixed_edges)

def identify_potential_edges(corr_matrix, column_names, threshold=0.2):
   """
   Identify potential edges for targeted intervention based on correlation.
   
   Args:
       corr_matrix (numpy.ndarray): Correlation matrix
       column_names (list): List of variable names
       threshold (float): Correlation threshold
       
   Returns:
       list: List of potential edges as (source, target) tuples
   """
   potential_edges = []
   
   # First get high correlation edges
   high_corr_edges = []
   for i in range(corr_matrix.shape[0]):
       # Skip source variables that can't be intervened on
       if column_names[i].lower() in {'energyconsumption', 'overallsatisfaction', 'collision'}:
           continue
           
       for j in range(corr_matrix.shape[1]):
           # Skip self-loops
           if i == j:
               continue
               
           # Add edges with correlation above threshold
           if corr_matrix[i, j] > threshold:
               high_corr_edges.append((column_names[i], column_names[j], corr_matrix[i, j]))
   
   # Sort by correlation (descending)
   high_corr_edges.sort(key=lambda x: x[2], reverse=True)
   
   # First prioritize edges with highest correlation
   for src, tgt, _ in high_corr_edges[:5]:
       potential_edges.append((src, tgt))
   
   # Then add moderate correlation edges
   mid_corr_edges = []
   for i in range(corr_matrix.shape[0]):
       # Skip source variables that can't be intervened on
       if column_names[i].lower() in {'energyconsumption', 'overallsatisfaction', 'collision'}:
           continue
           
       for j in range(corr_matrix.shape[1]):
           if i != j and threshold/2 < corr_matrix[i, j] <= threshold:
               mid_corr_edges.append((column_names[i], column_names[j], corr_matrix[i, j]))
   
   # Sort by correlation (descending)
   mid_corr_edges.sort(key=lambda x: x[2], reverse=True)
   
   # Add some medium correlation edges
   for src, tgt, _ in mid_corr_edges[:5]:
       if (src, tgt) not in potential_edges:
           potential_edges.append((src, tgt))
   
   return potential_edges

def update_edge_belief(G, src, tgt, pre_state, post_state):
    """Update belief about an edge based on intervention results without hard thresholds."""
    # Skip metadata fields
    if (src.lower() in {'timestamp', 'interventionapplied', 'weight'} or 
        tgt.lower() in {'timestamp', 'interventionapplied', 'weight'}):
        return
        
    # Check if variables are in the states
    if src not in pre_state or src not in post_state or \
       tgt not in pre_state or tgt not in post_state:
        return
    
    try:
        # Calculate changes
        src_pre = float(pre_state[src])
        src_post = float(post_state[src])
        src_change = abs(src_post - src_pre)
        
        tgt_pre = float(pre_state[tgt])
        tgt_post = float(post_state[tgt])
        tgt_change = abs(tgt_post - tgt_pre)
        
        # Use standardized change to account for different scales
        try:
            src_std = max(abs(src_pre), 0.1) * 0.1  # Estimate std if unknown
            tgt_std = max(abs(tgt_pre), 0.1) * 0.1
            
            std_src_change = src_change / src_std
            std_tgt_change = tgt_change / tgt_std
            
            # Only consider significant changes
            if std_src_change < 0.5:  # Source didn't change significantly
                return
                
            # Calculate standardized effect size
            if std_tgt_change > 0.5:  # Target changed significantly
                # Add edge based on intervention data
                G.add_edge(src, tgt)
            
        except (ZeroDivisionError, ValueError):
            pass
            
    except (ValueError, TypeError, ZeroDivisionError):
        # If values can't be converted to float or other error, do nothing
        pass

def apply_causal_constraints(G, intervention_data):
   """
   Apply causal constraints based on intervention data.
   
   Args:
       G (nx.DiGraph): Directed graph to update
       intervention_data (list): List of intervention results
   """
   # Create a map of causal effects by source-target pair
   causal_effects = {}
   
   for intervention in intervention_data:
       src = intervention['src']
       
       pre_state = intervention['pre_state']
       post_state = intervention['post_state']
       
       for var in post_state:
           if var == src or var not in pre_state:
               continue
               
           # Calculate effect size
           try:
               pre_val = float(pre_state[var])
               post_val = float(post_state[var])
               
               # Absolute change
               abs_change = abs(post_val - pre_val)
               
               # Relative change
               rel_change = abs_change / max(1e-6, abs(pre_val))
               
               # If significant change occurred, record it
               if abs_change > 1e-6 and rel_change > 0.05:
                   pair = (src, var)
                   if pair not in causal_effects:
                       causal_effects[pair] = []
                   causal_effects[pair].append(rel_change)
           except (ValueError, TypeError):
               continue
   
   # Apply constraints based on consistent causal effects
   for (src, tgt), effects in causal_effects.items():
       if len(effects) >= 2:  # Require at least 2 consistent observations
           avg_effect = sum(effects) / len(effects)
           if avg_effect > 0.1:
               # Add edge if consistent causal effect observed
               G.add_edge(src, tgt)
               # Ensure we maintain acyclicity
               if not nx.is_directed_acyclic_graph(G):
                   G.remove_edge(src, tgt)

def fix_edge_directions(edges):
   """
   Apply domain knowledge to fix edge directions.
   
   Args:
       edges (list): List of edges
       
   Returns:
       list: List of edges with corrected directions
   """
   fixed_edges = []
   
   for src, tgt in edges:
       # Check if edge direction needs to be flipped
       if src.lower() in {'energyconsumption', 'overallsatisfaction'} and \
          tgt.lower() in {'temperature', 'humidity', 'airquality'}:
           # Flip the edge direction
           fixed_edges.append((tgt, src))
       else:
           fixed_edges.append((src, tgt))
   
   # Make sure we still have a DAG
   return make_dag(fixed_edges)

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
       edges = run_notears_i(df)  # No simulation path for this test
       print(f"Discovered edges: {edges}")
   except ImportError:
       print("Skipping simplified NOTEARS-I test due to missing dependencies")