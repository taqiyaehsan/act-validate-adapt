#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PC Algorithm baseline implementation for comparison.
"""

import logging
import numpy as np
import pandas as pd
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from benchmarks.utils import test_edge_validation

logger = logging.getLogger(__name__)

def run_pc_baseline(data):
    """
    Run the PC algorithm without interventions on the provided data.
    
    Args:
        data (pd.DataFrame): Input data for causal discovery
        
    Returns:
        list: List of edges in the discovered causal graph
    """
    logger.info("Running PC baseline algorithm")
    
    try:
        # Select relevant columns (exclude metadata or non-variable columns)
        if 'Timestamp' in data.columns:
            data = data.drop(columns=['Timestamp'])
        if 'weight' in data.columns:
            weights = data['weight'].values
            data = data.drop(columns=['weight'])
            
        # Define PC algorithm parameters
        alpha = 0.05
        indep_test = fisherz
        stable = True
        uc_rule = 1  # Changed from 'pc' to 1
        uc_priority = 2  # Use priority 2: prioritize existing colliders
        mvpc = False
        correction_name = None
        background_knowledge = None
        verbose = False
        show_progress = False
            
        # Convert to numpy array for causallearn
        column_names = data.columns.tolist()
        
        # Check for missing values
        if data.isnull().values.any():
            logger.warning("Input data contains missing values. Dropping rows with missing values.")
            data = data.dropna()
        
        data_array = data.values
        
        # Handle weighted data if weights are available
        if 'weight' in locals():
            weighted_data = data * weights[:, np.newaxis]
            data_array = weighted_data.values
        
        logger.info(f"Input data shape: {data_array.shape}")
        logger.info(f"Variables: {column_names}")
        
        # Run PC algorithm with identical parameters to the main pipeline
        cg = pc(data=data_array, 
                alpha=alpha, 
                indep_test=indep_test, 
                stable=stable, 
                uc_rule=uc_rule, 
                uc_priority=uc_priority,
                mvpc=mvpc,
                correction_name=correction_name,
                background_knowledge=background_knowledge,
                verbose=verbose,
                show_progress=show_progress)
        
        # Extract edges using the same method as PCGenerator
        edges = _get_edges(cg.G.graph, column_names)
        
        logger.info(f"PC algorithm completed. Found {len(edges)} edges")
        logger.info(f"Edges: {edges}")
        
        return edges
        
    except Exception as e:
        logger.error(f"Error in PC baseline: {str(e)}")
        return []

def _get_edges(graph, column_names):
    """
    Convert PC graph to directed edges only, removing bidirectional edges 
    and retaining only the stronger direction.
    
    Args:
        graph (numpy.ndarray): Adjacency matrix from PC algorithm
        column_names (list): List of variable names
        
    Returns:
        list: List of directed edges as tuples (source, target)
    """
    edges = set()  # Use a set to avoid duplicate edges
    n_nodes = len(column_names)
    
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):  # Iterate only once per pair (i, j)
            if graph[i, j] in {1, -1} and graph[j, i] in {1, -1}:  # Bidirectional case
                # Retain the stronger direction (higher absolute value)
                if abs(graph[i, j]) >= abs(graph[j, i]):
                    edges.add((column_names[i], column_names[j]))
                else:
                    edges.add((column_names[j], column_names[i]))
            elif graph[i, j] in {1, -1}:  # One-way edge (i → j)
                edges.add((column_names[i], column_names[j]))
            elif graph[j, i] in {1, -1}:  # One-way edge (j → i)
                edges.add((column_names[j], column_names[i]))
    
    return list(edges)  # Convert back to list for final output

if __name__ == "__main__":
    # Small test for debugging
    logging.basicConfig(level=logging.INFO)
    
    # Generate random data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 3)  # 3 variables
    X[:, 2] = 0.8 * X[:, 0] + 0.2 * X[:, 1] + 0.1 * np.random.randn(n_samples)  # X2 = f(X0, X1)
    
    df = pd.DataFrame(X, columns=['X0', 'X1', 'X2'])
    
    edges = run_pc_baseline(df)
    print(f"Discovered edges: {edges}")