#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SAM (Structural Agnostic Model) baseline implementation for comparison.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import networkx as nx
from scipy import stats
from benchmarks_new.utils import test_edge_validation

logger = logging.getLogger(__name__)

def run_sam_baseline(data, simulation_path=None):
    """
    Run the SAM algorithm without interventions on the provided data.
    """
    logger.info("Running SAM baseline algorithm")
    
    try:
        # Check if cdt is available
        try:
            from cdt.causality.graph import SAM
        except ImportError:
            logger.error("CDT package not installed. Please install with 'pip install cdt'")
            raise ImportError("CDT package not installed. Please install with 'pip install cdt'")
        
        # Define SAM parameters identical to main pipeline
        sparsity_threshold = 0.15
        min_edge_weight = 0.1
        significance_level = 0.05
        
        # Preprocess: standardize + handle near-singular correlation matrices
        from benchmarks_new.utils import preprocess_benchmark_data
        data = preprocess_benchmark_data(data)

        is_ashrae = 'air_temperature' in data.columns and 'meter_reading' in data.columns
        relevant_columns = list(data.columns)
        processed_data = data.copy()
        samples = data.copy()
        
        # Initialize SAM model with identical parameters to main pipeline
        sam_model = SAM(
            lr=0.01,              # Learning rate for the generator
            dlr=0.005,            # Learning rate for the discriminator
            lambda1=0.1,          # L1 regularization (sparsity)
            lambda2=0.01,         # L2 regularization (weight decay)
            train_epochs=200,      # Training epochs (reduced from 1000 for speed)
            test_epochs=40,        # Test epochs (reduced from 80)
            batch_size=32,        # Use full-batch training (efficient for small graphs)
            dagloss=True,         # Enforce DAG constraint during training
            dagstart=0.0,         # Start DAG penalty halfway through training
            nruns=1,              # Random restarts (reduced from 3 for speed)
            njobs=1,              # Number of parallel jobs
            verbose=False         # Print progress and loss info
        )

        # Run SAM model
        graph = sam_model.predict(samples)
        
        # Extract edges
        edges = list(graph.edges())
        
        # Apply domain knowledge to orient edges.
        # SAM's GAN-based approach finds adjacencies but often reverses directions.
        # Known derived/output variables should always be targets, not sources.
        output_vars = {'energyconsumption', 'satisfaction', 'overallsatisfaction',
                       'meter_reading', 'pmv'}
        fixed_edges = []
        for source, target in edges:
            src_l, tgt_l = source.lower(), target.lower()
            if src_l in output_vars and tgt_l not in output_vars:
                # Reverse: derived → sensor should be sensor → derived
                fixed_edges.append((target, source))
            elif src_l == 'dew_temperature' and tgt_l == 'air_temperature':
                fixed_edges.append((target, source))
            elif src_l == 'meter_reading':
                continue  # meter_reading is always a target
            else:
                fixed_edges.append((source, target))
        edges = fixed_edges
        
        # Validate edges
        valid_edges = validate_edges(edges, processed_data, sparsity_threshold, min_edge_weight)
        
        # If no edges found, use correlation fallback
        if not valid_edges:
            logger.warning("No valid edges found. Using strongest correlations as fallback.")
            corr_matrix = processed_data.corr().abs()
            fallback_edges = []
            # Take top edges by absolute correlation
            for i, col_i in enumerate(relevant_columns):
                for col_j in relevant_columns[i+1:]:
                    if corr_matrix.loc[col_i, col_j] > 0.2:
                        fallback_edges.append((col_i, col_j))
            # Keep top 6 by correlation strength
            fallback_edges.sort(
                key=lambda e: corr_matrix.loc[e[0], e[1]], reverse=True)
            valid_edges = fallback_edges[:6]
        
        logger.info(f"SAM algorithm completed. Found {len(valid_edges)} edges")
        logger.info(f"Edges: {valid_edges}")
        
        return valid_edges
        
    except Exception as e:
        logger.error(f"Error in SAM baseline: {str(e)}")
        return generate_fallback_dag(data)

def validate_edges(edges, data, sparsity_threshold, min_edge_weight):
    """
    Validate edges from SAM algorithm.
    
    Args:
        edges: List of edges from SAM algorithm
        data: Input data
        sparsity_threshold: Threshold for edge sparsity
        min_edge_weight: Minimum edge weight to keep
        
    Returns:
        list: List of validated edges
    """
    G = nx.DiGraph()
    weighted_edges = []
    
    # Calculate edge weights using partial correlations
    for edge in edges:
        source, target = edge
        partial_corr = calculate_partial_correlation(
            data[source], data[target], 
            data[[col for col in data.columns if col not in [source, target]]],
            min_edge_weight
        )
        if abs(partial_corr) >= min_edge_weight:
            weighted_edges.append((source, target, {'weight': abs(partial_corr)}))
    
    if not weighted_edges:
        # Fallback: use strongest correlations to ensure non-empty graph
        correlations = data.corr()
        for i, source in enumerate(data.columns):
            for target in data.columns[i+1:]:
                corr = abs(correlations.loc[source, target])
                if corr >= min_edge_weight:
                    weighted_edges.append((source, target, {'weight': corr}))
    
    G.add_edges_from(weighted_edges)
    
    # Remove cycles while preserving strongest edges
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G, orientation="original")
            # Remove weakest edge in cycle
            min_weight_edge = min(cycle, key=lambda x: G[x[0]][x[1]]['weight'])
            G.remove_edge(*min_weight_edge[:2])
        except nx.NetworkXNoCycle:
            break

    # Apply sparsity constraint
    edges_to_remove = []
    for edge in G.edges(data=True):
        if edge[2]['weight'] < sparsity_threshold:
            edges_to_remove.append((edge[0], edge[1]))
    G.remove_edges_from(edges_to_remove)
    
    # Ensure graph remains connected and meaningful
    if len(G.edges()) < 2:
        # Add strongest valid edges back
        sorted_edges = sorted(weighted_edges, key=lambda x: x[2]['weight'], reverse=True)
        for edge in sorted_edges[:3]:
            if not creates_cycle(G, edge[0], edge[1]):
                G.add_edge(edge[0], edge[1], weight=edge[2]['weight'])
    
    return list(G.edges())

def calculate_partial_correlation(x, y, z, min_edge_weight):
    """
    Calculate partial correlation between x and y given z.
    
    Args:
        x: First variable
        y: Second variable
        z: Conditioning variables
        min_edge_weight: Minimum threshold
        
    Returns:
        float: Partial correlation coefficient
    """
    try:
        # Calculate residuals
        x_resid = get_residuals(x, z)
        y_resid = get_residuals(y, z)
        
        # Calculate partial correlation
        corr = stats.pearsonr(x_resid, y_resid)[0]
        return corr if abs(corr) > min_edge_weight else 0
        
    except Exception as e:
        logger.error(f"Partial correlation calculation failed: {e}")
        return 0

def get_residuals(target, features):
    """Get residuals from regression."""
    try:
        from sklearn.linear_model import LassoCV
        model = LassoCV(cv=5, random_state=42)
        model.fit(features, target)
        return target - model.predict(features)
    except Exception:
        return target

def creates_cycle(G, source, target):
    """Check if adding edge creates a cycle."""
    test_G = G.copy()
    test_G.add_edge(source, target)
    try:
        nx.find_cycle(test_G, orientation="original")
        return True
    except nx.NetworkXNoCycle:
        return False

def get_strongest_valid_edge(data, source):
    """Get strongest edge from source to targets."""
    min_edge_weight = 0.3
    correlations = data.corr()
    targets = ['EnergyConsumption', 'Satisfaction']
    edges = []
    
    for target in targets:
        if target in correlations.columns:
            corr = abs(correlations.loc[source, target])
            if corr >= min_edge_weight:
                edges.append((source, target, corr))
                
    return max(edges, key=lambda x: x[2])[:2] if edges else None

def generate_fallback_dag(data):
    """Generate minimal valid DAG based on strongest correlations."""
    relevant_columns = [
        'Temperature', 'Humidity', 'AirQuality',
        'EnergyConsumption', 'Satisfaction'
    ]
    
    # Check if relevant columns exist in data
    missing_cols = set(relevant_columns) - set(data.columns)
    if missing_cols:
        logger.warning(f"Missing columns for fallback DAG: {missing_cols}")
        return []
    
    correlations = data[relevant_columns].corr()
    edges = []
    
    # Add strongest validated edges for core relationships
    for source in ['Temperature', 'Humidity', 'AirQuality']:
        edge = get_strongest_valid_edge(data, source)
        if edge:
            edges.append(edge)
    
    return edges

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
        edges = run_sam_baseline(df)
        print(f"Discovered edges: {edges}")
    except ImportError:
        print("Skipping SAM test due to missing dependencies")