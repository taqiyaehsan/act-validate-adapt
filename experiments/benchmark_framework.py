#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark framework for comparing causal discovery methods.
This script orchestrates all comparison methods and reports results.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
import time
import argparse
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import traceback
import json
from scipy import stats
import random
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from src.metrics import MetricsCalculator, DAGMetrics

# Add project root to python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from src.ground_truth import GroundTruthDAG
from src.pipeline import CausalPipeline
from src.metrics import MetricsCalculator
from src.evaluator import HypothesisEvaluator, EdgeRanker
from src.metrics_viz import MetricsVisualizer, export_metrics_to_csv
from src.generators import PCGenerator, LLMGenerator, SAMGenerator

# Import comparison methods
from benchmarks.pc_baseline import run_pc_baseline
from benchmarks.sam_baseline import run_sam_baseline
from benchmarks.gies_comparison import run_gies
from benchmarks.jci_comparison import run_jci
from benchmarks.abcd_comparison import run_abcd
from benchmarks.causal_bandits import run_causal_bandits
from benchmarks.notears_i import run_notears_i
from benchmarks.icp_comparison import run_icp
from benchmarks.iid_comparison import run_iid
from benchmarks.utils import calculate_shd, calculate_precision_recall_f1, simulate_ashrae_intervention

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(project_root, "benchmark.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Color scheme for visualizations
COLORS = {
    'precision': '#4C72B0',  # Blue
    'recall': '#55A868',     # Green
    'f1_score': '#C44E52',   # Red
    'shd': '#F39C12',        # Orange 
    'runtime': '#8172B3',    # Purple
    'node_colors': {
        'Temperature': '#FF9966',     # Orange
        'Humidity': '#66B2FF',        # Blue
        'AirQuality': '#66CC66',      # Green
        'EnergyConsumption': '#FF6666', # Red
        'OverallSatisfaction': '#B266FF',  # Purple
        'air_temperature': '#FF9966',     # Orange
        'dew_temperature': '#66B2FF',     # Blue
        'sea_level_pressure': '#66CC66',  # Green
        'meter_reading': '#FF6666',       # Red
        'square_feet': '#AA66FF',         # Purple
        'year_built': '#FFCC66'           # Yellow
    }
}

def normalize_edges(edges, column_names=None):
    """Normalize edges to lowercase tuples for consistent comparison."""
    if edges is None:
        return set()
        
    normalized = set()
    for edge in edges:
        if isinstance(edge, (list, tuple)) and len(edge) == 2:
            src, tgt = edge
            src = str(src).strip().lower()
            tgt = str(tgt).strip().lower()
            
            # If column names are provided, validate edge elements
            if column_names:
                if src in column_names and tgt in column_names:
                    normalized.add((src, tgt))
            else:
                normalized.add((src, tgt))
    
    return normalized

def reverse_edges(edges):
    """Reverse edge directions"""
    return [(t, s) for s, t in edges]

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Benchmark causal discovery methods')
    
    parser.add_argument('--dataset', type=str, default='smart_room',
                        choices=['robot_arm', 'smart_room', 'causal_learn', 'bb_causal', 'ashrae'],
                        help='Dataset to use for benchmarking')
    
    parser.add_argument('--methods', type=str, nargs='+', 
                        default=['all'],
                        choices=['all', 'pc', 'sam', 'gies', 'jci', 'abcd',
                                'causal_bandits', 'notears_i', 'icp', 'iid', 'ours'],
                        help='Methods to benchmark')
    
    parser.add_argument('--iterations', type=int, default=60,
                        help='Number of iterations for iterative methods')
    
    parser.add_argument('--output_dir', type=str, default='benchmark_results',
                        help='Directory to save benchmark results')
    
    parser.add_argument('--api_key', type=str, default="YOUR_OPENAI_API_KEY",
                        help='OpenAI API key for LLM methods')
                        
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    parser.add_argument('--test_method', type=str, 
                        choices=['all', 'pc', 'sam', 'gies', 'jci', 'abcd',
                                'causal_bandits', 'notears_i', 'icp', 'iid', 'ours'],
                        help='Test a specific method with detailed debugging')
    
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging')
    
    parser.add_argument('--debug_metrics', action='store_true',
                        help='Print detailed metrics calculation information')
    
    return parser.parse_args()

def load_dataset(seed, dataset_name):
    """Load the selected dataset."""
    logger.info(f"Loading dataset: {dataset_name}")
    
    if dataset_name == 'ashrae':
        return load_ashrae_dataset(seed)
    elif dataset_name == 'smart_room':
        data_file = './data/smart-room-noisy_preprocessed.csv'
        if not os.path.exists(data_file):
            logger.error(f"Dataset file not found: {data_file}")
            raise FileNotFoundError(f"Dataset file not found: {data_file}")
        logger.info(f"Loaded data from {data_file} with shape {pd.read_csv(data_file).shape}")
        return pd.read_csv(data_file)
    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_ground_truth(dataset_name):
    """Get the ground truth DAG for the dataset with normalized edges."""
    if dataset_name == 'ashrae':
        return get_ashrae_ground_truth()
    elif dataset_name == 'smart_room':
        ground_truth = GroundTruthDAG()
        
        # Get the lowercase version of all nodes
        column_names = {node.lower() for node in ground_truth.nodes}
        
        # Normalize ground truth edges
        ground_truth_edges = normalize_edges(ground_truth.edges, column_names)
        
        # Update ground truth edges with normalized version
        ground_truth.normalized_edges = ground_truth_edges
        
        logger.info(f"Ground truth nodes: {ground_truth.nodes}")
        logger.info(f"Ground truth edges (original): {ground_truth.edges}")
        logger.info(f"Ground truth edges (normalized): {ground_truth.normalized_edges}")
        
        return ground_truth
    else:
        logger.error(f"Unknown dataset: {dataset_name}")
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_simulation_path(dataset_name):
    """Get the path to the simulation script for the dataset."""
    if dataset_name == 'smart_room':
        path = './js/smart_room_noisy.js'
    elif dataset_name == 'ashrae':
        path = None  # For ASHRAE, we use simulated interventions
    else:
        path = None
    return path

def run_ashrae_benchmarks(args):
    """Run benchmarks specifically tailored for ASHRAE dataset."""
    logger.info("Running ASHRAE-specific benchmarks")

    # Set random seed and prepare environment
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset and ground truth
    df = load_ashrae_dataset(args.seed)
    ground_truth = get_ashrae_ground_truth()
    column_names = {col.lower() for col in df.columns}

    # Determine which methods to run
    methods_to_run = ['ours', 'pc', 'sam', 'gies', 'jci', 'abcd', 'icp', 'iid', 'causal_bandits', 'notears_i'] if 'all' in args.methods else args.methods
    logger.info(f"Running methods: {methods_to_run}")

    # Dictionary to store results
    results = {}

    # Run each method
    for method in methods_to_run:
        logger.info(f"\n{'='*80}\nRunning method: {method}\n{'='*80}")
        
        if method == 'ours':
            try:
                # Initialize pipeline with dataset_type='ashrae' right from the start
                pipeline = CausalPipeline(
                    df, 
                    args.api_key, 
                    smart_room_path=None,  # No simulation path for ASHRAE
                    max_iterations=args.iterations,
                    dataset_type='ashrae',  # Set this at initialization
                    relevant_column=list(df.columns)  # Pass columns directly
                )
                
                # Create column mapping if needed
                pipeline.column_mapping = {col.lower(): col for col in df.columns}
                
                final_dag, _ = pipeline.run()
                edges = final_dag['edges']
                normalized_edges = normalize_edges(edges, column_names)
                
                # Get intervention results from pipeline if available
                intervention_results = getattr(pipeline.tester, 'intervention_results', {}) if hasattr(pipeline, 'tester') else {}
                method_support = getattr(pipeline.edge_ranker, 'edge_confidence', {edge: 0.5 for edge in normalized_edges}) if hasattr(pipeline, 'edge_ranker') else {edge: 0.5 for edge in normalized_edges}
                
                metrics = calculate_comprehensive_metrics(
                    normalized_edges, 
                    ground_truth.normalized_edges,
                    method,  # Fixed: use method instead of undefined method_name
                    intervention_results=intervention_results,
                    method_support=method_support,
                    ground_truth_obj=ground_truth,
                    debug=args.debug_metrics  # Fixed: use args.debug_metrics instead of undefined debug
                )
                                
                pipeline.cleanup()
                results[method] = {
                    'edges': edges,
                    'normalized_edges': normalized_edges,
                    'metrics': metrics,
                    'runtime': 0  # Not tracking runtime
                }
            except Exception as e:
                logger.error(f"Error in 'ours' method: {str(e)}")
                logger.error(traceback.format_exc())
                results[method] = {
                    'edges': [],
                    'normalized_edges': set(),
                    'metrics': {
                        'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                        'shd': len(ground_truth.normalized_edges),
                        'risk': 1.0, 'cost': 1.0  # Added default risk/cost
                    },
                    'runtime': 0.0,
                    'error': f"Failed: {str(e)}"
                }
        else:
            # For other methods, use standard approach
            result = test_individual_method(
                'ashrae', method, df, ground_truth, None, args.api_key,
                debug=args.debug_metrics, max_iterations=args.iterations
            )
            
            results[method] = result or {
                'edges': [],
                'normalized_edges': set(),
                'metrics': {
                    'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0,
                    'shd': len(ground_truth.normalized_edges),
                    'risk': 1.0, 'cost': 1.0  # Added default risk/cost
                },
                'runtime': 0.0,
                'error': "Failed to produce results"
            }

    # Save results as JSON
    with open(os.path.join(args.output_dir, 'ashrae_raw_results.json'), 'w') as f:
        serializable_results = {}
        for method, result in results.items():
            serializable_result = {}
            for key, value in result.items():
                if key == 'normalized_edges':
                    serializable_result[key] = [list(edge) for edge in value]
                elif isinstance(value, set):
                    serializable_result[key] = list(value)
                else:
                    serializable_result[key] = value
            serializable_results[method] = serializable_result
        
        json.dump(serializable_results, f, indent=4)

    # Visualize results
    visualize_ashrae_results(results, ground_truth, args.output_dir)

    return results

def calculate_comprehensive_metrics(predicted_edges, true_edges, method_name, intervention_results=None, method_support=None, ground_truth_obj=None, debug=True):
    """Calculate precision, recall, F1, and conditionally risk/cost."""
    
    # Standard metrics for all methods
    if not predicted_edges:
        return {
            'precision': 0.0, 'recall': 0.0 if true_edges else 1.0,
            'f1_score': 0.0, 'shd': len(true_edges)
        }
    
    true_positives = len(predicted_edges & true_edges)
    false_positives = len(predicted_edges - true_edges)
    false_negatives = len(true_edges - predicted_edges)
    
    precision = true_positives / len(predicted_edges) if predicted_edges else 0.0
    recall = true_positives / len(true_edges) if true_edges else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    shd = false_positives + false_negatives
    
    base_metrics = {
        'precision': precision, 'recall': recall, 'f1_score': f1_score, 'shd': shd
    }

    logger.info(f"Intervention results keys: {list(intervention_results.keys())}")
    logger.info(f"Sample intervention data: {list(intervention_results.values())[0][:1] if intervention_results else 'None'}")

    if debug:
        logger.info(f"Ground truth type: {type(ground_truth_obj)}")
        logger.info(f"Ground truth has get_shd: {hasattr(ground_truth_obj, 'get_shd')}")

    # Only calculate risk/cost for intervention methods
    intervention_methods = {'ours', 'gies', 'jci', 'abcd', 'causal_bandits', 'icp', 'iid', 'notears_i'}
    
    if method_name in intervention_methods and ground_truth_obj:
        try:
            metrics_calc = MetricsCalculator(ground_truth_obj, alpha=0.6, beta=0.6)  # Match paper
            dag_metrics = metrics_calc.calculate_metrics(
                method_name, predicted_edges, method_support or {}, intervention_results or {}
            )
            logger.info(f"risk: {dag_metrics.risk}, cost: {dag_metrics.cost}")
            base_metrics.update({'risk': dag_metrics.risk, 'cost': dag_metrics.cost})
        except Exception as e:
            if debug:
                logger.warning(f"Failed to calculate risk/cost for {method_name}: {e}")
    
    return base_metrics

def test_individual_method(dataset, method_name, df, ground_truth, simulation_path, api_key=None, debug=True, max_iterations=10):
   """Test a single method and report detailed results"""
   logger.info(f"Testing method: {method_name}")
   method_intervention_results = {}
   column_names = {col.lower() for col in df.columns}
   
   # Handle methods with and without intervention results
   def safe_unpack(result):
       if isinstance(result, tuple) and len(result) == 2:
           return result
       return result, {}
   
   if method_name == 'ours':
       if dataset == 'ashrae':
           try:
               pipeline = CausalPipeline(df, api_key, simulation_path=None, max_iterations=max_iterations)
               final_dag, _ = pipeline.run()
               edges = final_dag['edges']
               # Extract intervention results from pipeline
               method_intervention_results = getattr(pipeline.tester, 'intervention_results', {}) if hasattr(pipeline, 'tester') else {}
               runtime = 0
               pipeline.cleanup()
           except Exception as e:
               logger.error(f"Error running our method: {str(e)}")
               return {
                   'edges': [], 'normalized_edges': set(),
                   'metrics': {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'shd': len(ground_truth.normalized_edges)},
                   'runtime': 0.0, 'error': f"Failed: {str(e)}"
               }
       else:
           pipeline = CausalPipeline(df, api_key, simulation_path, max_iterations=max_iterations)
           final_dag, _ = pipeline.run()
           edges = final_dag['edges']
           method_intervention_results = getattr(pipeline.tester, 'intervention_results', {}) if hasattr(pipeline, 'tester') else {}
           runtime = 0
           pipeline.cleanup()
   elif method_name == 'pc':
       start_time = time.time()
       edges = run_pc_baseline(df)
       runtime = time.time() - start_time
   elif method_name == 'sam':
       start_time = time.time()
       edges = run_sam_baseline(df)
       runtime = time.time() - start_time
   elif method_name == 'gies':
       start_time = time.time()
       try:
           edges, method_intervention_results = safe_unpack(run_gies(df, simulation_path, max_intervention_sets=max_iterations))
       except ImportError as e:
           logger.error(f"GIES requires dependencies: {e}")
           return {
               'edges': [], 'normalized_edges': set(),
               'metrics': {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'shd': len(ground_truth.normalized_edges)},
               'runtime': 0.0, 'error': f"Dependencies not satisfied: {e}"
           }
       logger.info(f"Intervention results for {method_name}: {len(method_intervention_results)} edges")
       runtime = time.time() - start_time
   elif method_name == 'jci':
       start_time = time.time()
       edges, method_intervention_results = safe_unpack(run_jci(df, simulation_path, max_interventions=max_iterations))
       logger.info(f"Intervention results for {method_name}: {len(method_intervention_results)} edges")
       runtime = time.time() - start_time
   elif method_name == 'abcd':
       start_time = time.time()
       edges, method_intervention_results = safe_unpack(run_abcd(df, simulation_path, max_iterations=max_iterations))
       edges = reverse_edges(edges)
       logger.info(f"Intervention results for {method_name}: {len(method_intervention_results)} edges")
       runtime = time.time() - start_time
   elif method_name == 'causal_bandits':
       start_time = time.time()
       edges, method_intervention_results = safe_unpack(run_causal_bandits(df, simulation_path, num_iterations=max_iterations))
       edges = reverse_edges(edges)
       logger.info(f"Intervention results for {method_name}: {len(method_intervention_results)} edges")
       runtime = time.time() - start_time
   elif method_name == 'icp':
       start_time = time.time()
       edges, method_intervention_results = safe_unpack(run_icp(df, simulation_path, max_interventions=max_iterations))
       logger.info(f"Intervention results for {method_name}: {len(method_intervention_results)} edges")
       runtime = time.time() - start_time
   elif method_name == 'iid':
       start_time = time.time()
       edges, method_intervention_results = safe_unpack(run_iid(df, simulation_path, max_iterations=max_iterations))
       logger.info(f"Intervention results for {method_name}: {len(method_intervention_results)} edges")
       runtime = time.time() - start_time
   elif method_name == 'notears_i':
       start_time = time.time()
       edges, method_intervention_results = safe_unpack(run_notears_i(df, simulation_path, max_interventions=max_iterations))
       logger.info(f"Intervention results for {method_name}: {len(method_intervention_results)} edges")
       runtime = time.time() - start_time
   else:
       logger.error(f"Method {method_name} not implemented")
       return None
       
   normalized_edges = normalize_edges(edges, column_names)
   
   # Debug intervention results
   if debug and method_intervention_results:
       logger.info(f"Intervention results keys: {list(method_intervention_results.keys())}")
       logger.info(f"Sample intervention: {list(method_intervention_results.values())[0][:1] if method_intervention_results else 'None'}")
   
   metrics = calculate_comprehensive_metrics(
       normalized_edges, 
       ground_truth.normalized_edges,
       method_name,
       intervention_results=method_intervention_results,
       method_support={edge: 0.5 for edge in normalized_edges},
       ground_truth_obj=ground_truth,
       debug=debug
   )
   
   logger.info(f"Method: {method_name}, Runtime: {runtime:.2f}s, Metrics: {metrics}")
   
   return {
       'edges': edges,
       'normalized_edges': normalized_edges,
       'metrics': metrics,
       'runtime': runtime
   }

def visualize_ashrae_results(results, ground_truth, output_dir):
    """Create visualizations for ASHRAE benchmark with improved node coloring and layouts"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics for plots
    methods = list(results.keys())
    precision = [results[m]['metrics']['precision'] for m in methods]
    recall = [results[m]['metrics']['recall'] for m in methods]
    f1_score = [results[m]['metrics']['f1_score'] for m in methods]
    shd = [results[m]['metrics']['shd'] for m in methods]
    runtime = [results[m]['runtime'] for m in methods]

    # Create consolidated results CSV
    results_data = []
    for method in methods:
        method_results = results[method]
        metrics = method_results.get('metrics', {})
        
        row = {
            'Method': method, 
            'Runtime': method_results.get('runtime', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1_Score': metrics.get('f1_score', 0),
            'SHD': metrics.get('shd', 0),
            'Risk': metrics.get('risk', 'N/A'),
            'Cost': metrics.get('cost', 'N/A')
        }
        
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(output_dir, 'ashrae_benchmark_results.csv'), index=False)
    
    # Print summary table to console
    print("\n" + "="*80)
    print("ASHRAE BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80 + "\n")
    
    # Define node colors based on variable types
    node_colors = {
        'air_temperature': '#FF9966',     # Orange
        'dew_temperature': '#66B2FF',     # Blue
        'sea_level_pressure': '#66CC66',  # Green
        'meter_reading': '#FF6666',       # Red
        'square_feet': '#FFCC66',         # Yellow
        'year_built': '#A278B5'           # Purple
    }
    
    # Create dashboard visualization
    plt.figure(figsize=(15, 12))
    
    # 1. Compare precision, recall, F1 score
    plt.subplot(2, 2, 1)
    x = np.arange(len(methods))
    width = 0.25
    
    plt.bar(x - width, precision, width, label='Precision', color=COLORS['precision'])
    plt.bar(x, recall, width, label='Recall', color=COLORS['recall'])
    plt.bar(x + width, f1_score, width, label='F1 Score', color=COLORS['f1_score'])
    
    plt.xlabel('Method')
    plt.ylabel('Score')
    plt.title('Performance Metrics')
    plt.xticks(x, methods, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 2. Compare SHD (lower is better)
    plt.subplot(2, 2, 2)
    plt.bar(methods, shd, color=COLORS['shd'])
    plt.xlabel('Method')
    plt.ylabel('SHD')
    plt.title('Structural Hamming Distance (lower is better)')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 3. Compare runtime
    plt.subplot(2, 2, 3)
    plt.bar(methods, runtime, color=COLORS['runtime'])
    plt.xlabel('Method')
    plt.ylabel('Runtime (s)')
    plt.title('Runtime (lower is better)')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # 4. Create ground truth DAG visualization with better layout
    plt.subplot(2, 2, 4)
    G = nx.DiGraph()
    G.add_edges_from(ground_truth.edges)
    
    # Create a fixed layout based on variable types
    # Position input nodes at top, output nodes at bottom
    pos = {}
    input_nodes = [n for n in G.nodes() if n.lower() != 'meter_reading']
    output_nodes = [n for n in G.nodes() if n.lower() == 'meter_reading']
    
    # Position input nodes in a row at the top
    for i, node in enumerate(input_nodes):
        pos[node] = (i * 1.5, 1.0)
        
    # Position output nodes in a row at the bottom
    for i, node in enumerate(output_nodes):
        pos[node] = (len(input_nodes) / 2, 0)
    
    # Draw nodes with specific colors based on variable type
    for node_type, color in node_colors.items():
        # Find all nodes of this type (case-insensitive)
        nodes = [n for n in G.nodes() if n.lower() == node_type.lower()]
        if nodes:
            nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                                  node_color=color, node_size=1200, edgecolors='black')
    
    # Draw edges with curved arrows
    for edge in G.edges():
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[edge], 
            width=2.0,
            arrowsize=20,
            arrowstyle='-|>', 
            connectionstyle='arc3,rad=0.2',
            edge_color='black'
        )
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
    plt.title('Ground Truth DAG')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ashrae_results_summary.png'), dpi=300)
    
    # Create individual DAG visualizations
    plt.figure(figsize=(15, 10))
    
    n_methods = len(methods)
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols
    
    for i, method in enumerate(methods):
        plt.subplot(n_rows, n_cols, i+1)
        
        G = nx.DiGraph()
        G.add_edges_from(results[method]['edges'])
        
        # Create a similar fixed layout
        pos = {}
        input_nodes = [n for n in G.nodes() if n.lower() != 'meter_reading']
        output_nodes = [n for n in G.nodes() if n.lower() == 'meter_reading']
        
        # Position input nodes in a row at the top
        for j, node in enumerate(input_nodes):
            pos[node] = (j * 1.5, 1.0)
            
        # Position output nodes in a row at the bottom
        for j, node in enumerate(output_nodes):
            pos[node] = (len(input_nodes) / 2, 0)
        
        # Draw nodes with specific colors
        for node_type, color in node_colors.items():
            nodes = [n for n in G.nodes() if n.lower() == node_type.lower()]
            if nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                                      node_color=color, node_size=1000, edgecolors='black')
        
        # Draw edges with curved arrows
        for edge in G.edges():
            # Check if edge is in ground truth
            is_correct = any(edge[0].lower() == gt_edge[0].lower() and 
                             edge[1].lower() == gt_edge[1].lower() 
                             for gt_edge in ground_truth.edges)
            
            # Use green for correct edges, red for incorrect ones
            edge_color = 'green' if is_correct else 'red'
            
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[edge], 
                width=2.0,
                arrowsize=15,
                arrowstyle='-|>', 
                connectionstyle='arc3,rad=0.2',
                edge_color=edge_color
            )
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')
        
        correct_edges = len(set(results[method]['normalized_edges']) & 
                           set(ground_truth.normalized_edges))
        total_edges = len(results[method]['edges'])
        plt.title(f"{method}: {correct_edges}/{total_edges} correct edges")
    
    # Add a legend for node colors
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor=color, markersize=15, label=label) 
                      for label, color in node_colors.items()]
    
    plt.figlegend(handles=legend_elements, loc='lower center', 
                ncol=len(node_colors), bbox_to_anchor=(0.5, 0), fontsize=10)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust for legend at bottom
    plt.savefig(os.path.join(output_dir, 'ashrae_method_dags.png'), dpi=300)

def load_ashrae_dataset(seed=42):
    """Load ASHRAE building energy dataset with preprocessing for causal discovery"""
    try:
        # Set paths for data
        # data_path = os.path.join(project_root, "data", "ashrae_data.csv")
        data_path = './data/ashrae_data.csv'
        
        if os.path.exists(data_path):
            logger.info(f"Loading ASHRAE data from {data_path}")
            df = pd.read_csv(data_path)
        else:
            # Create synthetic dataset if real data not available
            logger.info("Creating synthetic ASHRAE-like dataset")
            np.random.seed(seed)
            n_samples = 5000
            
            # Create synthetic data with known causal structure
            air_temp = np.random.normal(22, 5, n_samples)  # air temperature in C
            dew_temp = air_temp * 0.7 + np.random.normal(0, 2, n_samples)  # dew point depends on air temp
            pressure = np.random.normal(1013, 5, n_samples)  # pressure in hPa
            square_feet = np.random.lognormal(9, 1, n_samples)  # building size
            year_built = np.random.randint(1950, 2020, n_samples)  # year built
            
            # Energy is affected by temp, dew point, size, and year
            meter_reading = (
                3 * air_temp + 
                0.5 * dew_temp + 
                0.0001 * square_feet +
                -0.1 * (year_built - 1950) +
                np.random.normal(0, 50, n_samples)
            )
            
            # Create dataframe
            df = pd.DataFrame({
                'air_temperature': air_temp,
                'dew_temperature': dew_temp, 
                'sea_level_pressure': pressure,
                'meter_reading': meter_reading,
                'square_feet': square_feet,
                'year_built': year_built
            })
            
            # Save the synthetic data
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            df.to_csv(data_path, index=False)
        
        # Standardize the data for the algorithms
        scaler = StandardScaler()
        num_cols = ['air_temperature', 'dew_temperature', 'sea_level_pressure',
                  'meter_reading', 'square_feet']
                  
        df_scaled = df.copy()
        df_scaled[num_cols] = scaler.fit_transform(df[num_cols])
        
        # Save scaling params for intervention
        scaling_params = pd.DataFrame({
            'column': num_cols,
            'mean': scaler.mean_,
            'std': scaler.scale_,
            'original_min': [df[col].min() for col in num_cols],
            'original_max': [df[col].max() for col in num_cols]
        })
        
        scaling_dir = os.path.join(project_root, "data")
        os.makedirs(scaling_dir, exist_ok=True)
        scaling_params.to_csv(os.path.join(scaling_dir, "ashrae_scaling_params.csv"), index=False)
        
        logger.info(f"Loaded ASHRAE data with shape {df_scaled.shape}")
        return df_scaled
        
    except Exception as e:
        logger.error(f"Error loading ASHRAE data: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def get_ashrae_ground_truth():
    """Define the ground truth causal structure for the ASHRAE dataset"""
    # Define causal structure based on domain knowledge
    edges = [
        ('air_temperature', 'meter_reading'),
        ('dew_temperature', 'meter_reading'),
        ('square_feet', 'meter_reading'),
        ('year_built', 'meter_reading'),
        ('air_temperature', 'dew_temperature')
    ]
    
    # Create nodes set
    nodes = set()
    for edge in edges:
        nodes.update(edge)
    
    # Create a custom ground truth class
    class ASHRAEGroundTruth:
        def __init__(self, edges, nodes):
            self.edges = set(edges)
            self.normalized_edges = normalize_edges(edges)
            self.nodes = nodes
        
        def get_shd(self, dag_edges):
            """Calculate Structural Hamming Distance"""
            normalized_dag_edges = normalize_edges(dag_edges)
            missing = len(self.normalized_edges - normalized_dag_edges)
            extra = len(normalized_dag_edges - self.normalized_edges)
            return missing + extra
    
    return ASHRAEGroundTruth(edges, nodes)

def statistical_significance_test(results, baseline_method='icp', alpha=0.05):
    """Perform statistical significance testing using McNemar's test."""
    try:
        from statsmodels.stats.contingency_tables import mcnemar
    except ImportError:
        logger.warning("statsmodels not installed, skipping significance testing")
        return []
        
    import numpy as np
    
    # Extract the ground truth edges
    ground_truth = None
    for method, result in results.items():
        if 'ground_truth' in result:
            ground_truth = result['ground_truth']
            break
    
    if not ground_truth:
        # Try to infer ground truth from the first method's metrics
        for method, result in results.items():
            if 'metrics' in result and 'true_edges' in result:
                ground_truth = result['true_edges']
                break
    
    if not ground_truth:
        logger.error("Could not extract ground truth edges")
        return []
    
    # Get baseline edges
    baseline_edges = set()
    for method, result in results.items():
        if method.lower() == baseline_method.lower():
            normalized_edges = normalize_edges(result.get('edges', []))
            baseline_edges = normalized_edges
            break
    
    if not baseline_edges:
        logger.error(f"Baseline method {baseline_method} not found in results")
        return []
    
    test_results = []
    
    # Compare each method against the baseline
    for method, result in results.items():
        if method.lower() == baseline_method.lower():
            continue
            
        method_edges = normalize_edges(result.get('edges', []))
        
        # Contingency table for McNemar's test
        both_correct = sum(1 for edge in ground_truth if edge in method_edges and edge in baseline_edges)
        ours_correct = sum(1 for edge in ground_truth if edge in method_edges and edge not in baseline_edges)
        baseline_correct = sum(1 for edge in ground_truth if edge not in method_edges and edge in baseline_edges)
        both_incorrect = len(ground_truth) - both_correct - ours_correct - baseline_correct
        
        contingency = np.array([[both_correct, baseline_correct], 
                               [ours_correct, both_incorrect]])
                               
        result = mcnemar(contingency, exact=False, correction=True)
        
        test_results.append({
            'Method': method,
            'p_value': result.pvalue,
            'statistic': result.statistic,
            'significant': result.pvalue < alpha,
            'contingency': str(contingency.tolist())
        })
    
    return test_results

def main():
    """Main function to run the benchmark."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check if API key is provided
    if args.api_key is None and ('all' in args.methods or 'ours' in args.methods):
        logger.warning("No API key provided. LLM-based methods will not work properly.")
        print("\nWARNING: No API key provided. LLM-based methods will not work properly.")
    
    # Special handling for ASHRAE dataset
    if args.dataset == 'ashrae':
        results = run_ashrae_benchmarks(args)
        print(f"ASHRAE benchmark completed. Results saved to: {args.output_dir}")
        return
    
    # For other datasets, use test_individual_method for each method
    df = load_dataset(args.seed, args.dataset)
    ground_truth = get_ground_truth(args.dataset)
    simulation_path = get_simulation_path(args.dataset)
    
    # Determine methods to run
    if 'all' in args.methods:
        methods_to_run = ['ours', 'pc', 'sam', 'gies', 'jci', 'abcd', 'causal_bandits', 'icp', 'iid', 'notears_i']
    else:
        methods_to_run = args.methods
    
    # Run each method individually
    results = {}
    for method in methods_to_run:
        logger.info(f"\n{'='*80}\nRunning method: {method}\n{'='*80}")
        
        result = test_individual_method(
            args.dataset, 
            method, 
            df, 
            ground_truth, 
            simulation_path, 
            args.api_key,
            debug=args.debug_metrics,
            max_iterations=args.iterations
        )
        
        if result:
            results[method] = result
        else:
            logger.error(f"Method {method} failed to produce results")
            results[method] = {
                'edges': [],
                'normalized_edges': set(),
                'metrics': {
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1_score': 0.0,
                    'shd': len(ground_truth.normalized_edges)
                },
                'runtime': 0.0,
                'error': f"Failed to produce results"
            }
    
    # Create visualization
    visualize_results(results, ground_truth, args.output_dir)
    
    print(f"Benchmark completed. Results saved to: {args.output_dir}")

def visualize_results(results, ground_truth, output_dir):
    """Create visualizations of benchmark results."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics for plotting
    methods = list(results.keys())
    precision = []
    recall = []
    f1_score = []
    shd = []
    runtime = []
    
    for method in methods:
        method_results = results[method]
        
        if 'metrics' in method_results:
            metrics = method_results['metrics']
            
            precision.append(metrics.get('precision', 0))
            recall.append(metrics.get('recall', 0))
            f1_score.append(metrics.get('f1_score', 0))
            shd.append(metrics.get('shd', 0))
        else:
            precision.append(0)
            recall.append(0)
            f1_score.append(0)
            shd.append(0)
        
        runtime.append(method_results.get('runtime', 0))
    
    # Plot core metrics (precision, recall, F1)
    plt.figure(figsize=(12, 6))
    x = np.arange(len(methods))
    width = 0.25
    
    plt.bar(x - width, precision, width, label='Precision', color=COLORS['precision'])
    plt.bar(x, recall, width, label='Recall', color=COLORS['recall'])
    plt.bar(x + width, f1_score, width, label='F1 Score', color=COLORS['f1_score'])
    
    plt.xlabel('Method')
    plt.ylabel('Performance Score')
    plt.title('Causal Discovery Performance Metrics')
    plt.xticks(x, methods, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_metrics.png'))
    plt.close()
    
    # Plot SHD (lower is better)
    plt.figure(figsize=(10, 6))
    plt.bar(methods, shd, color=COLORS['shd'])
    
    plt.xlabel('Method')
    plt.ylabel('Structural Hamming Distance (SHD)')
    plt.title('Structural Hamming Distance (lower is better)')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'shd.png'))
    plt.close()
    
    # Plot runtime (lower is better)
    plt.figure(figsize=(10, 6))
    plt.bar(methods, runtime, color=COLORS['runtime'])
    plt.xlabel('Method')
    plt.ylabel('Runtime (seconds)')
    plt.title('Computational Efficiency (lower is better)')
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'runtime.png'))
    plt.close()

    # Plot risk vs cost
    plt.figure(figsize=(10, 6))
    risk_values = [results[m]['metrics'].get('risk', 0) for m in methods]
    cost_values = [results[m]['metrics'].get('cost', 0) for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    plt.bar(x - width/2, risk_values, width, label='Risk', color='salmon')
    plt.bar(x + width/2, cost_values, width, label='Cost', color='skyblue')

    plt.xlabel('Method')
    plt.ylabel('Score')
    plt.title('Risk vs Cost Comparison')
    plt.xticks(x, methods, rotation=45)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'risk_cost.png'))
    plt.close()
    
    # Visualize the learned DAGs compared to ground truth
    plt.figure(figsize=(20, 15))
    
    # Determine max number of graphs for the grid
    n_graphs = len(methods) + 1  # +1 for ground truth
    n_cols = min(3, n_graphs)
    n_rows = (n_graphs + n_cols - 1) // n_cols
    
    # Setup the subplot layout
    for i, method in enumerate(['ground_truth'] + methods):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Create directed graph
        G = nx.DiGraph()
        
        if method == 'ground_truth':
            # Add ground truth edges
            edges = ground_truth.edges
            title = 'Ground Truth'
            node_color = 'lightblue'
        else:
            # Add method edges
            edges = results[method].get('edges', [])
            title = method
            
            # Color based on method type
            if method == 'ours':
                node_color = 'gold'
            elif method in ['pc', 'sam']:
                node_color = 'lightgreen'
            else:
                node_color = 'lightpink'
        
        G.add_edges_from(edges)
        
        # Draw the graph
        nx.draw(G, with_labels=True, node_color=node_color, node_size=800,
                font_weight='bold', arrows=True, arrowsize=15)
        plt.title(title)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learned_dags.png'))
    plt.close()
    
    # Create consolidated results CSV
    results_data = []
    for method in methods:
        method_results = results[method]
        
        metrics = method_results.get('metrics', {})
        
        row = {
            'Method': method, 
            'Runtime': method_results.get('runtime', 0),
            'Precision': metrics.get('precision', 0),
            'Recall': metrics.get('recall', 0),
            'F1_Score': metrics.get('f1_score', 0),
            'SHD': metrics.get('shd', 0),
            'Risk': metrics.get('risk', 'N/A'),
            'Cost': metrics.get('cost', 'N/A')
        }
        
        results_data.append(row)
    
    results_df = pd.DataFrame(results_data)
    results_df.to_csv(os.path.join(output_dir, 'benchmark_results.csv'), index=False)
    
    # Print summary table to console
    print("\n" + "="*80)
    print("BENCHMARK RESULTS SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80 + "\n")

if __name__ == "__main__":
    main()