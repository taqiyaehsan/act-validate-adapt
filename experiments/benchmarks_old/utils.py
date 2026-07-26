#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for benchmarking causal discovery methods.
"""

import logging
import numpy as np
import networkx as nx
import os
import tempfile
import json
import subprocess
import time
import psutil
from pathlib import Path
import sys

logger = logging.getLogger(__name__)

# Add project root to python path to ensure imports work
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def normalize_edge(edge):
    """
    Normalize an edge representation to a consistent format.
    
    Args:
        edge: Edge representation (tuple, list, or other format)
        
    Returns:
        tuple: Normalized edge as (source, target) tuple
    """
    if isinstance(edge, (list, tuple)):
        return tuple(str(x).strip() for x in edge)
    return edge

def calculate_shd(predicted_edges, true_edges):
    """
    Calculate Structural Hamming Distance between predicted and true edges.
    
    Args:
        predicted_edges: Set or list of predicted edges
        true_edges: Set or list of true edges
        
    Returns:
        int: Structural Hamming Distance
    """
    # Normalize edges
    predicted = {normalize_edge(edge) for edge in predicted_edges}
    true = {normalize_edge(edge) for edge in true_edges}
    
    # Calculate SHD as the sum of missing and extra edges
    missing = true - predicted  # Edges in true but not in predicted
    extra = predicted - true    # Edges in predicted but not in true
    
    shd = len(missing) + len(extra)
    
    logger.debug(f"SHD: {shd} (Missing: {len(missing)}, Extra: {len(extra)})")
    logger.debug(f"Missing edges: {missing}")
    logger.debug(f"Extra edges: {extra}")
    
    return shd

def calculate_precision_recall_f1(predicted_edges, true_edges):
    """
    Calculate precision, recall, and F1 score for edge prediction.
    
    Args:
        predicted_edges: Set or list of predicted edges
        true_edges: Set or list of true edges
        
    Returns:
        dict: Dictionary with precision, recall, F1 score, and SHD
    """
    # Normalize edges
    predicted = {normalize_edge(edge) for edge in predicted_edges}
    true = {normalize_edge(edge) for edge in true_edges}
    
    # Calculate true positives, false positives, and false negatives
    true_positives = len(predicted & true)
    false_positives = len(predicted - true)
    false_negatives = len(true - predicted)
    
    # Calculate precision, recall, and F1 score
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-10)
    
    # Calculate SHD
    shd = calculate_shd(predicted, true)
    
    # Calculate accuracy
    true_negatives = (len(true) * len(true)) - len(true) - false_positives - false_negatives
    accuracy = (true_positives + true_negatives) / max(true_positives + true_negatives + false_positives + false_negatives, 1)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'shd': shd,
        'accuracy': accuracy,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }

def is_dag(edges):
    """
    Check if a set of edges forms a directed acyclic graph (DAG).
    
    Args:
        edges: Set or list of edges
        
    Returns:
        bool: True if the graph is a DAG, False otherwise
    """
    G = nx.DiGraph()
    G.add_edges_from(edges)
    return nx.is_directed_acyclic_graph(G)

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
            # Properly extract the source and target nodes from the cycle edge
            u, v = cycle[0][0], cycle[0][1]
            G.remove_edge(u, v)
        except nx.NetworkXNoCycle:
            break
    
    return list(G.edges())

def perform_intervention(variable, value, simulator_path, duration=2000, intervention_time=100, timeout=60, dataset=None, dataset_name=None):
    """
    Perform an intervention using the provided simulator with improved error handling.
    
    Args:
        variable (str): Variable to intervene on
        value (float): Value to set the variable to
        simulator_path (str): Path to simulator script
        duration (int): Duration of simulation in ms
        intervention_time (int): Time to apply intervention in ms
        timeout (int): Timeout for the simulation in seconds
        
    Returns:
        dict: Result of the intervention
    """
    import subprocess
    import tempfile
    import json
    import os

    # If this is ASHRAE dataset, use simulated intervention
    if not simulator_path or simulator_path == 'None' or simulator_path is None:
        return simulate_ashrae_intervention(dataset, variable, value)
    
    # Validate simulator path
    if not simulator_path or simulator_path == 'None':
        logger.warning("No valid simulator path provided")
        return None
        
    if not os.path.exists(simulator_path):
        logger.error(f"Simulator file not found: {simulator_path}")
        return None
    
    # Create temporary JS file for this intervention
    with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as temp_js:
        temp_js_path = temp_js.name
        
        # Create intervention object in the format expected by simulateAndGetLatestData
        intervention_obj = [{
            "variable": variable,
            "action": "set",
            "value": value
        }]
        
        # Write simulation code with improved error handling
        temp_js.write(f'''
        const {{ simulateAndGetLatestData }} = require('{simulator_path}');
        
        async function runSimulation() {{
            try {{
                console.log("Starting simulation with {variable}={value}");
                const result = await simulateAndGetLatestData(
                    {duration},
                    {json.dumps(intervention_obj)},
                    {intervention_time}
                );
                
                // Validate simulation result
                if (!result) {{
                    throw new Error('Empty simulation result');
                }}
                
                if (!result.preInterventionData) {{
                    throw new Error('Missing preInterventionData');
                }}
                
                if (!result.postInterventionData) {{
                    throw new Error('Missing postInterventionData');
                }}
                
                // Check if source variable was actually changed
                const preVal = result.preInterventionData["{variable}"];
                const postVal = result.postInterventionData["{variable}"];
                
                console.log(`Pre-intervention {variable}: ${{preVal}}`);
                console.log(`Post-intervention {variable}: ${{postVal}}`);
                
                // Use marker for easy parsing
                console.log("SIMULATION_RESULT:" + JSON.stringify(result));
                process.exit(0);
            }} catch (error) {{
                console.error('SIMULATION_ERROR:', error.message);
                process.exit(1);
            }}
        }}

        // Handle unhandled promise rejections
        process.on('unhandledRejection', (error) => {{
            console.error('UNHANDLED_REJECTION:', error);
            process.exit(1);
        }});

        // Set a global timeout
        setTimeout(() => {{
            console.error('TIMEOUT_ERROR: Simulation did not complete in time');
            process.exit(1);
        }}, {timeout * 1000});

        runSimulation();
        ''')
    
    try:
        start_time = time.time()
        logger.debug(f"Starting intervention on {variable}={value}")
        
        # Run the Node.js process with memory limit
        process = subprocess.Popen(
            ['node', '--max-old-space-size=512', temp_js_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Process output
        simulation_result = None
        output, stderr = process.communicate(timeout=timeout)
        
        # Process all output lines
        for line in output.splitlines():
            logger.debug(f"Node output: {line}")  # Debug logging for all output
            if 'SIMULATION_RESULT:' in line:
                try:
                    # Extract and parse JSON result
                    result_json = line.split('SIMULATION_RESULT:', 1)[1]
                    simulation_result = json.loads(result_json)
                except (json.JSONDecodeError, IndexError) as e:
                    logger.error(f"JSON parsing error: {e}")
                    continue
        
        # Log any Node.js errors
        if stderr:
            logger.error(f"Node.js Errors: {stderr}")
        
        # Process valid simulation results
        if simulation_result and 'preInterventionData' in simulation_result and 'postInterventionData' in simulation_result:
            simulation_result['pre_state'] = simulation_result['preInterventionData']
            simulation_result['post_state'] = simulation_result['postInterventionData']

            # Check if the variable was actually changed
            pre_val = simulation_result['preInterventionData'].get(variable)
            post_val = simulation_result['postInterventionData'].get(variable)
            
            if pre_val is not None and post_val is not None:
                try:
                    pre_val_float = float(pre_val)
                    post_val_float = float(post_val)
                    diff = abs(post_val_float - pre_val_float)
                    
                    if diff < 0.001:
                        logger.warning(f"Intervention may not have worked: {variable} changed by only {diff}")
                    
                    logger.debug(f"Intervention complete: {variable} changed from {pre_val} to {post_val}")
                except (ValueError, TypeError):
                    logger.warning(f"Cannot convert {variable} values to float: pre={pre_val}, post={post_val}")
            
            # Log completion time
            end_time = time.time()
            logger.debug(f"Intervention took {end_time - start_time:.2f} seconds")
            
            return simulation_result
        else:
            logger.warning("Simulation returned invalid result structure")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error(f"Simulation timed out after {timeout} seconds")
        if process:
            try:
                # Kill process and all child processes
                parent = psutil.Process(process.pid)
                for child in parent.children(recursive=True):
                    child.kill()
                parent.kill()
            except:
                process.kill()
        return None
    except Exception as e:
        logger.error(f"Simulation error: {str(e)}")
        return None
    finally:
        # Clean up temporary files
        if os.path.exists(temp_js_path):
            try:
                os.remove(temp_js_path)
            except Exception as e:
                logger.warning(f"Error cleaning up {temp_js_path}: {str(e)}")
                
        # Ensure process is terminated
        if 'process' in locals() and process:
            try:
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    process.wait(3)  # Wait up to 3 seconds for termination
                    if process.poll() is None:  # Still running after terminate
                        process.kill()
            except:
                try:
                    process.kill()
                except:
                    pass

def simulate_ashrae_intervention(data, variable, value, scaling_params=None):
    """Simulate intervention on ASHRAE data using RandomForest"""
    from sklearn.ensemble import RandomForestRegressor
    
    # Create a copy of the data
    sample_data = data.sample(min(5000, len(data)))
    
    # Define child variables based on ground truth
    child_mapping = {
        'air_temperature': ['dew_temperature', 'meter_reading'],
        'dew_temperature': ['meter_reading'],
        'square_feet': ['meter_reading'],
        'year_built': ['meter_reading']
    }
    
    children = child_mapping.get(variable, [])
    
    # Record pre-state
    random_row = sample_data.sample(1).iloc[0]
    pre_state = {col: float(random_row[col]) for col in sample_data.columns}
    post_state = pre_state.copy()
    
    # Set intervened variable value
    std_dev = sample_data[variable].std()
    post_state[variable] = pre_state[variable] + (std_dev * 5 * (1 if value > 0 else -1))
    
    # For each child, use RandomForest to predict effect
    for child in children:
        try:
            # Train model on sample data
            X = sample_data[[variable]].values
            y = sample_data[child]
            
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            
            # Predict effect
            new_value = model.predict([[post_state[variable]]])[0]
            post_state[child] = new_value
            
        except Exception as e:
            logger.error(f"Error predicting {child}: {e}")
            # Fallback to simple multiplier if model fails
            effect_multiplier = 0.8
            change = (post_state[variable] - pre_state[variable]) * effect_multiplier
            post_state[child] = pre_state[child] + change
    
    result = {
            'preInterventionData': pre_state,
            'postInterventionData': post_state,
            'pre_state': pre_state,
            'post_state': post_state
        }
    
    return result

def test_edge_validation(edge, simulation_path, iterations=3, threshold=0.1):
    """
    Test if a causal edge exists by performing multiple interventions and measuring effects.
    
    Args:
        edge (tuple): The (source, target) edge to test
        simulation_path (str): Path to the simulation script
        iterations (int): Number of iterations to perform
        threshold (float): Effect size threshold to consider valid
    
    Returns:
        dict: Results with validation status, confidence, and detailed results
    """
    import numpy as np
    from benchmarks.utils import perform_intervention
    
    source, target = edge
    effect_sizes = []
    results = []
    
    # Try different values for the source variable
    for i in range(iterations):
        # For the smart room example, use sensible ranges for variables
        if source.lower() == 'temperature':
            # Temperature in range 18-30
            value = 18 + (i * 12 / max(1, iterations-1))
        elif source.lower() == 'humidity':
            # Humidity in range 30-70
            value = 30 + (i * 40 / max(1, iterations-1))
        elif source.lower() == 'airquality':
            # Air quality in range 0-500
            value = i * 500 / max(1, iterations-1)
        elif source.lower() in ['handx', 'ballx']:
            # X position in range 0-400
            value = i * 400 / max(1, iterations-1)
        elif source.lower() in ['handy', 'bally']:
            # Y position in range 0-300
            value = i * 300 / max(1, iterations-1)
        else:
            # Default: random value between 0-100
            value = i * 100 / max(1, iterations-1)
        
        # Perform the intervention
        result = perform_intervention(source, value, simulation_path)
        if not result:
            continue

        if result and 'preInterventionData' in result and 'postInterventionData' in result:
            result['pre_state'] = result['preInterventionData']
            result['post_state'] = result['postInterventionData']
            
        # Extract pre and post values
        try:
            pre_state = result.get('pre_state', result.get('preInterventionData', {}))
            post_state = result.get('post_state', result.get('postInterventionData', {}))
            
            # Calculate changes
            source_pre = float(pre_state.get(source, 0))
            source_post = float(post_state.get(source, 0))
            source_change = abs(source_post - source_pre)
            
            target_pre = float(pre_state.get(target, 0))
            target_post = float(post_state.get(target, 0))
            target_change = abs(target_post - target_pre)
            
            # If source changes, measure effect on target
            if source_change > 0.001:
                # Calculate effect size (normalized by source change)
                effect_size = target_change / source_change
                effect_sizes.append(effect_size)
                
                # Store detailed result
                results.append({
                    'source_value': value,
                    'source_pre': source_pre,
                    'source_post': source_post,
                    'source_change': source_change,
                    'target_pre': target_pre,
                    'target_post': target_post,
                    'target_change': target_change,
                    'effect_size': effect_size
                })
        except (ValueError, TypeError, ZeroDivisionError) as e:
            print(f"Error calculating effect size: {e}")
    
    # Calculate confidence based on effect sizes
    is_valid = False
    confidence = 0.0
    
    if effect_sizes:
        # Calculate average effect size
        avg_effect = np.mean(effect_sizes)
        
        # Calculate variation in effect sizes
        variation = np.std(effect_sizes) if len(effect_sizes) > 1 else 0
        
        # Edge is valid if average effect size is above threshold
        # and effect is reasonably consistent
        is_valid = avg_effect > threshold and variation < avg_effect * 2
        
        # Confidence is proportional to effect size and consistency
        consistency_factor = 1.0 / (1.0 + variation / max(avg_effect, 0.001))
        confidence = min(1.0, avg_effect * consistency_factor)
    
    return {
        'edge': edge,
        'valid': is_valid,
        'confidence': confidence,
        'effect_sizes': effect_sizes,
        'num_tests': len(effect_sizes),
        'detailed_results': results
    }