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

def preprocess_benchmark_data(data):
    """
    Standardize benchmark data preprocessing to match PolicyGRID's PCGenerator.
    Handles near-singular correlation matrices (adds small noise) and standardizes.
    Must be called by all benchmarks before running their algorithms.
    """
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    # Drop metadata
    drop_cols = [c for c in ['Timestamp', 'weight'] if c in data.columns]
    if drop_cols:
        data = data.drop(columns=drop_cols)

    if data.isnull().values.any():
        data = data.dropna()

    # Store raw data for RF-based interventions (before standardization)
    global _raw_benchmark_data
    _raw_benchmark_data = data.copy()

    # Check for near-singular correlation matrix
    corr_det = np.linalg.det(data.corr())
    if corr_det == 0 or np.isclose(corr_det, 0):
        logger.warning(f"Near-singular correlation matrix (det={corr_det:.2e}). Adding noise.")
        noise_scale = data.std() * 0.001
        for col in data.columns:
            data[col] = data[col] + np.random.normal(0, noise_scale[col], len(data))

    # Standardize
    column_names = list(data.columns)
    data_array = StandardScaler().fit_transform(data.values)
    data = pd.DataFrame(data_array, columns=column_names)

    return data


# Module-level raw data cache for RF-based interventions (ASHRAE etc.)
_raw_benchmark_data = None


def set_benchmark_data(data):
    """Store raw data for RF-based interventions when no simulator is available."""
    global _raw_benchmark_data
    import pandas as pd
    _raw_benchmark_data = data.copy()
    # Drop metadata
    drop_cols = [c for c in ['Timestamp', 'weight'] if c in _raw_benchmark_data.columns]
    if drop_cols:
        _raw_benchmark_data = _raw_benchmark_data.drop(columns=drop_cols)


def _rf_simulate_intervention(variable, value):
    """
    RandomForest-based intervention for datasets without a simulator (e.g., ASHRAE).
    Matches PolicyGRID's ASHRAETester._simulate_intervention in src/pipeline.py.
    Trains RF from parent → each potential child, predicts post-intervention state.
    """
    from sklearn.ensemble import RandomForestRegressor

    if _raw_benchmark_data is None:
        logger.warning("No benchmark data set for RF intervention. Call set_benchmark_data() first.")
        return None

    data = _raw_benchmark_data
    if variable not in data.columns:
        # Case-insensitive lookup
        for c in data.columns:
            if c.lower() == variable.lower():
                variable = c
                break
        else:
            logger.warning(f"Variable {variable} not in data columns: {list(data.columns)}")
            return None

    sample_data = data.sample(min(5000, len(data)))
    random_row = sample_data.sample(1).iloc[0]
    pre_state = {col: float(random_row[col]) for col in sample_data.columns}
    post_state = pre_state.copy()

    # Shift the intervened variable by 5 std devs
    std_dev = sample_data[variable].std()
    post_state[variable] = pre_state[variable] + (std_dev * 5 * (1 if value > 0 else -1))

    # For each other column, train RF to predict it from the intervened variable
    for child in data.columns:
        if child == variable:
            continue
        try:
            X = sample_data[[variable]].values
            y = sample_data[child].values
            model = RandomForestRegressor(n_estimators=50, random_state=42)
            model.fit(X, y)
            new_value = model.predict([[post_state[variable]]])[0]
            post_state[child] = new_value
        except Exception as e:
            logger.debug(f"RF predict failed for {variable}->{child}: {e}")
            # Fallback: simple linear extrapolation
            effect = 0.8 * (post_state[variable] - pre_state[variable])
            post_state[child] = pre_state[child] + effect

    # Add aliases for column name compatibility
    result_pre = dict(pre_state)
    result_post = dict(post_state)
    for col in list(result_post.keys()):
        result_pre[col.lower()] = result_pre[col]
        result_post[col.lower()] = result_post[col]
        # Title case
        result_pre[col.title().replace('_', '')] = result_pre[col]
        result_post[col.title().replace('_', '')] = result_post[col]

    return {
        'preInterventionData': result_pre,
        'postInterventionData': result_post,
        'pre_state': result_pre,
        'post_state': result_post,
    }


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

def orient_edges_by_regression(edges, data):
    """Orient undirected/ambiguous edges using regression R².

    For each edge (u, v), if regressing u→v gives higher R² than v→u,
    keep u→v; otherwise flip to v→u. This is a simple but effective
    heuristic for orienting edges discovered by constraint-based methods.
    """
    from sklearn.linear_model import LinearRegression
    oriented = []
    for u, v in edges:
        u_col = next((c for c in data.columns if c.lower() == str(u).lower()), None)
        v_col = next((c for c in data.columns if c.lower() == str(v).lower()), None)
        if u_col and v_col:
            xu = data[[u_col]].values
            xv = data[[v_col]].values
            r2_uv = LinearRegression().fit(xu, xv.ravel()).score(xu, xv.ravel())
            r2_vu = LinearRegression().fit(xv, xu.ravel()).score(xv, xu.ravel())
            if r2_uv >= r2_vu:
                oriented.append((u, v))
            else:
                oriented.append((v, u))
        else:
            oriented.append((u, v))
    return oriented


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

# Per-simulator intervenable variables.
# All other variables are determined by physics and cannot be directly set.
INTERVENABLE_VARS_BY_SIM = {
    'smart_building_rich': {'hvacpower', 'lightingpower'},
    'smart_room': {'temperature', 'humidity', 'airquality'},
    'smart_room_noise': {'temperature', 'humidity', 'airquality'},
    'smart_room_hidden_vars': {'temperature', 'humidity', 'airquality'},
    'open_window': {'temperature', 'humidity', 'airquality'},
}
# Default fallback (smart_building_rich for backward compatibility)
INTERVENABLE_VARS = {'hvacpower', 'lightingpower'}

# Active set — updated by _resolve_intervenable_vars when sim path is known
_active_intervenable = set(INTERVENABLE_VARS)


def _resolve_intervenable_vars(simulator_path=None, column_names=None):
    """Infer intervenable vars from simulator path or data columns."""
    global _active_intervenable
    if simulator_path:
        sim_name = os.path.basename(simulator_path).replace('.js', '')
        if sim_name in INTERVENABLE_VARS_BY_SIM:
            _active_intervenable = INTERVENABLE_VARS_BY_SIM[sim_name]
            return

    # No simulator — infer from column names
    if column_names:
        col_lower = {c.lower() for c in column_names}
        if 'meter_reading' in col_lower:
            # ASHRAE dataset
            _active_intervenable = col_lower - {'meter_reading', 'weight'}
            return
        # Physical testbed: Heater, Humidifier, Fan are actuators
        physical_actuators = {'heater', 'humidifier', 'fan'}
        if physical_actuators.issubset(col_lower):
            _active_intervenable = physical_actuators
            return

    _active_intervenable = set(INTERVENABLE_VARS)


def get_intervenable_vars(column_names, simulator_path=None):
    """Return the subset of column_names that the simulator can intervene on."""
    _resolve_intervenable_vars(simulator_path, column_names)
    return [c for c in column_names if c.lower() in _active_intervenable]


def perform_intervention(variable, value, simulator_path, duration=2000,
                         intervention_time=100, timeout=30, dataset=None,
                         dataset_name=None, hold_steps=3):
    """
    Perform an intervention using the --single-step CLI of smart_building_rich.js.

    Runs a baseline step (pre-state), then applies the intervention for
    hold_steps and records the post-state.

    Only actuator variables (hvacPower, lightingPower) can be intervened on.
    Returns None for non-intervenable variables.

    Args:
        variable (str): Variable to intervene on (e.g. 'HVACPower')
        value (float): Physical-unit value to set
        simulator_path (str): Absolute path to the JS simulator
        hold_steps (int): Number of steps to hold the intervention
        timeout (int): Per-subprocess timeout in seconds

    Returns:
        dict with keys 'preInterventionData', 'postInterventionData',
        'pre_state', 'post_state', or None on failure.
    """
    if not simulator_path or not os.path.exists(simulator_path):
        # Fall back to RF-based intervention (for ASHRAE etc.)
        if _raw_benchmark_data is not None:
            return _rf_simulate_intervention(variable, value)
        logger.warning(f"No valid simulator path: {simulator_path}")
        return None

    # Map DataFrame column names to JS simulator variable names (camelCase)
    _var_map = {
        'hvacpower': 'hvacPower', 'lightingpower': 'lightingPower',
        'temperature': 'temperature', 'humidity': 'humidity',
        'co2': 'co2', 'lightlevel': 'lightLevel',
        'noisedb': 'noiseDB', 'airquality': 'airQuality',
        'pmv': 'pmv', 'outdoortemp': 'outdoorTemp',
        'solarradiation': 'solarRadiation', 'occupancy': 'occupancy',
        'windowposition': 'windowPosition',
        'energyconsumption': 'energyConsumption',
        'satisfaction': 'satisfaction',
    }
    js_variable = _var_map.get(variable.lower(), variable)

    # Resolve intervenable set from sim path
    if simulator_path:
        _resolve_intervenable_vars(simulator_path)

    # Only actuator variables can be intervened on
    if variable.lower() not in _active_intervenable:
        logger.debug(f"Cannot intervene on {variable} — not an actuator. "
                     f"Intervenable: {_active_intervenable}")
        return None

    def _step(state_phys, intervention, elapsed_ms):
        cmd = ['node', simulator_path, '--single-step',
               '--elapsed-ms', str(int(elapsed_ms))]
        if state_phys:
            cmd += ['--state', json.dumps(state_phys)]
        if intervention:
            cmd += ['--intervention', json.dumps(intervention)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except (subprocess.TimeoutExpired, Exception) as e:
            logger.warning(f"Intervention step failed: {e}")
            return None
        for line in res.stdout.split('\n'):
            if 'RESULT:' in line:
                return json.loads(line.split('RESULT:', 1)[1].strip())
        return None

    try:
        # Baseline step (no intervention) to capture pre-state
        elapsed_ms = 8 * 3600 * 1000  # 8am
        pre_state = _step(None, None, elapsed_ms)
        if pre_state is None:
            return None

        # Apply intervention for hold_steps
        intervention = {js_variable: value}
        state = pre_state
        for h in range(hold_steps):
            elapsed_ms += 60 * 1000  # +1 minute per step
            state = _step(state, intervention, elapsed_ms)
            if state is None:
                return None

        post_state = state

        # Add key aliases so benchmarks can look up by any convention:
        # camelCase ('hvacPower'), DataFrame ('HVACPower'), or lowercase ('hvacpower')
        _col_names = {
            'hvacpower': 'HVACPower', 'lightingpower': 'LightingPower',
            'temperature': 'Temperature', 'humidity': 'Humidity',
            'co2': 'CO2', 'lightlevel': 'LightLevel',
            'noisedb': 'NoiseDB', 'airquality': 'AirQuality',
            'pmv': 'PMV', 'outdoortemp': 'OutdoorTemp',
            'solarradiation': 'SolarRadiation', 'occupancy': 'Occupancy',
            'windowposition': 'WindowPosition',
            'energyconsumption': 'EnergyConsumption',
            'satisfaction': 'Satisfaction',
            'overallsatisfaction': 'OverallSatisfaction',
        }

        def _add_aliases(d):
            extra = {}
            for k, v in d.items():
                extra[k.lower()] = v
                col = _col_names.get(k.lower())
                if col:
                    extra[col] = v
            # OverallSatisfaction alias for satisfaction (some benchmarks use this)
            if 'satisfaction' in d:
                extra['OverallSatisfaction'] = d['satisfaction']
                extra['overallSatisfaction'] = d['satisfaction']
                extra['overallsatisfaction'] = d['satisfaction']
            d.update(extra)
            return d

        pre_state = _add_aliases(dict(pre_state))
        post_state = _add_aliases(dict(post_state))

        return {
            'preInterventionData': pre_state,
            'postInterventionData': post_state,
            'pre_state': pre_state,
            'post_state': post_state,
        }
    except Exception as e:
        logger.error(f"perform_intervention error: {e}")
        return None

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
    from benchmarks_new.utils import perform_intervention
    
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