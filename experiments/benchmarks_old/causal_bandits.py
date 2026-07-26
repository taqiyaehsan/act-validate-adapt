#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Causal Bandits implementation for comparison.
This method frames causal discovery as a multi-armed bandit problem.
"""

import logging
import numpy as np
import pandas as pd
import time
import random
import networkx as nx
import os
from pathlib import Path
from benchmarks.utils import test_edge_validation
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

logger = logging.getLogger(__name__)

def run_causal_bandits(data, simulation_path=None, num_iterations=5, 
                      exploration_param=0.1, discount_factor=0.9):
    """Run the Causal Bandits algorithm with support for ASHRAE dataset."""
    logger.info("Running Causal Bandits algorithm")
    
    intervention_results = {}

    try:
        # Clean up data
        if 'Timestamp' in data.columns:
            data = data.drop(columns=['Timestamp'])
        if 'weight' in data.columns:
            data = data.drop(columns=['weight'])
            
        column_names = list(data.columns)
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Variables: {column_names}")
        
        # Detect dataset type
        is_ashrae = 'air_temperature' in data.columns and 'meter_reading' in data.columns
        
        # Create ASHRAE tester if needed
        if is_ashrae and simulation_path is None:
            # Create standalone ASHRAE tester
            class ASHRAETester:
                def __init__(self, dataset):
                    self.dataset = dataset
                    
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
        
        # Initialize with PC algorithm
        logger.info("Initializing with PC algorithm")
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
        
        # Extract initial edges
        adj_matrix = cg.G.graph
        edges = []
        for i in range(len(column_names)):
            for j in range(len(column_names)):
                if adj_matrix[i, j] == 1:
                    edges.append((column_names[i], column_names[j]))
        
        G = nx.DiGraph()
        G.add_edges_from(edges)
        
        # Initialize bandit arms
        arms = []
        
        # Define valid source variables
        if is_ashrae:
            valid_sources = [col for col in column_names if col.lower() != 'meter_reading']
        else:
            valid_sources = [col for col in column_names 
                            if col not in ['EnergyConsumption', 'OverallSatisfaction']]
        
        # Create arms for each potential edge
        for src in valid_sources:
            for tgt in column_names:
                if src != tgt:
                    # Define intervention values
                    low_val = data[src].quantile(0.25)
                    mid_val = data[src].quantile(0.5)
                    high_val = data[src].quantile(0.75)
                    
                    for val in [low_val, mid_val, high_val]:
                        arms.append({
                            'source': src,
                            'target': tgt,
                            'value': val,
                            'pulls': 0,
                            'reward': 0,
                            'edge_exists': (src, tgt) in edges
                        })
        
        # Run bandit algorithm if simulation path or ASHRAE tester is available
        if (simulation_path and os.path.exists(simulation_path)) or (is_ashrae and 'ashrae_tester' in locals()):
            logger.info("Starting causal bandit algorithm")
            
            for iteration in range(num_iterations):
                arm = select_arm_ucb(arms, exploration_param)
                
                logger.info(f"Iteration {iteration+1}/{num_iterations}: "
                          f"Testing edge {arm['source']}->{arm['target']} "
                          f"with value {arm['value']}")
                
                # Use appropriate intervention method
                if is_ashrae and 'ashrae_tester' in locals():
                    result = ashrae_tester._simulate_intervention(arm['source'], arm['value'])
                else:
                    result = perform_intervention(arm['source'], arm['value'], simulation_path)
                
                if result and 'postInterventionData' in result:
                    # Store intervention data
                    edge = (arm['source'], arm['target'])
                    intervention_data = extract_intervention_data(
                        result['preInterventionData'], 
                        result['postInterventionData']
                    )
                    
                    if edge not in intervention_results:
                        intervention_results[edge] = []
                    intervention_results[edge].append(intervention_data)
                    
                    reward = calculate_reward(arm, result['preInterventionData'], result['postInterventionData'])
                    
                    arm['pulls'] += 1
                    arm['reward'] = (arm['reward'] * (arm['pulls'] - 1) + reward) / arm['pulls']
                    
                    logger.info(f"Arm received reward: {reward}")
                else:
                    logger.warning("Intervention failed")
            
            # Update edge beliefs
            update_edge_beliefs(arms, G, discount_factor)
        else:
            logger.warning("No simulation path provided. Using only observational data.")
        
        # Extract final DAG
        edges = list(G.edges())
        edges = make_dag(edges)
        
        logger.info(f"Causal Bandits algorithm completed. Found {len(edges)} edges")
        logger.info(f"Edges: {edges}")
        
        return edges, intervention_results
        
    except Exception as e:
        logger.error(f"Error in Causal Bandits algorithm: {str(e)}")
        return []  # Return empty DAG on error

def select_arm_ucb(arms, exploration_param):
    """
    Select an arm using the UCB1 algorithm.
    
    Args:
        arms (list): List of arm dictionaries
        exploration_param (float): Exploration parameter
        
    Returns:
        dict: Selected arm
    """
    total_pulls = sum(arm['pulls'] for arm in arms)
    
    if total_pulls == 0:
        # If no arms have been pulled, select one randomly
        return random.choice(arms)
    
    best_arm = None
    best_ucb = -float('inf')
    
    for arm in arms:
        if arm['pulls'] == 0:
            # If this arm has never been pulled, it has infinite UCB value
            return arm
        
        # Calculate UCB value
        exploitation = arm['reward']
        exploration = exploration_param * np.sqrt(2 * np.log(total_pulls) / arm['pulls'])
        ucb = exploitation + exploration
        
        if ucb > best_ucb:
            best_arm = arm
            best_ucb = ucb
    
    return best_arm

def calculate_reward(arm, pre_state, post_state):
    """
    Calculate reward for a bandit arm based on intervention result.
    
    Args:
        arm (dict): Arm dictionary
        pre_state (dict): State before intervention
        post_state (dict): State after intervention
        
    Returns:
        float: Reward value
    """
    source = arm['source']
    target = arm['target']
    
    # Check if source and target are in the states
    if source not in pre_state or source not in post_state or \
       target not in pre_state or target not in post_state:
        # If variables are missing, return a neutral reward
        return 0.0
    
    try:
        # Calculate source change
        source_pre = float(pre_state[source])
        source_post = float(post_state[source])
        source_change = abs(source_post - source_pre)
        
        # Calculate target change
        target_pre = float(pre_state[target])
        target_post = float(post_state[target])
        target_change = abs(target_post - target_pre)
        
        # If source didn't change, return neutral reward
        if source_change < 1e-6:
            return 0.0
        
        # Calculate change ratio
        change_ratio = target_change / source_change
        
        # Calculate reward (higher for significant target changes)
        if change_ratio > 0.1:
            return change_ratio
        else:
            return 0.0
            
    except (ValueError, TypeError):
        # If values can't be converted to float, return a neutral reward
        return 0.0

def update_edge_beliefs(arms, graph, discount_factor):
    """
    Update edge beliefs in the graph based on arm rewards.
    
    Args:
        arms (list): List of arm dictionaries
        graph (nx.DiGraph): Directed graph to update
        discount_factor (float): Discount factor for rewards
    """
    # Group arms by edge
    edge_arms = {}
    for arm in arms:
        edge = (arm['source'], arm['target'])
        if edge not in edge_arms:
            edge_arms[edge] = []
        edge_arms[edge].append(arm)
    
    # Update edge beliefs based on arm rewards
    for edge, edge_arm_list in edge_arms.items():
        if not edge_arm_list:
            continue
            
        # Calculate average reward for this edge
        total_reward = sum(arm['reward'] * (discount_factor ** (1 / (arm['pulls'] + 1)))
                         for arm in edge_arm_list if arm['pulls'] > 0)
        total_pulls = sum(arm['pulls'] for arm in edge_arm_list)
        
        if total_pulls > 0:
            avg_reward = total_reward / len(edge_arm_list)
            
            # If average reward is high, add edge to graph
            if avg_reward > 0.1:
                graph.add_edge(*edge)
            # If average reward is low, remove edge from graph
            elif avg_reward < 0.05 and graph.has_edge(*edge):
                graph.remove_edge(*edge)

def perform_intervention(variable, value, simulator_path, duration=2000, intervention_time=100):
    """
    Perform an intervention using the provided simulator.
    
    Args:
        variable (str): Variable to intervene on
        value (float): Value to set the variable to
        simulator_path (str): Path to simulator script
        duration (int): Duration of simulation in ms
        intervention_time (int): Time to apply intervention in ms
        
    Returns:
        dict: Result of the intervention
    """
    import subprocess
    import tempfile
    import json
    
    # Validate simulator path
    if not simulator_path or simulator_path == 'None' or simulator_path == None:
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
        
        # Write simulation code
        temp_js.write(f'''
        const {{ simulateAndGetLatestData }} = require('{simulator_path}');
        
        async function runSimulation() {{
            try {{
                const result = await simulateAndGetLatestData(
                    {duration},
                    {json.dumps(intervention_obj)},
                    {intervention_time}
                );
                // Validate simulation result
                if (!result || !result.preInterventionData || !result.postInterventionData) {{
                    throw new Error('Invalid simulation result');
                }}
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
            console.error('SIMULATION_ERROR:', error);
            process.exit(1);
        }});

        runSimulation();
        ''')
    
    try:
        # Run the Node.js process with the same configuration as in tester.py
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
        output, stderr = process.communicate(timeout=60)  # 60 second timeout
        
        # Process all output lines
        for line in output.splitlines():
            print(line)  # Echo Node.js output for debugging
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
            logger.error(f"Node.js Errors:\n{stderr}")
        
        # Process valid simulation results
        if simulation_result and simulation_result.get('preInterventionData') and simulation_result.get('postInterventionData'):
            return simulation_result
        else:
            logger.warning("Simulation returned invalid result structure")
            return None
            
    except subprocess.TimeoutExpired:
        logger.error(f"Simulation timed out")
        if process:
            try:
                # Kill process
                process.kill()
            except:
                pass
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
            cycles = list(nx.simple_cycles(G))
            if cycles:
                # Remove first edge of first cycle
                cycle = cycles[0]
                if len(cycle) >= 2:
                    G.remove_edge(cycle[0], cycle[1])
            else:
                # No cycles found but still not a DAG?
                # This is unusual but let's break to avoid infinite loop
                break
        except nx.NetworkXNoCycle:
            break
    
    return list(G.edges())


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