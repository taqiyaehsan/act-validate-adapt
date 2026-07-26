#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fast Prompt Ablation: 3 Dataset Scenarios x 3 Prompt Variants
Tests: base, building, ashrae
Working directory: .
"""

import logging
import numpy as np
import pandas as pd
import json
import sys
import os
import networkx as nx
from pathlib import Path

# Set working directory
WORK_DIR = Path('.')
os.chdir(WORK_DIR)
sys.path.insert(0, str(WORK_DIR))

from src.gpt_client import GPTClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def normalize_edge(edge):
    """Normalize edge to tuple format"""
    if isinstance(edge, (list, tuple)):
        return tuple(str(x).strip() for x in edge)
    return edge


def calculate_shd(predicted_edges, true_edges):
    """Calculate Structural Hamming Distance"""
    predicted = {normalize_edge(edge) for edge in predicted_edges}
    true = {normalize_edge(edge) for edge in true_edges}
    
    missing = true - predicted
    extra = predicted - true
    
    return len(missing) + len(extra)


def calculate_precision_recall_f1(predicted_edges, true_edges):
    """Calculate precision, recall, F1, and SHD"""
    predicted = {normalize_edge(edge) for edge in predicted_edges}
    true = {normalize_edge(edge) for edge in true_edges}
    
    true_positives = len(predicted & true)
    false_positives = len(predicted - true)
    false_negatives = len(true - predicted)
    
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1_score = 2 * precision * recall / max(precision + recall, 1e-10)
    shd = calculate_shd(predicted, true)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'f1': f1_score,
        'shd': shd,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }


def make_dag(edges):
    """Convert edges to DAG by removing cycles"""
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
    while not nx.is_directed_acyclic_graph(G):
        try:
            cycle = nx.find_cycle(G, orientation="original")
            u, v = cycle[0][0], cycle[0][1]
            G.remove_edge(u, v)
        except nx.NetworkXNoCycle:
            break
    
    return list(G.edges())


def compute_metrics(true_dag, predicted_dag):
    """Compute metrics comparing predicted DAG to true DAG"""
    true_edges = list(true_dag.edges())
    pred_edges = list(predicted_dag.edges())
    
    return calculate_precision_recall_f1(pred_edges, true_edges)


# Dataset configurations
DATASETS = {
    'base': {
        'path': 'data/simulation_log_2025-03-04T16-42-08-821Z_preprocessed.csv',
        'columns': ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'OverallSatisfaction'],
        'ground_truth': [
            ('Temperature', 'EnergyConsumption'),
            ('Temperature', 'OverallSatisfaction'),
            ('Humidity', 'EnergyConsumption'),
            ('Humidity', 'OverallSatisfaction'),
            ('AirQuality', 'EnergyConsumption'),
            ('AirQuality', 'OverallSatisfaction')
        ]
    },
    'building': {
        'path': 'data/building_simulation_2025-06-09T01-07-57-790Z_processed.csv',
        'columns': ['Temperature', 'Humidity', 'AirQuality', 'LightingLevel', 'HVACSetpoint', 'OccupantCount',
                   'EnergyConsumption', 'OverallSatisfaction', 'ThermalComfort', 'VisualComfort', 
                   'AirQualityIndex', 'HVACPower', 'LightingPower'],
        'ground_truth': [
            ("Temperature", "HVACPower"),
            ("Humidity", "HVACPower"),
            ("HVACSetpoint", "HVACPower"),
            ("OccupantCount", "HVACPower"),
            ("LightingLevel", "LightingPower"),
            ("OccupantCount", "LightingPower"),
            ("Temperature", "ThermalComfort"),
            ("Humidity", "ThermalComfort"),
            ("LightingLevel", "VisualComfort"),
            ("AirQuality", "AirQualityIndex"),
            ("Temperature", "EnergyConsumption"),
            ("Humidity", "EnergyConsumption"),
            ("HVACSetpoint", "EnergyConsumption"),
            ("LightingLevel", "EnergyConsumption"),
            ("OccupantCount", "EnergyConsumption"),
            ("AirQuality", "EnergyConsumption"),
            ("Temperature", "OverallSatisfaction"),
            ("Humidity", "OverallSatisfaction"),
            ("HVACSetpoint", "OverallSatisfaction"),
            ("LightingLevel", "OverallSatisfaction"),
            ("OccupantCount", "OverallSatisfaction"),
            ("AirQuality", "OverallSatisfaction")
        ]
    },
    'ashrae': {
        'path': 'data/ashrae_data_processed.csv',
        'columns': ['air_temperature', 'dew_temperature', 'sea_level_pressure', 'meter_reading', 'square_feet'],
        'ground_truth': [
            ('air_temperature', 'meter_reading'),
            ('dew_temperature', 'meter_reading'),
            ('square_feet', 'meter_reading'),
            ('air_temperature', 'dew_temperature')
        ]
    }
}


class PromptVariants:
    """Three prompt variants"""
    
    @staticmethod
    def baseline(cols_list):
        """Original baseline prompt"""
        if len(cols_list) == 5 and 'meter_reading' in cols_list:
            # ASHRAE
            system = """You are an expert in causal discovery analyzing a smart room environment with 5 variables.
Generate a causal DAG based on physical principles and environmental systems.
Return your answer as a JSON object with this format exactly:
{"nodes": ['air_temperature', 'dew_temperature', 'sea_level_pressure', 'meter_reading', 'square_feet'], 
"edges": [["source", "target"], ...]}"""
            
            user = f"""Analyze this dataset with variables: {cols_list}

Rules:
1. Include directed edges based on likely causal mechanisms
2. Cycles are ok but no self-loops allowed
3. Focus on primary physical relationships, not secondary effects
4. Return only the JSON object, no additional text"""
        
        elif len(cols_list) == 13:
            # Building (large-scale)
            system = """You are an expert in causal discovery analyzing a smart building environment.
Generate a causal DAG based on HVAC systems, thermal comfort, and energy efficiency.
Variables: 'Temperature', 'Humidity', 'AirQuality', 'LightingLevel', 'HVACSetpoint', 'OccupantCount',
'EnergyConsumption', 'OverallSatisfaction', 'ThermalComfort', 'VisualComfort', 'AirQualityIndex', 'HVACPower', 'LightingPower']
Outcomes: 'EnergyConsumption', 'OverallSatisfaction', 'AirQualityIndex',
'ThermalComfort', 'VisualComfort', 'HVACPower', 'LightingPower'
Return JSON format: {"nodes": [...], "edges": [["source", "target"], ...]}
IMPORTANT: Return ONLY valid JSON, no markdown formatting or code blocks."""
            
            user = f"""Analyze this dataset with variables: {cols_list}

Rules:
1. Include directed edges based on likely causal mechanisms
2. Cycles are ok but no self-loops allowed
3. Focus on primary physical relationships, not secondary effects
4. Return only the JSON object, no additional text"""
        
        else:
            # Base (5 variables)
            system = """You are an expert in causal discovery. 
Generate a causal DAG based on the environment and energy consumption in buildings.
Return your answer as a JSON object with this format exactly:
{"nodes": ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'OverallSatisfaction'], "edges": [["source", "target"], ...]}
IMPORTANT: Return ONLY valid JSON, no markdown formatting or code blocks."""
            
            user = f"""Analyze this dataset with variables: {', '.join(cols_list)}

Rules:
1. Include directed edges based on likely causal mechanisms
2. Cycles are ok but no self-loops allowed
3. Focus on primary physical relationships, not secondary effects
4. Return only the JSON object, no additional text"""
        
        return system, user
    
    @staticmethod
    def minimal(cols_list):
        """Minimal - remove domain guidance"""
        system = """You are an expert in causal discovery.
Generate a causal DAG.
Return JSON: {"nodes": [...], "edges": [["source", "target"], ...]}
ONLY valid JSON, no markdown."""
        
        user = f"""Variables: {', '.join(cols_list)}

Rules:
1. Directed edges for causal relationships
2. No cycles or self-loops
3. Return only JSON"""
        
        return system, user
    
    @staticmethod
    def ablated(cols_list):
        """Ablated - remove causal reasoning guidance"""
        system = """You are an expert in data analysis.
Generate a directed graph showing relationships between variables.
Return JSON: {"nodes": [...], "edges": [["source", "target"], ...]}
ONLY valid JSON, no markdown."""
        
        user = f"""Variables: {', '.join(cols_list)}

Create a directed graph showing:
1. Relationships between variables
2. No cycles or self-loops
3. Return only JSON"""
        
        return system, user


def test_scenario_variant(api_key, scenario_name, variant_name, variant_func, n_runs=3):
    """Test single scenario-variant combination"""
    logger.info(f"\nTesting: {scenario_name} - {variant_name}")
    
    config = DATASETS[scenario_name]
    
    # Load data
    try:
        if not os.path.exists(config['path']):
            logger.warning(f"Data file not found: {config['path']}, skipping")
            return None
        data = pd.read_csv(config['path'])
    except Exception as e:
        logger.error(f"Failed to load {config['path']}: {e}")
        return None
    
    gpt_client = GPTClient(api_key)
    cols_list = config['columns']
    
    f1_scores = []
    shd_scores = []
    
    for run in range(n_runs):
        try:
            # Generate prompt
            system_prompt, user_message = variant_func(cols_list)
            
            # Get LLM response
            response = gpt_client.generate_response(system_prompt, user_message, temperature=1.0, top_p=0.8)
            
            # Parse (same cleaning as generators.py)
            cleaned = response.strip()
            
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            elif cleaned.startswith('```'):
                cleaned = cleaned[3:]
            elif cleaned.startswith('`json'):
                cleaned = cleaned[5:]
            elif cleaned.startswith('`'):
                cleaned = cleaned[1:]
                
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            elif cleaned.endswith('`'):
                cleaned = cleaned[:-1]
            
            cleaned = cleaned.strip()
            
            if not cleaned:
                logger.warning(f"Empty response for run {run+1}")
                continue
            
            parsed = json.loads(cleaned)
            
            if 'edges' not in parsed or 'nodes' not in parsed:
                logger.warning(f"Invalid response structure in run {run+1}")
                continue
            
            predicted_edges = parsed['edges']
            
            # Compute metrics
            pred_edges_dag = make_dag(predicted_edges)
            true_edges_dag = make_dag(config['ground_truth'])
            
            pred_dag = nx.DiGraph()
            pred_dag.add_edges_from(pred_edges_dag)
            
            true_dag = nx.DiGraph()
            true_dag.add_edges_from(true_edges_dag)
            
            metrics = compute_metrics(true_dag, pred_dag)
            
            f1_scores.append(metrics.get('f1', 0.0))
            shd_scores.append(metrics.get('shd', 0))
            
            logger.info(f"  Run {run+1}: F1={metrics.get('f1', 0):.3f}, SHD={metrics.get('shd', 0)}")
            
        except Exception as e:
            logger.error(f"  Run {run+1} failed: {e}")
            continue
    
    if not f1_scores:
        return None
    
    return {
        'scenario': scenario_name,
        'variant': variant_name,
        'n_runs': len(f1_scores),
        'f1_mean': float(np.mean(f1_scores)),
        'f1_std': float(np.std(f1_scores)),
        'shd_mean': float(np.mean(shd_scores)),
        'shd_std': float(np.std(shd_scores)),
        'f1_scores': [float(x) for x in f1_scores],
        'shd_scores': [int(x) for x in shd_scores]
    }


def run_full_ablation(api_key, n_runs_per_test=10):
    """Run complete ablation study"""
    logger.info("="*80)
    logger.info("PROMPT ABLATION STUDY: 3 Scenarios × 3 Variants")
    logger.info("="*80)
    
    variants = [
        ('baseline', PromptVariants.baseline),
        ('minimal', PromptVariants.minimal),
        ('ablated', PromptVariants.ablated)
    ]
    
    all_results = []
    
    for scenario_name in ['base', 'building', 'ashrae']:
        for variant_name, variant_func in variants:
            result = test_scenario_variant(
                api_key, scenario_name, variant_name, variant_func, n_runs_per_test
            )
            if result:
                all_results.append(result)
    
    return all_results


def print_summary(results):
    """Print formatted summary"""
    logger.info("\n" + "="*100)
    logger.info("RESULTS SUMMARY")
    logger.info("="*100)
    
    scenarios = ['base', 'building', 'ashrae']
    
    for scenario in scenarios:
        scenario_results = [r for r in results if r['scenario'] == scenario]
        if not scenario_results:
            continue
        
        logger.info(f"\n{'='*100}")
        logger.info(f"SCENARIO: {scenario.upper()}")
        logger.info(f"{'='*100}")
        
        print(f"\n{'Variant':<15} {'F1 Mean±Std':<15} {'SHD Mean±Std':<15} {'N Runs':<8}")
        print("-"*60)
        
        baseline_f1 = None
        for r in scenario_results:
            if r['variant'] == 'baseline':
                baseline_f1 = r['f1_mean']
            
            print(f"{r['variant']:<15} "
                  f"{r['f1_mean']:.3f}±{r['f1_std']:.3f}    "
                  f"{r['shd_mean']:.1f}±{r['shd_std']:.1f}      "
                  f"{r['n_runs']:<8}")
        
        if baseline_f1:
            print(f"\n{'Comparison vs Baseline:'}")
            for r in scenario_results:
                if r['variant'] != 'baseline':
                    delta = r['f1_mean'] - baseline_f1
                    pct = (delta / baseline_f1 * 100) if baseline_f1 > 0 else 0
                    print(f"  {r['variant']}: F1 Δ = {delta:+.3f} ({pct:+.1f}%)")


def save_results(results, output_path='ablation_results.json'):
    """Save results to JSON"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {output_path}")


def main():
    api_key = "YOUR_OPENAI_API_KEY"
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        return
    
    # Run ablation (3 runs per test for speed)
    results = run_full_ablation(api_key, n_runs_per_test=10)
    
    # Print and save
    print_summary(results)
    save_results(results)
    
    logger.info("\n" + "="*100)
    logger.info(f"COMPLETE: Tested {len(results)} scenario-variant combinations")
    logger.info("="*100)


if __name__ == "__main__":
    main()