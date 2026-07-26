#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Modular Benchmark Runner - Run setups individually with method selection
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import time
import logging
import argparse
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Add project paths
sys.path.append('.')
sys.path.append('./benchmarks')

# Import all baseline methods
from benchmarks_new.pc_baseline import run_pc_baseline
from benchmarks_new.sam_baseline import run_sam_baseline
from benchmarks_new.jci_comparison import run_jci
from benchmarks_new.abcd_comparison import run_abcd
from benchmarks_new.gies_comparison import run_gies
from benchmarks_new.causal_bandits import run_causal_bandits
from benchmarks_new.icp_comparison import run_icp
from benchmarks_new.iid_comparison import run_iid
from benchmarks_new.notears_i import run_notears_i
from src.generators import LLMGenerator

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Setup:
    name: str
    data_path: str
    ground_truth: List[Tuple[str, str]]
    description: str

class SetupBenchmark:
    """Run individual setups with method selection."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.setups = self._define_setups()
        self.all_methods = [
            'pc', 'sam', 'gies', 'jci', 'abcd', 
            'causal_bandits', 'icp', 'iid', 'notears_i'
        ]
        if api_key:
            self.all_methods.append('llm')
    
    def _define_setups(self) -> Dict[str, Setup]:
        """Define all 6 benchmark setups."""
        setups = [
            Setup(
                name="base",
                data_path="data_regen/smart_room_processed.csv",
                ground_truth=[
                    ("Temperature", "EnergyConsumption"),
                    ("Temperature", "Satisfaction"),
                    ("Humidity", "EnergyConsumption"),
                    ("Humidity", "Satisfaction"),
                    ("AirQuality", "EnergyConsumption"),
                    ("AirQuality", "Satisfaction")
                ],
                description="Basic smart room scenario"
            ),
            Setup(
                name="noisy",
                data_path="data_regen/smart_room_noise_processed.csv",
                ground_truth=[
                    ("Temperature", "EnergyConsumption"),
                    ("Temperature", "Satisfaction"),
                    ("Humidity", "EnergyConsumption"),
                    ("Humidity", "Satisfaction"),
                    ("AirQuality", "EnergyConsumption"),
                    ("AirQuality", "Satisfaction")
                ],
                description="Smart room with noise"
            ),
            Setup(
                name="hidden_vars",
                data_path="data_regen/hidden_vars_processed.csv",
                ground_truth=[
                    ("Temperature", "EnergyConsumption"),
                    ("Temperature", "Satisfaction"),
                    ("Humidity", "EnergyConsumption"),
                    ("Humidity", "Satisfaction"),
                    ("AirQuality", "EnergyConsumption"),
                    ("AirQuality", "Satisfaction")
                ],
                description="Smart room with hidden variables"
            ),
            Setup(
                name="ashrae",
                data_path="data/ashrae_data.csv",
                ground_truth=[
                    ("square_feet", "meter_reading"),
                    ("year_built", "meter_reading"),
                    ("air_temperature", "dew_temperature"),
                    ("air_temperature", "meter_reading"),
                    ("dew_temperature", "meter_reading"),
                ],
                description="ASHRAE building energy dataset"
            ),
            Setup(
                name="smart_building",
                data_path="data/building_simulation_2025-06-09T01-07-57-790Z_processed.csv",
                ground_truth=[
                    # HVAC load depends on internal and external environmental conditions
                    ("Temperature", "HVACPower"),         # More deviation from setpoint increases HVAC load
                    ("Humidity", "HVACPower"),            # Humidity control affects HVAC power
                    ("HVACSetpoint", "HVACPower"),        # Directly sets the thermal target
                    ("OccupantCount", "HVACPower"),       # More occupants → more internal heat gain

                    # Lighting power depends on control and presence
                    ("LightingLevel", "LightingPower"),   # Higher level → more lighting energy
                    ("OccupantCount", "LightingPower"),   # Occupancy triggers lighting systems

                    # Comfort measures depend on sensed conditions
                    ("Temperature", "ThermalComfort"),    # PMV model — thermal perception
                    ("Humidity", "ThermalComfort"),       # Humidity affects heat transfer

                    ("LightingLevel", "VisualComfort"),   # More light → better visual conditions

                    ("AirQuality", "AirQualityIndex"),    # Direct transformation of AQI to satisfaction

                    # Energy consumption is driven by all controls and conditions
                    ("Temperature", "EnergyConsumption"),     # Temperature deviation increases HVAC load
                    ("Humidity", "EnergyConsumption"),        # Humidity control costs energy
                    ("HVACSetpoint", "EnergyConsumption"),    # Setpoint defines thermal demand
                    ("LightingLevel", "EnergyConsumption"),   # Direct lighting energy use
                    ("OccupantCount", "EnergyConsumption"),   # Both HVAC and lighting respond to people
                    ("AirQuality", "EnergyConsumption"),      # Ventilation loads affect energy

                    # Satisfaction depends directly on controllable factors
                    ("Temperature", "OverallSatisfaction"),   # Impacts thermal comfort
                    ("Humidity", "OverallSatisfaction"),      # Impacts perceived air quality & comfort
                    ("HVACSetpoint", "OverallSatisfaction"),  # Affects comfort if poorly configured
                    ("LightingLevel", "OverallSatisfaction"), # Affects visual experience
                    ("OccupantCount", "OverallSatisfaction"), # Indirectly affects crowding/comfort
                    ("AirQuality", "OverallSatisfaction")     # Perceived air quality and health
                ],
                description="Complex smart building with multiple subsystems"
            ),
            Setup(
                name="physical",
                data_path="data/sensor_data_processed.csv",
                ground_truth=[
                    ("Temperature", "EnergyConsumption"),
                    ("Humidity", "EnergyConsumption"),
                    ("AirQuality", "EnergyConsumption"),
                    ("Temperature", "OverallSatisfaction"),
                    ("Humidity", "OverallSatisfaction"),
                    ("AirQuality", "OverallSatisfaction"),
                    ("Temperature", "Humidity"),
                    ("Humidity", "AirQuality")
                ],
                description="Physical smart room with causal dependencies"
            )
        ]
        return {setup.name: setup for setup in setups}

    def reverse_edges(self, edges, setup_name=None):
        """
        Reverse edge directions only if the source is an outcome variable.
        
        Args:
            edges: List or set of edges as tuples (source, target)
            setup_name: String indicating the setup ('base', 'noisy', 'hidden_vars', 'ashrae', 'smart_building', 'physical')
        
        Returns:
            List of edges with conditional reversals applied
        """
        
        # Define outcome variables for different setups
        outcome_variables = {
            'base': {'energyconsumption', 'overallsatisfaction'},
            'noisy': {'energyconsumption', 'overallsatisfaction'},
            'hidden_vars': {'energyconsumption', 'overallsatisfaction'},
            'smart_building': {
                'energyconsumption', 'overallsatisfaction', 'thermalcomfort', 
                'visualcomfort', 'airqualityindex', 'hvacpower', 'lightingpower'
            },
            'physical': {
                'energyconsumption', 'overallsatisfaction', 'thermalcomfort', 
                'visualcomfort', 'airqualityindex', 'hvacpower', 'lightingpower'
            },
            'ashrae': {'meter_reading'}
        }
        
        # Variable name normalization mapping
        variable_mapping = {
            'energyconsumption': 'energyconsumption',
            'energy_consumption': 'energyconsumption',
            'EnergyConsumption': 'energyconsumption',
            'ENERGYCONSUMPTION': 'energyconsumption',
            
            'overallsatisfaction': 'overallsatisfaction',
            'overall_satisfaction': 'overallsatisfaction',
            'OverallSatisfaction': 'overallsatisfaction',
            'OVERALLSATISFACTION': 'overallsatisfaction',
            
            'meter_reading': 'meter_reading',
            'meterreading': 'meter_reading',
            'MeterReading': 'meter_reading',
            'METER_READING': 'meter_reading',
            
            'thermalcomfort': 'thermalcomfort',
            'ThermalComfort': 'thermalcomfort',
            'thermal_comfort': 'thermalcomfort',
            
            'visualcomfort': 'visualcomfort',
            'VisualComfort': 'visualcomfort',
            'visual_comfort': 'visualcomfort',
            
            'airqualityindex': 'airqualityindex',
            'AirQualityIndex': 'airqualityindex',
            'air_quality_index': 'airqualityindex',
            
            'hvacpower': 'hvacpower',
            'HVACPower': 'hvacpower',
            'hvac_power': 'hvacpower',
            
            'lightingpower': 'lightingpower',
            'LightingPower': 'lightingpower',
            'lighting_power': 'lightingpower'
        }
        
        def normalize_variable_name(var_name):
            var_str = str(var_name).strip()
            return variable_mapping.get(var_str, var_str.lower())
        
        def infer_setup_from_edges(edges):
            all_vars = set()
            for edge in edges:
                if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    all_vars.add(normalize_variable_name(edge[0]))
                    all_vars.add(normalize_variable_name(edge[1]))
            
            if 'meter_reading' in all_vars:
                return 'ashrae'
            elif any(var in all_vars for var in ['hvacpower', 'lightingpower', 'thermalcomfort']):
                return 'smart_building'
            else:
                return 'base'  # Default for base/noisy/hidden_vars
        
        # Determine setup if not provided
        if setup_name is None:
            setup_name = infer_setup_from_edges(edges)
            print(f"Inferred setup: {setup_name}")
        
        # Get outcome variables for the setup
        outcome_vars = outcome_variables.get(setup_name, set())
        
        # Process edges
        processed_edges = []
        reversal_log = []
        
        for edge in edges:
            if not isinstance(edge, (list, tuple)) or len(edge) != 2:
                processed_edges.append(edge)
                continue
                
            source, target = edge
            normalized_source = normalize_variable_name(source)
            
            # Check if source is an outcome variable
            if normalized_source in outcome_vars:
                reversed_edge = (target, source)
                processed_edges.append(reversed_edge)
                reversal_log.append(f"Reversed {source} -> {target} to {target} -> {source}")
            else:
                processed_edges.append(edge)
        
        # Log reversals
        if reversal_log:
            print(f"Edge reversals applied ({setup_name}):")
            for log_entry in reversal_log:
                print(f"  - {log_entry}")
        else:
            print(f"No edge reversals needed for {setup_name}")
        
        return processed_edges
    
    def run_method(self, method: str, data: pd.DataFrame, setup_name: str) -> Dict:
        """Run a specific method on data."""
        start_time = time.time()
        
        try:
            # Run method and extract edges consistently
            if method == 'pc':
                result = run_pc_baseline(data.copy())
                edges = result if isinstance(result, list) else []
            elif method == 'sam':
                result = run_sam_baseline(data.copy())
                edges = result if isinstance(result, list) else []
            elif method == 'gies':
                result = run_gies(data.copy())
                edges = result[0] if isinstance(result, tuple) else result
            elif method == 'jci':
                result = run_jci(data.copy())
                edges = result[0] if isinstance(result, tuple) else result
            elif method == 'abcd':
                result = run_abcd(data.copy())
                edges = result[0] if isinstance(result, tuple) else result
            elif method == 'causal_bandits':
                result = run_causal_bandits(data.copy())
                edges = result[0] if isinstance(result, tuple) else result
            elif method == 'icp':
                result = run_icp(data.copy())
                edges = result[0] if isinstance(result, tuple) else result
            elif method == 'iid':
                result = run_iid(data.copy())
                edges = result[0] if isinstance(result, tuple) else result
            elif method == 'notears_i':
                result = run_notears_i(data.copy())
                edges = result[0] if isinstance(result, tuple) else result
            elif method == 'llm':
                relevant_columns = list(data.columns)
                llm_gen = LLMGenerator(self.api_key, relevant_columns=relevant_columns)
                result = llm_gen.generate(data)
                edges = result.get('edges', []) if isinstance(result, dict) else []
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Ensure edges is a list
            if edges is None:
                edges = []
            elif not isinstance(edges, list):
                edges = list(edges) if hasattr(edges, '__iter__') else []
            
            # Normalize edges to (str, str) tuples
            normalized_edges = []
            for edge in edges:
                if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                    normalized_edges.append((str(edge[0]), str(edge[1])))
            
            print(f"Method {method}: {len(normalized_edges)} edges extracted")
            
            # Reverse edges for specific methods
            if method in ['sam', 'gies', 'jci', 'abcd', 'causal_bandits', 'icp', 'iid', 'notears_i']:
                normalized_edges = self.reverse_edges(normalized_edges, setup_name)
                    
            runtime = time.time() - start_time
            
            return {
                'method': method,
                'edges': normalized_edges,
                'runtime': runtime,
                'success': True
            }
            
        except Exception as e:
            runtime = time.time() - start_time
            logger.error(f"Method {method} failed: {e}")
            return {
                'method': method,
                'edges': [],
                'runtime': runtime,
                'success': False,
                'error': str(e)
            }
    
    def calculate_metrics(self, predicted_edges: List[Tuple[str, str]], 
                         ground_truth: List[Tuple[str, str]]) -> Dict:
        """Calculate evaluation metrics with consistent edge normalization."""
        
        # Normalize edges to lowercase for consistent comparison
        def normalize_edge(edge):
            return tuple(str(x).strip().lower() for x in edge)
        
        predicted_set = {normalize_edge(edge) for edge in predicted_edges}
        ground_truth_set = {normalize_edge(edge) for edge in ground_truth}
        
        # Calculate confusion matrix elements
        true_positives = len(predicted_set.intersection(ground_truth_set))
        false_positives = len(predicted_set - ground_truth_set)
        false_negatives = len(ground_truth_set - predicted_set)
        
        # Calculate metrics following project conventions
        precision = true_positives / max(len(predicted_set), 1)
        recall = true_positives / len(ground_truth_set) if ground_truth_set else 0
        f1_score = 2 * (precision * recall) / max(precision + recall, 1e-10)
        
        # SHD: missing edges + extra edges
        shd = false_positives + false_negatives
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'shd': shd,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def load_data(self, setup: Setup) -> pd.DataFrame:
        """Load and clean data for a setup."""
        # Try multiple paths
        data_paths = [
            setup.data_path,
            os.path.join("data", setup.name, os.path.basename(setup.data_path)),
            os.path.join(setup.name, os.path.basename(setup.data_path))
        ]
        
        data = None
        for path in data_paths:
            if os.path.exists(path):
                data = pd.read_csv(path)
                break
        
        if data is None:
            raise FileNotFoundError(f"Data file not found. Tried: {data_paths}")
        
        # Clean data based on setup type
        if setup.name == 'ashrae':
            ashrae_cols = ['air_temperature', 'dew_temperature', 'sea_level_pressure', 
                          'meter_reading', 'square_feet', 'year_built']
            available_cols = [col for col in ashrae_cols if col in data.columns]
        else:
            cols_to_drop = ['Timestamp', 'timestamp', 'weight', 'InterventionApplied']
            data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])
            available_cols = list(data.columns)
        
        data = data[available_cols]
        
        # Handle missing values
        if data.isnull().any().any():
            data = data.fillna(data.mean())
        
        return data
    
    def print_results_table(self, results: List[Dict], setup_name: str):
        """Print formatted results table."""
        print(f"\n{'='*73}")
        print(f"Setup: {setup_name.upper()}")
        print(f"{'='*73}")
        print(f"{'Method':>12} {'Runtime':>8} {'Precision':>10} {'Recall':>8} {'F1_Score':>8} {'SHD':>5}")
        print(f"{'='*73}")
        
        for result in results:
            method = result['method']
            runtime = result['runtime']
            metrics = result['metrics']
            
            print(f"{method:>12} {runtime:>8.6f} {metrics['precision']:>10.6f} "
                  f"{metrics['recall']:>8.6f} {metrics['f1_score']:>8.6f} {metrics['shd']:>5}")
        
        print(f"{'='*73}")
    
    def run_setup(self, setup_name: str, methods: List[str] = None):
        """Run benchmark on a specific setup."""
        if setup_name not in self.setups:
            print(f"Unknown setup: {setup_name}")
            print(f"Available setups: {list(self.setups.keys())}")
            return
        
        setup = self.setups[setup_name]
        
        if methods is None:
            methods = self.all_methods
        
        # Validate methods
        invalid_methods = [m for m in methods if m not in self.all_methods]
        if invalid_methods:
            print(f"Invalid methods: {invalid_methods}")
            print(f"Available methods: {self.all_methods}")
            return
        
        try:
            data = self.load_data(setup)
            print(f"Loaded data: {data.shape[0]} samples, {data.shape[1]} variables")
        except Exception as e:
            print(f"Failed to load data: {e}")
            return
        
        # Run methods
        results = []
        for method in methods:
            result = self.run_method(method, data, setup_name)
            if result['success']:
                result['metrics'] = self.calculate_metrics(result['edges'], setup.ground_truth)
            else:
                result['metrics'] = self.calculate_metrics([], setup.ground_truth)
            results.append(result)
        
        # Print table
        self.print_results_table(results, setup_name)
        
        # Save results
        output_dir = f"results_{setup_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to JSON
        with open(os.path.join(output_dir, "results.json"), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save to CSV
        csv_data = []
        for result in results:
            metrics = result['metrics']
            csv_data.append({
                'Setup': setup_name,
                'Method': result['method'],
                'Runtime': result['runtime'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1_Score': metrics['f1_score'],
                'SHD': metrics['shd'],
                'Success': result['success'],
                'Error': result.get('error', '')
            })
        
        pd.DataFrame(csv_data).to_csv(os.path.join(output_dir, "results.csv"), index=False)
        print(f"Results saved to {output_dir}/")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Run causal discovery benchmark')
    parser.add_argument('setup', choices=['base', 'noisy', 'hidden_vars', 'ashrae', 'smart_building', 'physical'], 
                       help='Setup to run')
    parser.add_argument('--methods', nargs='+', 
                       choices=['pc', 'sam', 'gies', 'jci', 'abcd', 'causal_bandits', 'icp', 'iid', 'notears_i', 'llm'],
                       help='Methods to run (default: all)')
    parser.add_argument('--api-key', default="YOUR_OPENAI_API_KEY",
                        help='OpenAI API key for LLM method')
    
    args = parser.parse_args()
    
    benchmark = SetupBenchmark(args.api_key)
    benchmark.run_setup(args.setup, args.methods)


if __name__ == "__main__":
    main()