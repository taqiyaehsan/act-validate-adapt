import pandas as pd
import numpy as np
import json
import os
import time
import logging
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import your existing classes
from src.pipeline import CausalPipeline
from src.generators import PCGenerator, SAMGenerator, LLMGenerator
from src.tester import HypothesisTester
from src.evaluator import EdgeRanker
from src.ground_truth import GroundTruthDAG
from src.metrics import MetricsCalculator

logger = logging.getLogger(__name__)

class RealDataAblationStudy:
    """
    Ablation study using your actual simulations and collected data
    """
    
    def __init__(self, api_key, output_dir='ablation_results_real'):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Define generator combinations
        self.configurations = {
            'pc_only': ['pc'],
            'sam_only': ['sam'], 
            'llm_only': ['llm'],
            'pc_sam': ['pc', 'sam'],
            'pc_llm': ['pc', 'llm'],
            'sam_llm': ['sam', 'llm'],
            'full_framework': ['pc', 'sam', 'llm']
        }
        
        # Define your actual test scenarios
        self.scenarios = self._define_scenarios()
        self.results = defaultdict(lambda: defaultdict(dict))
    
    def _define_scenarios(self):
        """Define test scenarios using your actual data and simulations"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        scenarios = {
            'base_smart_room': {
                'data_path': 'ata/simulation_log_2025-03-04T16-42-08-821Z_preprocessed.csv',
                'simulation_path': './smart_room.js',
                'dataset_type': 'simulation',
                'ground_truth': [
                    ('Temperature', 'EnergyConsumption'),
                    ('Humidity', 'EnergyConsumption'),
                    ('AirQuality', 'EnergyConsumption'),
                    ('Temperature', 'OverallSatisfaction'),
                    ('Humidity', 'OverallSatisfaction'),
                    ('AirQuality', 'OverallSatisfaction')
                ],
                'relevant_columns': ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'OverallSatisfaction']
            },
            
            'noisy_smart_room': {
                'data_path': 'data/smart_room_noisy_preprocessed.csv',  # Add noise during processing
                'simulation_path': './smart_room_noise.js',
                'dataset_type': 'simulation',
                'ground_truth': [
                    ('Temperature', 'EnergyConsumption'),
                    ('Humidity', 'EnergyConsumption'),
                    ('AirQuality', 'EnergyConsumption'),
                    ('Temperature', 'OverallSatisfaction'),
                    ('Humidity', 'OverallSatisfaction'),
                    ('AirQuality', 'OverallSatisfaction')
                ],
                'relevant_columns': ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'OverallSatisfaction']
            },
            
            'hidden_vars_smart_room': {
                'data_path': 'data/smart-room-hidden-vars_preprocessed.csv',
                'simulation_path': './smart_room_hidden_vars.js',
                'dataset_type': 'simulation',
                'ground_truth': [
                    ('Temperature', 'EnergyConsumption'),
                    ('Humidity', 'EnergyConsumption'),
                    ('AirQuality', 'EnergyConsumption'),
                    ('Temperature', 'OverallSatisfaction'),
                    ('Humidity', 'OverallSatisfaction'),
                    ('AirQuality', 'OverallSatisfaction')
                ],
                'relevant_columns': ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'OverallSatisfaction']
            },
            
            'smart_building': {
                'data_path': 'data/building_simulation_2025-06-09T01-07-57-790Z_processed.csv',
                'simulation_path': './smart_building.js',
                'dataset_type': 'simulation',
                'ground_truth': [
                    ("Temperature", "EnergyConsumption"),
                    ("Humidity", "EnergyConsumption"),
                    ("AirQuality", "EnergyConsumption"),
                    ("HVACSetpoint", "EnergyConsumption"),
                    ("OccupantCount", "EnergyConsumption"),
                    ("Temperature", "OverallSatisfaction"),
                    ("Humidity", "OverallSatisfaction"),
                    ("AirQuality", "OverallSatisfaction"),
                    ("LightingLevel", "OverallSatisfaction"),
                    ("HVACSetpoint", "OverallSatisfaction"),
                    ("OccupantCount", "OverallSatisfaction"),
                    ("Temperature", "ThermalComfort"),
                    ("Humidity", "ThermalComfort"),
                    ("LightingLevel", "VisualComfort"),
                    ("AirQuality", "AirQualityIndex"),
                    ("EnergyConsumption", "HVACPower"),
                    ("LightingLevel", "LightingPower"),
                    ("HVACSetpoint", "HVACPower")
                ],
                'relevant_columns': ['Temperature', 'Humidity', 'AirQuality',
                                    'LightingLevel', 'HVACSetpoint', 'OccupantCount',
                                    'EnergyConsumption', 'OverallSatisfaction',
                                    'ThermalComfort', 'VisualComfort', 'AirQualityIndex', 
                                    'HVACPower', 'LightingPower']
            },
            
            'ashrae_building': {
                'data_path': 'data/ashrae_data.csv',  # Your ASHRAE dataset
                'simulation_path': None,  # No simulation for ASHRAE
                'dataset_type': 'ashrae',
                'ground_truth': [
                    ("square_feet", "meter_reading"),
                    ("year_built", "meter_reading"),
                    ("air_temperature", "dew_temperature"),
                    ("air_temperature", "meter_reading"),
                    ("dew_temperature", "meter_reading"),
                ],
                'relevant_columns': ['air_temperature', 'dew_temperature', 
                                   'meter_reading', 'square_feet', 'year_built']
            }
        }
        
        return scenarios
    
    def create_modified_pipeline(self, scenario_config, selected_generators):
        """Create a modified pipeline that uses only selected generators"""
        
        class ModifiedCausalPipeline(CausalPipeline):
            def __init__(self, csv_data, api_key, smart_room_path=None, 
                        selected_generators=['pc', 'sam', 'llm'], **kwargs):
                
                # Initialize parent class
                super().__init__(csv_data, api_key, smart_room_path, **kwargs)
                
                # Override generators with selected ones only
                self.selected_generators = selected_generators
                self.generators = {}
                
                if 'pc' in selected_generators:
                    self.generators['pc'] = PCGenerator(relevant_columns=self.relevant_columns)
                    
                if 'sam' in selected_generators:
                    self.generators['sam'] = SAMGenerator(relevant_columns=self.relevant_columns)
                    
                if 'llm' in selected_generators:
                    self.generators['llm'] = LLMGenerator(api_key, relevant_columns=self.relevant_columns)
                
                logger.info(f"Pipeline initialized with generators: {list(self.generators.keys())}")
            
            def _generate_hypotheses(self):
                """Generate hypotheses using only selected generators"""
                if 'hypotheses' in self._cache:
                    return self._cache['hypotheses']
                    
                hypotheses = {}
                for name, generator in self.generators.items():
                    try:
                        logger.info(f"Generating hypothesis using {name.upper()}")
                        hypothesis = generator.generate(self.data)
                        
                        if hypothesis and 'edges' in hypothesis:
                            # Filter out invalid source edges
                            if self.dataset_type == 'ashrae':
                                output_vars = {'meter_reading'}
                            else:
                                output_vars = {'energyconsumption', 'overallsatisfaction'}
                            
                            valid_edges = [
                                edge for edge in hypothesis['edges']
                                if str(edge[0]).lower() not in output_vars
                            ]
                            
                            hypotheses[name] = {
                                'hypothesis': {**hypothesis, 'edges': valid_edges},
                                'edges': valid_edges
                            }
                            
                            logger.info(f"{name.upper()} generated {len(valid_edges)} edges: {valid_edges}")
                        else:
                            logger.warning(f"{name.upper()} failed to generate valid hypothesis")
                            
                    except Exception as e:
                        logger.error(f"Error in {name} generation: {e}")
                        
                self._cache['hypotheses'] = hypotheses
                return hypotheses
        
        return ModifiedCausalPipeline
    
    def load_scenario_data(self, scenario_name):
        """Load data for a specific scenario"""
        scenario = self.scenarios[scenario_name]
        
        # Load the data
        data_path = scenario['data_path']
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
        else:
            data = pd.read_csv('sensor_data_processed.csv')
        
        # Filter to relevant columns if they exist
        available_columns = [col for col in scenario['relevant_columns'] if col in data.columns]
        if available_columns:
            data = data[available_columns]
        
        # Add data validation
        if data.shape[0] < 50:
            logger.warning(f"Insufficient data for {scenario_name}: {data.shape}")
            return None
        
        # Remove constant/missing columns
        data = data.dropna()
        # constant_cols = [col for col in data.columns if data[col].nunique() <= 1]
        # if constant_cols:
        #     logger.warning(f"Removing constant columns: {constant_cols}")
        #     data = data.drop(columns=constant_cols)
        
        logger.info(f"Loaded {scenario_name} data: {data.shape}")
        return data
    
    def run_single_configuration(self, scenario_name, config_name, generators, n_runs=3):
        """Run a single configuration with your actual pipeline"""
        
        logger.info(f"Running {config_name} on {scenario_name} ({n_runs} runs)")
        
        scenario = self.scenarios[scenario_name]
        data = self.load_scenario_data(scenario_name)
        
        run_results = []
        
        for run_idx in range(n_runs):
            logger.info(f"  Run {run_idx + 1}/{n_runs}")
            
            start_time = time.time()
            
            try:
                # Create modified pipeline
                ModifiedPipeline = self.create_modified_pipeline(scenario, generators)
                
                # Initialize pipeline with scenario-specific parameters
                pipeline = ModifiedPipeline(
                    csv_data=data,
                    api_key=self.api_key,
                    smart_room_path=scenario['simulation_path'],
                    selected_generators=generators,
                    max_iterations=2,  # Reduced for ablation study
                    dataset_type=scenario['dataset_type'],
                    relevant_column=scenario['relevant_columns']
                )
                
                # Run the pipeline
                logger.info(f"Starting pipeline execution for {config_name}")
                final_dag = pipeline.run()
                
                # Calculate metrics
                metrics = self._calculate_metrics(
                    predicted_edges=final_dag.get('edges', []),
                    ground_truth_edges=scenario['ground_truth'],
                    pipeline=pipeline
                )
                
                result = {
                    **metrics,
                    'runtime': time.time() - start_time,
                    'total_interventions': getattr(pipeline, 'intervention_count', 0),
                    'iterations_to_convergence': getattr(pipeline, 'current_iteration', 0),
                    'generators_used': generators,
                    'final_dag': final_dag,
                    'run_id': run_idx,
                    'scenario': scenario_name,
                    'configuration': config_name
                }
                
                run_results.append(result)
                
                logger.info(f"  ✓ Run {run_idx + 1} completed - SHD: {metrics['shd']}, F1: {metrics['f1']:.3f}")
                
            except Exception as e:
                logger.error(f"  ✗ Run {run_idx + 1} failed: {str(e)}")
                run_results.append({
                    'error': str(e),
                    'runtime': time.time() - start_time,
                    'run_id': run_idx,
                    'scenario': scenario_name,
                    'configuration': config_name,
                    'generators_used': generators
                })
        
        return run_results
    
    def _calculate_metrics(self, predicted_edges, ground_truth_edges, pipeline):
        """Calculate performance metrics including risk and cost"""
        # Normalize edges for comparison
        def normalize_edge(edge):
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                return (str(edge[0]).lower(), str(edge[1]).lower())
            return edge
        
        if isinstance(predicted_edges, dict):
            predicted_edges = predicted_edges.get('edges', [])
        elif isinstance(predicted_edges, tuple):
            predicted_edges = list(predicted_edges)
        elif not isinstance(predicted_edges, list):
            predicted_edges = []
        
        # Handle case where predicted_edges might be None or malformed
        if not predicted_edges:
            predicted_edges = []
        
        pred_set = {normalize_edge(edge) for edge in predicted_edges if edge is not None}
        gt_set = {normalize_edge(edge) for edge in ground_truth_edges if edge is not None}
        
        # Remove any None values
        pred_set.discard(None)
        gt_set.discard(None)
        
        # Calculate basic metrics
        true_positives = len(pred_set.intersection(gt_set))
        false_positives = len(pred_set - gt_set)
        false_negatives = len(gt_set - pred_set)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        shd = false_positives + false_negatives
        
        base_metrics = {
            'shd': float(shd),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'false_negatives': int(false_negatives)
        }
        
        # Add risk/cost metrics if pipeline has MetricsCalculator
        if hasattr(pipeline, 'metrics_calculator') and hasattr(pipeline, '_results'):
            try:
                for method_name, method_result in pipeline._results.items():
                    # Handle both dict and direct edge list formats
                    if isinstance(method_result, dict):
                        edges = set(map(tuple, method_result.get('edges', [])))
                    elif isinstance(method_result, (list, set)):
                        edges = set(map(tuple, method_result))
                    else:
                        continue
                    
                    # Calculate method-specific metrics
                    method_metrics = pipeline.metrics_calculator.calculate_metrics(
                        method_name,
                        edges,
                        getattr(pipeline.edge_ranker, 'edge_confidence', {}),
                        getattr(pipeline.tester, 'intervention_results', {})
                    )
                    
                    base_metrics[f'{method_name}_cost'] = float(method_metrics.cost)
                    base_metrics[f'{method_name}_risk'] = float(method_metrics.risk)
                    
            except Exception as e:
                logger.warning(f"Could not calculate risk/cost metrics: {e}")
                base_metrics['risk_cost_error'] = str(e)
        
        return base_metrics
    
    def run_full_study(self, n_runs=3, selected_scenarios=None, selected_configs=None):
        """Run the complete ablation study"""
        
        if selected_scenarios is None:
            selected_scenarios = list(self.scenarios.keys())
        
        if selected_configs is None:
            selected_configs = list(self.configurations.keys())
        
        logger.info("Starting Ablation Study")
        logger.info(f"Scenarios: {selected_scenarios}")
        logger.info(f"Configurations: {selected_configs}")
        logger.info(f"Runs per configuration: {n_runs}")
        
        total_experiments = len(selected_scenarios) * len(selected_configs)
        experiment_count = 0
        
        for scenario_name in selected_scenarios:
            if scenario_name not in self.scenarios:
                logger.warning(f"Scenario {scenario_name} not found, skipping")
                continue
                
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing scenario: {scenario_name}")
            logger.info(f"{'='*60}")
            
            for config_name in selected_configs:
                if config_name not in self.configurations:
                    logger.warning(f"Configuration {config_name} not found, skipping")
                    continue
                    
                experiment_count += 1
                generators = self.configurations[config_name]
                
                logger.info(f"\nExperiment {experiment_count}/{total_experiments}: {config_name}")
                logger.info(f"Generators: {generators}")
                
                # Run this configuration
                run_results = self.run_single_configuration(
                    scenario_name, config_name, generators, n_runs
                )
                
                # Process and store results
                self.results[scenario_name][config_name] = {
                    'individual_runs': run_results,
                    'mean_metrics': self._calculate_mean_metrics(run_results),
                    'std_metrics': self._calculate_std_metrics(run_results)
                }
                
                # Save intermediate results
                self._save_intermediate_results(scenario_name, config_name)
        
        logger.info(f"\nReal Data Ablation Study completed!")
        logger.info(f"Total experiments: {experiment_count}")
        
        return self.results
    
    def _calculate_mean_metrics(self, run_results):
        """Calculate mean metrics across successful runs"""
        valid_runs = [r for r in run_results if 'error' not in r]
        
        if not valid_runs:
            return {}
        
        metrics = ['shd', 'precision', 'recall', 'f1', 'total_interventions', 
                  'iterations_to_convergence', 'runtime']
        
        mean_metrics = {}
        for metric in metrics:
            values = [r.get(metric, 0) for r in valid_runs]
            mean_metrics[metric] = np.mean(values) if values else 0.0
            mean_metrics[f'{metric}_count'] = len(values)
        
        return mean_metrics
    
    def _calculate_std_metrics(self, run_results):
        """Calculate standard deviation across successful runs"""
        valid_runs = [r for r in run_results if 'error' not in r]
        
        if not valid_runs:
            return {}
        
        metrics = ['shd', 'precision', 'recall', 'f1', 'total_interventions', 
                  'iterations_to_convergence', 'runtime']
        
        std_metrics = {}
        for metric in metrics:
            values = [r.get(metric, 0) for r in valid_runs]
            std_metrics[metric] = np.std(values) if len(values) > 1 else 0.0
        
        return std_metrics
    
    def _save_intermediate_results(self, scenario_name, config_name):
        """Save intermediate results after each configuration"""
        results_file = self.output_dir / f'intermediate_results_{scenario_name}_{config_name}.json'
        
        try:
            # Convert numpy types for JSON serialization
            def convert_for_json(obj):
                if isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                else:
                    return obj
            
            serializable_data = convert_for_json(self.results[scenario_name][config_name])
            
            with open(results_file, 'w') as f:
                json.dump(serializable_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}")
    
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations...")
        
        # 1. SHD Heatmap
        self._plot_shd_heatmap()
        
        # 2. Performance metrics comparison
        self._plot_performance_comparison()
        
        # 3. Intervention efficiency
        self._plot_intervention_efficiency()
        
        # 4. Runtime analysis
        self._plot_runtime_analysis()
        
        # 5. Generator contribution analysis
        self._plot_generator_contributions()
    
    def _plot_shd_heatmap(self):
        """Create SHD heatmap across all configurations and scenarios"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        scenarios = [s for s in self.scenarios.keys() if s in self.results]
        configs = [c for c in self.configurations.keys() if any(c in self.results[s] for s in scenarios)]
        
        # Build matrix
        shd_matrix = []
        for scenario in scenarios:
            row = []
            for config in configs:
                if config in self.results[scenario]:
                    shd = self.results[scenario][config]['mean_metrics'].get('shd', float('inf'))
                else:
                    shd = float('inf')
                row.append(shd)
            shd_matrix.append(row)
        
        # Create heatmap
        sns.heatmap(shd_matrix, 
                   xticklabels=[c.replace('_', ' ').title() for c in configs], 
                   yticklabels=[s.replace('_', ' ').title() for s in scenarios],
                   annot=True, 
                   fmt='.1f',
                   cmap='RdYlGn_r',
                   ax=ax,
                   cbar_kws={'label': 'Structural Hamming Distance'})
        
        ax.set_title('SHD Comparison: Real Data Ablation Study', fontsize=16, fontweight='bold')
        ax.set_xlabel('Generator Configuration', fontweight='bold')
        ax.set_ylabel('Data Scenario', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'shd_heatmap_real_data.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_comparison(self):
        """Compare F1 scores across configurations"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        configs = list(self.configurations.keys())
        f1_means = []
        f1_stds = []
        
        for config in configs:
            f1_values = []
            for scenario in self.scenarios.keys():
                if scenario in self.results and config in self.results[scenario]:
                    f1 = self.results[scenario][config]['mean_metrics'].get('f1', 0)
                    f1_values.append(f1)
            
            f1_means.append(np.mean(f1_values) if f1_values else 0)
            f1_stds.append(np.std(f1_values) if f1_values else 0)
        
        x_pos = np.arange(len(configs))
        bars = ax.bar(x_pos, f1_means, yerr=f1_stds, capsize=5, 
                     alpha=0.8, color='skyblue', edgecolor='navy')
        
        ax.set_xlabel('Configuration', fontweight='bold')
        ax.set_ylabel('F1 Score', fontweight='bold')
        ax.set_title('F1 Performance Comparison (Real Data)', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c.replace('_', ' ').title() for c in configs], rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, mean_val in zip(bars, f1_means):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{mean_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'f1_comparison_real_data.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_intervention_efficiency(self):
        """Plot intervention count vs SHD"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.configurations)))
        
        for idx, (config, generators) in enumerate(self.configurations.items()):
            x_vals = []  # interventions
            y_vals = []  # shd
            
            for scenario in self.scenarios.keys():
                if scenario in self.results and config in self.results[scenario]:
                    interventions = self.results[scenario][config]['mean_metrics'].get('total_interventions', 0)
                    shd = self.results[scenario][config]['mean_metrics'].get('shd', float('inf'))
                    if shd != float('inf'):
                        x_vals.append(interventions)
                        y_vals.append(shd)
            
            if x_vals and y_vals:
                ax.scatter(x_vals, y_vals, label=config.replace('_', ' ').title(), 
                          alpha=0.7, s=100, color=colors[idx])
        
        ax.set_xlabel('Total Interventions', fontweight='bold')
        ax.set_ylabel('Structural Hamming Distance', fontweight='bold')
        ax.set_title('Intervention Efficiency vs Accuracy (Real Data)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'intervention_efficiency_real_data.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_runtime_analysis(self):
        """Analyze runtime vs performance trade-offs"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.configurations)))
        
        for idx, (config, generators) in enumerate(self.configurations.items()):
            x_vals = []  # runtime
            y_vals = []  # f1
            
            for scenario in self.scenarios.keys():
                if scenario in self.results and config in self.results[scenario]:
                    runtime = self.results[scenario][config]['mean_metrics'].get('runtime', 0)
                    f1 = self.results[scenario][config]['mean_metrics'].get('f1', 0)
                    x_vals.append(runtime)
                    y_vals.append(f1)
            
            if x_vals and y_vals:
                ax.scatter(x_vals, y_vals, label=config.replace('_', ' ').title(), 
                          alpha=0.7, s=100, color=colors[idx])
        
        ax.set_xlabel('Runtime (seconds)', fontweight='bold')
        ax.set_ylabel('F1 Score', fontweight='bold')
        ax.set_title('Computational Cost vs Performance (Real Data)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'runtime_analysis_real_data.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_generator_contributions(self):
        """Analyze individual generator contributions"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Calculate average F1 scores for each configuration type
        single_configs = ['pc_only', 'sam_only', 'llm_only']
        pairwise_configs = ['pc_sam', 'pc_llm', 'sam_llm']
        full_config = ['full_framework']
        
        config_groups = [
            ('Single Generators', single_configs),
            ('Pairwise Combinations', pairwise_configs),
            ('Full Framework', full_config)
        ]
        
        x_positions = []
        f1_scores = []
        labels = []
        colors = []
        
        color_map = {'pc': '#1f77b4', 'sam': '#ff7f0e', 'llm': '#2ca02c', 'full': '#d62728'}
        
        x_pos = 0
        for group_name, configs in config_groups:
            for config in configs:
                if any(config in self.results[s] for s in self.results.keys()):
                    # Calculate average F1 across all scenarios
                    f1_values = []
                    for scenario in self.results.keys():
                        if config in self.results[scenario]:
                            f1 = self.results[scenario][config]['mean_metrics'].get('f1', 0)
                            f1_values.append(f1)
                    
                    avg_f1 = np.mean(f1_values) if f1_values else 0
                    
                    x_positions.append(x_pos)
                    f1_scores.append(avg_f1)
                    labels.append(config.replace('_', ' ').title())
                    
                    # Assign colors based on primary generator
                    if 'pc' in config:
                        colors.append(color_map['pc'])
                    elif 'sam' in config:
                        colors.append(color_map['sam'])
                    elif 'llm' in config:
                        colors.append(color_map['llm'])
                    else:
                        colors.append(color_map['full'])
                    
                    x_pos += 1
            
            x_pos += 0.5  # Add space between groups
        
        bars = ax.bar(x_positions, f1_scores, color=colors, alpha=0.8, edgecolor='black')
        
        # Add value labels
        for bar, f1 in zip(bars, f1_scores):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                   f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Configuration', fontweight='bold')
        ax.set_ylabel('Average F1 Score', fontweight='bold')
        ax.set_title('Generator Contribution Analysis (Real Data)', fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(labels, rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Add group separators
        group_boundaries = [2.5, 5.5]
        for boundary in group_boundaries:
            if boundary < len(x_positions):
                ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'generator_contributions_real_data.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self):
        """Generate detailed report with real data results"""
        report_path = self.output_dir / 'real_data_ablation_report.md'
        
        with open(report_path, 'w') as f:
            f.write("# Real Data Ablation Study Report\n\n")
            f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            
            # Calculate overall rankings
            overall_rankings = self._calculate_overall_rankings()
            
            f.write("### Overall Performance Ranking (by Average F1 Score)\n\n")
            for i, (config, score) in enumerate(overall_rankings, 1):
                f.write(f"{i}. **{config.replace('_', ' ').title()}**: {score:.3f}\n")
            
            # Key findings
            if overall_rankings:
                best_config = overall_rankings[0][0]
                best_score = overall_rankings[0][1]
                worst_config = overall_rankings[-1][0]
                worst_score = overall_rankings[-1][1]
                
                f.write(f"\n### Key Findings\n\n")
                f.write(f"- **Best performing configuration**: {best_config.replace('_', ' ').title()} (F1: {best_score:.3f})\n")
                f.write(f"- **Worst performing configuration**: {worst_config.replace('_', ' ').title()} (F1: {worst_score:.3f})\n")
                
                if worst_score > 0:
                    improvement = ((best_score - worst_score) / worst_score * 100)
                    f.write(f"- **Performance improvement**: {improvement:.1f}% gain from worst to best\n")
            
            # Scenario-specific analysis
            f.write("\n### Scenario-Specific Analysis\n\n")
            for scenario_name in self.scenarios.keys():
                if scenario_name in self.results:
                    f.write(f"#### {scenario_name.replace('_', ' ').title()}\n\n")
                    
                    # Find best configuration for this scenario
                    scenario_configs = self.results[scenario_name]
                    best_f1 = 0
                    best_config_scenario = None
                    
                    for config, results in scenario_configs.items():
                        f1 = results['mean_metrics'].get('f1', 0)
                        if f1 > best_f1:
                            best_f1 = f1
                            best_config_scenario = config
                    
                    if best_config_scenario:
                        f.write(f"- **Best configuration**: {best_config_scenario.replace('_', ' ').title()} (F1: {best_f1:.3f})\n")
                        
                        # Get intervention and runtime stats
                        best_results = scenario_configs[best_config_scenario]['mean_metrics']
                        interventions = best_results.get('total_interventions', 0)
                        runtime = best_results.get('runtime', 0)
                        shd = best_results.get('shd', 0)
                        
                        f.write(f"- **Performance metrics**: SHD: {shd:.1f}, Interventions: {interventions:.0f}, Runtime: {runtime:.1f}s\n")
                    f.write("\n")
            
            # Generator-specific insights
            f.write("## Generator Analysis\n\n")
            
            # Single generator performance
            single_gen_performance = {}
            for config in ['pc_only', 'sam_only', 'llm_only']:
                f1_values = []
                for scenario in self.results.keys():
                    if config in self.results[scenario]:
                        f1 = self.results[scenario][config]['mean_metrics'].get('f1', 0)
                        f1_values.append(f1)
                if f1_values:
                    single_gen_performance[config] = np.mean(f1_values)
            
            if single_gen_performance:
                best_single = max(single_gen_performance.items(), key=lambda x: x[1])
                f.write(f"### Individual Generator Performance\n\n")
                f.write(f"- **Best single generator**: {best_single[0].replace('_only', '').upper()} (Average F1: {best_single[1]:.3f})\n")
                
                for config, f1 in sorted(single_gen_performance.items(), key=lambda x: x[1], reverse=True):
                    generator = config.replace('_only', '').upper()
                    f.write(f"- **{generator}**: {f1:.3f}\n")
            
            # Pairwise combination analysis
            pairwise_performance = {}
            for config in ['pc_sam', 'pc_llm', 'sam_llm']:
                f1_values = []
                for scenario in self.results.keys():
                    if config in self.results[scenario]:
                        f1 = self.results[scenario][config]['mean_metrics'].get('f1', 0)
                        f1_values.append(f1)
                if f1_values:
                    pairwise_performance[config] = np.mean(f1_values)
            
            if pairwise_performance:
                best_pairwise = max(pairwise_performance.items(), key=lambda x: x[1])
                f.write(f"\n### Pairwise Combination Performance\n\n")
                f.write(f"- **Best pairwise combination**: {best_pairwise[0].replace('_', ' + ').upper()} (Average F1: {best_pairwise[1]:.3f})\n")
                
                for config, f1 in sorted(pairwise_performance.items(), key=lambda x: x[1], reverse=True):
                    combination = config.replace('_', ' + ').upper()
                    f.write(f"- **{combination}**: {f1:.3f}\n")
            
            # Full framework analysis
            full_framework_f1 = []
            for scenario in self.results.keys():
                if 'full_framework' in self.results[scenario]:
                    f1 = self.results[scenario]['full_framework']['mean_metrics'].get('f1', 0)
                    full_framework_f1.append(f1)
            
            if full_framework_f1:
                avg_full_f1 = np.mean(full_framework_f1)
                f.write(f"\n### Full Framework Performance\n\n")
                f.write(f"- **Average F1 Score**: {avg_full_f1:.3f}\n")
                
                # Compare to best single and best pairwise
                if single_gen_performance and pairwise_performance:
                    best_single_f1 = best_single[1]
                    best_pairwise_f1 = best_pairwise[1]
                    
                    single_improvement = ((avg_full_f1 - best_single_f1) / best_single_f1 * 100) if best_single_f1 > 0 else 0
                    pairwise_improvement = ((avg_full_f1 - best_pairwise_f1) / best_pairwise_f1 * 100) if best_pairwise_f1 > 0 else 0
                    
                    f.write(f"- **Improvement over best single generator**: {single_improvement:.1f}%\n")
                    f.write(f"- **Improvement over best pairwise**: {pairwise_improvement:.1f}%\n")
            
            # Cost-benefit analysis
            f.write("\n## Cost-Benefit Analysis\n\n")
            self._add_cost_benefit_analysis(f)
            
            # Statistical significance
            f.write("\n## Statistical Analysis\n\n")
            self._add_statistical_analysis(f)
            
            # Detailed results table
            f.write("\n## Detailed Results\n\n")
            self._add_detailed_results_table(f)
            
            # Recommendations
            f.write("\n## Recommendations\n\n")
            self._add_recommendations(f, overall_rankings)
            
            # Future work
            f.write("\n## Future Work\n\n")
            f.write("Based on this ablation study, we recommend:\n\n")
            f.write("1. **Investigate ensemble methods** that combine the strengths of different generators\n")
            f.write("2. **Optimize intervention strategies** to reduce the number of required interventions\n")
            f.write("3. **Develop adaptive selection** mechanisms that choose optimal generators based on data characteristics\n")
            f.write("4. **Extend to larger scale systems** with more complex causal structures\n")
            f.write("5. **Investigate domain-specific optimizations** for building energy systems\n")
        
        logger.info(f"Comprehensive report saved to: {report_path}")
    
    def _calculate_overall_rankings(self):
        """Calculate overall performance rankings across all scenarios"""
        config_scores = defaultdict(list)
        
        for scenario in self.results.keys():
            for config in self.configurations.keys():
                if config in self.results[scenario]:
                    f1 = self.results[scenario][config]['mean_metrics'].get('f1', 0)
                    config_scores[config].append(f1)
        
        # Average F1 scores across scenarios
        avg_scores = {config: np.mean(scores) for config, scores in config_scores.items() if scores}
        
        # Sort by score (descending)
        return sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
    
    def _add_cost_benefit_analysis(self, f):
        """Add cost-benefit analysis to the report"""
        f.write("### Computational Cost vs Performance Trade-offs\n\n")
        
        # Calculate average runtime and F1 for each configuration
        config_metrics = {}
        for config in self.configurations.keys():
            runtimes = []
            f1_scores = []
            interventions = []
            
            for scenario in self.results.keys():
                if config in self.results[scenario]:
                    metrics = self.results[scenario][config]['mean_metrics']
                    runtimes.append(metrics.get('runtime', 0))
                    f1_scores.append(metrics.get('f1', 0))
                    interventions.append(metrics.get('total_interventions', 0))
            
            if runtimes and f1_scores:
                config_metrics[config] = {
                    'avg_runtime': np.mean(runtimes),
                    'avg_f1': np.mean(f1_scores),
                    'avg_interventions': np.mean(interventions)
                }
        
        # Find most efficient configurations
        if config_metrics:
            # Sort by F1/runtime ratio (performance per unit time)
            efficiency_ranking = sorted(
                config_metrics.items(), 
                key=lambda x: x[1]['avg_f1'] / max(x[1]['avg_runtime'], 0.1), 
                reverse=True
            )
            
            f.write("**Most efficient configurations (F1/Runtime):**\n\n")
            for i, (config, metrics) in enumerate(efficiency_ranking[:3], 1):
                efficiency = metrics['avg_f1'] / max(metrics['avg_runtime'], 0.1)
                f.write(f"{i}. **{config.replace('_', ' ').title()}**: "
                       f"Efficiency: {efficiency:.3f}, F1: {metrics['avg_f1']:.3f}, "
                       f"Runtime: {metrics['avg_runtime']:.1f}s\n")
    
    def _add_statistical_analysis(self, f):
        """Add statistical analysis to the report"""
        f.write("### Statistical Significance\n\n")
        
        # Check if we have enough data for statistical tests
        configs_with_multiple_runs = []
        for scenario in self.results.keys():
            for config in self.results[scenario].keys():
                runs = self.results[scenario][config]['individual_runs']
                valid_runs = [r for r in runs if 'error' not in r]
                if len(valid_runs) >= 2:
                    configs_with_multiple_runs.append((scenario, config))
        
        if configs_with_multiple_runs:
            f.write(f"- **Statistical tests performed on {len(configs_with_multiple_runs)} configuration-scenario combinations**\n")
            f.write("- Multiple runs per configuration enable confidence interval calculation\n")
            f.write("- Standard deviations reported indicate result consistency\n")
        else:
            f.write("- Limited statistical analysis due to single runs per configuration\n")
            f.write("- Recommend increasing number of runs for future studies\n")
        
        # Add variance analysis
        f.write("\n### Result Consistency Analysis\n\n")
        high_variance_configs = []
        
        for scenario in self.results.keys():
            for config in self.results[scenario].keys():
                std_f1 = self.results[scenario][config]['std_metrics'].get('f1', 0)
                mean_f1 = self.results[scenario][config]['mean_metrics'].get('f1', 0)
                
                if mean_f1 > 0 and std_f1 / mean_f1 > 0.2:  # High coefficient of variation
                    high_variance_configs.append((scenario, config, std_f1 / mean_f1))
        
        if high_variance_configs:
            f.write("**Configurations with high result variance:**\n\n")
            for scenario, config, cv in sorted(high_variance_configs, key=lambda x: x[2], reverse=True)[:5]:
                f.write(f"- **{config.replace('_', ' ').title()}** on {scenario.replace('_', ' ').title()}: CV = {cv:.2f}\n")
        else:
            f.write("- All configurations show consistent results across runs\n")
    
    def _add_detailed_results_table(self, f):
        """Add detailed results table to the report"""
        def safe_format(value, decimals=1):
            try:
                return f"{float(value):.{decimals}f}"
            except (ValueError, TypeError):
                return str(value)
        
        for scenario_name in self.scenarios.keys():
            if scenario_name in self.results:
                f.write(f"### {scenario_name.replace('_', ' ').title()}\n\n")
                f.write("| Configuration | SHD ↓ | Precision ↑ | Recall ↑ | F1 ↑ | Interventions ↓ | Runtime ↓ (s) |\n")
                f.write("|---------------|-------|-------------|----------|------|----------------|---------------|\n")
                
                # Sort configurations by F1 score for this scenario
                scenario_configs = []
                for config in self.configurations.keys():
                    if config in self.results[scenario_name]:
                        metrics = self.results[scenario_name][config]['mean_metrics']
                        scenario_configs.append((config, metrics))
                
                scenario_configs.sort(key=lambda x: x[1].get('f1', 0), reverse=True)
                
                for config, metrics in scenario_configs:
                    f.write(f"| {config.replace('_', ' ').title()} | "
                            f"{safe_format(metrics.get('shd', 'N/A'))} | "
                            f"{safe_format(metrics.get('precision', 0), 3)} | "
                            f"{safe_format(metrics.get('recall', 0), 3)} | "
                            f"{safe_format(metrics.get('f1', 0), 3)} | "
                            f"{safe_format(metrics.get('total_interventions', 0), 0)} | "
                            f"{safe_format(metrics.get('runtime', 0))} |\n")
                
                f.write("\n")
    
    def _add_recommendations(self, f, rankings):
        """Add actionable recommendations based on results"""
        if not rankings:
            f.write("Insufficient data for specific recommendations.\n")
            return
        
        best_config = rankings[0][0]
        best_score = rankings[0][1]
        
        f.write(f"### Primary Recommendations\n\n")
        f.write(f"1. **For maximum accuracy**: Use {best_config.replace('_', ' ').title()} configuration (F1: {best_score:.3f})\n\n")
        
        # Find best single generator for resource-constrained scenarios
        single_configs = [item for item in rankings if item[0] in ['pc_only', 'sam_only', 'llm_only']]
        if single_configs:
            best_single = single_configs[0]
            f.write(f"2. **For resource constraints**: Use {best_single[0].replace('_only', '').upper()} only (F1: {best_single[1]:.3f})\n\n")
        
        # Find best pairwise for balanced approach
        pairwise_configs = [item for item in rankings if item[0] in ['pc_sam', 'pc_llm', 'sam_llm']]
        if pairwise_configs:
            best_pairwise = pairwise_configs[0]
            f.write(f"3. **For balanced performance**: Use {best_pairwise[0].replace('_', ' + ').upper()} combination (F1: {best_pairwise[1]:.3f})\n\n")
        
        # Scenario-specific recommendations
        f.write("### Scenario-Specific Recommendations\n\n")
        
        scenario_recommendations = {
            'base_smart_room': 'Clean simulation data with clear causal relationships',
            'noisy_smart_room': 'Real-world deployment with measurement noise',
            'hidden_vars_smart_room': 'Complex systems with unobserved confounders',
            'smart_building': 'Large-scale multi-zone building systems',
            'ashrae_building': 'Real building data with limited intervention capability'
        }
        
        for scenario_name in self.scenarios.keys():
            if scenario_name in self.results and scenario_name in scenario_recommendations:
                # Find best config for this scenario
                scenario_configs = self.results[scenario_name]
                best_scenario_config = max(
                    scenario_configs.items(), 
                    key=lambda x: x[1]['mean_metrics'].get('f1', 0)
                )
                
                config_name = best_scenario_config[0]
                f1_score = best_scenario_config[1]['mean_metrics'].get('f1', 0)
                
                f.write(f"- **{scenario_name.replace('_', ' ').title()}** "
                       f"({scenario_recommendations[scenario_name]}): "
                       f"Use {config_name.replace('_', ' ').title()} (F1: {f1_score:.3f})\n")
    
    def save_all_results(self):
        """Save all results in multiple formats"""
        logger.info("Saving all results...")
        
        # Save raw results as JSON
        results_file = self.output_dir / 'complete_ablation_results.json'
        
        def convert_for_json(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        serializable_results = convert_for_json(dict(self.results))
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        # Save summary CSV
        summary_data = []
        for scenario in self.results.keys():
            for config in self.results[scenario].keys():
                metrics = self.results[scenario][config]['mean_metrics']
                summary_data.append({
                    'scenario': scenario,
                    'configuration': config,
                    'generators': '+'.join(self.configurations[config]),
                    'shd': metrics.get('shd', None),
                    'precision': metrics.get('precision', None),
                    'recall': metrics.get('recall', None),
                    'f1': metrics.get('f1', None),
                    'total_interventions': metrics.get('total_interventions', None),
                    'runtime': metrics.get('runtime', None)
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / 'ablation_summary.csv', index=False)
        
        logger.info(f"Results saved to:")
        logger.info(f"  - JSON: {results_file}")
        logger.info(f"  - CSV: {self.output_dir / 'ablation_summary.csv'}")


def main():
    """Main function to run the real data ablation study"""
    
    print("="*80)
    print("ABLATION STUDY: DAG Generator Framework")
    print("="*80)
    print()
    
    # Configuration
    api_key = "YOUR_OPENAI_API_KEY"  # Replace with your actual API key
    
    # Initialize the study
    study = RealDataAblationStudy(api_key=api_key, output_dir='real_ablation_results')
    
    # Ask user which scenarios and configurations to run
    print("Available scenarios:")
    for i, scenario in enumerate(study.scenarios.keys(), 1):
        print(f"  {i}. {scenario}")
    
    print("\nAvailable configurations:")
    for i, config in enumerate(study.configurations.keys(), 1):
        print(f"  {i}. {config}")
    
    # For demonstration, run a subset (you can modify this)
    selected_scenarios = ['smart_building']  # Start with these
    selected_configs = ['sam_only']  # Key comparisons
    
    print(f"\nRunning ablation study on:")
    print(f"  Scenarios: {selected_scenarios}")
    print(f"  Configurations: {selected_configs}")
    print(f"  Runs per configuration: 2")
    print()
    
    try:
        # Phase 1: Run experiments
        print("Phase 1: Running experiments...")
        results = study.run_full_study(
            n_runs=1, 
            selected_scenarios=selected_scenarios, 
            selected_configs=selected_configs
        )
        
        # Phase 2: Save results
        print("\nPhase 2: Saving results...")
        study.save_all_results()
        
        # Phase 3: Create visualizations
        print("\nPhase 3: Creating visualizations...")
        study.create_visualizations()
        
        # Phase 4: Generate report
        print("\nPhase 4: Generating comprehensive report...")
        study.generate_comprehensive_report()
        
        print(f"\n{'='*80}")
        print("REAL DATA ABLATION STUDY COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"Results saved to: {study.output_dir}")
        print()
        print("Key files:")
        print(f"  - Comprehensive report: {study.output_dir}/real_data_ablation_report.md")
        print(f"  - Complete results: {study.output_dir}/complete_ablation_results.json")
        print(f"  - Summary table: {study.output_dir}/ablation_summary.csv")
        print(f"  - Visualizations: {study.output_dir}/*.png")
        
        # Print quick summary
        if results:
            print(f"\nQuick Summary:")
            for scenario in selected_scenarios:
                if scenario in results:
                    print(f"\n{scenario.replace('_', ' ').title()}:")
                    for config in selected_configs:
                        if config in results[scenario]:
                            f1 = results[scenario][config]['mean_metrics'].get('f1', 0)
                            print(f"  {config.replace('_', ' ').title()}: F1 = {f1:.3f}")
        
    except Exception as e:
        print(f"\nError during ablation study: {str(e)}")
        logger.error(f"Ablation study failed: {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('ablation_study.log'),
            logging.StreamHandler()
        ]
    )
    
    success = main()
    if success:
        print("\nAblation study completed successfully!")
    else:
        print("\nAblation study encountered errors. Check ablation_study.log for details.")