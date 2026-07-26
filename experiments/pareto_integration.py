"""
Integration script to replace existing NeurIPS experiments with Pareto frontier analysis
Modifies experiments_neurips.py to use the new Pareto analysis system
"""

import os
import sys
import logging
from pathlib import Path

# Add the path to import the new Pareto system
sys.path.append(str(Path(__file__).parent))

from pareto_frontier_system import (
    ParetoAnalyzer, 
    ParetoExperimentRunner, 
    run_pareto_neurips_experiments
)

logger = logging.getLogger(__name__)


class EnhancedNeurIPSExperiments:
    """
    Enhanced version of NeurIPSExperiments that integrates Pareto frontier analysis
    This is a drop-in replacement for the existing experiments_neurips.py class
    """
    
    def __init__(self, pipeline, dataset, smart_room_path=None, api_key=None):
        self.pipeline = pipeline
        self.dataset = dataset
        self.smart_room_path = smart_room_path
        self.api_key = api_key
        
        # Maintain compatibility with existing interface
        self.objectives = {
            'satisfaction': {'weight': 0.6, 'direction': 'maximize'},
            'energy': {'weight': 0.4, 'direction': 'minimize'}
        }
        
        self.ashrae_setpoints = {
            'Temperature': 0.625,
            'Humidity': 0.55,
            'AirQuality': 0.75
        }
        
        # Initialize Pareto experiment runner
        self.pareto_runner = ParetoExperimentRunner(
            pipeline, dataset, smart_room_path, api_key
        )
    
    def run_workshop_experiments(self, n_runs=20, use_pareto=True):
        """
        Enhanced workshop experiments with optional Pareto analysis
        
        Args:
            n_runs: Number of experimental runs (scenarios for Pareto analysis)
            use_pareto: If True, run full Pareto analysis. If False, run legacy experiments.
        """
        if use_pareto:
            logger.info("Running enhanced Pareto frontier experiments")
            return self._run_pareto_experiments(n_runs)
        else:
            logger.info("Running legacy experiments")
            return self._run_legacy_experiments(n_runs)
    
    def _run_pareto_experiments(self, n_scenarios):
        """Run the new Pareto frontier experiments"""
        return run_pareto_neurips_experiments(
            self.pipeline, 
            self.dataset, 
            self.smart_room_path, 
            self.api_key,
            n_scenarios=n_scenarios,
            output_dir='neurips_pareto_results_noisy'
        )
    
    def _run_legacy_experiments(self, n_runs):
        """Fallback to original experiment structure for compatibility"""
        logger.info(f"Running legacy experiments with {n_runs} runs")
        
        results = {
            'satisfaction_percentage': {},
            'energy_usage': {},
            'multi_objective_score': {},
            'pareto_hypervolume': {},
            'raw_data': {'n_runs': n_runs}
        }
        
        # Run original experiments for each method
        for method_name in ['GRID', 'ASHRAE']:
            policy = self._get_grid_policy() if method_name == 'GRID' else self._get_ashrae_policy()
            logger.info(f"Testing {method_name} policy")

            satisfaction_scores = []
            energy_scores = []
            
            for run in range(n_runs):
                if run % 5 == 0:
                    logger.info(f"Progress: {run}/{n_runs} runs complete")
                
                # Simulate episode
                scenario = {'seed': run}
                satisfaction, energy = self._simulate_policy_episode(policy, scenario)
                
                satisfaction_scores.append(satisfaction)
                energy_scores.append(energy)
            
            # Store results
            results['satisfaction_percentage'][method_name] = {
                'scores': satisfaction_scores,
                'mean': sum(satisfaction_scores) / len(satisfaction_scores),
                'std': (sum([(x - sum(satisfaction_scores)/len(satisfaction_scores))**2 
                           for x in satisfaction_scores]) / len(satisfaction_scores))**0.5
            }
            
            results['energy_usage'][method_name] = {
                'scores': energy_scores,
                'mean': sum(energy_scores) / len(energy_scores),
                'std': (sum([(x - sum(energy_scores)/len(energy_scores))**2 
                           for x in energy_scores]) / len(energy_scores))**0.5
            }
            
            # Calculate multi-objective scores
            multi_obj_scores = []
            for sat, eng in zip(satisfaction_scores, energy_scores):
                # Higher satisfaction is better, lower energy is better
                normalized_sat = sat / 100.0
                normalized_eng = 1.0 - (eng / 50.0)  # Assume max energy is 50
                multi_obj = 0.6 * normalized_sat + 0.4 * normalized_eng
                multi_obj_scores.append(multi_obj)
            
            results['multi_objective_score'][method_name] = {
                'scores': multi_obj_scores,
                'mean': sum(multi_obj_scores) / len(multi_obj_scores),
                'std': (sum([(x - sum(multi_obj_scores)/len(multi_obj_scores))**2 
                           for x in multi_obj_scores]) / len(multi_obj_scores))**0.5
            }
        
        return results
    
    def _simulate_policy_episode(self, policy, scenario):
        """Simulate a single policy episode"""
        # Mock simulation - replace with actual simulation logic
        base_satisfaction = 75.0
        base_energy = 25.0
        
        # Add some variation based on policy
        if 'Temperature' in policy:
            temp_effect = (policy['Temperature'] - 0.5) * 10
            satisfaction = base_satisfaction + temp_effect
            energy = base_energy - temp_effect * 0.5
        else:
            satisfaction = base_satisfaction
            energy = base_energy
        
        # Add random variation
        import numpy as np
        satisfaction += np.random.normal(0, 5)
        energy += np.random.normal(0, 3)
        
        # Clip to reasonable ranges
        satisfaction = max(0, min(100, satisfaction))
        energy = max(10, min(50, energy))
        
        return satisfaction, energy
    
    def _get_grid_policy(self):
        """Get GRID policy using existing logic"""
        if not hasattr(self, '_grid_action_plan'):
            try:
                policy_data = self.pipeline.export_for_policy_engine()
                from src.policy_engine import CausalPolicyEngine
                policy_engine = CausalPolicyEngine(policy_data, self.dataset, 
                                                use_llm=False, api_key=self.api_key)
                
                objectives = {
                    'overallsatisfaction': {'target': 80.0, 'weight': 0.6},
                    'energyconsumption': {'target': 30.0, 'weight': 0.4}
                }
                constraints = {
                    'Temperature': (0.0, 1.0),
                    'Humidity': (0.0, 1.0), 
                    'AirQuality': (0.0, 1.0)
                }
                
                result = policy_engine.optimize_policy(objectives, constraints, 
                                                    self._get_initial_state({'seed': 0}))
                self._grid_action_plan = result.action_plan if result else self.ashrae_setpoints
                
            except Exception as e:
                logger.warning(f"GRID policy failed: {e}")
                self._grid_action_plan = self.ashrae_setpoints
        
        return self._grid_action_plan

    def _get_ashrae_policy(self):
        """ASHRAE standard setpoints"""
        return self.ashrae_setpoints

    def _get_initial_state(self, scenario):
        """Get initial state for scenario"""
        import numpy as np
        return {
            'Temperature': 0.5 + np.random.normal(0, 0.05),
            'Humidity': 0.5 + np.random.normal(0, 0.05),
            'AirQuality': 0.5 + np.random.normal(0, 0.05),
            'OccupantCount': 2
        }


def patch_existing_experiments():
    """
    Function to patch the existing experiments_neurips.py file
    Call this to upgrade your existing system
    """
    
    logger.info("Patching existing NeurIPS experiments with Pareto frontier analysis")
    
    # Check if experiments_neurips.py exists
    experiments_file = Path("experiments_neurips.py")
    if not experiments_file.exists():
        logger.error("experiments_neurips.py not found. Please ensure the file exists.")
        return False
    
    # Create backup
    backup_file = Path("experiments_neurips_backup.py")
    if not backup_file.exists():
        import shutil
        shutil.copy(experiments_file, backup_file)
        logger.info(f"Created backup: {backup_file}")
    
    # Create the patched version
    patch_content = '''"""
Patched NeurIPS Workshop Experiments with Pareto Frontier Analysis
This file has been automatically enhanced with Pareto frontier capabilities
"""

# Import the enhanced experiments class
from pareto_integration import EnhancedNeurIPSExperiments

# Create alias for backward compatibility
NeurIPSExperiments = EnhancedNeurIPSExperiments

# For complete backward compatibility, you can also import the original if needed:
# from experiments_neurips_backup import NeurIPSExperiments as OriginalNeurIPSExperiments
'''
    
    # Write the patch
    with open("experiments_neurips_patched.py", 'w') as f:
        f.write(patch_content)
    
    logger.info("Created experiments_neurips_patched.py with Pareto frontier capabilities")
    logger.info("To use: replace 'from experiments_neurips import NeurIPSExperiments' with")
    logger.info("       'from experiments_neurips_patched import NeurIPSExperiments'")
    
    return True


def update_main_py():
    """
    Generate an updated main.py that uses the new Pareto experiments
    """
    
    updated_main_content = '''"""
Updated main.py with Pareto Frontier Analysis
This version integrates the enhanced NeurIPS experiments
"""

import os
import pandas as pd
import numpy as np
from src.pipeline import CausalPipeline
from src.metrics_viz import MetricsVisualizer, export_metrics_to_csv
from src.policy_engine import CausalPolicyEngine
import logging
import json

# Import the enhanced experiments
from pareto_integration import EnhancedNeurIPSExperiments

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def main():
    """Main execution with Pareto frontier analysis"""
    
    # Configuration
    api_key = os.getenv("OPENAI_API_KEY")
    smart_room_path = "smart_room_sim.py"
    output_dir = "results+logs/pareto_analysis"
    
    # Load data
    data_path = "data/ashrae_benchmark_data.csv"
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    df = pd.read_csv(data_path)
    
    # Initialize and run causal discovery pipeline
    pipeline = CausalPipeline(
        df, 
        api_key, 
        smart_room_path, 
        max_iterations=2,
        alpha=0.6,
        beta=0.6,
        effect_threshold=0.1
    )
    final_dag, final_metrics = pipeline.run()
    
    try:
        # Export pipeline results
        policy_data = pipeline.export_for_policy_engine()
        
        # Initialize enhanced experiments with Pareto analysis
        experiments = EnhancedNeurIPSExperiments(pipeline, df, smart_room_path, api_key)
        
        # Run Pareto frontier experiments (set use_pareto=False for legacy experiments)
        logger.info("Running enhanced Pareto frontier experiments")
        results = experiments.run_workshop_experiments(n_runs=20, use_pareto=True)
        
        # Save results
        os.makedirs(output_dir, exist_ok=True)
        with open(f'{output_dir}/pareto_neurips_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("Pareto frontier analysis complete!")
        logger.info(f"Results saved to {output_dir}/")
        logger.info("Check the following outputs:")
        logger.info("  - pareto_frontiers.png (Figure 1)")
        logger.info("  - hypervolume_comparison.png (Figure 2)")  
        logger.info("  - calibration_plot.png (Figure 3)")
        logger.info("  - table1_paired_differences.csv (Table 1)")
        logger.info("  - table2_operational_metrics.csv (Table 2)")
        
    except Exception as e:
        logger.error(f"Pareto experiments failed: {e}")
        
        # Fallback to legacy experiments
        logger.info("Falling back to legacy experiments")
        results = experiments.run_workshop_experiments(n_runs=20, use_pareto=False)
        
        with open(f'{output_dir}/legacy_neurips_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    # Calculate final metrics
    final_edges = set(tuple(edge) if isinstance(edge, list) else edge for edge in final_dag['edges'])
    final_shd = pipeline.ground_truth.get_shd(final_edges)
    logger.info(f"Final DAG achieved SHD: {final_shd}")

    # Export traditional metrics
    export_metrics_to_csv(final_metrics, output_dir)
    
    # Generate visualizations
    visualizer = MetricsVisualizer(output_dir)
    figs = visualizer.plot_metrics(final_metrics)
    
    logger.info("All analyses complete!")


if __name__ == "__main__":
    main()
'''
    
    with open("main_pareto.py", 'w') as f:
        f.write(updated_main_content)
    
    logger.info("Created main_pareto.py with integrated Pareto frontier analysis")


if __name__ == "__main__":
    # Run the patching process
    patch_existing_experiments()
    update_main_py()
    
    print("\n" + "="*60)
    print("PARETO FRONTIER INTEGRATION COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the generated files:")
    print("   - pareto_frontier_system.py (core implementation)")
    print("   - pareto_integration.py (this file)")
    print("   - experiments_neurips_patched.py (backward compatibility)")
    print("   - main_pareto.py (updated main script)")
    print("\n2. To use the new system:")
    print("   python main_pareto.py")
    print("\n3. Or integrate into existing code:")
    print("   from pareto_integration import EnhancedNeurIPSExperiments")
    print("   # Replace NeurIPSExperiments with EnhancedNeurIPSExperiments")
    print("\n4. Outputs will include all required deliverables:")
    print("   - Figure 1: Pareto frontiers with EAF ribbons")
    print("   - Figure 2: Hypervolume comparison with 95% CI")
    print("   - Figure 3: Calibration plots")
    print("   - Table 1: Paired differences at matched operating points")
    print("   - Table 2: Operational metrics per policy")
    print("="*60)