import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
import logging
from pathlib import Path
import time
import argparse

from src.pipeline import CausalPipeline
from src.metrics import MetricsCalculator
from src.ground_truth import GroundTruthDAG

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("parameter_optimization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def grid_search_parameters(
    data_path="data/simulation_log_2025-03-04T16-42-08-821Z_preprocessed.csv",
    api_key="YOUR_OPENAI_API_KEY",
    simulation_path="./smart_room.js",
    output_dir="parameter_tuning_results",
    alphas=[0.4, 0.5, 0.6],           # Reduced number of alpha values
    betas=[0.4, 0.5, 0.6],            # Reduced number of beta values
    effect_sizes=[0.05, 0.1, 0.2],     # Reduced number of effect sizes
    # alphas=[0.3, 0.4, 0.5, 0.6, 0.7],
    # betas=[0.3, 0.4, 0.5, 0.6, 0.7],
    # effect_sizes=[0.05, 0.1, 0.15, 0.2, 0.25],
    max_iterations=1,
    parallel=True                # Add parallel processing option
    ):

    """Perform grid search for optimal parameters in causal discovery pipeline"""
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset - only load a subset for faster processing
    logger.info(f"Loading dataset from {data_path}")
    data = pd.read_csv(data_path)
    
    # Optional: Use a sample of the data to speed up processing
    # If the dataset is large, use a random subset
    if len(data) > 500:
        logger.info(f"Using random subset of {min(500, len(data))} rows from original {len(data)} rows")
        data = data.sample(min(500, len(data)), random_state=42)
    
    # Storage for results
    results = []
    
    # Track progress
    total_combinations = len(alphas) * len(betas) * len(effect_sizes)
    current = 0
    
    # Use parallel processing if enabled
    if parallel and total_combinations > 1:
        try:
            from joblib import Parallel, delayed
            import multiprocessing
            
            # num_cores = max(1, multiprocessing.cpu_count() - 1)  # Use all but one core
            num_cores = min(4, max(1, multiprocessing.cpu_count() - 1))
            logger.info(f"Using parallel processing with {num_cores} cores")
            
            def process_combination(alpha, beta, effect_size):
                logger.info(f"Processing parameter combination: α={alpha}, β={beta}, effect_size={effect_size}")
                try:
                    import subprocess
                    subprocess.run(["pkill", "-f", "node"], stderr=subprocess.DEVNULL)

                    # Initialize pipeline with parameters
                    pipeline = CausalPipeline(
                        data, 
                        api_key, 
                        simulation_path,
                        max_iterations=max_iterations,
                        alpha=alpha,
                        beta=beta,
                        effect_threshold=effect_size
                    )
                    
                    # Run pipeline
                    run_start = time.time()
                    final_dag, metrics = pipeline.run()
                    run_time = time.time() - run_start
                    
                    # Clean up
                    pipeline.cleanup()
                    
                    # Return results
                    return {
                        'alpha': alpha,
                        'beta': beta,
                        'effect_size': effect_size,
                        'f1_score': metrics['final'].f1_score,
                        'shd': metrics['final'].shd,
                        'cost': metrics['final'].cost,
                        'edge_count': len(final_dag['edges']),
                        'runtime': run_time
                    }
                    
                except Exception as e:
                    logger.error(f"Error with parameters α={alpha}, β={beta}, effect_size={effect_size}: {str(e)}")
                    return {
                        'alpha': alpha, 'beta': beta, 'effect_size': effect_size,
                        'f1_score': 0, 'shd': float('inf'), 'cost': 1.0,
                        'edge_count': 0, 'runtime': 0, 'error': str(e)
                    }
                finally:
                    # Ensure cleanup happens
                    if 'pipeline' in locals() and pipeline:
                        pipeline.cleanup()
            
            # Run parameter combinations in parallel
            all_combinations = list(product(alphas, betas, effect_sizes))
            parallel_results = Parallel(n_jobs=num_cores)(
                delayed(process_combination)(alpha, beta, effect_size) 
                for alpha, beta, effect_size in all_combinations
            )
            
            # Filter out None results (errors)
            results = [r for r in parallel_results if r is not None]
            
        except ImportError:
            logger.warning("Parallel processing requested but joblib not available. Falling back to sequential processing.")
            parallel = False
    
    # Sequential processing if parallel is disabled or failed
    if not parallel or not results:
        for alpha, beta, effect_size in product(alphas, betas, effect_sizes):
            current += 1
            logger.info(f"Testing combination {current}/{total_combinations}: α={alpha}, β={beta}, effect_size={effect_size}")
            logger.info(f"Parameters: α={alpha}, β={beta}, effect_size={effect_size}")

            # Initialize pipeline with parameters
            pipeline = CausalPipeline(
                data, 
                api_key, 
                simulation_path,
                max_iterations=max_iterations,
                alpha=alpha,
                beta=beta,
                effect_threshold=effect_size
            )
            
            # Run pipeline
            try:
                run_start = time.time()
                final_dag, metrics = pipeline.run()
                run_time = time.time() - run_start
                
                # Store results
                results.append({
                        'alpha': alpha,
                        'beta': beta,
                        'effect_size': effect_size,
                        'f1_score': metrics['final'].f1_score,
                        'shd': metrics['final'].shd,
                        'cost': metrics['final'].cost,
                        'edge_count': len(final_dag['edges']),
                        'runtime': run_time
                    })
                
                # Save intermediate results after each run
                pd.DataFrame(results).to_csv(os.path.join(output_dir, "intermediate_results.csv"), index=False)
                
                # Clean up after each run
                pipeline.cleanup()
                
            except Exception as e:
                logger.error(f"Error with parameters α={alpha}, β={beta}, effect_size={effect_size}: {str(e)}")
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save all results to CSV
    results_file_path = os.path.join(output_dir, "parameter_grid_search_results.csv")
    results_df.to_csv(results_file_path, index=False)
    logger.info(f"All results saved to {results_file_path}")
    
    # Save final results
    if not results_df.empty:
        # Normalize metrics to 0-1 scale for balanced comparison
        results_df['norm_f1'] = results_df['f1_score']
        results_df['norm_shd'] = 1 - (results_df['shd'] / results_df['shd'].max())
        results_df['norm_risk'] = 1 - results_df['risk']
        results_df['norm_cost'] = 1 - results_df['cost']
        
        # Combined metric with weighted importance
        results_df['combined_score'] = (
            0.33 * results_df['norm_f1'] +
            0.33 * results_df['norm_shd'] +
            0.33 * results_df['norm_cost']
        )
                
        # Select parameters based on combined score
        best_params = results_df.loc[results_df['combined_score'].idxmax()]
        return best_params
    else:
        logger.warning("No valid results found in grid search")
        return None

def create_sensitivity_plots(results_df, output_dir):
    """Create sensitivity analysis plots focused on F1 score, SHD, and cost"""
    if results_df.empty:
        logger.warning("No results to plot")
        return

    logger.info(f"Creating visualization plots in {output_dir}")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Fix data types if needed
    for col in ['alpha', 'beta', 'effect_size', 'f1_score', 'shd', 'cost']:
        if col in results_df.columns:
            results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
    
    # Create parameter combination labels
    results_df['parameter_combo'] = results_df.apply(
        lambda x: f"α={x['alpha']}, β={x['beta']}, e={x['effect_size']}", 
        axis=1
    )
    
    # 1. F1 Score visualization
    plt.figure(figsize=(12, 6))
    sorted_df = results_df.sort_values('f1_score', ascending=False)
    ax = sns.barplot(x='parameter_combo', y='f1_score', data=sorted_df)
    plt.xticks(rotation=90)
    plt.title('F1 Score by Parameter Combination')
    
    # Add value labels
    for i, v in enumerate(sorted_df['f1_score']):
        ax.text(i, v + 0.005, f"{v:.3f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_score_comparison.png"), dpi=200)
    plt.close()
    
    # 2. SHD visualization
    plt.figure(figsize=(12, 6))
    sorted_df = results_df.sort_values('shd')  # Lower SHD is better
    ax = sns.barplot(x='parameter_combo', y='shd', data=sorted_df)
    plt.xticks(rotation=90)
    plt.title('Structural Hamming Distance (SHD) by Parameter Combination')
    
    # Add value labels
    for i, v in enumerate(sorted_df['shd']):
        ax.text(i, v + 0.1, f"{v:.1f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shd_comparison.png"), dpi=200)
    plt.close()
    
    # 3. Cost visualization
    plt.figure(figsize=(12, 6))
    sorted_df = results_df.sort_values('cost')  # Lower cost is better
    ax = sns.barplot(x='parameter_combo', y='cost', data=sorted_df)
    plt.xticks(rotation=90)
    plt.title('Intervention Cost by Parameter Combination')
    
    # Add value labels
    for i, v in enumerate(sorted_df['cost']):
        ax.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cost_comparison.png"), dpi=200)
    plt.close()
    
    # 4. Parameter effect plots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # F1 Score by parameters
    sns.boxplot(x='alpha', y='f1_score', data=results_df, ax=ax1)
    ax1.set_title('F1 Score by Alpha (α)')
    
    sns.boxplot(x='beta', y='f1_score', data=results_df, ax=ax2)
    ax2.set_title('F1 Score by Beta (β)')
    
    sns.boxplot(x='effect_size', y='f1_score', data=results_df, ax=ax3)
    ax3.set_title('F1 Score by Effect Size')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_score_parameters.png"), dpi=200)
    plt.close()
    
    # 5. SHD by parameters
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.boxplot(x='alpha', y='shd', data=results_df, ax=ax1)
    ax1.set_title('SHD by Alpha (α)')
    
    sns.boxplot(x='beta', y='shd', data=results_df, ax=ax2)
    ax2.set_title('SHD by Beta (β)')
    
    sns.boxplot(x='effect_size', y='shd', data=results_df, ax=ax3)
    ax3.set_title('SHD by Effect Size')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shd_parameters.png"), dpi=200)
    plt.close()
    
    # 6. Cost by parameters
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    sns.boxplot(x='alpha', y='cost', data=results_df, ax=ax1)
    ax1.set_title('Cost by Alpha (α)')
    
    sns.boxplot(x='beta', y='cost', data=results_df, ax=ax2)
    ax2.set_title('Cost by Beta (β)')
    
    sns.boxplot(x='effect_size', y='cost', data=results_df, ax=ax3)
    ax3.set_title('Cost by Effect Size')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cost_parameters.png"), dpi=200)
    plt.close()
    
    # 7. Combined score visualization
    if 'combined_score' in results_df.columns:
        plt.figure(figsize=(12, 6))
        sorted_df = results_df.sort_values('combined_score', ascending=False)
        ax = sns.barplot(x='parameter_combo', y='combined_score', data=sorted_df)
        plt.xticks(rotation=90)
        plt.title('Combined Score (F1, SHD, Cost)')
        
        # Add value labels
        for i, v in enumerate(sorted_df['combined_score']):
            ax.text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "combined_score.png"), dpi=200)
        plt.close()
    
    logger.info("Visualization plots created successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parameter optimization for causal discovery pipeline')
    parser.add_argument('--data', type=str, default="data/simulation_log_2025-03-04T16-42-08-821Z_preprocessed.csv", 
                        help='Path to preprocessed data file')
    parser.add_argument('--api_key', type=str, 
                        default="YOUR_OPENAI_API_KEY", 
                        help='API key for LLM')
    parser.add_argument('--simulation', type=str, default="./smart_room.js", 
                        help='Path to simulation script')
    parser.add_argument('--output', type=str, default="./parameter_tuning_results", 
                        help='Output directory for results')
    parser.add_argument('--iterations', type=int, default=1, 
                        help='Maximum iterations for each pipeline run')
    parser.add_argument('--sequential', action='store_true',
                        help='Disable parallel processing')
    parser.add_argument('--visualize_only', action='store_true',
                        help='Only run visualization on existing CSV data')
    parser.add_argument('--results_file', type=str, default="./parameter_tuning_results/parameter_grid_search_results.csv",
                        help='CSV file with results for visualization (used with --visualize_only)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    if args.visualize_only:
        csv_path = os.path.join(args.output, args.results_file)
        if not os.path.exists(csv_path):
            logger.error(f"Results file not found: {csv_path}")
            print(f"Error: Results file not found: {csv_path}")
            exit(1)
            
        print(f"Loading results from {csv_path}")
        results_df = pd.read_csv(csv_path)
        
        # Print basic statistics
        print(f"Loaded {len(results_df)} parameter combinations")
        if not results_df.empty:
            best_params = results_df.loc[results_df['f1_score'].idxmax()]
            print(f"\nBest parameters found:")
            print(f"  α = {best_params['alpha']}")
            print(f"  β = {best_params['beta']}")
            print(f"  effect_size = {best_params['effect_size']}")
            print(f"  F1 Score = {best_params['f1_score']:.3f}")
            print(f"  SHD = {best_params['shd']}")
            print(f"  Cost = {best_params['cost']:.3f}")
        
        # Create visualization
        create_sensitivity_plots(results_df, args.output)
        print(f"Visualization completed. Results saved to {args.output}")
    else:
        # Run grid search
        best_params = grid_search_parameters(
            data_path=args.data,
            api_key=args.api_key,
            simulation_path=args.simulation,
            output_dir=args.output,
            max_iterations=args.iterations,
            parallel=not args.sequential
        )
        
        print("\nGrid search complete!")
        if best_params is not None:
            print(f"Best parameters found:")
            print(f"  α = {best_params['alpha']}")
            print(f"  β = {best_params['beta']}")
            print(f"  effect_size = {best_params['effect_size']}")
            print(f"  F1 Score = {best_params['f1_score']:.3f}")
            print(f"  SHD = {best_params['shd']}")
            print(f"  Cost = {best_params['cost']:.3f}")