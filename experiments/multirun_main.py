"""
Multi-Run Experiment Orchestrator with CWM Integration
========================================================

Runs the full PolicyGRID pipeline (discovery → validation → monitoring →
policy evaluation) across multiple seeds and aggregates results with
confidence intervals.

Workflow per seed:
  1. Load & preprocess simulator data (or ASHRAE dataset)
  2. Run CWMIntegratedPipeline: PC + LLM + VARLiNGAM discovery, then
     intervention-based validation, then CWM confidence tracking
  3. Train H_0/H_1 predictive models and regime monitor
  4. Evaluate all 8 policy arms via NeurIPSExperiments (B1)
  5. Log per-episode metrics (Satisfaction%, Energy%, MO, kWh, DH)

Aggregation:
  - Mean ± std across seeds for each arm × simulator × comfort_target
  - Exports raw CSV (per-episode) and summary CSV (per-arm means)

DOMAIN-AGNOSTIC KNOBS:
  - ``N_DISCOVERY_ITERATIONS``: How many discover-validate cycles (default 25).
  - ``N_DATA_ROWS``:            Number of rows to collect from the simulator.
  - ``COMFORT_TARGETS``:        List of ε values for Pareto frontier sweep.
  - ``SIMULATORS``:             Dict mapping simulator names → JS file paths.
  - ``BASE_SEED``:              Starting seed for reproducibility.
  - ``N_SEEDS``:                Number of independent seeds to run.

Entry point: ``python multirun_main.py``
"""

import os
import sys
import time
import resource
import platform
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import random
import logging
import json
import atexit
from datetime import datetime
from typing import Dict, List
from collections import defaultdict

from src.pipeline_cwm import create_cwm_pipeline
from src.metrics_viz import MetricsVisualizer, export_metrics_to_csv
from src.policy_engine import CausalPolicyEngine
from experiments_neurips import NeurIPSExperiments
from energyplus_interface import EnergyPlusInterface

from calibration_analysis import (
    compute_calibration_metrics,
    compute_suite_metrics,
    plot_single_run_calibration,
    create_aggregate_table
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def cleanup_processes():
    """Cleanup any running processes"""
    logger.info("Cleaning up processes...")


class ReproducibleExperiment:
    """Manage reproducible experiments with seed tracking"""
    
    def __init__(self, base_seed: int = 42):
        self.base_seed = base_seed
        self.run_seeds = []
        
    def get_seed(self, run_id: int) -> int:
        seed = self.base_seed + run_id
        self.run_seeds.append(seed)
        return seed
    
    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


def run_single_execution(run_id: int, df: pd.DataFrame, config: dict, seed: int, output_dir: str) -> Dict:
    """Run single framework execution with CWM integration"""
    
    logger.info(f"\nRun {run_id + 1}/3 (seed={seed})")
    
    # Set seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Create run-specific directory
    run_dir = os.path.join(output_dir, f'run_{run_id:03d}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Initialize pipeline with CWM
    pipeline = create_cwm_pipeline(
        csv_data=df, 
        api_key=config['api_key'],
        smart_room_path=config.get('smart_room_path'),
        dataset_type=config['dataset_type'],
        max_iterations=5,
        alpha=config['alpha'],
        beta=config['beta'],
        effect_threshold=config['effect_threshold']
    )
    
    # Run with CWM (discovery + reconstruction only)
    _wall_start = time.time()
    _cpu_start  = time.process_time()
    final_dag, final_metrics = pipeline.run_with_cwm()

    # After discovery phase
    logger.info("Discovery phase over...")
    discovery_df = pd.DataFrame(pipeline.probability_history)
    discovery_df.to_csv(os.path.join(run_dir, 'discovery_probabilities.csv'), index=False)

    from src.pipeline_cwm import plot_calibration_analysis
    plot_calibration_analysis(discovery_df, os.path.join(run_dir, 'discovery_calibration.png'))

    # Print monitoring setup
    logger.info("\n=== Monitoring Configuration ===")
    logger.info(f"H0 model edges: {list(pipeline.H0_graph.edges())}")
    logger.info(f"H1 model edges: {list(pipeline.H1_graph.edges())}")
    logger.info(f"Prior probabilities: P(H0)={pipeline.P_H0:.3f}, P(H1)={pipeline.P_H1:.3f}")

    # Monitoring phase
    logger.info("Starting monitoring phase...")
    monitoring_df = pipeline.monitor_regime_changes(duration=300, sample_interval=5)
    monitoring_df.to_csv(os.path.join(run_dir, 'monitoring_probabilities.csv'), index=False)
    plot_calibration_analysis(monitoring_df, os.path.join(run_dir, 'monitoring_calibration.png'))

    # ── Compute resource tracking ─────────────────────────────────────────────
    _wall_end    = time.time()
    _cpu_end     = time.process_time()
    _rss_raw     = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports ru_maxrss in bytes; Linux in kilobytes
    _peak_mem_mb = (_rss_raw / (1024 * 1024)
                    if platform.system() == 'Darwin'
                    else _rss_raw / 1024)

    # ── LLM call counting ────────────────────────────────────────────────────
    # Sum call_count from all GPTClient instances in the pipeline
    _bp_early    = pipeline.pipeline
    _n_llm_calls = 0
    _llm_gen     = getattr(_bp_early, 'generators', {}).get('llm', None)
    if _llm_gen is not None:
        _n_llm_calls += getattr(getattr(_llm_gen, 'gpt_client', None), 'call_count', 0)
    _tester_gc   = getattr(getattr(_bp_early, 'tester', None), 'gpt_client', None)
    if _tester_gc is not None:
        _n_llm_calls += getattr(_tester_gc, 'call_count', 0)

    # ── H0/H1 learned SEM parameters (what each model captured) ──────────────
    for model_name, model in [('h0', getattr(pipeline, 'H0_model', None)),
                               ('h1', getattr(pipeline, 'H1_model', None))]:
        if model is not None and hasattr(model, 'parameters') and model.parameters:
            rows = []
            for node, params in model.parameters.items():
                row = {'variable': node,
                       'intercept': round(float(params.get('intercept', float('nan'))), 6)}
                for parent, weight in params.get('weights', {}).items():
                    row[f'w_{parent}'] = round(float(weight), 6)
                rows.append(row)
            pd.DataFrame(rows).to_csv(
                os.path.join(run_dir, f'{model_name}_parameters.csv'), index=False)

    # ── Per-run model metadata (likelihood σ, priors) ─────────────────────────
    pd.DataFrame([{
        'run_id':            run_id,
        'seed':              seed,
        'likelihood_sigma':  getattr(pipeline.cwm, 'likelihood_sigma', float('nan'))
                             if pipeline.cwm else float('nan'),
        'p_h0_prior':        pipeline.P_H0,
        'p_h1_prior':        pipeline.P_H1,
        'h0_edges':          len(list(pipeline.H0_graph.edges()))
                             if hasattr(pipeline, 'H0_graph') else -1,
        'h1_edges':          len(list(pipeline.H1_graph.edges()))
                             if hasattr(pipeline, 'H1_graph') else -1,
    }]).to_csv(os.path.join(run_dir, 'model_metadata.csv'), index=False)

    # Access base pipeline for metrics
    base_pipeline = pipeline.pipeline
    
    # Calculate final SHD
    final_edges = set(tuple(edge) if isinstance(edge, list) else edge for edge in final_dag['edges'])
    final_shd = base_pipeline.ground_truth.get_shd(final_edges)
    
    # === CWM-SPECIFIC LOGGING ===
    cwm_metrics = {}
    if pipeline.cwm:
        cwm_summary = pipeline.cwm.get_state_summary()
        cwm_metrics = {
            'final_uncertainty': cwm_summary.get('uncertainty', 0),
            'mean_confidence': cwm_summary.get('mean_confidence', 0),
            'num_hypotheses': cwm_summary.get('num_hypotheses', 0),
            'data_buffer_size': cwm_summary.get('data_buffer_size', 0)
        }
        
        # Save CWM confidence matrix
        conf_matrix = pipeline.cwm.get_confidence_matrix()
        conf_matrix.to_csv(os.path.join(run_dir, 'cwm_confidence_matrix.csv'))
        
        # Save low-confidence edges
        low_conf_edges = pipeline.cwm.get_low_confidence_edges(threshold=0.6)
        if low_conf_edges:
            low_conf_df = pd.DataFrame(low_conf_edges, columns=['source', 'target', 'confidence'])
            low_conf_df.to_csv(os.path.join(run_dir, 'cwm_low_confidence_edges.csv'), index=False)
    
    # === DETAILED LOGGING FOR CUSTOM VISUALIZATIONS ===
    
    # 1. Iteration-by-iteration metrics
    iteration_data = []
    for iter_num in range(base_pipeline.current_iteration):
        iter_metrics = {
            'run_id': run_id,
            'seed': seed,
            'iteration': iter_num + 1,
            'pc_shd': base_pipeline.shd_history.get('pc', [])[iter_num] if iter_num < len(base_pipeline.shd_history.get('pc', [])) else None,
            'sam_shd': base_pipeline.shd_history.get('sam', [])[iter_num] if iter_num < len(base_pipeline.shd_history.get('sam', [])) else None,
            'llm_shd': base_pipeline.shd_history.get('llm', [])[iter_num] if iter_num < len(base_pipeline.shd_history.get('llm', [])) else None,
            'varlingam_shd': base_pipeline.shd_history.get('varlingam', [])[iter_num] if iter_num < len(base_pipeline.shd_history.get('varlingam', [])) else None,
        }
        iteration_data.append(iter_metrics)
    
    iteration_df = pd.DataFrame(iteration_data)
    iteration_df.to_csv(os.path.join(run_dir, 'iteration_metrics.csv'), index=False)
    
    # 2. Edge-level details with CWM confidence
    edge_data = []
    for edge in final_dag['edges']:
        edge_info = {
            'run_id': run_id,
            'seed': seed,
            'source': edge[0],
            'target': edge[1],
            'confidence': final_dag['confidence_scores'].get(f"{edge[0]}->{edge[1]}", 0),
            'num_interventions': len(base_pipeline.tester.intervention_results.get(edge, [])),
            'in_ground_truth': edge in base_pipeline.ground_truth.edges
        }
        
        # Add CWM confidence if available
        if pipeline.cwm and edge in pipeline.cwm.edge_confidences:
            edge_info['cwm_confidence'] = pipeline.cwm.edge_confidences[edge].confidence
        
        # Add intervention outcomes if available
        if edge in base_pipeline.tester.intervention_results:
            outcomes = base_pipeline.tester.intervention_results[edge]
            edge_info['avg_effect_size'] = np.mean([abs(r.get('effect_size', 0)) for r in outcomes]) if outcomes else 0
        
        edge_data.append(edge_info)
    
    edge_df = pd.DataFrame(edge_data)
    edge_df.to_csv(os.path.join(run_dir, 'edge_details.csv'), index=False)
    
    # 3. Method DAG comparison (now includes VARLiNGAM)
    method_comparison = []
    for method in ['pc', 'sam', 'llm', 'varlingam', 'final']:
        if method == 'final':
            edges = set(tuple(e) for e in final_dag['edges'])
            shd = final_shd
        else:
            if method in base_pipeline._results:
                edges = set(tuple(e) if not isinstance(e, dict) else e.get('edge', e) 
                           for e in base_pipeline._extract_edges(base_pipeline._results[method]))
                shd = base_pipeline.ground_truth.get_shd(edges)
            else:
                continue
        
        method_comparison.append({
            'run_id': run_id,
            'seed': seed,
            'method': method,
            'num_edges': len(edges),
            'shd': shd,
            'precision': final_metrics[method].precision if method in final_metrics else None,
            'recall': final_metrics[method].recall if method in final_metrics else None,
            'f1': final_metrics[method].f1_score if method in final_metrics else None
        })
    
    method_df = pd.DataFrame(method_comparison)
    method_df.to_csv(os.path.join(run_dir, 'method_comparison.csv'), index=False)
    
    # 4. Full intervention log
    intervention_log = []
    for edge, results in base_pipeline.tester.intervention_results.items():
        for idx, result in enumerate(results):
            intervention_log.append({
                'run_id': run_id,
                'seed': seed,
                'source': edge[0],
                'target': edge[1],
                'intervention_num': idx + 1,
                'intervention_value': result.get('intervention_value'),
                'baseline_mean': result.get('baseline_mean'),
                'intervention_mean': result.get('intervention_mean'),
                'effect_size': result.get('effect_size'),
                'p_value': result.get('p_value')
            })
    
    if intervention_log:
        intervention_df = pd.DataFrame(intervention_log)
        intervention_df.to_csv(os.path.join(run_dir, 'interventions.csv'), index=False)
    
    # 5. Save final DAG structure
    dag_structure = {
        'run_id': run_id,
        'seed': seed,
        'edges': [{'source': e[0], 'target': e[1], 
                   'confidence': final_dag['confidence_scores'].get(f"{e[0]}->{e[1]}", 0)} 
                  for e in final_dag['edges']],
        'validated_edges': [{'source': e[0], 'target': e[1]} for e in base_pipeline.validated_edges],
        'cwm_metrics': cwm_metrics
    }
    
    with open(os.path.join(run_dir, 'dag_structure.json'), 'w') as f:
        json.dump(dag_structure, f, indent=2)
    
    # Summary metrics
    run_metrics = {
        'run_id': run_id,
        'seed': seed,
        'shd': final_shd,
        'precision': final_metrics['final'].precision,
        'recall': final_metrics['final'].recall,
        'f1': final_metrics['final'].f1_score,
        'cost': final_metrics['final'].cost,
        'risk': final_metrics['final'].risk,
        'total_interventions': sum(len(r) for r in base_pipeline.tester.intervention_results.values()),
        'validated_edges': len(base_pipeline.validated_edges),
        'iterations_completed': base_pipeline.current_iteration,
        # Compute resources
        'wall_time_s':  _wall_end - _wall_start,
        'cpu_time_s':   _cpu_end  - _cpu_start,
        'peak_mem_mb':  _peak_mem_mb,
        'n_llm_calls':  _n_llm_calls,
        **cwm_metrics  # Add CWM metrics
    }

    logger.info(f"  SHD={run_metrics['shd']}, F1={run_metrics['f1']:.3f}, "
                f"Cost={run_metrics['cost']:.3f}, Risk={run_metrics['risk']:.3f}, "
                f"wall={run_metrics['wall_time_s']:.1f}s, "
                f"LLM_calls={run_metrics['n_llm_calls']}, "
                f"mem={run_metrics['peak_mem_mb']:.1f}MB")
    if cwm_metrics:
        logger.info(f"  CWM Uncertainty={cwm_metrics.get('final_uncertainty', 0):.3f}, "
                   f"Mean Confidence={cwm_metrics.get('mean_confidence', 0):.3f}")
    
    if final_shd == 0:
        logger.info(f"  *** PERFECT MATCH ACHIEVED ***")
    
    base_pipeline.cleanup()
    return run_metrics, final_dag, pipeline


def compute_statistics(values: np.ndarray, n_bootstrap: int = 1000, confidence: float = 0.95) -> Dict:
    """Compute comprehensive statistics including bootstrap CI"""
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    median = np.median(values)
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    
    # Bootstrap CI
    bootstrap_means = []
    for _ in range(n_bootstrap):
        resample = np.random.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(resample))
    
    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(bootstrap_means, alpha * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha) * 100)
    
    return {
        'n': len(values),
        'mean': mean,
        'std': std,
        'median': median,
        'q25': q25,
        'q75': q75,
        'iqr': iqr,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_width': ci_upper - ci_lower,
        'raw_values': values
    }


def create_distribution_plots(all_metrics: List[Dict], output_dir: str):
    """Create violin + box plots and KDE for all metrics with nested structure support"""
    
    # Handle nested metric structure (intervention_only, full_dag, methods)
    def extract_metrics(metrics_list, metric_type='full_dag'):
        """Extract flattened metrics from potentially nested structure"""
        flattened = []
        for m in metrics_list:
            if isinstance(m, dict):
                # Check for nested structure
                if metric_type in m:
                    flattened.append(m[metric_type])
                elif 'intervention_only' in m:
                    flattened.append(m['intervention_only'])
                else:
                    # Assume flat structure
                    flattened.append(m)
            else:
                flattened.append(m)
        return flattened
    
    # Extract metrics based on what's available
    flattened_metrics = extract_metrics(all_metrics, metric_type='full_dag')
    
    # Debug logging
    logger.info(f"Total metrics to plot: {len(flattened_metrics)}")
    if flattened_metrics:
        logger.info(f"First metric structure: {flattened_metrics[0].keys() if isinstance(flattened_metrics[0], dict) else type(flattened_metrics[0])}")
    
    metrics = ['shd', 'precision', 'recall', 'f1', 'cost', 'risk']
    
    # Check which metrics have valid data
    available_metrics = []
    for metric in metrics:
        values = [m.get(metric, np.nan) for m in flattened_metrics if isinstance(m, dict) and metric in m]
        values = [v for v in values if not np.isnan(v) and v is not None]
        if len(values) > 0:
            available_metrics.append(metric)
        else:
            logger.warning(f"No valid data for metric: {metric}")
    
    if not available_metrics:
        logger.error("No valid metrics found for plotting")
        return
    
    # Create plots only for available metrics
    n_metrics = len(available_metrics)
    n_cols = 3
    n_rows = int(np.ceil(n_metrics / n_cols))
    
    # Violin + Box plots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    fig.suptitle('Metric Distributions: Violin + Box Plots\n(Multiple Runs)', 
                 fontsize=16, fontweight='bold')
    
    # Flatten axes if needed
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flat
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        values = [m.get(metric, np.nan) for m in flattened_metrics if isinstance(m, dict)]
        values = [v for v in values if not np.isnan(v) and v is not None]
        
        if len(values) < 2:
            logger.warning(f"Insufficient data for {metric} (n={len(values)}), skipping")
            ax.text(0.5, 0.5, f'Insufficient data\n(n={len(values)})', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{metric.upper()}', fontsize=14, fontweight='bold')
            continue
        
        # Violin plot
        parts = ax.violinplot([values], positions=[0], widths=0.7, 
                              showmeans=True, showmedians=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#8da0cb')
            pc.set_alpha(0.7)
        
        # Box plot overlay
        bp = ax.boxplot([values], positions=[0], widths=0.3, patch_artist=True,
                        boxprops=dict(facecolor='white', alpha=0.8),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(color='black', linewidth=1.5),
                        capprops=dict(color='black', linewidth=1.5))
        
        mean_val = np.mean(values)
        ax.axhline(mean_val, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        
        ax.set_title(f'{metric.upper()}', fontsize=14, fontweight='bold')
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right')
    
    # Hide unused subplots
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_violin_box.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # KDE plots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    fig.suptitle('Kernel Density Estimates', fontsize=16, fontweight='bold')
    
    # Flatten axes if needed
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes
    else:
        axes = axes.flat
    
    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        values = [m.get(metric, np.nan) for m in flattened_metrics if isinstance(m, dict)]
        values = [v for v in values if not np.isnan(v) and v is not None]
        
        if len(values) < 2:
            ax.text(0.5, 0.5, f'Insufficient data\n(n={len(values)})', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{metric.upper()}', fontsize=14, fontweight='bold')
            continue
        
        sns.kdeplot(values, ax=ax, fill=True, color='#8da0cb', alpha=0.6, linewidth=2)
        
        mean_val = np.mean(values)
        median_val = np.median(values)
        
        ax.axvline(mean_val, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='red', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
        
        stats_dict = compute_statistics(np.array(values))
        ax.axvspan(stats_dict['ci_lower'], stats_dict['ci_upper'], alpha=0.2, color='green', 
                   label=f"95% CI")
        
        ax.set_title(f'{metric.upper()}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_kde.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Distribution plots saved to {output_dir}")


def create_summary_table(statistics: Dict[str, Dict], output_dir: str):
    """Create formatted summary table"""
    if not statistics:
        logger.warning("No statistics to create table")
        return pd.DataFrame()  # Return empty DataFrame
    
    # Create DataFrame
    df = pd.DataFrame(statistics).T
    
    if df.empty:
        logger.warning("Statistics DataFrame is empty")
        return df
    
    table_data = []
    for metric, stats in statistics.items():
        table_data.append({
            'Metric': metric.upper(),
            'N': stats['n'],
            'Mean': f"{stats['mean']:.4f}",
            'Std': f"{stats['std']:.4f}",
            'Median': f"{stats['median']:.4f}",
            'IQR': f"{stats['iqr']:.4f}",
            '95% CI': f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]",
            'Mean±CI': f"{stats['mean']:.4f} ± {(stats['ci_upper']-stats['ci_lower'])/2:.4f}"
        })
    
    df = pd.DataFrame(table_data)
    df.to_csv(os.path.join(output_dir, 'summary_statistics_table.csv'), index=False)
    
    # Create table image
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df.values, colLabels=df.columns, 
                     cellLoc='center', loc='center',
                     colColours=['#4C72B0']*len(df.columns))
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4C72B0')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Summary Statistics (30 Runs, 60 Iterations Each)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, 'summary_statistics_table.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


def save_method_summary_stats(all_method_metrics, output_dir):
    """Save summary statistics CSV for each method"""
    
    for method_name, metrics_dict in all_method_metrics.items():
        stats_rows = []
        
        for metric_name in ['shd', 'precision', 'recall', 'f1', 'accuracy', 'cost', 'risk']:
            if metric_name not in metrics_dict:
                continue
                
            values = np.array(metrics_dict[metric_name])
            stats = compute_statistics(values)
            ci_range = (stats['ci_upper'] - stats['ci_lower']) / 2

            stats_rows.append({
                'Metric': metric_name.upper(),
                'N': stats['n'],
                'Mean': f"{stats['mean']:.4f}",
                'Std': f"{stats['std']:.4f}",
                'Median': f"{stats['median']:.4f}",
                'IQR': f"{stats['iqr']:.4f}",
                '95% CI': f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]",
                'Mean±CI': f"{stats['mean']:.4f} ± {ci_range:.4f}"
            })
        
        df = pd.DataFrame(stats_rows)
        csv_path = os.path.join(output_dir, f'{method_name}_summary_statistics.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {method_name} summary to {csv_path}")


def plot_final_dags(pipeline, final_dag, final_shd, output_dir):
    """Create DAG comparison plot (includes VARLiNGAM)"""
    
    fig, axes = plt.subplots(2, 3, figsize=(30, 20))
    axes = axes.flatten()
    
    base_pipeline = pipeline.pipeline
    
    # Plot each method's DAG
    methods = ['pc', 'sam', 'llm', 'varlingam', 'final']
    for idx, method in enumerate(methods):
        ax = axes[idx]
        
        if method == 'final':
            G = nx.DiGraph()
            G.add_edges_from(final_dag['edges'])
            title = f"Final Validated DAG (SHD: {final_shd})"
            if final_shd == 0:
                title += " - PERFECT MATCH!"
            color = 'gold'
        else:
            if method not in base_pipeline._results:
                ax.axis('off')
                continue
            G = nx.DiGraph()
            edges = base_pipeline._extract_edges(base_pipeline._results[method])
            # Handle VARLiNGAM edges with lag info
            edge_tuples = []
            for edge in edges:
                if isinstance(edge, dict) and 'edge' in edge:
                    edge_tuples.append(edge['edge'])
                else:
                    edge_tuples.append(edge)
            G.add_edges_from(edge_tuples)
            shd = base_pipeline.shd_history.get(method, [0])
            title = f"{method.upper()} DAG (SHD: {shd[-1] if shd else 'N/A'})"
            color = {'pc': 'lightblue', 'sam': 'lightgreen', 'llm': 'lightpink', 'varlingam': 'lavender'}[method]
        
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, ax=ax, with_labels=True, node_color=color, node_size=1000,
                font_size=10, font_weight='bold')
        
        if method == 'final':
            edge_labels = {(u, v): f"{final_dag['confidence_scores'].get(f'{u}->{v}', 0):.2f}" 
                          for (u, v) in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=9)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    # Hide unused subplot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_dags_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_dags_json(cwm_pipeline, run_id, output_dir, final_dag):
    """Save all DAGs including H0/H1 hypotheses as JSON"""
    
    pipeline = cwm_pipeline.pipeline if hasattr(cwm_pipeline, 'pipeline') else cwm_pipeline
    
    dags = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'methods': {},
        'hypotheses': {},
        'edge_confidences': {},
        'hypothesis_weights': {}
    }
    
    # Method DAGs
    if hasattr(pipeline, '_results') and pipeline._results:
        for method_name, dag in pipeline._results.items():
            edges = pipeline._extract_edges(dag)
            dags['methods'][method_name] = {
                'edges': [list(e) if isinstance(e, tuple) else e for e in edges],
                'num_edges': len(edges)
            }
    
    # H0 and H1 from CWM hypotheses
    if hasattr(cwm_pipeline, 'cwm') and cwm_pipeline.cwm and cwm_pipeline.cwm.hypotheses:
        for hyp in cwm_pipeline.cwm.hypotheses:
            if 'H0' in hyp.name or 'H1' in hyp.name:
                dags['hypotheses'][hyp.name] = {
                    'edges': [list(e) for e in hyp.graph.edges()],
                    'num_edges': len(hyp.graph.edges()),
                    'learned_edges': {
                        target: list(hyp.predictive_model.parameters[target]['weights'].keys())
                        for target in hyp.predictive_model.variables
                        if (
                            target in hyp.predictive_model.parameters and
                            hyp.predictive_model.parameters[target].get('weights')
                        )
                    }
                }
                dags['hypothesis_weights'][hyp.name] = float(hyp.weight)
    
    # Edge confidences
    if hasattr(cwm_pipeline, 'cwm') and cwm_pipeline.cwm:
        for (src, tgt), conf in cwm_pipeline.cwm.edge_confidences.items():
            dags['edge_confidences'][f"{src}->{tgt}"] = float(conf.confidence)
    
    # Final validated DAG
    dags['final'] = {
        'edges': [list(e) for e in final_dag.get('edges', [])],
        'confidence_scores': final_dag.get('confidence_scores', {}),
        'num_edges': len(final_dag.get('edges', []))
    }
    
    json_path = os.path.join(output_dir, f'all_dags_run_{run_id}.json')
    with open(json_path, 'w') as f:
        json.dump(dags, f, indent=2)
    
    logger.info(f"Saved DAGs with H0/H1 hypotheses to: {json_path}")



def main():
    """Main execution with CWM integration"""
    
    try:
        logger.info("=== Starting 30-Run Statistical Framework Evaluation with CWM ===")
        
        # Register cleanup
        atexit.register(cleanup_processes)
        
        # Configuration
        api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # smart_room_path = "./smart_room.js"
        smart_room_path = './open_window.js'

        config = {
            'api_key': api_key,
            'smart_room_path': smart_room_path,
            'dataset_type': 'open_window',
            'alpha': 0.6,
            'beta': 0.4,
            'effect_threshold': 0.1
        }
        
        # Setup output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(current_dir, f"results/3runs_cwm_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize EnergyPlus (if needed)
        eplus = EnergyPlusInterface(
            idf_template_path=os.path.join(current_dir, "templates", "smart_room.idf"),
            weather_file=os.path.join(current_dir, "weather", "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw")
        )
        
        # Load data
        # data_path = "data/simulation_log_2025-03-04T16-42-08-821Z_preprocessed.csv"
        data_path = 'data/challenge1_data_10k_processed.csv' 

        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        df = pd.read_csv(data_path)
        logger.info(f"Loaded data with {len(df)} samples")
        
        # Initialize reproducible experiment
        experiment = ReproducibleExperiment(base_seed=42)
        
        # Run 30 executions
        n_runs = 2
        all_metrics = []
        all_calibration_metrics = []  
        all_method_metrics = defaultdict(lambda: defaultdict(list))
        last_pipeline = None
        last_final_dag = None
        
        logger.info(f"\nStarting {n_runs} runs (60 iterations each) WITH CWM")
        logger.info(f"Base seed: {experiment.base_seed}\n")
        
        for run_id in range(n_runs):
            try:
                seed = experiment.get_seed(run_id)
                run_metrics, final_dag, pipeline = run_single_execution(run_id, df, config, seed, output_dir)
                # (all_metrics.append deferred until after suite merge below)

                # Extract calibration data — compare RF signal vs CWM signal
                _suite = {}  # will be populated below if calibration_tracker exists
                if hasattr(pipeline, 'calibration_tracker'):
                    tracker = pipeline.calibration_tracker

                    n_ts   = len(tracker['timestamps'])
                    n_act  = len(tracker['actual_window_state'])
                    n_rf   = len(tracker.get('P_H1_RF', []))
                    n_cwm  = len(tracker.get('P_H1', []))

                    if n_ts > 0 and n_act == n_ts and n_rf == n_ts:
                        cal_metrics_rf = compute_calibration_metrics(
                            tracker['timestamps'],
                            tracker['P_H1_RF'],
                            tracker['actual_window_state']
                        )
                        all_calibration_metrics.append(cal_metrics_rf)
                    else:
                        cal_metrics_rf = {'brier_score': float('nan'), 'log_loss': float('nan')}
                        logger.warning(f"Run {run_id+1}: RF calibration skipped — "
                                       f"length mismatch ts={n_ts} act={n_act} rf={n_rf}")

                    # CWM structural Bayesian signal (H0/H1 model competition)
                    if n_ts > 0 and n_act == n_ts and n_cwm == n_ts:
                        cal_metrics_cwm = compute_calibration_metrics(
                            tracker['timestamps'],
                            tracker['P_H1'],
                            tracker['actual_window_state']
                        )
                    else:
                        cal_metrics_cwm = {'brier_score': float('nan'), 'log_loss': float('nan')}
                        logger.warning(f"Run {run_id+1}: CWM calibration skipped — "
                                       f"length mismatch ts={n_ts} act={n_act} cwm={n_cwm}")

                    plot_single_run_calibration(
                        tracker,
                        save_path=os.path.join(output_dir, f'calibration_run_{run_id+1}.png'),
                        window_open_time=30
                    )

                    logger.info(
                        f"Run {run_id+1} Calibration — "
                        f"RF Brier: {cal_metrics_rf['brier_score']:.3f}  |  "
                        f"CWM Brier: {cal_metrics_cwm['brier_score']:.3f}"
                    )

                    # ── Per-run calibration CSV — full suite from saved CSV ───
                    # Load monitoring_probabilities.csv (source of truth) and
                    # compute Brier + log-loss for every hypothesis signal,
                    # both on all timesteps and on steady-state timesteps only
                    # (steady-state excludes ±transition_buffer_min around each
                    # regime change, isolating convergence from detection latency).
                    _mon_csv = os.path.join(output_dir, f'run_{run_id:03d}',
                                            'monitoring_probabilities.csv')
                    try:
                        _mon_df    = pd.read_csv(_mon_csv)
                        _suite     = compute_suite_metrics(_mon_df,
                                                           transition_buffer_min=0.25)
                        _suite_row = {'run_id': run_id + 1, **_suite}
                    except Exception as _e:
                        logger.warning(f"Suite metrics failed for run {run_id}: {_e}")
                        _suite     = {
                            'n_total':            n_ts,
                            'CWM_binary_brier':   cal_metrics_cwm['brier_score'],
                            'CWM_binary_log_loss':cal_metrics_cwm['log_loss'],
                            'RF_brier':           cal_metrics_rf['brier_score'],
                            'RF_log_loss':        cal_metrics_rf['log_loss'],
                        }
                        _suite_row = {'run_id': run_id + 1, **_suite}

                    pd.DataFrame([_suite_row]).to_csv(
                        os.path.join(output_dir, f'run_{run_id:03d}',
                                     'calibration_metrics.csv'),
                        index=False
                    )

                # Merge suite calibration metrics into run_metrics, then append
                run_metrics.update(_suite)
                all_metrics.append(run_metrics)

                # Save last successful run for visualization
                last_pipeline = pipeline
                last_final_dag = final_dag

                save_dags_json(pipeline, run_id, output_dir, final_dag) 

                if hasattr(pipeline, 'method_metrics'):
                    logger.info(f"Found method_metrics: {pipeline.method_metrics}")
                    for method, metrics in pipeline.method_metrics.items():
                        for metric_name, value in metrics.items():
                            all_method_metrics[method][metric_name].append(value)
                elif hasattr(pipeline, 'pipeline') and hasattr(pipeline.pipeline, 'method_metrics'):
                    logger.info(f"Found method_metrics in nested pipeline")
                    for method, metrics in pipeline.pipeline.method_metrics.items():
                        for metric_name, value in metrics.items():
                            all_method_metrics[method][metric_name].append(value)
                
                # Save intermediate results after every run (crash-safe)
                pd.DataFrame(all_metrics).to_csv(
                    os.path.join(output_dir, 'intermediate_results.csv'), index=False
                )
                
            except Exception as e:
                logger.error(f"Error in run {run_id + 1}: {str(e)}", exc_info=True)
                continue

        logger.info(f"Method metrics collected: {dict(all_method_metrics)}")

        if all_method_metrics:
            save_method_summary_stats(all_method_metrics, output_dir)
        else:
            logger.warning("No method metrics collected")

        # Compute statistics (including CWM metrics if present)
        logger.info(f"\n{'='*80}")
        logger.info("Computing comprehensive statistics...")
        logger.info(f"{'='*80}\n")
        
        metrics = ['shd', 'precision', 'recall', 'f1', 'cost', 'risk']
        
        # Add CWM metrics if available
        if all_metrics and 'final_uncertainty' in all_metrics[0]:
            metrics.extend(['final_uncertainty', 'mean_confidence'])
        
        statistics = {}
        
        for metric in metrics:
            values = np.array([m.get(metric, 0) for m in all_metrics if metric in m])
            if len(values) > 0:
                statistics[metric] = compute_statistics(values)
        
        # Create visualizations
        create_distribution_plots(all_metrics, output_dir)
        summary_df = create_summary_table(statistics, output_dir)

        # Plot DAGs from last run
        if last_pipeline and last_final_dag:
            base_pipeline = last_pipeline.pipeline
            final_edges = set(tuple(edge) if isinstance(edge, list) else edge 
                            for edge in last_final_dag['edges'])
            final_shd = base_pipeline.ground_truth.get_shd(final_edges)
            plot_final_dags(last_pipeline, last_final_dag, final_shd, output_dir)
            
            # Export metrics from last run
            visualizer = MetricsVisualizer(output_dir)
            
            # Save edge metrics
            metrics_df = pd.DataFrame({
                'Edge': [f"{e[0]}->{e[1]}" for e in last_final_dag['edges']],
                'Confidence': [base_pipeline.edge_ranker.edge_confidence.get(e, 0) 
                             for e in last_final_dag['edges']],
                'Num_Interventions': [len(base_pipeline.tester.intervention_results.get(e, []))
                                    for e in last_final_dag['edges']]
            })
            metrics_df.to_csv(os.path.join(output_dir, 'edge_metrics.csv'), index=False)
        
        # Save detailed results
        runs_df = pd.DataFrame(all_metrics)
        runs_df.to_csv(os.path.join(output_dir, 'all_runs_detailed.csv'), index=False)

        # Save seed info for reproducibility
        seed_info = {
            'base_seed': experiment.base_seed,
            'run_seeds': experiment.run_seeds,
            'n_runs': len(all_metrics),
            'cwm_enabled': True
        }
        with open(os.path.join(output_dir, 'seed_info.json'), 'w') as f:
            json.dump(seed_info, f, indent=2)
        
        # Complete results JSON
        complete_results = {
            'experiment_info': {
                'n_runs': len(all_metrics),
                'iterations_per_run': 5,
                'base_seed': experiment.base_seed,
                'timestamp': datetime.now().isoformat(),
                'dataset': config['dataset_type'],
                'cwm_enabled': True
            },
            'statistics': {
                metric: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                        for k, v in stats.items() if k != 'raw_values'}
                for metric, stats in statistics.items()
            },
            'individual_runs': all_metrics
        }
        
        with open(os.path.join(output_dir, 'complete_results.json'), 'w') as f:
            json.dump(complete_results, f, indent=2)
        
        if all_calibration_metrics:
            create_aggregate_table(
                all_calibration_metrics,
                os.path.join(output_dir, 'calibration_summary.png')
            )
            # CSV alongside the PNG — one row per run, RF signal
            pd.DataFrame(all_calibration_metrics).to_csv(
                os.path.join(output_dir, 'calibration_summary.csv'), index=False
            )

        # ── Aggregate per-run calibration_metrics.csv files into one table ───
        _cal_rows = []
        for _rid in range(n_runs):
            _p = os.path.join(output_dir, f'run_{_rid:03d}', 'calibration_metrics.csv')
            if os.path.exists(_p):
                _cal_rows.append(pd.read_csv(_p))
        if _cal_rows:
            pd.concat(_cal_rows, ignore_index=True).to_csv(
                os.path.join(output_dir, 'all_calibration_metrics.csv'), index=False
            )

        # ── Comprehensive all-runs summary: mean ± std + 95% CI for every metric ─
        if all_metrics:
            _df_all      = pd.DataFrame(all_metrics)
            _numeric_cols = _df_all.select_dtypes(include=[np.number]).columns.tolist()
            _stat_cols   = [c for c in _numeric_cols if c not in ('run_id', 'seed')]

            _summary_rows = []
            for _col in _stat_cols:
                _vals = _df_all[_col].dropna().values
                if len(_vals) == 0:
                    continue
                _s       = compute_statistics(_vals)
                _ci_half = (_s['ci_upper'] - _s['ci_lower']) / 2
                _summary_rows.append({
                    'metric':      _col,
                    'n':           _s['n'],
                    'mean':        round(float(_s['mean']),     6),
                    'std':         round(float(_s['std']),      6),
                    'mean_pm_std': f"{_s['mean']:.4f} +/- {_s['std']:.4f}",
                    'mean_pm_ci':  f"{_s['mean']:.4f} +/- {_ci_half:.4f}",
                    'ci_lower':    round(float(_s['ci_lower']), 6),
                    'ci_upper':    round(float(_s['ci_upper']), 6),
                    'median':      round(float(_s['median']),   6),
                    'q25':         round(float(_s['q25']),      6),
                    'q75':         round(float(_s['q75']),      6),
                })

            if _summary_rows:
                _summ_df = pd.DataFrame(_summary_rows)
                _summ_df.to_csv(os.path.join(output_dir, 'all_runs_summary.csv'),
                                index=False)
                logger.info(f"all_runs_summary.csv: {len(_summary_rows)} metrics "
                            f"across {len(_df_all)} runs")

                # Print key metrics to log
                _key = ['shd', 'f1', 'wall_time_s', 'cpu_time_s', 'peak_mem_mb',
                        'n_llm_calls', 'total_interventions',
                        'CWM_binary_brier', 'CWM_binary_brier_ss',
                        'RF_brier',         'RF_brier_ss']
                logger.info("Key metrics (mean +/- 95% CI half-width):")
                for _km in _key:
                    _row = _summ_df[_summ_df['metric'] == _km]
                    if not _row.empty:
                        logger.info(f"  {_km:30s}: {_row['mean_pm_ci'].values[0]}")

        # Print final summary
        logger.info(f"\n{'='*80}")
        logger.info("FINAL STATISTICAL SUMMARY (WITH CWM)")
        logger.info(f"{'='*80}")
        logger.info(f"\nCompleted: {len(all_metrics)}/{n_runs} runs")
        logger.info(f"Base seed: {experiment.base_seed}")
        logger.info(f"\n{summary_df.to_string(index=False)}")
        logger.info(f"\n{'='*80}")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"{'='*80}\n")
        
    except Exception as e:
        logger.error(f"Error in pipeline execution: {str(e)}", exc_info=True)
        raise
        
    finally:
        cleanup_processes()
        if 'last_pipeline' in locals() and last_pipeline:
            last_pipeline.pipeline.cleanup()
        if 'eplus' in locals():
            eplus.cleanup()


if __name__ == "__main__":
    main()