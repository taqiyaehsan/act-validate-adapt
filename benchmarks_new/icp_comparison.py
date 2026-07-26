#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ICP (Invariant Causal Prediction) implementation for comparison.
"""

import logging
import numpy as np
import pandas as pd
import subprocess
import tempfile
import os
import networkx as nx
from benchmarks_new.utils import perform_intervention, make_dag, get_intervenable_vars, _resolve_intervenable_vars
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz

logger = logging.getLogger(__name__)

def run_icp(data, simulation_path=None, max_interventions=10, alpha=0.075):
    """Run the ICP algorithm with ASHRAE support."""
    logger.info("Running ICP algorithm")

    if simulation_path:
        _resolve_intervenable_vars(simulation_path)

    try:
        # Keep raw data for intervention values, preprocess for algorithm
        from benchmarks_new.utils import preprocess_benchmark_data
        raw_data = data.copy()
        if 'Timestamp' in raw_data.columns: raw_data = raw_data.drop(columns=['Timestamp'])
        if 'weight' in raw_data.columns: raw_data = raw_data.drop(columns=['weight'])
        data = preprocess_benchmark_data(data)

        column_names = data.columns.tolist()
        logger.info(f"Input data shape: {data.shape}")
        logger.info(f"Variables: {column_names}")

        intervention_results = {}
        
        # Detect dataset type
        is_ashrae = 'air_temperature' in data.columns and 'meter_reading' in data.columns
        
        # Create ASHRAE tester if needed
        if is_ashrae and simulation_path is None:
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
                            
                            model = RandomForestRegressor(n_estimators=20, random_state=42)
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
        
        # Run PC algorithm
        logger.info("Initializing with PC algorithm")
        cg = pc(data.values, indep_test='fisherz', alpha=0.05, stable=True, uc_rule=1, uc_priority=2)
        
        # Extract initial DAG
        adj_matrix = cg.G.graph
        edges = []
        for i in range(len(column_names)):
            for j in range(len(column_names)):
                if adj_matrix[i, j] == 1:
                    edges.append((column_names[i], column_names[j]))
        
        G = nx.DiGraph()
        G.add_edges_from(edges)
        
        # Perform interventions if possible
        from benchmarks_new.utils import _raw_benchmark_data
        if (simulation_path and os.path.exists(simulation_path)) or (is_ashrae and 'ashrae_tester' in locals()) or (_raw_benchmark_data is not None):
            logger.info("Starting interventions")
            
            # Setup for environments
            intervention_datasets = [data]
            intervention_environments = [np.zeros(len(data))]
            
            num_interventions = 0
            env_id = 1
            
            # Only actuator variables can be intervened on
            intervention_vars = get_intervenable_vars(column_names, simulation_path)
            
            for var in intervention_vars:
                if num_interventions >= max_interventions: break
                
                # Define intervention values from RAW data (not standardized)
                # Use extreme quantiles or fixed 0.1/0.9 for normalized data
                raw_col = [c for c in raw_data.columns if c.lower() == var.lower()]
                if raw_col:
                    low_val = raw_data[raw_col[0]].quantile(0.1)
                    high_val = raw_data[raw_col[0]].quantile(0.9)
                    # If data has low variance, use wider range
                    if abs(high_val - low_val) < 0.05:
                        low_val = max(0.0, raw_data[raw_col[0]].min() - 0.1)
                        high_val = min(1.0, raw_data[raw_col[0]].max() + 0.1)
                else:
                    low_val = 0.1
                    high_val = 0.9
                
                env_data_low = []
                env_data_high = []
                
                # Perform interventions
                for _ in range(3):
                    if is_ashrae and 'ashrae_tester' in locals():
                        result_low = ashrae_tester._simulate_intervention(var, low_val)
                        result_high = ashrae_tester._simulate_intervention(var, high_val)
                    else:
                        result_low = perform_intervention(var, low_val, simulation_path)
                        result_high = perform_intervention(var, high_val, simulation_path)
                    
                    # Store intervention data for MetricsCalculator
                    for result, value in [(result_low, low_val), (result_high, high_val)]:
                        if result and 'postInterventionData' in result and 'preInterventionData' in result:
                            # Store for all potential targets
                            for target in column_names:
                                if target != var:
                                    edge = (var, target)
                                    intervention_data = extract_intervention_data(
                                        result['preInterventionData'], 
                                        result['postInterventionData']
                                    )
                                    if edge not in intervention_results:
                                        intervention_results[edge] = []
                                    intervention_results[edge].append(intervention_data)
                    
                    if result_low and 'postInterventionData' in result_low:
                        env_data_low.append(result_low['postInterventionData'])
                        num_interventions += 1
                        
                    if result_high and 'postInterventionData' in result_high:
                        env_data_high.append(result_high['postInterventionData'])
                        num_interventions += 1
                        
                    if num_interventions >= max_interventions: break
                
                # Create environment datasets, aligning columns to training data
                for env_data in [env_data_low, env_data_high]:
                    if not env_data:
                        continue
                    # Extract only the columns we need from each dict
                    aligned_rows = []
                    for row_dict in env_data:
                        row = {}
                        for col in column_names:
                            # Try exact match, then case-insensitive, then Satisfaction alias
                            if col in row_dict:
                                row[col] = float(row_dict[col])
                            else:
                                found = False
                                for k, v in row_dict.items():
                                    if k.lower() == col.lower():
                                        row[col] = float(v)
                                        found = True
                                        break
                                if not found and col.lower() == 'satisfaction':
                                    for k, v in row_dict.items():
                                        if 'satisfaction' in k.lower():
                                            row[col] = float(v)
                                            found = True
                                            break
                                if not found:
                                    row[col] = np.nan
                        aligned_rows.append(row)
                    env_df = pd.DataFrame(aligned_rows)
                    env_df = env_df.dropna(axis=1)  # drop cols with NaN
                    if len(env_df.columns) == len(column_names) and len(env_df) > 0:
                        intervention_datasets.append(env_df)
                        intervention_environments.append(np.full(len(env_df), env_id))
                        env_id += 1
            
            logger.info(f"Created {env_id-1} intervention environments")
            
            # Apply ICP to identify invariant predictors
            edges = []
            
            if len(intervention_datasets) >= 3:  # Need at least 3 environments for ICP
                try:
                    for target in column_names:
                        # Test all variables as potential targets
                        if True:
                            invariant_predictors = compute_icp_for_target(
                                intervention_datasets, 
                                intervention_environments, 
                                target, 
                                column_names,
                                alpha
                            )
                            
                            for predictor in invariant_predictors:
                                edges.append((predictor, target))
                except Exception as e:
                    logger.warning(f"ICP computation failed: {str(e)}")
                    edges = list(G.edges())
            else:
                logger.warning("Not enough environments for ICP. Using initial DAG.")
                edges = list(G.edges())
        else:
            logger.warning("No simulation path provided. Using only observational data.")
            edges = list(G.edges())
        
        edges = make_dag(edges)
        
        logger.info(f"ICP algorithm completed. Found {len(edges)} edges")
        logger.info(f"Edges: {edges}")
        
        return edges
        
    except Exception as e:
        logger.error(f"Error in ICP algorithm: {str(e)}")
        return []

def compute_icp_for_target(intervention_datasets, intervention_environments, target, 
                         feature_names, alpha=0.05):
    """
    Compute ICP for a specific target variable.
    
    Args:
        intervention_datasets (list): List of pandas DataFrames for different environments
        intervention_environments (list): List of environment IDs for each dataset
        target (str): Target variable
        feature_names (list): List of all feature names
        alpha (float): Significance level
        
    Returns:
        list: List of invariant predictor variables
    """
    # Combine datasets
    combined_data = pd.concat(intervention_datasets, ignore_index=True)
    
    # Combine environment indicators
    environments = np.concatenate(intervention_environments)
    
    # Potential predictor variables (exclude target)
    predictors = [col for col in feature_names if col != target]
    
    try:
        # Check if we have the R library installed
        has_R = check_r_installed()
        
        if has_R:
            # Use R implementation if available (more reliable)
            return compute_icp_r(combined_data, environments, target, predictors, alpha)
        else:
            # Fall back to Python implementation
            return compute_icp_python(combined_data, environments, target, predictors, alpha)
            
    except Exception as e:
        logger.warning(f"ICP computation error: {str(e)}")
        return []  # Return empty list if ICP fails

def check_r_installed():
    """Check if R and InvariantCausalPrediction package are installed."""
    try:
        # Check if R is installed
        r_process = subprocess.run(['Rscript', '--version'], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Check if InvariantCausalPrediction package is installed
        r_script = 'if (requireNamespace("InvariantCausalPrediction", quietly = TRUE)) { cat("TRUE") } else { cat("FALSE") }'
        r_check = subprocess.run(['Rscript', '-e', r_script], 
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        return "TRUE" in r_check.stdout
    except FileNotFoundError:
        return False

def compute_icp_r(data, environments, target, predictors, alpha):
    """Compute ICP using R implementation."""
    # Create temporary directory for R files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save data to CSV
        data_path = os.path.join(temp_dir, 'data.csv')
        data.to_csv(data_path, index=False)
        
        # Save environments to CSV
        env_path = os.path.join(temp_dir, 'environments.csv')
        pd.DataFrame({'environment': environments}).to_csv(env_path, index=False)
        
        # Create R script
        r_script_path = os.path.join(temp_dir, 'run_icp.R')
        with open(r_script_path, 'w') as f:
            f.write(f'''
            # Load required packages
            if (!requireNamespace("InvariantCausalPrediction", quietly = TRUE)) {{
                install.packages("InvariantCausalPrediction")
            }}
            library(InvariantCausalPrediction)
            
            # Load data
            data <- read.csv("{data_path}")
            environments <- read.csv("{env_path}")$environment
            
            # Set target and predictors
            target <- "{target}"
            predictors <- c({', '.join([f'"{p}"' for p in predictors])})
            
            # Run ICP
            X <- as.matrix(data[, predictors])
            y <- as.numeric(data[, target])
            
            icp_result <- ICP(X, y, environments, alpha = {alpha})
            
            # Save results
            invariant_predictors <- predictors[icp_result$maximinCoef != 0]
            write.table(invariant_predictors, "{os.path.join(temp_dir, 'icp_results.txt')}", 
                       row.names = FALSE, col.names = FALSE, quote = FALSE)
            ''')
        
        # Run R script
        result = subprocess.run(['Rscript', r_script_path],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            logger.warning(f"R script error: {result.stderr}")
            return []
        
        # Read results
        results_path = os.path.join(temp_dir, 'icp_results.txt')
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                invariant_predictors = [line.strip() for line in f.readlines()]
            return invariant_predictors
        else:
            return []

def compute_icp_python(data, environments, target, predictors, alpha):
    """
    Compute ICP using Python implementation.
    This is a simplified version that tests invariance using linear regression.
    """
    # Target variable as numpy array
    y = data[target].values
    
    # Potential causal parents
    causal_parents = []
    
    # For each predictor, test invariance across environments
    for predictor in predictors:
        # Skip self-loops
        if predictor == target:
            continue
        
        # Predictor variable as numpy array
        X = data[predictor].values.reshape(-1, 1)
        
        # Test invariance across environments
        p_values = []
        unique_envs = np.unique(environments)
        
        # Skip if we have fewer than 2 environments
        if len(unique_envs) < 2:
            continue
        
        for env in unique_envs:
            # Get data for this environment
            X_env = X[environments == env]
            y_env = y[environments == env]
            
            # Skip if we have too few samples
            if len(X_env) < 10:
                continue
            
            # Fit linear model on all data
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(X, y)
            
            # Predict on environment data
            y_pred = model.predict(X_env)
            
            # Compute residuals
            residuals = y_env - y_pred
            
            # Test if residuals have mean zero (invariance test)
            from scipy.stats import ttest_1samp
            t_stat, p_value = ttest_1samp(residuals, 0)
            
            p_values.append(p_value)
        
        # If all p-values are above alpha, predictor is invariant
        if all(p > alpha for p in p_values):
            causal_parents.append(predictor)
    
    return causal_parents

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

if __name__ == "__main__":
    # Small test for debugging
    logging.basicConfig(level=logging.INFO)
    
    # Generate random data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(n_samples, 3)  # 3 variables
    X[:, 2] = 0.8 * X[:, 0] + 0.2 * X[:, 1] + 0.1 * np.random.randn(n_samples)  # X2 = f(X0, X1)
    
    df = pd.DataFrame(X, columns=['X0', 'X1', 'X2'])
    
    try:
        edges = run_icp(df)  # No simulation path for this test
        print(f"Discovered edges: {edges}")
    except ImportError:
        print("Skipping ICP test due to missing dependencies")