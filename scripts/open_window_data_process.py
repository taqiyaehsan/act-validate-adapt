#!/usr/bin/env python3
"""
Challenge 1 Data Preprocessing
Handles open_window.js simulation data with intervention support
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_challenge1_data(input_file: str, output_file: str = None, 
                               normalize: bool = True) -> pd.DataFrame:
    """
    Preprocess Challenge 1 open window simulation data.
    
    Args:
        input_file: Path to challenge1_data.csv
        output_file: Path for processed output
        normalize: Whether to normalize features
        
    Returns:
        Processed DataFrame
    """
    logger.info(f"Loading Challenge 1 data from {input_file}")
    
    # Load data
    df = pd.read_csv(input_file)
    
    # Rename columns to match causal discovery pipeline
    column_mapping = {
        'temperature': 'Temperature',
        'humidity': 'Humidity', 
        'airQuality': 'AirQuality',
        'pmv': 'PMV',
        'energyConsumption': 'EnergyConsumption',
        'satisfaction': 'Satisfaction',
        'windowOpen': 'WindowOpen',
        'outdoorTemperature': 'OutdoorTemperature'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Recalculate PMV if it's all zeros
    if 'PMV' in df.columns and df['PMV'].abs().max() < 0.001:
        logger.info("PMV appears to be all zeros, recalculating...")
        df['PMV'] = (df['Temperature'] - 22) * 0.3 + (df['Humidity'] - 50) * 0.01
    
    # Select relevant columns (exclude timestamp)
    required_cols = ['Temperature', 'Humidity', 'AirQuality', 'PMV',
                     'EnergyConsumption', 'Satisfaction', 'WindowOpen', 
                     'OutdoorTemperature']
    
    # Add elapsedTime if present (for time-series analysis)
    # if 'elapsedTime' in df.columns:
    #     required_cols.append('elapsedTime')
    
    # Add intervention marker if present
    if 'intervention' in df.columns:
        required_cols.append('intervention')
        
    df = df[[col for col in required_cols if col in df.columns]].copy()
    
    # Handle missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        logger.warning(f"Missing values:\n{missing[missing > 0]}")
        df = df.ffill().bfill()
    
    # Normalize if requested
    if normalize:
        logger.info("Normalizing features...")
        
        # Don't normalize binary WindowOpen or intervention
        numeric_cols = [col for col in df.columns 
                       if col not in ['WindowOpen', 'intervention', 'elapsedTime']]
        
        scaler = MinMaxScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        
        # Save scaling parameters
        if output_file:
            base_name = output_file.rsplit('.', 1)[0]
            scaling_df = pd.DataFrame({
                'feature': numeric_cols,
                'data_min': scaler.data_min_,
                'data_max': scaler.data_max_,
                'scale': scaler.scale_
            })
            scaling_df.to_csv(f"{base_name}_scaling.csv", index=False)
    
    # Save processed data
    if output_file:
        df.to_csv(output_file, index=False)
        logger.info(f"Saved processed data to {output_file}")
    
    logger.info(f"Processed shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def analyze_challenge1_structure(df: pd.DataFrame) -> dict:
    """Analyze causal structure of Challenge 1 data."""
    
    analysis = {
        'total_samples': len(df),
        'window_open_pct': (df['WindowOpen'].sum() / len(df)) * 100,
        'variables': list(df.columns),
        'intervention_vars': ['WindowOpen', 'OutdoorTemperature'],
        'outcome_vars': ['Temperature', 'EnergyConsumption', 'Satisfaction'],
        'intermediate_vars': ['PMV'] if 'PMV' in df.columns else [],
        'correlations': {}
    }
    
    # Key correlations
    if 'WindowOpen' in df.columns:
        closed = df[df['WindowOpen'] == 0]
        open_df = df[df['WindowOpen'] == 1]
        
        analysis['correlations']['TOA_T1_closed'] = closed['OutdoorTemperature'].corr(closed['Temperature'])
        analysis['correlations']['TOA_T1_open'] = open_df['OutdoorTemperature'].corr(open_df['Temperature'])
        analysis['correlations']['W1_Energy'] = df['WindowOpen'].corr(df['EnergyConsumption'])
        
        # PMV correlation if available
        if 'PMV' in df.columns:
            analysis['correlations']['T1_PMV'] = df['Temperature'].corr(df['PMV'])
    
    return analysis


if __name__ == "__main__":
    # Test preprocessing
    df = preprocess_challenge1_data(
        'challenge1_data_10k.csv',
        'challenge1_data_10k_processed.csv',
        normalize=True
    )
    
    # Analyze
    analysis = analyze_challenge1_structure(df)
    print("\nChallenge 1 Data Analysis:")
    print(f"  Samples: {analysis['total_samples']}")
    print(f"  Window open: {analysis['window_open_pct']:.1f}%")
    print(f"  Key correlations:")
    for k, v in analysis['correlations'].items():
        print(f"    {k}: {v:.3f}")