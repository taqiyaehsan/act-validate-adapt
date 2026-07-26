#!/usr/bin/env python3
"""
Preprocess regenerated simulator data for B1 experiments.

Takes raw sim data from data_regen/ (already in [0,1] for T/H/AQ,
Energy/Satisfaction as fractions) and produces:
  - *_processed.csv   (MinMax-normalized [0,1])
  - *_processed_scaling.csv  (feature, data_min, data_max, scale)

Format matches challenge1_data_10k_processed.csv so all B1 experiments
use consistent column names and scaling.
"""
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('preprocess_regen')

REGEN_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_regen')

# Dataset configs: raw_file -> (output_base, columns_to_keep, rename_map)
DATASETS = {
    'smart_room': {
        'raw': 'smart_room_sim_data.csv',
        'out_base': 'smart_room_processed',
        'keep_cols': ['Temperature', 'Humidity', 'AirQuality',
                      'EnergyConsumption', 'OverallSatisfaction'],
        'rename': {'OverallSatisfaction': 'Satisfaction'},
    },
    'smart_room_noise': {
        'raw': 'smart_room_noise_sim_data.csv',
        'out_base': 'smart_room_noise_processed',
        'keep_cols': ['Temperature', 'Humidity', 'AirQuality',
                      'EnergyConsumption', 'OverallSatisfaction'],
        'rename': {'OverallSatisfaction': 'Satisfaction'},
    },
    'hidden_vars': {
        'raw': 'hidden_vars_sim_data.csv',
        'out_base': 'hidden_vars_processed',
        'keep_cols': ['Temperature', 'Humidity', 'AirQuality',
                      'EnergyConsumption', 'OverallSatisfaction',
                      'OutdoorTemperature'],
        'rename': {'OverallSatisfaction': 'Satisfaction'},
    },
}


def preprocess_dataset(name, cfg):
    raw_path = os.path.join(REGEN_DIR, cfg['raw'])
    if not os.path.exists(raw_path):
        logger.error(f"[{name}] Raw file not found: {raw_path}")
        return None

    df = pd.read_csv(raw_path)
    logger.info(f"[{name}] Loaded {len(df)} rows, cols={list(df.columns)}")

    # Keep only required columns
    available = [c for c in cfg['keep_cols'] if c in df.columns]
    df = df[available].copy()

    # Rename columns
    if cfg.get('rename'):
        df = df.rename(columns=cfg['rename'])

    # Handle missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        logger.warning(f"[{name}] {missing} missing values, forward-filling")
        df = df.ffill().bfill()

    # Print raw stats
    logger.info(f"[{name}] Raw stats:")
    for col in df.columns:
        logger.info(f"  {col:25s} mean={df[col].mean():.4f}  "
                     f"min={df[col].min():.4f}  max={df[col].max():.4f}")

    # MinMax normalize all columns to [0,1]
    scaler = MinMaxScaler()
    numeric_cols = list(df.columns)
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Save processed CSV
    out_csv = os.path.join(REGEN_DIR, f"{cfg['out_base']}.csv")
    df.to_csv(out_csv, index=False)
    logger.info(f"[{name}] Saved {len(df)} rows to {out_csv}")

    # Save scaling params (same format as open_window)
    scaling_df = pd.DataFrame({
        'feature': numeric_cols,
        'data_min': scaler.data_min_,
        'data_max': scaler.data_max_,
        'scale': scaler.scale_,
    })
    scaling_path = os.path.join(REGEN_DIR, f"{cfg['out_base']}_scaling.csv")
    scaling_df.to_csv(scaling_path, index=False)
    logger.info(f"[{name}] Saved scaling to {scaling_path}")

    # Print normalized stats
    logger.info(f"[{name}] Normalized stats:")
    for col in df.columns:
        logger.info(f"  {col:25s} mean={df[col].mean():.4f}  "
                     f"min={df[col].min():.4f}  max={df[col].max():.4f}")

    return df


def main():
    for name, cfg in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"  Preprocessing: {name}")
        print(f"{'='*60}")
        preprocess_dataset(name, cfg)

    print(f"\nDone. Files in {REGEN_DIR}/:")
    for f in sorted(os.listdir(REGEN_DIR)):
        size = os.path.getsize(os.path.join(REGEN_DIR, f))
        print(f"  {f:50s} {size:>10,d} bytes")


if __name__ == '__main__':
    main()
