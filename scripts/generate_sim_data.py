#!/usr/bin/env python3
"""
Generate training datasets from JS simulators.

Runs each JS simulator with diverse interventions (Latin-hypercube sampling
over the action space) and records the outputs.  The resulting datasets
accurately reflect each simulator's current dynamics, ensuring SEM coefficients
match the simulator's interventional effects.

Usage:
  python generate_sim_data.py                          # all JS sims, 2000 samples
  python generate_sim_data.py --sim smart_room         # one simulator
  python generate_sim_data.py --n-samples 5000         # more samples
"""
import argparse
import json
import logging
import os
import subprocess
import sys

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('generate_sim_data')
logger.setLevel(logging.INFO)

# ── Simulator configs ────────────────────────────────────────────────────────

SIM_CONFIGS = {
    'open_window': {
        'sim_path': 'open_window.js',
        'output_path': 'data/open_window_sim_data.csv',
        'use_elapsed_ms': True,
    },
    'smart_room': {
        'sim_path': 'js/smart_room.js',
        'output_path': 'data/smart_room_sim_data.csv',
        'use_elapsed_ms': False,
    },
    'smart_room_noise': {
        'sim_path': 'js/smart_room_noise.js',
        'output_path': 'data/smart_room_noise_sim_data.csv',
        'use_elapsed_ms': False,
    },
    'hidden_vars': {
        'sim_path': 'js/smart_room_hidden_vars.js',
        'output_path': 'data/hidden_vars_sim_data.csv',
        'use_elapsed_ms': False,
    },
}

# Physical-unit ranges (matching experiments_neurips.py)
RANGES = {
    'Temperature': (18.0, 30.0),
    'Humidity':    (30.0, 70.0),
    'AirQuality':  (200.0, 1000.0),
}


def norm_to_phys(var, norm_val):
    lo, hi = RANGES[var]
    return float(np.clip(norm_val, 0, 1)) * (hi - lo) + lo


def phys_to_norm(var, phys_val):
    lo, hi = RANGES[var]
    return float(np.clip((phys_val - lo) / (hi - lo), 0, 1))


def generate_dataset(sim_name, cfg, n_samples=2000, seed=42):
    """Generate dataset from a JS simulator via Latin-hypercube-like sampling."""
    sim_path = cfg['sim_path']
    if not os.path.exists(sim_path):
        logger.error(f"Simulator not found: {sim_path}")
        return None

    rng = np.random.RandomState(seed)
    data = []
    n_failed = 0

    # Latin-hypercube-like: stratified random sampling over [0,1]^3
    n_per_dim = max(1, int(np.cbrt(n_samples)))
    grid = np.linspace(0.05, 0.95, n_per_dim)
    n_grid = n_per_dim ** 3

    # Generate grid + random samples to reach n_samples
    interventions = []

    # Stratified grid
    for t in grid:
        for h in grid:
            for aq in grid:
                interventions.append((t, h, aq))

    # Fill remaining with random samples
    n_remaining = max(0, n_samples - len(interventions))
    for _ in range(n_remaining):
        interventions.append(tuple(rng.uniform(0.05, 0.95, 3)))

    # Shuffle and truncate
    rng.shuffle(interventions)
    interventions = interventions[:n_samples]

    logger.info(f"[{sim_name}] Generating {len(interventions)} samples from {sim_path}")

    for i, (t_norm, h_norm, aq_norm) in enumerate(interventions):
        if (i + 1) % 500 == 0:
            logger.info(f"  {i+1}/{len(interventions)} samples generated")

        t_phys = norm_to_phys('Temperature', t_norm)
        h_phys = norm_to_phys('Humidity', h_norm)
        aq_phys = norm_to_phys('AirQuality', aq_norm)

        # Use a neutral starting state
        state_payload = json.dumps({
            'temperature': 24.0,
            'humidity': 50.0,
            'airQuality': 500.0,
        })

        intervention_payload = json.dumps({
            'temperature': round(t_phys, 2),
            'humidity': round(h_phys, 2),
            'airQuality': round(aq_phys, 2),
        })

        cmd = ['node', sim_path, '--single-step',
               '--state', state_payload,
               '--intervention', intervention_payload]

        if cfg.get('use_elapsed_ms'):
            # Random elapsed time to cover different window states
            elapsed_ms = rng.randint(0, 300) * 1000
            cmd.extend(['--elapsed-ms', str(elapsed_ms)])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            for line in result.stdout.splitlines():
                if line.startswith('RESULT:'):
                    sim = json.loads(line[len('RESULT:'):])
                    row = {
                        'Temperature': t_norm,
                        'Humidity': h_norm,
                        'AirQuality': aq_norm,
                        'EnergyConsumption': sim.get('energyConsumption', 50.0) / 100.0,
                        'OverallSatisfaction': sim.get('overallSatisfaction', 50.0) / 100.0,
                    }
                    # Include extra fields if present
                    if 'windowOpen' in sim:
                        row['WindowOpen'] = sim['windowOpen']
                    if 'outdoorTemperature' in sim:
                        row['OutdoorTemperature'] = phys_to_norm(
                            'Temperature', sim['outdoorTemperature'])
                    data.append(row)
                    break
            else:
                n_failed += 1
        except Exception as e:
            n_failed += 1
            if n_failed <= 3:
                logger.warning(f"  Sample {i} failed: {e}")

    if n_failed > 0:
        logger.warning(f"  {n_failed}/{len(interventions)} samples failed")

    if not data:
        logger.error(f"  No data generated for {sim_name}!")
        return None

    df = pd.DataFrame(data)

    # Also add Satisfaction alias for open_window compatibility
    if 'OverallSatisfaction' in df.columns:
        df['Satisfaction'] = df['OverallSatisfaction']

    # Save
    output_path = cfg['output_path']
    df.to_csv(output_path, index=False)
    logger.info(f"  Saved {len(df)} rows to {output_path}")
    logger.info(f"  Columns: {list(df.columns)}")

    # Quick quality check
    for col in ['Temperature', 'Humidity', 'AirQuality',
                'EnergyConsumption', 'OverallSatisfaction']:
        if col in df.columns:
            logger.info(f"    {col:25s} mean={df[col].mean():.3f}  "
                        f"std={df[col].std():.3f}")

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Generate training datasets from JS simulators')
    parser.add_argument('--sim', type=str, default=None,
                        choices=list(SIM_CONFIGS.keys()),
                        help='Generate for one simulator (default: all)')
    parser.add_argument('--n-samples', type=int, default=2000,
                        help='Number of samples per simulator (default: 2000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')
    args = parser.parse_args()

    sims = [args.sim] if args.sim else list(SIM_CONFIGS.keys())

    for name in sims:
        cfg = SIM_CONFIGS[name]
        print(f"\n{'='*60}")
        print(f"  Generating data: {name}")
        print(f"  Simulator: {cfg['sim_path']}")
        print(f"  Samples: {args.n_samples}")
        print(f"{'='*60}")

        df = generate_dataset(name, cfg, n_samples=args.n_samples, seed=args.seed)
        if df is not None:
            print(f"  -> {len(df)} rows saved to {cfg['output_path']}")

    print(f"\nDone.")


if __name__ == '__main__':
    main()
