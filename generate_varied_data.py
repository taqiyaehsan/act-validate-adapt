#!/usr/bin/env python3
"""Generate varied training data for all simulators."""
import subprocess, json, os, sys
import pandas as pd
import numpy as np

def generate_data(sim_path, output_prefix, n_steps=2000,
                  action_vars=None, low_vals=None, high_vals=None,
                  extra_cols=None, persistent_state=True):
    """Generate data with 80% default + 10% low + 10% high actuator variation.

    If persistent_state=True, each step passes the previous state via --state
    so that dynamics accumulate (e.g. window effects on temperature compound
    over time). This matches how the sim runs in production.
    """
    sim = os.path.abspath(sim_path)
    rows = []
    np.random.seed(42)
    prev_state = None  # carried between steps when persistent_state=True

    for i in range(n_steps):
        elapsed_ms = (6 * 3600 + i * 60) * 1000

        r = np.random.random()
        if r < 0.1:
            intervention = dict(zip(action_vars, low_vals))
        elif r < 0.2:
            intervention = dict(zip(action_vars, high_vals))
        else:
            intervention = {}

        cmd = ['node', sim, '--single-step', '--elapsed-ms', str(int(elapsed_ms))]
        if persistent_state and prev_state:
            cmd += ['--state', json.dumps(prev_state)]
        if intervention:
            cmd += ['--intervention', json.dumps(intervention)]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            for line in result.stdout.split('\n'):
                if 'RESULT:' in line:
                    d = json.loads(line.split('RESULT:', 1)[1].strip())
                    rows.append(d)
                    # Carry state forward for next step
                    if persistent_state:
                        prev_state = {
                            'temperature': d.get('temperature', 22),
                            'humidity': d.get('humidity', 50),
                            'airQuality': d.get('airQuality', 300),
                        }
                    break
        except:
            pass

        if (i + 1) % 500 == 0:
            print(f'  {output_prefix}: {i+1}/{n_steps}')

    df = pd.DataFrame(rows)

    # Select and rename columns based on sim
    if extra_cols:
        keep = action_vars + ['energyConsumption', 'overallSatisfaction'] + extra_cols
    else:
        keep = action_vars + ['energyConsumption', 'overallSatisfaction']

    # Only keep columns that exist
    keep = [c for c in keep if c in df.columns]
    df = df[keep]

    # Standardize column names
    col_map = {
        'temperature': 'Temperature', 'humidity': 'Humidity',
        'airQuality': 'AirQuality', 'energyConsumption': 'EnergyConsumption',
        'overallSatisfaction': 'Satisfaction',
        'pmv': 'PMV', 'windowOpen': 'WindowOpen',
        'outdoorTemperature': 'OutdoorTemperature',
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # Save raw
    raw_path = f'data_regen/{output_prefix}_data_varied.csv'
    df.to_csv(raw_path, index=False)

    # Normalize and save
    scaling = pd.DataFrame({
        'feature': df.columns,
        'data_min': df.min().values,
        'data_max': df.max().values,
    })
    scaling.to_csv(f'data_regen/{output_prefix}_varied_scaling.csv', index=False)

    df = df.apply(pd.to_numeric, errors='coerce')
    df_norm = (df - df.min()) / (df.max() - df.min() + 1e-12)
    df_norm.to_csv(f'data_regen/{output_prefix}_varied_processed.csv', index=False)

    print(f'  {output_prefix}: {len(df)} rows saved')
    for c in df.columns:
        print(f'    {c}: mean={df[c].mean():.1f}, std={df[c].std():.1f}')
    return df


if __name__ == '__main__':
    # smart_room_noise: same vars as smart_room
    print('=== smart_room_noise ===')
    generate_data('js/smart_room_noise.js', 'smart_room_noise',
                  action_vars=['temperature', 'humidity', 'airQuality'],
                  low_vals=[18, 30, 150],
                  high_vals=[30, 70, 500])

    # smart_room_hidden_vars: same actuators, has extra hidden vars
    print('\n=== smart_room_hidden_vars ===')
    generate_data('js/smart_room_hidden_vars.js', 'smart_room_hidden_vars',
                  action_vars=['temperature', 'humidity', 'airQuality'],
                  low_vals=[18, 30, 150],
                  high_vals=[30, 70, 500])

    # open_window: same actuators + window/outdoor/pmv context
    print('\n=== open_window ===')
    generate_data('js/open_window.js', 'open_window',
                  action_vars=['temperature', 'humidity', 'airQuality'],
                  low_vals=[18, 30, 150],
                  high_vals=[30, 70, 500],
                  extra_cols=['pmv', 'windowOpen', 'outdoorTemperature'])

    # smart_building: 5 zones, actuators are HVACSetpoint and LightingLevel
    print('\n=== smart_building ===')
    generate_data('js/smart_building.js', 'smart_building',
                  action_vars=['HVACSetpoint', 'LightingLevel'],
                  low_vals=[18, 0.2],
                  high_vals=[28, 1.0],
                  extra_cols=['Temperature', 'Humidity', 'AirQuality',
                              'OccupantCount', 'ThermalComfort', 'VisualComfort',
                              'AirQualityIndex', 'HVACPower', 'LightingPower'])
