#!/usr/bin/env python3
"""Quick diagnostic: what PMV/Satisfaction values does the monitoring simulator produce?
Compare with training data distribution to understand the SEM prediction gap."""
import os, sys, json, subprocess, tempfile, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load scaling params
scaling = pd.read_csv('data/challenge1_data_10k_processed_scaling.csv')
print("Scaling params:")
for _, row in scaling.iterrows():
    print(f"  {row['feature']}: min={row['data_min']}, max={row['data_max']}")

# Load training data stats
train = pd.read_csv('data/challenge1_data_10k_processed.csv')
print(f"\nTraining data stats (already normalized 0-1):")
for col in ['Temperature', 'Humidity', 'AirQuality', 'PMV', 'Satisfaction']:
    if col in train.columns:
        c = col.lower()
        vals = train[col]
        print(f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}, "
              f"min={vals.min():.4f}, max={vals.max():.4f}")

# Check closed-window training data
if 'WindowOpen' in train.columns:
    closed = train[train['WindowOpen'] == 0]
    print(f"\nClosed-window training data ({len(closed)} rows):")
    for col in ['Temperature', 'Humidity', 'AirQuality', 'PMV', 'Satisfaction']:
        if col in closed.columns:
            vals = closed[col]
            print(f"  {col}: mean={vals.mean():.4f}, std={vals.std():.4f}")

# Run simulator for 30 seconds and capture raw values
sim_path = os.path.abspath('open_window.js')
with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
    f.write(f'''
    const {{ SmartRoom }} = require('{sim_path}');
    const room = new SmartRoom();
    room._initialize().then(() => {{
        room.start();
        const interval = setInterval(() => {{
            room.updateDependentVariables().then(() => {{
                const state = room.getState();
                console.log('STATE:' + JSON.stringify(state));
            }});
        }}, 2000);
        setTimeout(() => {{
            clearInterval(interval);
            room.stop();
            room.cleanup().then(() => process.exit(0));
        }}, 30000);
    }});
    ''')
    temp_path = f.name

print(f"\nRunning simulator for 30s...")
proc = subprocess.Popen(['node', temp_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
raw_states = []
for line in iter(proc.stdout.readline, ''):
    if 'STATE:' in line:
        state = json.loads(line.split('STATE:', 1)[1])
        raw_states.append(state)
proc.wait()
os.remove(temp_path)

print(f"Captured {len(raw_states)} states from simulator\n")

# Normalize and compare
def normalize(val, feature):
    row = scaling[scaling['feature'].str.lower() == feature.lower()]
    if len(row) == 0:
        return val
    return (val - row['data_min'].values[0]) / (row['data_max'].values[0] - row['data_min'].values[0])

for i, state in enumerate(raw_states[:10]):
    t = i * 2  # 2s intervals
    T_raw = state.get('Temperature', state.get('temperature', 0))
    PMV_raw = state.get('PMV', state.get('pmv', 0))
    Sat_raw = state.get('Satisfaction', state.get('satisfaction', 0))
    W = state.get('WindowOpen', state.get('windowOpen', 0))

    T_norm = normalize(T_raw, 'Temperature')
    PMV_norm = normalize(PMV_raw, 'PMV')
    Sat_norm = normalize(Sat_raw, 'Satisfaction')

    # SEM prediction
    T_pred = -0.015 + 0.854 * PMV_norm + 0.165 * Sat_norm

    print(f"t={t:3d}s W={int(W)} | Raw: T={T_raw:6.2f} PMV={PMV_raw:6.3f} Sat={Sat_raw:5.1f}% | "
          f"Norm: T={T_norm:.4f} PMV={PMV_norm:.4f} Sat={Sat_norm:.4f} | "
          f"SEM T_pred={T_pred:.4f} err={abs(T_norm - T_pred):.4f}")

print(f"\nTraining mean (closed): T={closed['Temperature'].mean():.4f} "
      f"PMV={closed['PMV'].mean():.4f} Sat={closed['Satisfaction'].mean():.4f}")
