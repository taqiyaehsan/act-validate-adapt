#!/usr/bin/env python
"""Ultra-quick policy smoke test: 1 episode, 1 comfort target, all sims."""

import os, sys, time, json
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))

from experiments_neurips import NeurIPSExperiments

# Fixed validated edges (from neurips_results_old/app.log seed 0)
# + WindowOpen/OutdoorTemperature edges (real causal structure that a
#   proper CWM discovery run would find — needed to deconfound AQ→Sat)
VALIDATED_EDGES = [
    ('airquality', 'satisfaction'),
    ('temperature', 'satisfaction'),
    ('temperature', 'airquality'),
    ('temperature', 'pmv'),
    ('airquality', 'energyconsumption'),
    ('temperature', 'energyconsumption'),
    ('humidity', 'airquality'),
    ('humidity', 'energyconsumption'),
    ('humidity', 'satisfaction'),
    ('temperature', 'humidity'),
    # Window/outdoor edges — deconfound the AQ↔Satisfaction relationship
    ('windowopen', 'airquality'),
    ('windowopen', 'temperature'),
    ('windowopen', 'energyconsumption'),
    ('outdoortemperature', 'temperature'),
]


# Intervention-observed effect directions.
# NOTE: AirQuality is AQI (pollution index) — higher = worse.
#   SEM's β=-0.77 for AQ→Sat is the true causal effect, not confounding.
# Only constrain directions we are certain about from simulator physics.
INTERVENTION_DIRECTIONS = {
    ('temperature', 'satisfaction'): +1,      # warmer (in range) → higher satisfaction
}


class FakePipeline:
    def __init__(self, edges):
        self.validated_edges = set(edges)

    def export_for_policy_engine(self):
        return {
            'validated_edges': list(self.validated_edges),
            'intervention_directions': INTERVENTION_DIRECTIONS,
        }

    def get_method_dags(self):
        import networkx as nx
        G = nx.DiGraph()
        G.add_edges_from(self.validated_edges)
        return {'fixed': G}


SIM_CONFIGS = {
    'smart_room': {
        'sim_path': 'js/smart_room.js',
        'obs_data_path': 'data_regen/smart_room_processed.csv',
        'scaling_path': 'data_regen/smart_room_processed_scaling.csv',
        'dataset_type': 'open_window',
        'data_in_physical_units': False,
    },
    'hidden_vars': {
        'sim_path': 'js/smart_room_hidden_vars.js',
        'obs_data_path': 'data_regen/hidden_vars_processed.csv',
        'scaling_path': 'data_regen/hidden_vars_processed_scaling.csv',
        'dataset_type': 'open_window',
        'data_in_physical_units': False,
    },
    'noise': {
        'sim_path': 'js/smart_room_noise.js',
        'obs_data_path': 'data_regen/smart_room_noise_processed.csv',
        'scaling_path': 'data_regen/smart_room_noise_processed_scaling.csv',
        'dataset_type': 'open_window',
        'data_in_physical_units': False,
    },
}

CT = 1.0  # single comfort target
N_EP = 1  # single episode


def load_scaling(path):
    if not path or not os.path.exists(path):
        return None
    sc = pd.read_csv(path).set_index('feature')
    return {row: {'min': sc.loc[row, 'data_min'], 'max': sc.loc[row, 'data_max']}
            for row in sc.index}


def collect_intervention_data(exp, n_per_var=20):
    """Run controlled single-variable interventions via the simulator.

    Mimics what the CWM pipeline collects during discovery iterations:
    for each action variable, sweep it across its range while holding
    others at multiple base contexts.  Returns a DataFrame in the same
    [0,1]-normalised scale as the observational data.
    """
    rows = []
    # Multiple base states to cover different operating regimes
    base_seeds = [0, 7, 21, 42, 99]

    for seed in base_seeds:
        base_state = exp._get_initial_state({'seed': seed})
        base_action = dict(exp.ashrae_setpoints)

        for var in ['Temperature', 'Humidity', 'AirQuality']:
            for level in np.linspace(0.05, 0.95, n_per_var):
                action = dict(base_action)
                action[var] = level
                ns = exp._step_simulator(base_state, action, {'seed': seed}, 0)
                rows.append({
                    'Temperature': ns.get('Temperature', 0.5),
                    'Humidity': ns.get('Humidity', 0.5),
                    'AirQuality': ns.get('AirQuality', 0.5),
                    # Sim returns [0,100]; obs data is [0,1]
                    'Satisfaction': ns.get('Satisfaction',
                                           ns.get('OverallSatisfaction', 50)) / 100.0,
                    'EnergyConsumption': ns.get('EnergyConsumption', 50) / 100.0,
                })

    df = pd.DataFrame(rows)
    print(f"  Collected {len(df)} intervention samples "
          f"(sat range [{df['Satisfaction'].min():.2f}, {df['Satisfaction'].max():.2f}])")
    return df


def run_sim(label, cfg):
    print(f"\n{'='*60}")
    print(f"  {label}  (ε={CT}, {N_EP} ep)")
    print(f"{'='*60}")

    obs_df = pd.read_csv(cfg['obs_data_path'])
    scaling = load_scaling(cfg.get('scaling_path'))
    sim_path = cfg['sim_path'] if cfg['sim_path'] and os.path.exists(cfg['sim_path']) else None

    # Phase 1: build a temporary experiment to collect intervention data
    tmp_exp = NeurIPSExperiments(
        pipeline=FakePipeline(VALIDATED_EDGES),
        dataset=obs_df,
        smart_room_path=sim_path,
        comfort_target=CT,
        obs_only_edges=VALIDATED_EDGES,
        h0_edges=None, h1_edges=None,
        dataset_type=cfg['dataset_type'],
        scaling_ranges=scaling,
        data_in_physical_units=cfg['data_in_physical_units'],
    )
    int_data = collect_intervention_data(tmp_exp)

    # Phase 2: build combined dataset (obs + upweighted intervention rows)
    # Per the CWM framework: intervention data is added to the observational
    # dataset with higher weight before fitting SEMs.
    combined_df = pd.concat([
        obs_df.assign(weight=1.0),
        int_data.assign(weight=2.0),
    ], ignore_index=True)

    exp = NeurIPSExperiments(
        pipeline=FakePipeline(VALIDATED_EDGES),
        dataset=combined_df,              # GRID gets combined data for SEM
        smart_room_path=sim_path,
        comfort_target=CT,
        obs_only_edges=VALIDATED_EDGES,
        h0_edges=None, h1_edges=None,
        dataset_type=cfg['dataset_type'],
        scaling_ranges=scaling,
        data_in_physical_units=cfg['data_in_physical_units'],
        obs_only_dataset=obs_df,          # GRID_ObsOnly gets obs-only data
    )
    exp._build_policy_arms()

    # Remove slow arms
    for slow in ['DDPC_PETS']:
        exp.policy_arms.pop(slow, None)

    print(f"  Arms: {list(exp.policy_arms.keys())}")

    # Also dump SEM coefficients for GRID's engine
    engine = exp._policy_engine
    if engine:
        print(f"\n  SEM coefficients (with interaction terms):")
        for target, sem_entry in engine._sem.items():
            intercept, lin_c, quad_c, inter_c, r2 = sem_entry
            print(f"    {target}: R²={r2:.4f}")
            for p in sorted(lin_c):
                grad_at_05 = lin_c[p] + 2 * quad_c.get(p, 0) * 0.5
                print(f"      {p}: β={lin_c[p]:+.4f} γ={quad_c.get(p,0):+.4f} ∂/∂|0.5={grad_at_05:+.4f}")
            for (pi, pj), d in inter_c.items():
                print(f"      {pi}×{pj}: δ={d:+.4f}")

    rows = []
    static_arms = {'ASHRAE', 'Static_Baseline'}

    for arm_name, fn in exp.policy_arms.items():
        t0 = time.time()
        sat, eng, cv, cvr, kwh, dh = \
            exp._evaluate_policy_episode_with_cv(fn, arm_name, 0)
        mo = 0.6 * (sat / 100) + 0.4 * (1 - eng / 100)
        wall = time.time() - t0
        rows.append({
            'arm': arm_name, 'sat': sat, 'eng': eng, 'mo': mo,
            'kwh': kwh, 'dh': dh, 'wall_s': wall,
        })
        print(f"  {arm_name:25s}  sat={sat:5.1f}  eng={eng:5.1f}  "
              f"MO={mo:.4f}  kWh={kwh:.1f}  DH={dh:.1f}  ({wall:.1f}s)")

    return pd.DataFrame(rows)


def main():
    for label, cfg in SIM_CONFIGS.items():
        if not os.path.exists(cfg['obs_data_path']):
            print(f"  Skipping {label}: data not found")
            continue
        run_sim(label, cfg)

    print("\nDone.")


if __name__ == '__main__':
    main()
