#!/usr/bin/env python3
"""
Unified Policy Comparison for ALL simulators.
==============================================
Uses discovered edges from run_discovery_monitoring.py (or GT edges as fallback).
Runs PolicyGRID + all baselines across epsilon-constraint sweep.

Usage:
    python run_policy_all.py                                   # all sims
    python run_policy_all.py --sim smart_building_rich          # one sim
    python run_policy_all.py --sim smart_room --edges gt        # use GT edges
    python run_policy_all.py --sim all --edges discovered       # use discovered edges

Output: results/policy/<sim_name>/
    - policy_metrics.csv
    - summary.txt
    - figures/
"""

import sys, os, json, subprocess, time, warnings, argparse
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

from src.policy_engine import CausalPolicyEngine

# ═══════════════════════════════════════════════════════════════════════════════
# SIM CONFIGS
# ═══════════════════════════════════════════════════════════════════════════════

SIM_CONFIGS = {
    'smart_room': {
        'sim_path': 'js/smart_room.js',
        'data_path': 'data_regen/smart_room_varied_processed.csv',
        'scaling_path': 'data_regen/smart_room_varied_scaling.csv',
        'gt_key': 'smart_room',
        'action_vars': ['Temperature', 'Humidity', 'AirQuality'],
        'output_vars': ['EnergyConsumption', 'Satisfaction'],
        'state_vars': ['Temperature', 'Humidity', 'AirQuality',
                       'EnergyConsumption', 'Satisfaction'],
        'js_var_map': {'temperature': 'temperature', 'humidity': 'humidity',
                       'airquality': 'airQuality', 'energyconsumption': 'energyConsumption',
                       'satisfaction': 'overallSatisfaction'},
        'sat_key': 'overallSatisfaction', 'energy_key': 'energyConsumption',
        'policy_config': {'linear_only': False, 'lambda': 5.0,
                         'intervention_directions': None},
    },
    'smart_room_noise': {
        'sim_path': 'js/smart_room_noise.js',
        'data_path': 'data_regen/smart_room_noise_varied_processed.csv',
        'scaling_path': 'data_regen/smart_room_noise_varied_scaling.csv',
        'gt_key': 'smart_room_noise',
        'action_vars': ['Temperature', 'Humidity', 'AirQuality'],
        'output_vars': ['EnergyConsumption', 'Satisfaction'],
        'state_vars': ['Temperature', 'Humidity', 'AirQuality',
                       'EnergyConsumption', 'Satisfaction'],
        'js_var_map': {'temperature': 'temperature', 'humidity': 'humidity',
                       'airquality': 'airQuality', 'energyconsumption': 'energyConsumption',
                       'satisfaction': 'overallSatisfaction'},
        'sat_key': 'overallSatisfaction', 'energy_key': 'energyConsumption',
        'policy_config': {'linear_only': False, 'lambda': 5.0,
                         'intervention_directions': None},
    },
    'smart_room_hidden_vars': {
        'sim_path': 'js/smart_room_hidden_vars.js',
        'data_path': 'data_regen/smart_room_hidden_vars_varied_processed.csv',
        'scaling_path': 'data_regen/smart_room_hidden_vars_varied_scaling.csv',
        'gt_key': 'hidden_vars',
        'action_vars': ['Temperature', 'Humidity', 'AirQuality'],
        'output_vars': ['EnergyConsumption', 'Satisfaction'],
        'state_vars': ['Temperature', 'Humidity', 'AirQuality',
                       'EnergyConsumption', 'Satisfaction'],
        'js_var_map': {'temperature': 'temperature', 'humidity': 'humidity',
                       'airquality': 'airQuality', 'energyconsumption': 'energyConsumption',
                       'satisfaction': 'overallSatisfaction'},
        'sat_key': 'overallSatisfaction', 'energy_key': 'energyConsumption',
        'policy_config': {'linear_only': False, 'lambda': 5.0,
                         'intervention_directions': None},
    },
    'open_window': {
        'sim_path': 'js/open_window.js',
        'data_path': 'data_regen/open_window_varied_processed.csv',
        'scaling_path': 'data_regen/open_window_varied_scaling.csv',
        'gt_key': 'open_window',
        'action_vars': ['Temperature', 'Humidity', 'AirQuality'],
        'output_vars': ['EnergyConsumption', 'Satisfaction'],
        'state_vars': ['Temperature', 'Humidity', 'AirQuality', 'PMV',
                       'EnergyConsumption', 'Satisfaction', 'WindowOpen',
                       'OutdoorTemperature'],
        'js_var_map': {'temperature': 'temperature', 'humidity': 'humidity',
                       'airquality': 'airQuality', 'energyconsumption': 'energyConsumption',
                       'satisfaction': 'overallSatisfaction', 'pmv': 'pmv',
                       'windowopen': 'windowOpen', 'outdoortemperature': 'outdoorTemperature'},
        'sat_key': 'overallSatisfaction', 'energy_key': 'energyConsumption',
        'policy_config': {'linear_only': False, 'lambda': 5.0,
                         'intervention_directions': None},
    },
    'smart_building_rich': {
        'sim_path': 'js/smart_building_rich.js',
        'data_path': 'data_regen/smart_building_rich_processed.csv',
        'scaling_path': 'data_regen/smart_building_rich_processed_scaling.csv',
        'gt_key': 'smart_building_rich',
        'action_vars': ['HVACPower', 'LightingPower'],
        'output_vars': ['EnergyConsumption', 'Satisfaction'],
        'state_vars': ['OutdoorTemp', 'SolarRadiation', 'Temperature', 'Humidity',
                       'CO2', 'LightLevel', 'NoiseDB', 'AirQuality', 'PMV',
                       'HVACPower', 'LightingPower', 'EnergyConsumption', 'Satisfaction'],
        'js_var_map': {'hvacpower': 'hvacPower', 'lightingpower': 'lightingPower'},
        'sat_key': 'satisfaction', 'energy_key': 'energyConsumption',
        'latent_vars': {'occupancy', 'windowposition'},
        'policy_config': {
            'linear_only': True, 'lambda': 0.5,
            'intervention_directions': {
                ('HVACPower', 'Temperature'): +1,
                ('HVACPower', 'EnergyConsumption'): +1,
                ('LightingPower', 'LightLevel'): +1,
                ('LightingPower', 'EnergyConsumption'): +1,
            },
        },
    },
}

COMFORT_TARGETS = [0.5, 0.8, 1.0, 1.5, 2.0]
N_RUNS = 3
N_STEPS = 120


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_scaling(path):
    sp = pd.read_csv(path)
    feat_col = 'feature' if 'feature' in sp.columns else 'column'
    min_col = 'data_min' if 'data_min' in sp.columns else 'min'
    max_col = 'data_max' if 'data_max' in sp.columns else 'max'
    smap = {}
    for _, row in sp.iterrows():
        smap[row[feat_col].lower()] = (row[min_col], row[max_col])
    return smap


def norm_to_phys(val, var, smap):
    if var in smap:
        mn, mx = smap[var]
        return mn + val * (mx - mn)
    return val


def phys_to_norm(val, var, smap):
    if var in smap:
        mn, mx = smap[var]
        return (val - mn) / (mx - mn + 1e-12)
    return val


def step_sim(sim_path, state_phys, action_norm, smap, js_var_map, elapsed_ms=None):
    """Step sim via --single-step CLI."""
    sim_abs = os.path.abspath(sim_path)
    intervention = {}
    for var, val in action_norm.items():
        phys_val = norm_to_phys(val, var.lower(), smap)
        js_name = js_var_map.get(var.lower(), var)
        intervention[js_name] = phys_val

    cmd = ['node', sim_abs, '--single-step']
    if elapsed_ms is not None:
        cmd += ['--elapsed-ms', str(int(elapsed_ms))]
    if state_phys:
        cmd += ['--state', json.dumps(state_phys)]
    if intervention:
        cmd += ['--intervention', json.dumps(intervention)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except Exception:
        return None, None

    for line in result.stdout.split('\n'):
        if 'RESULT:' in line:
            new_phys = json.loads(line.split('RESULT:', 1)[1].strip())
            new_norm = {}
            for k, v in new_phys.items():
                if isinstance(v, (int, float)):
                    new_norm[k.lower()] = phys_to_norm(v, k.lower(), smap)
            return new_phys, new_norm
    return None, None


def run_episode(policy_fn, cfg, smap, n_steps=N_STEPS, start_hour=6.0):
    """Run one policy episode."""
    state_phys = None
    records = []
    default_action = {v.lower(): 0.5 for v in cfg['action_vars']}
    latent = cfg.get('latent_vars', set())

    for t in range(n_steps):
        elapsed_ms = t * 60 * 1000

        if state_phys is not None:
            state_norm = {}
            for k, v in state_phys.items():
                if isinstance(v, (int, float)):
                    state_norm[k.lower()] = phys_to_norm(v, k.lower(), smap)
            # Filter latent vars for policy input
            state_obs = {k: v for k, v in state_norm.items() if k not in latent}
            action = policy_fn(state_obs, t)
        else:
            action = dict(default_action)

        new_phys, new_norm = step_sim(cfg['sim_path'], state_phys, action,
                                       smap, cfg['js_var_map'], elapsed_ms=elapsed_ms)
        if new_phys is None:
            continue
        state_phys = new_phys
        records.append({
            'step': t,
            'satisfaction': new_phys.get(cfg['sat_key'], 50),
            'energy': new_phys.get(cfg['energy_key'], 50),
        })
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# POLICY BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_grid_policy(edges, data, cfg, comfort_target):
    """Build PolicyGRID policy from edges."""
    pcfg = cfg['policy_config']
    kwargs = {
        'use_llm': False,
        'action_vars': cfg['action_vars'],
        'reg_lambda': pcfg['lambda'],
        'n_gradient_steps': 1,
        'linear_only': pcfg['linear_only'],
    }
    if pcfg.get('intervention_directions'):
        kwargs['intervention_directions'] = pcfg['intervention_directions']

    try:
        engine = CausalPolicyEngine({'validated_edges': edges}, data, **kwargs)
    except Exception as e:
        logger.warning(f"    PolicyEngine init failed: {e}")
        return None

    sat_weight = max(0.3, min(0.85, 1.0 - comfort_target * 0.35))
    eng_weight = 1.0 - sat_weight
    cm = {v.lower(): v for v in data.columns}

    def policy_fn(state_obs, t):
        try:
            state_proper = {cm.get(k, k): v for k, v in state_obs.items()}
            result = engine.optimize_policy(
                objectives={'satisfaction': {'target': 70, 'weight': sat_weight},
                            'energy': {'target': 15, 'weight': eng_weight}},
                constraints={}, current_state=state_proper)
            return {k.lower(): max(0.0, min(1.0, v))
                    for k, v in result.action_plan.items()}
        except Exception:
            return {v.lower(): 0.5 for v in cfg['action_vars']}

    return policy_fn


def make_pid_policy(action_vars):
    def policy_fn(state_obs, t):
        return {v.lower(): max(0, min(1, 0.5 + 1.0 * (0.5 - state_obs.get(v.lower(), 0.5))))
                for v in action_vars}
    return policy_fn


def make_baselines(data, cfg, edges):
    """Build all baselines, patching module vars for this sim."""
    import src.ddpc_baseline as ddpc
    import src.mpc_baseline as mpc
    import src.cmbpo_baseline as cmbpo

    sv = cfg['state_vars']
    av = cfg['action_vars']
    ov = cfg['output_vars']

    ddpc.STATE_VARS = [v for v in sv]; ddpc.ACTION_VARS = [v for v in av]
    ddpc.OUTPUT_VARS = [v for v in ov]; ddpc.N_U = len(av); ddpc.N_Y = len(ov)

    mpc.STATE_VARS = [v for v in sv]; mpc.ACTION_VARS = [v for v in av]
    mpc.OUTPUT_VARS = [v for v in ov]; mpc.N_U = len(av); mpc.N_Y = len(ov)
    mpc.N_X = len(sv)

    cmbpo.ACTION_VARS = [v for v in av]; cmbpo.OUTPUT_VARS = [v for v in ov]
    cmbpo.SENSOR_VARS = [v for v in av]; cmbpo.N_U = len(av); cmbpo.N_Y = len(ov)

    from src.ddpc_baseline import create_ddpc_suite
    from src.mpc_baseline import LinearMPCController
    from src.cmbpo_baseline import CMBPOController

    baselines = {}
    try:
        suite = create_ddpc_suite(data)
        for name, ctrl in suite.items():
            baselines[name] = ctrl
    except Exception as e:
        logger.warning(f"    DDPC suite failed: {e}")

    try:
        baselines['MPC_Linear'] = LinearMPCController().fit(data)
    except Exception as e:
        logger.warning(f"    MPC failed: {e}")

    try:
        cm = CMBPOController(edges=edges)
        cm.fit(data)
        baselines['C-MBPO'] = cm
    except Exception as e:
        logger.warning(f"    C-MBPO failed: {e}")

    return baselines


def wrap_baseline(ctrl, comfort_target, action_vars):
    def policy_fn(state_obs, t):
        if t == 0 and hasattr(ctrl, 'reset'):
            ctrl.reset()
        action = ctrl.get_action(state_obs, comfort_target)
        return {k.lower(): v for k, v in action.items()}
    return policy_fn


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_sim_policy(sim_name, edge_source='discovered'):
    cfg = SIM_CONFIGS[sim_name]
    out_dir = f'results/policy/{sim_name}'
    os.makedirs(out_dir, exist_ok=True)

    logger.info(f"\n{'='*70}")
    logger.info(f"  POLICY: {sim_name}")
    logger.info(f"{'='*70}")

    if not os.path.exists(cfg['data_path']):
        logger.error(f"  Data not found: {cfg['data_path']}")
        return
    if cfg['sim_path'] is None:
        logger.info(f"  No simulator — skipping policy evaluation")
        return

    data = pd.read_csv(cfg['data_path'])
    smap = load_scaling(cfg['scaling_path'])

    # Load edges
    if edge_source == 'discovered':
        disc_path = f'results/discovery_monitoring/{sim_name}/discovered_edges.json'
        if os.path.exists(disc_path):
            disc = json.load(open(disc_path))
            edges = {(s, t) for s, t in disc['consensus_validated']}
            logger.info(f"  Using discovered edges: {len(edges)} (from {disc_path})")
        else:
            logger.warning(f"  Discovered edges not found at {disc_path}, falling back to GT")
            edge_source = 'gt'

    if edge_source == 'gt':
        with open('ground_truth_graphs.json') as f:
            gt_raw = json.load(f)[cfg['gt_key']]['edges']
        edges = {(s, t) for s, t in gt_raw}
        logger.info(f"  Using GT edges: {len(edges)}")

    logger.info(f"  Data: {data.shape}, Actions: {cfg['action_vars']}")

    # Fit baselines
    logger.info("  Fitting baselines...")
    baselines = make_baselines(data, cfg, edges)
    pid_fn = make_pid_policy(cfg['action_vars'])
    logger.info(f"  Baselines fitted: {list(baselines.keys())}")

    # Sweep
    t_start = time.time()
    policy_rows = []

    for ct_idx, ct in enumerate(COMFORT_TARGETS):
        logger.info(f"\n  ── ε={ct} ({ct_idx+1}/{len(COMFORT_TARGETS)}) ──")

        grid_fn = make_grid_policy(edges, data, cfg, comfort_target=ct)

        arms = {}
        if ct_idx == 0:
            arms['PID'] = pid_fn
        if grid_fn:
            arms['PolicyGRID'] = grid_fn
        for name, ctrl in baselines.items():
            arms[name] = wrap_baseline(ctrl, ct, cfg['action_vars'])

        for arm_name, pol_fn in arms.items():
            sats, engs = [], []
            for run_i in range(N_RUNS):
                recs = run_episode(pol_fn, cfg, smap, n_steps=N_STEPS,
                                   start_hour=6.0 + run_i * 2)
                if recs:
                    sats.append(np.mean([r['satisfaction'] for r in recs]))
                    engs.append(np.mean([r['energy'] for r in recs]))

            if sats:
                policy_rows.append({
                    'method': arm_name, 'comfort_target': ct,
                    'satisfaction': round(np.mean(sats), 1),
                    'energy': round(np.mean(engs), 1),
                    'sat_std': round(np.std(sats), 2),
                    'eng_std': round(np.std(engs), 2),
                })
                logger.info(f"    {arm_name:<20} Sat={np.mean(sats):.1f}±{np.std(sats):.1f}  "
                            f"Eng={np.mean(engs):.1f}")

        # Duplicate PID
        if ct_idx > 0:
            pid_rows = [r for r in policy_rows
                        if r['method'] == 'PID' and r['comfort_target'] == COMFORT_TARGETS[0]]
            for pr in pid_rows:
                policy_rows.append({**pr, 'comfort_target': ct})

    elapsed = time.time() - t_start
    df = pd.DataFrame(policy_rows)
    df.to_csv(f'{out_dir}/policy_metrics.csv', index=False)

    # Summary
    logger.info(f"\n  {'='*50}")
    logger.info(f"  RESULTS: {sim_name} ({elapsed/60:.1f} min)")
    logger.info(f"  {'='*50}")
    logger.info(f"  {'Method':<20} {'ε':>4}  {'Sat':>6}  {'Eng':>6}")
    logger.info(f"  {'-'*40}")
    for _, r in df.iterrows():
        logger.info(f"  {r['method']:<20} {r['comfort_target']:>4.1f}  "
                    f"{r['satisfaction']:>6.1f}  {r['energy']:>6.1f}")

    # Save summary
    with open(f'{out_dir}/summary.txt', 'w') as f:
        f.write(f"Policy Comparison: {sim_name}\n")
        f.write(f"Edge source: {edge_source} ({len(edges)} edges)\n")
        f.write(f"Runtime: {elapsed/60:.1f} min\n\n")
        f.write(df.to_string(index=False))

    logger.info(f"\n  Results saved to {out_dir}/")
    return df


def main():
    parser = argparse.ArgumentParser(description='Run policy comparison for all sims')
    parser.add_argument('--sim', default='all',
                        choices=list(SIM_CONFIGS.keys()) + ['all'])
    parser.add_argument('--edges', default='discovered',
                        choices=['discovered', 'gt'],
                        help='Edge source: discovered (from discovery run) or gt')
    args = parser.parse_args()

    t_start = time.time()

    if args.sim == 'all':
        for sim_name in SIM_CONFIGS:
            if SIM_CONFIGS[sim_name]['sim_path'] is not None:
                run_sim_policy(sim_name, args.edges)
    else:
        run_sim_policy(args.sim, args.edges)

    elapsed = time.time() - t_start
    logger.info(f"\n{'='*70}")
    logger.info(f"  TOTAL TIME: {elapsed/60:.1f} minutes")
    logger.info(f"{'='*70}")


if __name__ == '__main__':
    main()
