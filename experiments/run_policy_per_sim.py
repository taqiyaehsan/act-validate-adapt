#!/usr/bin/env python3
"""
Per-Simulator Policy Comparison — ALL baselines
=================================================
Mirrors run_full_policy_comparison.py but for each simulator independently.
Uses GT edges for PolicyGRID (near-perfect discovery on simpler sims).

Baselines: PolicyGRID, PolicyGRID-O (obs=GT too for simple sims), PID,
           DDPC_Behavioral, DDPC_Subspace, DDPC_Neural, DDPC_PETS,
           MPC_Linear, C-MBPO

Usage:
    python run_policy_per_sim.py --sim smart_room
    python run_policy_per_sim.py --sim open_window
    python run_policy_per_sim.py --sim all
"""

import sys, os, json, subprocess, time, warnings, argparse
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATOR CONFIGS
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
        'sensor_vars': ['Temperature', 'Humidity', 'AirQuality'],
        'js_var_map': {
            'temperature': 'temperature', 'humidity': 'humidity',
            'airquality': 'airQuality', 'energyconsumption': 'energyConsumption',
            'satisfaction': 'overallSatisfaction',
        },
        'sat_key': 'overallSatisfaction',
        'energy_key': 'energyConsumption',
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
        'sensor_vars': ['Temperature', 'Humidity', 'AirQuality'],
        'js_var_map': {
            'temperature': 'temperature', 'humidity': 'humidity',
            'airquality': 'airQuality', 'energyconsumption': 'energyConsumption',
            'satisfaction': 'overallSatisfaction',
        },
        'sat_key': 'overallSatisfaction',
        'energy_key': 'energyConsumption',
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
        'sensor_vars': ['Temperature', 'Humidity', 'AirQuality'],
        'js_var_map': {
            'temperature': 'temperature', 'humidity': 'humidity',
            'airquality': 'airQuality', 'energyconsumption': 'energyConsumption',
            'satisfaction': 'overallSatisfaction',
        },
        'sat_key': 'overallSatisfaction', 'energy_key': 'energyConsumption',
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
        'sensor_vars': ['Temperature', 'Humidity', 'AirQuality', 'PMV'],
        'js_var_map': {
            'temperature': 'temperature', 'humidity': 'humidity',
            'airquality': 'airQuality', 'energyconsumption': 'energyConsumption',
            'satisfaction': 'overallSatisfaction', 'pmv': 'pmv',
            'windowopen': 'windowOpen',
            'outdoortemperature': 'outdoorTemperature',
        },
        'sat_key': 'overallSatisfaction',
        'energy_key': 'energyConsumption',
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


class PersistentSim:
    """Keeps a Node.js simulator process alive across steps.
    Same physics as --single-step, ~10x faster (no process respawn)."""

    def __init__(self, sim_path):
        self.proc = subprocess.Popen(
            ['node', 'js/persistent_runner.js', sim_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1)
        # Wait for READY
        while True:
            line = self.proc.stdout.readline().strip()
            if line == 'READY':
                break

    def step(self, intervention=None, state=None, elapsed_ms=None):
        cmd = {}
        if intervention:
            cmd['intervention'] = intervention
        if state:
            cmd['state'] = state
        if elapsed_ms is not None:
            cmd['elapsed_ms'] = elapsed_ms
        self.proc.stdin.write(json.dumps(cmd) + '\n')
        self.proc.stdin.flush()
        line = self.proc.stdout.readline().strip()
        if line.startswith('RESULT:'):
            return json.loads(line.split('RESULT:', 1)[1])
        return None

    def reset(self):
        self.proc.stdin.write(json.dumps({'reset': True}) + '\n')
        self.proc.stdin.flush()
        self.proc.stdout.readline()  # consume RESULT from reset step

    def close(self):
        try:
            self.proc.stdin.write('QUIT\n')
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def step_sim(persistent_sim, action_norm, smap, js_var_map, elapsed_ms=None):
    """Step using persistent sim. Returns (phys_state, norm_state)."""
    intervention = {}
    for var, val in action_norm.items():
        phys_val = norm_to_phys(val, var.lower(), smap)
        js_name = js_var_map.get(var.lower(), var)
        intervention[js_name] = phys_val

    new_phys = persistent_sim.step(intervention=intervention, elapsed_ms=elapsed_ms)
    if new_phys is None:
        return None, None

    new_norm = {}
    for k, v in new_phys.items():
        if isinstance(v, (int, float)):
            new_norm[k.lower()] = phys_to_norm(v, k.lower(), smap)
    return new_phys, new_norm


def run_episode(policy_fn, psim, cfg, smap, n_steps=120, start_hour=6.0):
    """Run one episode using the persistent sim."""
    records = []
    default_action = {v.lower(): 0.5 for v in cfg['action_vars']}
    state_phys = None

    for t in range(n_steps):
        # elapsed_ms relative to episode start (not time-of-day)
        # so window schedule (e.g., open at 30min) works within the episode
        elapsed_ms = t * 60 * 1000

        if state_phys is not None:
            state_norm = {}
            for k, v in state_phys.items():
                if isinstance(v, (int, float)):
                    state_norm[k.lower()] = phys_to_norm(v, k.lower(), smap)
            action = policy_fn(state_norm, t)
        else:
            action = dict(default_action)

        new_phys, new_norm = step_sim(psim, action, smap, cfg['js_var_map'],
                                       elapsed_ms=elapsed_ms)
        if new_phys is None:
            continue
        state_phys = new_phys
        rec = {
            'step': t,
            'satisfaction': new_phys.get(cfg['sat_key'], 50),
            'energy': new_phys.get(cfg['energy_key'], 50),
        }
        # Track regime info for open_window
        if 'windowOpen' in new_phys:
            rec['windowOpen'] = new_phys['windowOpen']
        if 'outdoorTemperature' in new_phys:
            rec['outdoorTemp'] = new_phys['outdoorTemperature']
        records.append(rec)
    return records


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURE SRC BASELINES FOR THIS SIM
# ═══════════════════════════════════════════════════════════════════════════════

def patch_baseline_modules(cfg):
    """Monkey-patch src/ baseline modules to use this sim's variable space."""
    import src.ddpc_baseline as ddpc
    import src.mpc_baseline as mpc
    import src.cmbpo_baseline as cmbpo

    sv = cfg['state_vars']
    av = cfg['action_vars']
    ov = cfg['output_vars']
    sens = cfg['sensor_vars']

    ddpc.STATE_VARS = list(sv)
    ddpc.ACTION_VARS = list(av)
    ddpc.OUTPUT_VARS = list(ov)
    ddpc.N_U = len(av)
    ddpc.N_Y = len(ov)

    mpc.STATE_VARS = list(sv)
    mpc.ACTION_VARS = list(av)
    mpc.OUTPUT_VARS = list(ov)
    mpc.N_U = len(av)
    mpc.N_Y = len(ov)
    mpc.N_X = len(sv)

    cmbpo.ACTION_VARS = list(av)
    cmbpo.OUTPUT_VARS = list(ov)
    cmbpo.SENSOR_VARS = list(sens)
    cmbpo.N_U = len(av)
    cmbpo.N_Y = len(ov)


# ═══════════════════════════════════════════════════════════════════════════════
# POLICY BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def make_grid_policy(gt_edges, data, action_vars, comfort_target=0.8):
    from src.policy_engine import CausalPolicyEngine

    # Quadratic SEM, no forced directions — let the SEM learn from varied data.
    # λ=5 keeps optimizer near comfort zone (tuned across sims).
    try:
        engine = CausalPolicyEngine(
            {'validated_edges': gt_edges}, data,
            use_llm=False, action_vars=action_vars,
            reg_lambda=5.0, n_gradient_steps=1,
            linear_only=False,
        )
    except Exception as e:
        logger.warning(f"  PolicyEngine init failed: {e}")
        return None

    sat_weight = max(0.3, min(0.85, 1.0 - comfort_target * 0.35))
    eng_weight = 1.0 - sat_weight

    def policy_fn(state_obs, t):
        try:
            col_map = {v.lower(): v for v in data.columns}
            state_proper = {col_map.get(k, k): v for k, v in state_obs.items()}
            result = engine.optimize_policy(
                objectives={'satisfaction': {'target': 70, 'weight': sat_weight},
                            'energy': {'target': 80, 'weight': eng_weight}},
                constraints={}, current_state=state_proper)
            return {k.lower(): max(0.0, min(1.0, v))
                    for k, v in result.action_plan.items()}
        except Exception:
            return {v.lower(): 0.5 for v in action_vars}
    return policy_fn


def make_pid_policy(action_vars):
    def policy_fn(state_obs, t):
        action = {}
        for v in action_vars:
            current = state_obs.get(v.lower(), 0.5)
            action[v.lower()] = max(0.0, min(1.0, 0.5 + 1.0 * (0.5 - current)))
        return action
    return policy_fn


def _wrap_baseline(ctrl, comfort_target, action_vars):
    def policy_fn(state_obs, t):
        if t == 0 and hasattr(ctrl, 'reset'):
            ctrl.reset()
        action = ctrl.get_action(state_obs, comfort_target)
        return {k.lower(): v for k, v in action.items()}
    return policy_fn


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def run_sim_comparison(sim_name):
    cfg = SIM_CONFIGS[sim_name]
    t_start = time.time()

    logger.info(f"\n{'='*70}")
    logger.info(f"  POLICY COMPARISON: {sim_name}")
    logger.info(f"{'='*70}")

    # Load data
    data = pd.read_csv(cfg['data_path'])
    smap = load_scaling(cfg['scaling_path'])

    # Load GT
    with open('ground_truth_graphs.json') as f:
        gt_raw = json.load(f)[cfg['gt_key']]['edges']
    gt_edges = {(s, t) for s, t in gt_raw}

    logger.info(f"  Data: {data.shape}, GT: {len(gt_edges)} edges")
    logger.info(f"  Actions: {cfg['action_vars']}")

    # Patch module-level vars for this sim
    patch_baseline_modules(cfg)

    # Fit ALL baselines
    from src.ddpc_baseline import create_ddpc_suite
    from src.mpc_baseline import LinearMPCController
    from src.cmbpo_baseline import CMBPOController

    logger.info("  Fitting baselines...")
    ddpc_suite = create_ddpc_suite(data)
    mpc_linear = LinearMPCController().fit(data)
    cmbpo = CMBPOController(edges=gt_edges)
    cmbpo.fit(data)  # offline only, no sim access needed

    pid_fn = make_pid_policy(cfg['action_vars'])
    logger.info("  All baselines fitted.\n")

    # Start persistent sim (one Node process for all episodes)
    psim = PersistentSim(cfg['sim_path'])
    logger.info("  Persistent sim started.\n")

    # Sweep
    policy_rows = []

    for ct_idx, ct in enumerate(COMFORT_TARGETS):
        logger.info(f"  ── ε={ct} ({ct_idx+1}/{len(COMFORT_TARGETS)}) ──")

        grid_fn = make_grid_policy(gt_edges, data, cfg['action_vars'], comfort_target=ct)

        arms = {}
        if ct_idx == 0:
            arms['PID'] = pid_fn
        if grid_fn:
            arms['PolicyGRID'] = grid_fn
        arms['C-MBPO'] = _wrap_baseline(cmbpo, ct, cfg['action_vars'])
        for name in ['DDPC_Behavioral', 'DDPC_Subspace']:
            if name in ddpc_suite:
                arms[name] = _wrap_baseline(ddpc_suite[name], ct, cfg['action_vars'])
        arms['MPC_Linear'] = _wrap_baseline(mpc_linear, ct, cfg['action_vars'])
        for name in ['DDPC_Neural', 'DDPC_PETS']:
            if name in ddpc_suite:
                arms[name] = _wrap_baseline(ddpc_suite[name], ct, cfg['action_vars'])

        for arm_name, pol_fn in arms.items():
            sats, engs = [], []
            for run_i in range(N_RUNS):
                psim.reset()  # fresh sim state per episode
                recs = run_episode(pol_fn, psim, cfg, smap, n_steps=N_STEPS)
                if recs:
                    sats.append(np.mean([r['satisfaction'] for r in recs]))
                    engs.append(np.mean([r['energy'] for r in recs]))

            if sats:
                policy_rows.append({
                    'method': arm_name, 'comfort_target': ct,
                    'satisfaction': round(np.mean(sats), 1),
                    'energy': round(np.mean(engs), 1),
                })
                logger.info(f"    {arm_name:<20} Sat={np.mean(sats):.1f}  Eng={np.mean(engs):.1f}")

        # Duplicate PID for other ε
        if ct_idx > 0:
            pid_rows = [r for r in policy_rows
                        if r['method'] == 'PID' and r['comfort_target'] == COMFORT_TARGETS[0]]
            for pr in pid_rows:
                policy_rows.append({**pr, 'comfort_target': ct})

    psim.close()

    df = pd.DataFrame(policy_rows)
    out_dir = f'results/policy_{sim_name}'
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(f'{out_dir}/policy_metrics.csv', index=False)

    elapsed = time.time() - t_start
    logger.info(f"\n  {'='*50}")
    logger.info(f"  RESULTS: {sim_name} ({elapsed/60:.1f} min)")
    logger.info(f"  {'='*50}")
    logger.info(f"  {'Method':<20} {'ε':>4}  {'Sat':>6}  {'Eng':>6}")
    logger.info(f"  {'-'*40}")
    for _, r in df.iterrows():
        logger.info(f"  {r['method']:<20} {r['comfort_target']:>4.1f}  "
                     f"{r['satisfaction']:>6.1f}  {r['energy']:>6.1f}")

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sim', default='smart_room',
                        choices=list(SIM_CONFIGS.keys()) + ['all'])
    args = parser.parse_args()

    if args.sim == 'all':
        for sim_name in SIM_CONFIGS:
            run_sim_comparison(sim_name)
    else:
        run_sim_comparison(args.sim)


if __name__ == '__main__':
    main()
