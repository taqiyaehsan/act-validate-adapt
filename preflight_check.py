#!/usr/bin/env python3
"""Preflight check: verify everything works before server deployment."""
import sys, os, json, subprocess, time
sys.path.insert(0, '.')

PASS = 0
FAIL = 0

def ok(msg):
    global PASS; PASS += 1; print(f"  ✓ {msg}")

def fail(msg):
    global FAIL; FAIL += 1; print(f"  ✗ {msg}")

def check(cond, msg):
    if cond: ok(msg)
    else: fail(msg)

print("=" * 60)
print("  POLICYGRID PREFLIGHT CHECK")
print("=" * 60)

# ── 1. File existence ─────────────────────────────────────────
print("\n1. Required files...")
for f in [
    'run_full_pipeline.py', 'neurips_final_experiments.py',
    'regime_aware_validation.py', 'generate_varied_data.py',
    'ground_truth_graphs.json', 'requirements.txt', 'package.json',
    'epluszsz.csv',
    'src/pipeline.py', 'src/pipeline_cwm.py', 'src/policy_engine.py',
    'src/causal_world_model.py', 'src/tester.py', 'src/generators.py',
    'src/evaluator.py', 'src/metrics.py', 'src/gpt_client.py',
    'src/ddpc_baseline.py', 'src/mpc_baseline.py', 'src/cmbpo_baseline.py',
    'js/smart_room.js', 'js/smart_room_noise.js',
    'js/smart_room_hidden_vars.js', 'js/open_window.js',
    'js/smart_building_rich.js',
    'benchmarks_new/pc_baseline.py', 'benchmarks_new/sam_baseline.py',
    'benchmarks_new/utils.py',
]:
    check(os.path.exists(f), f)

# ── 2. Training data ──────────────────────────────────────────
print("\n2. Training data...")
import pandas as pd
for name, path, min_rows in [
    ('smart_room', 'data_regen/smart_room_varied_processed.csv', 100),
    ('smart_room_noise', 'data_regen/smart_room_noise_varied_processed.csv', 100),
    ('hidden_vars', 'data_regen/smart_room_hidden_vars_varied_processed.csv', 100),
    ('open_window', 'data_regen/open_window_varied_processed.csv', 100),
    ('rich_building', 'data_regen/smart_building_rich_processed.csv', 100),
    ('ashrae', 'data/ashrae_data_processed.csv', 100),
]:
    if os.path.exists(path):
        df = pd.read_csv(path)
        check(len(df) >= min_rows, f"{name}: {len(df)} rows ({path})")
    else:
        fail(f"{name}: MISSING {path}")

# ── 3. Ground truth graphs ────────────────────────────────────
print("\n3. Ground truth graphs...")
gt = json.load(open('ground_truth_graphs.json'))
for key in ['smart_room', 'smart_room_noise', 'hidden_vars',
            'open_window', 'smart_building_rich', 'ashrae']:
    if key in gt:
        n = len(gt[key]['edges'])
        check(n > 0, f"{key}: {n} GT edges")
    else:
        fail(f"{key}: MISSING from ground_truth_graphs.json")

# ── 4. Python imports ─────────────────────────────────────────
print("\n4. Python imports...")
imports = [
    ('numpy', 'import numpy'),
    ('pandas', 'import pandas'),
    ('sklearn', 'from sklearn.linear_model import Ridge'),
    ('scipy', 'from scipy import stats'),
    ('networkx', 'import networkx'),
    ('causal-learn (PC)', 'from causallearn.search.ConstraintBased.PC import pc'),
    ('lingam (VARLiNGAM)', 'import lingam'),
    ('src.pipeline', 'from src.pipeline import CausalPipeline'),
    ('src.pipeline_cwm', 'from src.pipeline_cwm import create_cwm_pipeline'),
    ('src.policy_engine', 'from src.policy_engine import CausalPolicyEngine'),
    ('src.ddpc_baseline', 'from src.ddpc_baseline import create_ddpc_suite'),
    ('src.mpc_baseline', 'from src.mpc_baseline import LinearMPCController'),
    ('src.cmbpo_baseline', 'from src.cmbpo_baseline import CMBPOController'),
    ('run_full_pipeline', 'from run_full_pipeline import SIM_CONFIGS, run_full_pipeline'),
    ('benchmarks_new.pc_baseline', 'from benchmarks_new.pc_baseline import run_pc_baseline'),
]
for name, stmt in imports:
    try:
        exec(stmt)
        ok(name)
    except Exception as e:
        fail(f"{name}: {e}")

# ── 5. Node.js simulators ────────────────────────────────────
print("\n5. Node.js simulators...")
for sim_name, sim_path in [
    ('smart_room', 'js/smart_room.js'),
    ('smart_room_noise', 'js/smart_room_noise.js'),
    ('hidden_vars', 'js/smart_room_hidden_vars.js'),
    ('open_window', 'js/open_window.js'),
    ('smart_building_rich', 'js/smart_building_rich.js'),
]:
    try:
        r = subprocess.run(
            ['node', os.path.abspath(sim_path), '--single-step',
             '--elapsed-ms', '3600000'],
            capture_output=True, text=True, timeout=15)
        found = False
        for line in r.stdout.split('\n'):
            if 'RESULT:' in line:
                d = json.loads(line.split('RESULT:', 1)[1].strip())
                n_vars = len([k for k, v in d.items()
                             if isinstance(v, (int, float))])
                check(n_vars >= 3, f"{sim_name}: {n_vars} variables returned")
                found = True
                break
        if not found:
            fail(f"{sim_name}: no RESULT in output. stderr: {r.stderr[:100]}")
    except subprocess.TimeoutExpired:
        fail(f"{sim_name}: timeout (>15s)")
    except Exception as e:
        fail(f"{sim_name}: {e}")

# ── 6. SIM_CONFIGS consistency ────────────────────────────────
print("\n6. SIM_CONFIGS consistency...")
from run_full_pipeline import SIM_CONFIGS, load_scaling
for sim_name, cfg in SIM_CONFIGS.items():
    issues = []
    if cfg.get('data_path') and not os.path.exists(cfg['data_path']):
        issues.append(f"data missing: {cfg['data_path']}")
    if cfg.get('scaling_path') and not os.path.exists(cfg['scaling_path']):
        issues.append(f"scaling missing: {cfg['scaling_path']}")
    if cfg.get('sim_path') and not os.path.exists(cfg['sim_path']):
        issues.append(f"sim missing: {cfg['sim_path']}")
    if cfg.get('gt_key') and cfg['gt_key'] not in gt:
        issues.append(f"GT key '{cfg['gt_key']}' not in ground_truth_graphs.json")
    if issues:
        fail(f"{sim_name}: {'; '.join(issues)}")
    else:
        ok(f"{sim_name}: config OK")

# ── 7. Quick discovery smoke test ─────────────────────────────
print("\n7. Quick smoke test (PC on smart_room, ~5s)...")
try:
    from benchmarks_new.pc_baseline import run_pc_baseline
    data = pd.read_csv('data_regen/smart_room_varied_processed.csv')
    t0 = time.time()
    result = run_pc_baseline(data)
    elapsed = time.time() - t0
    if isinstance(result, tuple):
        edges = result[0]
    elif hasattr(result, 'edges'):
        edges = list(result.edges())
    else:
        edges = result
    check(len(edges) > 0, f"PC found {len(edges)} edges in {elapsed:.1f}s")
except Exception as e:
    fail(f"PC smoke test: {e}")

# ── 8. Policy engine smoke test ───────────────────────────────
print("\n8. Quick smoke test (PolicyEngine, ~2s)...")
try:
    from src.policy_engine import CausalPolicyEngine
    data = pd.read_csv('data_regen/smart_room_varied_processed.csv')
    edges = {('Temperature', 'Satisfaction'), ('Humidity', 'Satisfaction')}
    engine = CausalPolicyEngine(
        {'validated_edges': edges}, data,
        use_llm=False, action_vars=['Temperature', 'Humidity', 'AirQuality'])
    result = engine.optimize_policy(
        {'satisfaction': {'target': 70, 'weight': 0.7},
         'energy': {'target': 15, 'weight': 0.3}},
        {}, {'temperature': 0.5, 'humidity': 0.5, 'airquality': 0.5})
    check(hasattr(result, 'action_plan'), f"PolicyEngine OK: {result.action_plan}")
except Exception as e:
    fail(f"PolicyEngine smoke test: {e}")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  RESULTS: {PASS} passed, {FAIL} failed")
if FAIL == 0:
    print("  ✓ ALL CHECKS PASSED — ready for server deployment")
else:
    print(f"  ✗ {FAIL} ISSUES — fix before uploading")
print("=" * 60)
sys.exit(FAIL)
