#!/usr/bin/env python3
"""
Pre-flight check — run on the server before experiments to verify
that all files, dependencies, and simulators are in place.

Usage: python preflight_check.py
"""

import sys
import os
import importlib
import subprocess
import shutil
import json
import csv

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

passed = 0
failed = 0
warnings = 0


def ok(msg):
    global passed
    passed += 1
    print(f"  [OK]   {msg}")


def fail(msg):
    global failed
    failed += 1
    print(f"  [FAIL] {msg}")


def warn(msg):
    global warnings
    warnings += 1
    print(f"  [WARN] {msg}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Required files
# ══════════════════════════════════════════════════════════════════════════════
print("\n1. Checking required files...")

REQUIRED_FILES = {
    # Core framework (src/)
    "src/__init__.py": "src package init",
    "src/pipeline.py": "base causal pipeline",
    "src/pipeline_cwm.py": "CWM-enhanced pipeline",
    "src/generators.py": "causal discovery algorithms (PC, SAM, LLM, VARLiNGAM)",
    "src/tester.py": "intervention validation (do-operator)",
    "src/evaluator.py": "edge ranking",
    "src/causal_world_model.py": "CWM: confidence, hypotheses, Bayesian filter",
    "src/policy_engine.py": "SEM policy engine (quadratic + interactions)",
    "src/ddpc_baseline.py": "DDPC baselines (DeePC, N4SID, MLP, PETS)",
    "src/ground_truth.py": "ground-truth DAGs",
    "src/metrics.py": "DAG quality metrics (SHD, F1)",
    "src/metrics_viz.py": "metrics visualization",
    "src/gpt_client.py": "LLM API client (OpenAI)",
    # Experiment harness
    "experiments_neurips.py": "B1 experiment harness (8 arms × 4 sims)",
    "run_exp_suite_B_prelim.py": "B1+B2 experiment suite (full pipeline)",
    "multirun_main.py": "multi-seed runner with aggregation",
    "main.py": "single-run entry point",
    "regime_aware_validation.py": "RF regime classifier + BMA switching",
    "calibration_analysis.py": "monitoring calibration metrics",
    "data_processing_enhanced.py": "intervention-aware data preprocessing",
    "energyplus_interface.py": "EnergyPlus interface",
    "smoke_test_policy.py": "smoke test",
    # JS simulators
    "open_window.js": "open_window simulator (Challenge 1)",
    "open_window_generate_data.js": "open_window data generator",
    "gen_data.js": "generic data generator with interventions",
    "python_interface.js": "Python IPC bridge (PMV/PPD)",
    "js/smart_room.js": "smart_room simulator",
    "js/smart_room_hidden_vars.js": "hidden_vars simulator",
    "js/smart_room_noise.js": "smart_room_noise simulator",
    "js/smart_building.js": "smart_building simulator",
    "js/data_collection_from_sim.js": "bulk data collection harness",
    # Data & config
    "epluszsz.csv": "EnergyPlus zone sizing lookup table",
    "ground_truth_graphs.json": "ground-truth DAG definitions",
    "package.json": "Node.js dependencies",
    "requirements.txt": "Python dependencies",
    "README.md": "project documentation",
}

for fpath, desc in REQUIRED_FILES.items():
    full = os.path.join(ROOT, fpath)
    if os.path.isfile(full):
        size_kb = os.path.getsize(full) / 1024
        ok(f"{fpath} ({size_kb:.1f} KB) — {desc}")
    else:
        fail(f"{fpath} MISSING — {desc}")

# ══════════════════════════════════════════════════════════════════════════════
# 2. Required directories
# ══════════════════════════════════════════════════════════════════════════════
print("\n2. Checking required directories...")

REQUIRED_DIRS = {
    "src": "core framework modules",
    "js": "JS simulators",
    "data": "static datasets (ASHRAE)",
    "data_regen": "regenerated simulator datasets",
    "data/obs_only_edges": "cached observational DAGs",
    "weather": "TMY3 weather files",
    "node_modules": "Node.js packages (run npm install if missing)",
}

for dpath, desc in REQUIRED_DIRS.items():
    full = os.path.join(ROOT, dpath)
    if os.path.isdir(full):
        n = len(os.listdir(full))
        ok(f"{dpath}/ ({n} items) — {desc}")
    else:
        fail(f"{dpath}/ MISSING — {desc}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. data_regen/ contents
# ══════════════════════════════════════════════════════════════════════════════
print("\n3. Checking data_regen/ datasets...")

EXPECTED_DATA_REGEN = [
    ("challenge1_data_10k.csv", "open_window raw"),
    ("challenge1_data_10k_processed.csv", "open_window preprocessed"),
    ("challenge1_data_10k_processed_scaling.csv", "open_window scaling params"),
    ("smart_room_sim_data.csv", "smart_room raw"),
    ("smart_room_processed.csv", "smart_room preprocessed"),
    ("smart_room_processed_scaling.csv", "smart_room scaling params"),
    ("smart_room_noise_sim_data.csv", "smart_room_noise raw"),
    ("smart_room_noise_processed.csv", "smart_room_noise preprocessed"),
    ("smart_room_noise_processed_scaling.csv", "smart_room_noise scaling params"),
    ("hidden_vars_sim_data.csv", "hidden_vars raw"),
    ("hidden_vars_processed.csv", "hidden_vars preprocessed"),
    ("hidden_vars_processed_scaling.csv", "hidden_vars scaling params"),
]

for f, desc in EXPECTED_DATA_REGEN:
    fpath = os.path.join(ROOT, "data_regen", f)
    if os.path.isfile(fpath):
        size_kb = os.path.getsize(fpath) / 1024
        # Check it's not empty and has rows
        with open(fpath) as fh:
            n_lines = sum(1 for _ in fh)
        ok(f"data_regen/{f} ({size_kb:.0f} KB, {n_lines} lines) — {desc}")
    else:
        fail(f"data_regen/{f} MISSING — {desc}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. ASHRAE data
# ══════════════════════════════════════════════════════════════════════════════
print("\n4. Checking ASHRAE data...")

for f in ["ashrae_data.csv", "ashrae_data_processed.csv", "ashrae_scaling_params.csv"]:
    fpath = os.path.join(ROOT, "data", f)
    if os.path.isfile(fpath):
        size_kb = os.path.getsize(fpath) / 1024
        ok(f"data/{f} ({size_kb:.0f} KB)")
    else:
        fail(f"data/{f} MISSING")

# ══════════════════════════════════════════════════════════════════════════════
# 5. obs_only_edges/ JSON files
# ══════════════════════════════════════════════════════════════════════════════
print("\n5. Checking observational DAG caches...")

for sim in ["open_window", "smart_room", "smart_room_noise", "hidden_vars"]:
    fpath = os.path.join(ROOT, "data", "obs_only_edges", f"{sim}.json")
    if os.path.isfile(fpath):
        with open(fpath) as fh:
            edges = json.load(fh)
        n = len(edges) if isinstance(edges, list) else len(edges.get('edges', edges))
        ok(f"data/obs_only_edges/{sim}.json ({n} edges)")
    elif os.path.isfile(os.path.join(ROOT, "data", "obs_only_edges", f"{sim.replace('_', '')}.json")):
        warn(f"data/obs_only_edges/{sim}.json — found with alternate naming")
    else:
        warn(f"data/obs_only_edges/{sim}.json MISSING (will be regenerated)")

# ══════════════════════════════════════════════════════════════════════════════
# 6. ground_truth_graphs.json integrity
# ══════════════════════════════════════════════════════════════════════════════
print("\n6. Checking ground_truth_graphs.json...")

gt_path = os.path.join(ROOT, "ground_truth_graphs.json")
if os.path.isfile(gt_path):
    try:
        with open(gt_path) as fh:
            gt = json.load(fh)
        for sim_name in ["open_window", "smart_room", "smart_room_noise",
                         "hidden_vars"]:
            if sim_name in gt:
                n_edges = len(gt[sim_name].get('edges', []))
                ok(f"ground_truth['{sim_name}'] — {n_edges} edges")
            else:
                warn(f"ground_truth['{sim_name}'] not found")
    except json.JSONDecodeError as e:
        fail(f"ground_truth_graphs.json — invalid JSON: {e}")
else:
    fail("ground_truth_graphs.json MISSING")

# ══════════════════════════════════════════════════════════════════════════════
# 7. epluszsz.csv integrity
# ══════════════════════════════════════════════════════════════════════════════
print("\n7. Checking epluszsz.csv...")

eplus_path = os.path.join(ROOT, "epluszsz.csv")
if os.path.isfile(eplus_path):
    with open(eplus_path) as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        n_rows = sum(1 for _ in reader)
    ok(f"epluszsz.csv — {n_rows} data rows, {len(header)} columns")
else:
    fail("epluszsz.csv MISSING")

# ══════════════════════════════════════════════════════════════════════════════
# 8. Python dependencies
# ══════════════════════════════════════════════════════════════════════════════
print("\n8. Checking Python dependencies...")

print(f"  Python: {sys.version.split()[0]} ({sys.executable})")

PY_DEPS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "statsmodels": "statsmodels",
    "causallearn": "causal-learn (PC algorithm)",
    "lingam": "lingam (VARLiNGAM)",
    "torch": "PyTorch (SAM + DDPC_Neural)",
    "networkx": "networkx",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "pyvis": "pyvis (DAG visualization)",
    "pythermalcomfort": "pythermalcomfort (PMV/PPD)",
    "dotenv": "python-dotenv",
    "joblib": "joblib",
    "tqdm": "tqdm",
    "psutil": "psutil",
    "requests": "requests (OpenAI API)",
    "graphviz": "graphviz (DAG rendering)",
}

for module, name in PY_DEPS.items():
    try:
        mod = importlib.import_module(module)
        ver = getattr(mod, '__version__', '?')
        ok(f"{name} ({ver})")
    except ImportError:
        fail(f"{name} — pip install {name.split('(')[0].strip()}")

# SAM (cdt) — required now that SAM is uncommented
for module, name in [("cdt", "cdt (SAM algorithm)")]:
    try:
        mod = importlib.import_module(module)
        ver = getattr(mod, '__version__', '?')
        ok(f"{name} ({ver})")
    except ImportError:
        fail(f"{name} — required for SAMGenerator; pip install cdt")

# Optional
for module, name in [("eppy", "eppy (EnergyPlus IDF parsing)")]:
    try:
        mod = importlib.import_module(module)
        ver = getattr(mod, '__version__', '?')
        ok(f"{name} ({ver})")
    except ImportError:
        warn(f"{name} not installed (optional — only for EnergyPlus)")

# ══════════════════════════════════════════════════════════════════════════════
# 9. Node.js & npm
# ══════════════════════════════════════════════════════════════════════════════
print("\n9. Checking Node.js runtime...")

node = shutil.which("node")
if node:
    ver = subprocess.check_output(["node", "--version"], text=True).strip()
    major = int(ver.lstrip('v').split('.')[0])
    if major >= 18:
        ok(f"node {ver}")
    else:
        fail(f"node {ver} — need v18+ (current: {ver})")
else:
    fail("node not found — install Node.js 18+")

npm = shutil.which("npm")
if npm:
    ver = subprocess.check_output(["npm", "--version"], text=True).strip()
    ok(f"npm {ver}")
else:
    fail("npm not found")

if os.path.isdir(os.path.join(ROOT, "node_modules")):
    # Check key packages exist
    for pkg in ["express", "papaparse", "matter-js"]:
        if os.path.isdir(os.path.join(ROOT, "node_modules", pkg)):
            ok(f"node_modules/{pkg}")
        else:
            fail(f"node_modules/{pkg} MISSING — run: npm install")
else:
    fail("node_modules/ MISSING — run: npm install")

# ══════════════════════════════════════════════════════════════════════════════
# 10. Test ALL JS simulators
# ══════════════════════════════════════════════════════════════════════════════
print("\n10. Testing JS simulators...")

JS_SIMS = [
    "open_window.js",
    "js/smart_room.js",
    "js/smart_room_noise.js",
    "js/smart_room_hidden_vars.js",
]

for sim_name in JS_SIMS:
    if not os.path.isfile(os.path.join(ROOT, sim_name)):
        fail(f"{sim_name} — file not found, skipping test")
        continue
    # All JS sims use --single-step mode (JSON in/out), not --rows
    cmd = [
        "node", sim_name, "--single-step",
        "--state", '{"temperature":20,"humidity":50,"airQuality":80}',
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=30, cwd=ROOT
        )
        stdout = result.stdout.strip()
        if result.returncode == 0 and "RESULT:" in stdout:
            # Parse the JSON result
            result_line = [l for l in stdout.split("\n") if l.startswith("RESULT:")][0]
            result_json = json.loads(result_line.replace("RESULT:", ""))
            keys = list(result_json.keys())
            ok(f"{sim_name} — returns {len(keys)} fields: {', '.join(keys[:5])}...")
        elif result.returncode == 0:
            warn(f"{sim_name} — runs but no RESULT: line in output")
        else:
            stderr_snip = (result.stderr or "")[:200]
            fail(f"{sim_name} — exit code {result.returncode}: {stderr_snip}")
    except subprocess.TimeoutExpired:
        fail(f"{sim_name} — timed out (30s)")
    except FileNotFoundError:
        fail(f"{sim_name} — node not found")

# ══════════════════════════════════════════════════════════════════════════════
# 11. Python imports — full chain
# ══════════════════════════════════════════════════════════════════════════════
print("\n11. Testing Python imports (full chain)...")

IMPORT_TESTS = [
    ("src.pipeline", "CausalPipeline"),
    ("src.pipeline_cwm", "CWMIntegratedPipeline"),
    ("src.pipeline_cwm", "create_cwm_pipeline"),
    ("src.generators", "PCGenerator"),
    ("src.generators", "SAMGenerator"),
    ("src.generators", "LLMGenerator"),
    ("src.generators", "VARLiNGAMGenerator"),
    ("src.tester", "HypothesisTester"),
    ("src.evaluator", "HypothesisEvaluator"),
    ("src.evaluator", "EdgeRanker"),
    ("src.causal_world_model", "CausalWorldModel"),
    ("src.policy_engine", "CausalPolicyEngine"),
    ("src.ddpc_baseline", "create_ddpc_suite"),
    ("src.ground_truth", "GroundTruthDAG"),
    ("src.ground_truth", "Challenge1GroundTruth"),
    ("src.metrics", "MetricsCalculator"),
    ("src.metrics_viz", "MetricsVisualizer"),
    ("src.gpt_client", "GPTClient"),
    ("experiments_neurips", "NeurIPSExperiments"),
    ("regime_aware_validation", "WindowStatePredictor"),
    ("regime_aware_validation", "RegimeAwareValidator"),
    ("calibration_analysis", "compute_calibration_metrics"),
]

for module_name, class_name in IMPORT_TESTS:
    try:
        mod = importlib.import_module(module_name)
        obj = getattr(mod, class_name)
        ok(f"from {module_name} import {class_name}")
    except ImportError as e:
        fail(f"from {module_name} import {class_name} — ImportError: {e}")
    except AttributeError:
        fail(f"from {module_name} import {class_name} — class/function not found")
    except Exception as e:
        fail(f"from {module_name} import {class_name} — {type(e).__name__}: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 12. SIM_CONFIGS paths in run_exp_suite_B_prelim.py
# ══════════════════════════════════════════════════════════════════════════════
print("\n12. Checking SIM_CONFIGS data paths...")

try:
    # Import and check all data paths exist
    spec = importlib.util.spec_from_file_location(
        "run_exp", os.path.join(ROOT, "run_exp_suite_B_prelim.py"))
    run_exp = importlib.util.module_from_spec(spec)
    # Don't execute (has argparse main), just read the source for SIM_CONFIGS
    with open(os.path.join(ROOT, "run_exp_suite_B_prelim.py")) as fh:
        source = fh.read()

    # Extract paths referenced in SIM_CONFIGS
    import re
    for match in re.finditer(r"'(obs_data_path|scaling_path|sim_path)'\s*:\s*'([^']+)'", source):
        key, path = match.groups()
        if path == 'None' or path is None:
            continue
        full = os.path.join(ROOT, path)
        if os.path.isfile(full):
            ok(f"SIM_CONFIGS.{key} = '{path}'")
        else:
            fail(f"SIM_CONFIGS.{key} = '{path}' — FILE NOT FOUND")
except Exception as e:
    warn(f"Could not parse SIM_CONFIGS: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 13. SAM generator enabled check
# ══════════════════════════════════════════════════════════════════════════════
print("\n13. Checking SAM generator is enabled...")

try:
    with open(os.path.join(ROOT, "src", "pipeline.py")) as fh:
        pipeline_src = fh.read()
    # Check if SAM line is commented out
    import re
    sam_match = re.search(r"^(\s*#?\s*)'sam'\s*:\s*SAMGenerator", pipeline_src, re.MULTILINE)
    if sam_match:
        prefix = sam_match.group(1).strip()
        if prefix.startswith('#'):
            fail("SAMGenerator is COMMENTED OUT in src/pipeline.py")
        else:
            ok("SAMGenerator is enabled in src/pipeline.py")
    else:
        warn("Could not find SAMGenerator line in src/pipeline.py")
except Exception as e:
    warn(f"Could not check SAM status: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# 14. .env file (API keys)
# ══════════════════════════════════════════════════════════════════════════════
print("\n14. Checking environment configuration...")

env_path = os.path.join(ROOT, ".env")
if os.path.isfile(env_path):
    with open(env_path) as fh:
        env_content = fh.read()
    if "OPENAI_API_KEY" in env_content:
        # Check it's not just the placeholder
        for line in env_content.split('\n'):
            if line.startswith('OPENAI_API_KEY') and 'sk-' in line:
                ok(".env has OPENAI_API_KEY (needed for LLM generator)")
                break
        else:
            warn(".env has OPENAI_API_KEY but value looks like placeholder")
    else:
        warn(".env exists but no OPENAI_API_KEY (LLM generator will be skipped)")
else:
    warn(".env not found — create one with OPENAI_API_KEY=sk-... for LLM generator")

# Also check env var directly
api_key = os.environ.get("OPENAI_API_KEY")
if api_key and api_key.startswith("sk-"):
    ok("OPENAI_API_KEY set in environment")
elif api_key:
    warn("OPENAI_API_KEY set but doesn't start with sk-")

# ══════════════════════════════════════════════════════════════════════════════
# 15. Disk space
# ══════════════════════════════════════════════════════════════════════════════
print("\n15. Checking disk space...")

try:
    stat = os.statvfs(ROOT)
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
    if free_gb > 5:
        ok(f"{free_gb:.1f} GB free disk space")
    elif free_gb > 1:
        warn(f"{free_gb:.1f} GB free — experiments may need more space")
    else:
        fail(f"{free_gb:.1f} GB free — insufficient disk space")
except Exception:
    warn("Could not check disk space")

# ══════════════════════════════════════════════════════════════════════════════
# 16. Write permissions
# ══════════════════════════════════════════════════════════════════════════════
print("\n16. Checking write permissions...")

test_dir = os.path.join(ROOT, "neurips_results")
try:
    os.makedirs(test_dir, exist_ok=True)
    test_file = os.path.join(test_dir, ".preflight_test")
    with open(test_file, 'w') as fh:
        fh.write("test")
    os.remove(test_file)
    ok(f"Can write to neurips_results/")
except PermissionError:
    fail(f"Cannot write to neurips_results/ — check permissions")
except Exception as e:
    fail(f"Write test failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  PASSED:   {passed}")
print(f"  FAILED:   {failed}")
print(f"  WARNINGS: {warnings}")
print("=" * 60)

if failed == 0 and warnings == 0:
    print("  All checks passed — ready to run experiments.")
elif failed == 0:
    print(f"  Ready to run (review {warnings} warning(s) above).")
else:
    print(f"  Fix {failed} failure(s) before running experiments.")

print()
print("  Quick start:")
print("    python smoke_test_policy.py                          # smoke test")
print("    python run_exp_suite_B_prelim.py --all-sims --include-pets --seeds 5")
print()

sys.exit(1 if failed else 0)
