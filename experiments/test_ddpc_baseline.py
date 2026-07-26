"""
Diagnostic script for src/ddpc_baseline.py
Run from machine_causation/ directory:
    python test_ddpc_baseline.py

Tests (in order):
    1. Data loading
    2. ARX helpers (_build_hankel, _normalize_state, _extract_arrays)
    3. DeePCController  — fit + get_action x5 steps
    4. SubspaceDDPCController — fit + get_action x5 steps
    5. NeuralDDPCController  — fit + get_action x5 steps
    6. create_ddpc_suite factory
    7. Full episode simulation (12 steps, receding horizon, no JS simulator)
    8. Scale invariance check (_normalize_state)
"""

import sys, os, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def check(label, condition, detail=""):
    if condition:
        print(f"  {PASS}  {label}")
    else:
        print(f"  {FAIL}  {label}  {'← ' + detail if detail else ''}")
    return condition

def run(label, fn):
    """Run fn(); catch and print any exception."""
    try:
        result = fn()
        print(f"  {PASS}  {label}")
        return result
    except Exception as e:
        print(f"  {FAIL}  {label}")
        traceback.print_exc()
        return None


# ── 1. Imports ───────────────────────────────────────────────────────────────
section("1. Imports")

try:
    from src.ddpc_baseline import (
        DeePCController,
        SubspaceDDPCController,
        NeuralDDPCController,
        PETSController,
        create_ddpc_suite,
        _build_hankel,
        _normalize_state,
        _extract_arrays,
        STATE_VARS, ACTION_VARS, OUTPUT_VARS,
        N_U, N_Y, HORIZON,
    )
    print(f"  {PASS}  src.ddpc_baseline imported successfully")
except Exception:
    print(f"  {FAIL}  Import failed")
    traceback.print_exc()
    sys.exit(1)


# ── 2. Data loading ──────────────────────────────────────────────────────────
section("2. Data Loading")

DATA_PATH = "data/smart_room_noisy_preprocessed.csv"

df = run("Read CSV", lambda: pd.read_csv(DATA_PATH))
if df is None:
    print("Cannot continue without data."); sys.exit(1)

check("Expected columns present",
      all(c in df.columns for c in STATE_VARS),
      f"Missing: {[c for c in STATE_VARS if c not in df.columns]}")
check("≥ 100 rows", len(df) >= 100, f"got {len(df)}")
check("No NaNs in STATE_VARS", df[STATE_VARS].isna().sum().sum() == 0)
check("Values roughly in [0,1]",
      df[STATE_VARS].max().max() <= 1.5,
      f"max={df[STATE_VARS].max().max():.3f}")

print(f"\n  Data shape : {df.shape}")
print(f"  Columns    : {list(df.columns)}")
print(f"  Value range: {df[STATE_VARS].min().min():.3f} – {df[STATE_VARS].max().max():.3f}")


# ── 3. Helper functions ──────────────────────────────────────────────────────
section("3. Helpers")

def test_hankel():
    seq = np.random.randn(50, 3)
    H = _build_hankel(seq, depth=5)
    assert H.shape == (3*5, 50-5+1), f"Expected (15,46) got {H.shape}"
    return H

run("_build_hankel shape", test_hankel)

def test_normalize():
    raw = {'Temperature': 0.5, 'Humidity': 0.4,
           'AirQuality': 0.7, 'EnergyConsumption': 45.0, 'OverallSatisfaction': 72.0}
    norm = _normalize_state(raw)
    assert norm['EnergyConsumption'] < 1.0,   "E not normalized"
    assert norm['OverallSatisfaction'] < 1.0, "S not normalized"
    assert norm['Temperature'] == 0.5,        "T should be unchanged"

run("_normalize_state (% → [0,1])", test_normalize)

def test_normalize_already_scaled():
    already = {'Temperature': 0.5, 'Humidity': 0.4,
               'AirQuality': 0.7, 'EnergyConsumption': 0.6, 'OverallSatisfaction': 0.8}
    norm = _normalize_state(already)
    assert norm['EnergyConsumption'] == 0.6, "Should leave [0,1] values unchanged"

run("_normalize_state (already [0,1] → unchanged)", test_normalize_already_scaled)

u_data, y_data, x_data = _extract_arrays(df)
check("_extract_arrays u_data shape", u_data.shape[1] == N_U, str(u_data.shape))
check("_extract_arrays y_data shape", y_data.shape[1] == N_Y, str(y_data.shape))
check("_extract_arrays x_data shape", x_data.shape[1] == len(STATE_VARS), str(x_data.shape))


# ── Shared test state ────────────────────────────────────────────────────────
# Representative normalized state (as from simulator)
STATE_NORM = {
    'Temperature': 0.55, 'Humidity': 0.45, 'AirQuality': 0.72,
    'EnergyConsumption': 0.48, 'OverallSatisfaction': 0.51,
}
# Same state in simulator % scale (E and S in 0-100)
STATE_PCT = {
    'Temperature': 0.55, 'Humidity': 0.45, 'AirQuality': 0.72,
    'EnergyConsumption': 48.0, 'OverallSatisfaction': 51.0,
}
COMFORT_TARGETS = [0.5, 0.8, 1.0]


def check_action(action, label):
    ok = (isinstance(action, dict)
          and set(action.keys()) == set(ACTION_VARS)
          and all(0.0 <= v <= 1.0 for v in action.values()))
    check(f"{label} → valid action in [0,1]³", ok, str(action))
    return ok


# ── 4. DeePCController ───────────────────────────────────────────────────────
section("4. DeePCController (DeePC / Behavioral)")

deepc = run("Instantiate", lambda: DeePCController(lam_u=1e-2))

if deepc:
    run("fit(u_data, y_data)", lambda: deepc.fit(u_data, y_data))

    check("Q3 shape",
          deepc.Q3 is not None and deepc.Q3.shape == (N_Y * HORIZON, N_U * HORIZON),
          str(deepc.Q3.shape if deepc.Q3 is not None else "None"))
    check("Q12 shape",
          deepc.Q12 is not None and deepc.Q12.shape[0] == N_Y * HORIZON,
          str(deepc.Q12.shape if deepc.Q12 is not None else "None"))

    deepc.reset()
    print("\n  -- get_action x5 (normalized state) --")
    for ct in COMFORT_TARGETS:
        a = run(f"  get_action(comfort={ct})", lambda ct=ct: deepc.get_action(STATE_NORM, ct))
        if a: check_action(a, f"    comfort={ct}")

    print("\n  -- get_action (simulator-% scale state) --")
    deepc.reset()
    a_pct = run("  get_action(% state)", lambda: deepc.get_action(STATE_PCT, 0.8))
    if a_pct: check_action(a_pct, "    % state")

    # History window grows correctly
    deepc.reset()
    for _ in range(6):
        deepc.get_action(STATE_NORM, 0.8)
    check("History deque saturated (T_INI entries)",
          len(deepc._u_hist) == deepc.T_INI and len(deepc._y_hist) == deepc.T_INI)


# ── 5. SubspaceDDPCController ────────────────────────────────────────────────
section("5. SubspaceDDPCController (N4SID-style + linear MPC)")

subsp = run("Instantiate", lambda: SubspaceDDPCController(lag_y=3, lag_u=3))

if subsp:
    run("fit(u_data, y_data)", lambda: subsp.fit(u_data, y_data))

    check("A matrix fitted",  subsp.A is not None and subsp.A.shape[0] == subsp.n_x)
    check("B matrix fitted",  subsp.B is not None and subsp.B.shape[1] == N_U)
    check("C matrix fitted",  subsp.C is not None and subsp.C.shape[0] == N_Y)
    check("F_u cached",       subsp._F_u is not None and subsp._F_u.shape == (HORIZON*N_Y, HORIZON*N_U))
    check("F_x cached",       subsp._F_x is not None)

    subsp.reset()
    print("\n  -- get_action x3 comfort targets --")
    for ct in COMFORT_TARGETS:
        a = run(f"  get_action(comfort={ct})", lambda ct=ct: subsp.get_action(STATE_NORM, ct))
        if a: check_action(a, f"    comfort={ct}")

    print("\n  -- get_action (% state) --")
    subsp.reset()
    a_pct = run("  get_action(% state)", lambda: subsp.get_action(STATE_PCT, 0.8))
    if a_pct: check_action(a_pct, "    % state")

    # State update: _x_cur changes after each call
    subsp.reset()
    x_before = subsp._x_cur.copy() if subsp._x_cur is not None else None
    subsp.get_action(STATE_NORM, 0.8)
    if x_before is not None:
        check("_x_cur updated after get_action",
              not np.allclose(subsp._x_cur, x_before))


# ── 6. NeuralDDPCController ──────────────────────────────────────────────────
section("6. NeuralDDPCController (MLP + CEM)")

neural = run("Instantiate", lambda: NeuralDDPCController(
    hidden=(32, 16), population=50, n_iter=5   # small for speed
))

if neural:
    run("fit(x_data, u_data)", lambda: neural.fit(x_data, u_data))
    check("MLP fitted", neural.model is not None)
    check("mean_state shape", neural.mean_state is not None and len(neural.mean_state) == len(STATE_VARS))

    neural.reset()
    print("\n  -- get_action x3 comfort targets (CEM pop=50, iter=5) --")
    for ct in COMFORT_TARGETS:
        a = run(f"  get_action(comfort={ct})", lambda ct=ct: neural.get_action(STATE_NORM, ct))
        if a: check_action(a, f"    comfort={ct}")

    print("\n  -- get_action (% state) --")
    a_pct = run("  get_action(% state)", lambda: neural.get_action(STATE_PCT, 0.8))
    if a_pct: check_action(a_pct, "    % state")

    # Rollout shape check
    def test_rollout():
        seq = np.full((HORIZON, N_U), 0.5)
        init = np.array([STATE_NORM[v] for v in STATE_VARS])
        traj = neural._rollout(init, seq)
        assert traj.shape == (HORIZON, len(STATE_VARS)), f"Got {traj.shape}"
        assert (traj >= 0).all() and (traj <= 1).all(), "Trajectory out of [0,1]"
    run("_rollout shape + clip", test_rollout)


# ── 7. PETSController ────────────────────────────────────────────────────────
section("7. PETSController (Ensemble + TS-∞ + CEM)")

pets = run("Instantiate", lambda: PETSController(
    n_ensemble=3, hidden=(32, 16), n_particles=5, population=50, n_iter=3
))

if pets:
    run("fit(x_data, u_data)", lambda: pets.fit(x_data, u_data))
    check("Ensemble fitted",
          len(pets.models) == 3 and all(m is not None for m in pets.models))
    check("mean_state shape",
          pets.mean_state is not None and len(pets.mean_state) == len(STATE_VARS))

    pets.reset()
    print("\n  -- get_action x3 comfort targets (ensemble=3, particles=5, iter=3) --")
    for ct in COMFORT_TARGETS:
        a = run(f"  get_action(comfort={ct})", lambda ct=ct: pets.get_action(STATE_NORM, ct))
        if a: check_action(a, f"    comfort={ct}")

    print("\n  -- get_action (% state) --")
    a_pct = run("  get_action(% state)", lambda: pets.get_action(STATE_PCT, 0.8))
    if a_pct: check_action(a_pct, "    % state")

    # TS-∞ rollout: check shape + all values in [0,1]
    def test_pets_rollout():
        seq  = np.full((HORIZON, N_U), 0.5)
        init = np.array([STATE_NORM[v] for v in STATE_VARS])
        traj = pets._rollout_ts(init, seq)
        assert traj.shape == (HORIZON, len(STATE_VARS)), f"Got {traj.shape}"
        assert (traj >= 0).all() and (traj <= 1).all(), "Trajectory out of [0,1]"
    run("_rollout_ts shape + clip", test_pets_rollout)


# ── 8. create_ddpc_suite factory ─────────────────────────────────────────────
section("8. create_ddpc_suite factory")

suite = run("create_ddpc_suite(df)", lambda: create_ddpc_suite(
    df,
    cem_population=50,   # small for speed
    cem_n_iter=3,
    pets_ensemble=3,
    pets_particles=5,
))

if suite:
    check("Returns 4 controllers", len(suite) == 4, str(list(suite.keys())))
    check("DDPC_Behavioral key",  'DDPC_Behavioral' in suite)
    check("DDPC_Subspace key",    'DDPC_Subspace'   in suite)
    check("DDPC_Neural key",      'DDPC_Neural'     in suite)
    check("DDPC_PETS key",        'DDPC_PETS'       in suite)

    for name, ctrl in suite.items():
        ctrl.reset()
        a = run(f"  {name}.get_action()", lambda ctrl=ctrl: ctrl.get_action(STATE_NORM, 0.8))
        if a: check_action(a, f"    {name}")


# ── 9. Full 12-step episode simulation ───────────────────────────────────────
section("9. Full 12-step episode (receding horizon, no JS simulator)")

def simulate_episode(controller, comfort_target=0.8, seed=42):
    """Minimal stand-in for ParetoOptimizer._simulate_episode_ddpc."""
    np.random.seed(seed)
    state = {
        'Temperature':       0.5 + np.random.normal(0, 0.05),
        'Humidity':          0.5 + np.random.normal(0, 0.05),
        'AirQuality':        0.7 + np.random.normal(0, 0.05),
        'EnergyConsumption': 30.0,    # % scale (as real simulator would give)
        'OverallSatisfaction': 60.0,  # % scale
    }
    controller.reset()
    actions, energies, satisfactions = [], [], []

    for t in range(12):
        action = controller.get_action(state, comfort_target)
        actions.append(action)

        # Minimal fallback simulation (mirrors _fallback_simulation)
        T_sp = action['Temperature']
        H_sp = action['Humidity']
        AQ_sp = action['AirQuality']
        temp_penalty = abs(T_sp - 0.55) * 30
        humidity_penalty = abs(H_sp - 0.50) * 25
        aq_bonus = AQ_sp * 20
        sat = max(0, min(100, 80 - temp_penalty - humidity_penalty + aq_bonus))
        t_diff = abs(T_sp - state.get('Temperature', 0.5))
        h_diff = abs(H_sp - state.get('Humidity', 0.5))
        energy = t_diff * 40 + h_diff * 20 + AQ_sp * 15

        state = {
            'Temperature':       np.clip(T_sp + np.random.normal(0, 0.02), 0, 1),
            'Humidity':          np.clip(H_sp + np.random.normal(0, 0.02), 0, 1),
            'AirQuality':        np.clip(AQ_sp + np.random.normal(0, 0.01), 0, 1),
            'EnergyConsumption': energy,
            'OverallSatisfaction': sat,
        }
        energies.append(energy)
        satisfactions.append(sat)

    return actions, energies, satisfactions

if suite:
    for name, ctrl in suite.items():
        def ep(ctrl=ctrl, name=name):
            acts, energies, sats = simulate_episode(ctrl)
            print(f"\n  {name}:")
            print(f"    mean energy : {np.mean(energies):.3f}")
            print(f"    mean sat    : {np.mean(sats):.1f} %")
            print(f"    action range: "
                  f"T=[{min(a['Temperature'] for a in acts):.2f}, {max(a['Temperature'] for a in acts):.2f}]  "
                  f"H=[{min(a['Humidity'] for a in acts):.2f}, {max(a['Humidity'] for a in acts):.2f}]  "
                  f"AQ=[{min(a['AirQuality'] for a in acts):.2f}, {max(a['AirQuality'] for a in acts):.2f}]")
            assert len(acts) == 12, "episode must be 12 steps"
            return acts

        run(f"{name} 12-step episode", ep)


# ── Summary ──────────────────────────────────────────────────────────────────
section("Done")