#!/usr/bin/env python3
"""
Run policy baselines on physical testbed data (offline).

Uses the saved baseline training data + validated edges to fit each method's
model, then evaluates what actions each method recommends at 3 comfort targets.
Since we can't step a live simulator, we use the SEM fitted on validated edges
to predict outcomes — the same SEM that PolicyGRID uses for gradient optimization.
"""
import sys, os, json, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

BASELINE_CSV = 'smart_room_physical/deployed/data/baseline_data.csv'
EDGES_PATH = 'smart_room_physical/deployed/results/discovery/seed_42/validated_edges.json'
SEM_PATH = 'smart_room_physical/deployed/results/policy/sem_coefficients.json'
OUTPUT_DIR = 'smart_room_physical/deployed/results/policy'

# Physical testbed variables
ACTION_VARS = ['Heater', 'Humidifier', 'Fan']
STATE_VARS = ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'Satisfaction']
ALL_VARS = ACTION_VARS + STATE_VARS


def load_data():
    data = pd.read_csv(BASELINE_CSV)
    cols = [c for c in data.columns if c in ALL_VARS + [v.lower() for v in ALL_VARS]]
    # Normalize column names
    col_map = {}
    for c in data.columns:
        for v in ALL_VARS:
            if c.lower() == v.lower():
                col_map[c] = v
    data = data.rename(columns=col_map)
    return data[[v for v in ALL_VARS if v in data.columns]]


def fit_sem(data, edges):
    """Fit Ridge SEM on validated edges (same as PolicyGRID policy engine)."""
    parent_map = {v: [] for v in ALL_VARS}
    for s, t in edges:
        if s in ALL_VARS and t in ALL_VARS:
            if s not in parent_map[t]:
                parent_map[t].append(s)

    models = {}
    for target, parents in parent_map.items():
        if not parents or target not in data.columns:
            continue
        parent_cols = [p for p in parents if p in data.columns]
        if not parent_cols:
            continue
        X = data[parent_cols].values
        y = data[target].values
        valid = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        if valid.sum() < 20:
            continue
        reg = Ridge(alpha=0.5).fit(X[valid], y[valid])
        models[target] = {'ridge': reg, 'parents': parent_cols}

    return models


def predict_sem(sem, state):
    """Predict next state given current state using SEM."""
    predicted = dict(state)
    for target, m in sem.items():
        x = np.array([state.get(p, 0.0) for p in m['parents']]).reshape(1, -1)
        predicted[target] = float(np.clip(m['ridge'].predict(x)[0], 0, 1))
    return predicted


def evaluate_action(sem, base_state, action):
    """Apply action and predict outcomes."""
    state = dict(base_state)
    state.update(action)
    predicted = predict_sem(sem, state)
    return predicted.get('Satisfaction', 0), predicted.get('EnergyConsumption', 0)


# ── Policy methods ──

def policygrid_policy(sem, base_state, comfort_weight):
    """SEM gradient descent (same as live deployment)."""
    # Initialize at midpoint
    actions = {a: 0.5 for a in ACTION_VARS}

    for step in range(50):
        lr = 0.1 * (1 - step / 50)
        for a in ACTION_VARS:
            # Numerical gradient
            eps = 0.01
            a_plus = dict(actions); a_plus[a] = min(1, actions[a] + eps)
            a_minus = dict(actions); a_minus[a] = max(0, actions[a] - eps)

            s_plus = dict(base_state); s_plus.update(a_plus)
            s_minus = dict(base_state); s_minus.update(a_minus)

            sat_plus = predict_sem(sem, s_plus).get('Satisfaction', 0)
            sat_minus = predict_sem(sem, s_minus).get('Satisfaction', 0)
            energy_plus = predict_sem(sem, s_plus).get('EnergyConsumption', 0)
            energy_minus = predict_sem(sem, s_minus).get('EnergyConsumption', 0)

            d_sat = (sat_plus - sat_minus) / (2 * eps)
            d_energy = (energy_plus - energy_minus) / (2 * eps)

            grad = comfort_weight * d_sat - (1 - comfort_weight) * d_energy
            actions[a] = np.clip(actions[a] + lr * grad, 0, 1)

    # Threshold to binary for physical actuators
    return {a: 1.0 if v > 0.5 else 0.0 for a, v in actions.items()}


def pid_policy(base_state, setpoint=0.5):
    """PID: simple proportional control on satisfaction error."""
    sat = base_state.get('Satisfaction', 0.5)
    error = setpoint - sat
    # Proportional gain: turn on heater if satisfaction too low
    heater = 1.0 if error > 0.1 else 0.0
    return {'Heater': heater, 'Humidifier': 0.0, 'Fan': 0.0}


def mpc_linear_policy(data, base_state, comfort_weight):
    """Linear MPC: fit linear dynamics, optimize via enumeration (binary)."""
    # Fit simple linear model: next_state = A * state + B * action
    X = data[ACTION_VARS + STATE_VARS].values[:-1]
    Y_sat = data['Satisfaction'].values[1:]
    Y_energy = data['EnergyConsumption'].values[1:]

    reg_sat = Ridge(alpha=1.0).fit(X, Y_sat)
    reg_energy = Ridge(alpha=1.0).fit(X, Y_energy)

    # Enumerate all 8 binary action combinations
    best_obj, best_action = -1e9, {a: 0.0 for a in ACTION_VARS}
    for h in [0, 1]:
        for hu in [0, 1]:
            for f in [0, 1]:
                action = {'Heater': float(h), 'Humidifier': float(hu), 'Fan': float(f)}
                state_vec = np.array([action.get(v, base_state.get(v, 0.5))
                                      for v in ACTION_VARS + STATE_VARS]).reshape(1, -1)
                sat = reg_sat.predict(state_vec)[0]
                energy = reg_energy.predict(state_vec)[0]
                obj = comfort_weight * sat - (1 - comfort_weight) * energy
                if obj > best_obj:
                    best_obj = obj
                    best_action = action
    return best_action


def ddpc_behavioral_policy(data, base_state, comfort_weight):
    """DDPC Behavioral: Willems' lemma with Hankel matrix, binary threshold."""
    T = len(data)
    u = data[ACTION_VARS].values  # (T, 3)
    y_sat = data['Satisfaction'].values
    y_energy = data['EnergyConsumption'].values

    # Build Hankel matrices
    depth = min(10, T // 4)
    n_col = T - depth + 1
    if n_col < 5:
        return {a: 0.0 for a in ACTION_VARS}

    # Simple approach: regress output on recent action history
    # (approximation of the full Willems' lemma QP for binary case)
    X = np.column_stack([u[i:T-depth+i+1] for i in range(depth)])
    y_s = y_sat[depth-1:T]
    y_e = y_energy[depth-1:T]

    reg_sat = Ridge(alpha=1.0).fit(X, y_s)
    reg_energy = Ridge(alpha=1.0).fit(X, y_e)

    # Enumerate binary actions (current step only, assume rest = recent history)
    recent_u = u[-depth+1:].flatten()
    best_obj, best_action = -1e9, {a: 0.0 for a in ACTION_VARS}
    for h in [0, 1]:
        for hu in [0, 1]:
            for f in [0, 1]:
                test_u = np.concatenate([recent_u, [h, hu, f]]).reshape(1, -1)
                if test_u.shape[1] != X.shape[1]:
                    continue
                sat = reg_sat.predict(test_u)[0]
                energy = reg_energy.predict(test_u)[0]
                obj = comfort_weight * sat - (1 - comfort_weight) * energy
                if obj > best_obj:
                    best_obj = obj
                    best_action = {'Heater': float(h), 'Humidifier': float(hu), 'Fan': float(f)}
    return best_action


def cmbpo_policy(sem, base_state, comfort_weight):
    """C-MBPO: same graph as PolicyGRID, but random-search optimizer (no gradient)."""
    best_obj, best_action = -1e9, {a: 0.0 for a in ACTION_VARS}

    # Random search over binary actions (8 combinations for 3 binary actuators)
    for h in [0, 1]:
        for hu in [0, 1]:
            for f in [0, 1]:
                action = {'Heater': float(h), 'Humidifier': float(hu), 'Fan': float(f)}
                sat, energy = evaluate_action(sem, base_state, action)
                obj = comfort_weight * sat - (1 - comfort_weight) * energy
                if obj > best_obj:
                    best_obj = obj
                    best_action = action
    return best_action


# ── Main ──

def main():
    print("=" * 70)
    print("  Physical Testbed Policy Baselines (Offline Evaluation)")
    print("=" * 70)

    data = load_data()
    print(f"Data: {len(data)} rows, columns: {list(data.columns)}")

    with open(EDGES_PATH) as f:
        edges = [tuple(e) for e in json.load(f)]
    print(f"Validated edges: {len(edges)}")

    sem = fit_sem(data, edges)
    print(f"SEM targets: {list(sem.keys())}")

    # Use mean state as baseline condition
    base_state = {v: data[v].mean() for v in ALL_VARS if v in data.columns}
    print(f"\nBaseline state: Temp={base_state.get('Temperature',0):.3f}, "
          f"Sat={base_state.get('Satisfaction',0):.3f}, "
          f"Energy={base_state.get('EnergyConsumption',0):.3f}")

    # Comfort weights: low=energy-saving, high=comfort-prioritizing
    # Maps to epsilon in the paper
    targets = [
        ('eps=0.5 (energy)', 0.3),
        ('eps=1.0 (balanced)', 0.5),
        ('eps=2.0 (comfort)', 0.8),
    ]

    methods = {
        'PolicyGRID': lambda cw: policygrid_policy(sem, base_state, cw),
        'C-MBPO': lambda cw: cmbpo_policy(sem, base_state, cw),
        'MPC_Linear': lambda cw: mpc_linear_policy(data, base_state, cw),
        'DDPC_Behavioral': lambda cw: ddpc_behavioral_policy(data, base_state, cw),
        'PID': lambda cw: pid_policy(base_state),
        'All-OFF': lambda cw: {a: 0.0 for a in ACTION_VARS},
        'All-ON': lambda cw: {a: 1.0 for a in ACTION_VARS},
    }

    results = []
    print(f"\n{'Method':20s} {'Target':22s} {'Action':30s} {'Sat':>6s} {'Energy':>8s}")
    print("-" * 90)

    for target_name, comfort_weight in targets:
        for method_name, policy_fn in methods.items():
            action = policy_fn(comfort_weight)
            sat, energy = evaluate_action(sem, base_state, action)

            action_str = ', '.join(f'{k}={v:.0f}' for k, v in action.items())
            print(f'{method_name:20s} {target_name:22s} {action_str:30s} {sat:6.3f} {energy:8.3f}')

            results.append({
                'method': method_name, 'target': target_name,
                'comfort_weight': comfort_weight,
                'heater': action.get('Heater', 0),
                'humidifier': action.get('Humidifier', 0),
                'fan': action.get('Fan', 0),
                'predicted_sat': round(sat, 4),
                'predicted_energy': round(energy, 4),
            })

    df = pd.DataFrame(results)
    out = os.path.join(OUTPUT_DIR, 'policy_baselines.csv')
    df.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == '__main__':
    main()
