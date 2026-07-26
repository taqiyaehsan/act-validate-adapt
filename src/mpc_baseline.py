"""
MPC Baseline for PolicyGRID
============================
Standard Model Predictive Control with learned dynamics + iterated CEM.

Unlike the DDPC baselines (which use Willems'/subspace/neural approaches from
the data-driven control literature), this is classical MPC: explicitly learn
a dynamics model y_{t+1} = f(x_t, u_t), then optimise actions over a planning
horizon via Cross-Entropy Method.

Two variants:
  - LinearMPC:  Ridge regression dynamics.  Fast, interpretable.
  - EnsembleMPC: Ensemble of MLPs (like PETS but without TS-∞).
                 Captures nonlinearity + model uncertainty.

Both use the same CEM optimizer and share the DDPC evaluation interface
(ACTION_VARS, OUTPUT_VARS, HORIZON, get_action).

Reference
---------
Camacho & Bordons. "Model Predictive Control." Springer, 2007.
CEM: Botev, Kroese, Rubinstein & L'Ecuyer. "The Cross-Entropy Method
     for Optimization." Handbook of Statistics 31, 2013.
Ensemble dynamics: Kurutach, Clavera, Duan, Tamar & Abbeel.
     "Model-Ensemble Trust-Region Policy Optimization." ICLR 2018.
     https://arxiv.org/abs/1802.10592
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Shared with DDPC baselines — same action/output variables
ACTION_VARS = ['HVACPower', 'LightingPower']
OUTPUT_VARS = ['EnergyConsumption', 'Satisfaction']
N_U = 2
N_Y = 2
HORIZON = 12

# Full state variables (for dynamics model input) — all observable columns
STATE_VARS = [
    'OutdoorTemp', 'SolarRadiation', 'Temperature', 'Humidity', 'CO2',
    'LightLevel', 'NoiseDB', 'AirQuality', 'PMV', 'HVACPower',
    'LightingPower', 'EnergyConsumption', 'Satisfaction',
]
N_X = len(STATE_VARS)


def _unpack_action(u: np.ndarray) -> Dict[str, float]:
    return {v: float(np.clip(u[i], 0.0, 1.0)) for i, v in enumerate(ACTION_VARS)}


def _extract_data(df: pd.DataFrame):
    """Extract state, action, and output arrays from training DataFrame."""
    col_map = {c.lower(): c for c in df.columns}

    x_cols = [col_map.get(v.lower(), v) for v in STATE_VARS if v.lower() in col_map]
    u_cols = [col_map.get(v.lower(), v) for v in ACTION_VARS if v.lower() in col_map]
    y_cols = [col_map.get(v.lower(), v) for v in OUTPUT_VARS if v.lower() in col_map]

    x_data = df[x_cols].values if x_cols else df.values
    u_data = df[u_cols].values if u_cols else np.zeros((len(df), N_U))
    y_data = df[y_cols].values if y_cols else np.zeros((len(df), N_Y))

    return x_data, u_data, y_data


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Linear MPC — Ridge dynamics + CEM
# ═══════════════════════════════════════════════════════════════════════════════

class LinearMPCController:
    """MPC with linear Ridge dynamics and iterated CEM optimizer.

    Dynamics: y_{t+1} = W [x_t; u_t] + b  (Ridge regression).
    Optimizer: CEM over H-step horizon.

    This is the standard "learn a model, plan with CEM" approach.
    No causal structure, no ensemble, no uncertainty.
    """

    def __init__(
        self,
        population:      int   = 120,
        elite_frac:      float = 0.2,
        n_iter:          int   = 8,
        comfort_penalty: float = 100.0,
        ridge_alpha:     float = 2.0,
        tuned:           bool  = False,
    ):
        self.pop             = population
        self.elite           = max(5, int(population * elite_frac))
        self.n_iter          = n_iter
        self.comfort_penalty = comfort_penalty
        self.ridge_alpha     = ridge_alpha
        # When True, scales comfort/energy/control cost weights by ε via the same
        # w_s(ε), w_e(ε) map PolicyGRID uses (see run_closedloop_server.py:277).
        # Default False preserves the paper-version behavior for all baseline runs.
        self.tuned           = tuned

        self.model = None
        self.mean_x = None
        self.comfort_target = 0.8

    def fit(self, df: pd.DataFrame) -> 'LinearMPCController':
        x_data, u_data, y_data = _extract_data(df)
        T = len(x_data) - 1

        # Features: [x_t, u_t] → x_{t+1}
        feat = np.hstack([x_data[:T], u_data[:T]])
        tgt = x_data[1:]

        self.model = Ridge(alpha=self.ridge_alpha)
        self.model.fit(feat, tgt)
        self.mean_x = x_data.mean(axis=0)

        r2 = self.model.score(feat, tgt)
        logger.info(f"LinearMPC: Ridge dynamics R²={r2:.3f}, "
                    f"features={feat.shape[1]}, targets={tgt.shape[1]}")
        return self

    def set_comfort_target(self, ct: float):
        self.comfort_target = ct

    def _rollout(self, x0: np.ndarray, u_seq: np.ndarray) -> np.ndarray:
        """Roll out dynamics for H steps. Returns (H+1, n_x) trajectory."""
        traj = [x0.copy()]
        x = x0.copy()
        for t in range(len(u_seq)):
            feat = np.concatenate([x, u_seq[t]]).reshape(1, -1)
            x = self.model.predict(feat)[0]
            traj.append(x.copy())
        return np.array(traj)

    def _cost(self, trajectory: np.ndarray, u_seq: np.ndarray) -> float:
        """Compute MPC cost over trajectory.

        When self.tuned=True, scales Q (comfort), energy, and R (control) terms
        by ε via the same w_s/w_e map PolicyGRID uses. Otherwise uses the fixed
        paper-version weights.
        """
        sat_floor = max(0.3, 1.0 - self.comfort_target)

        if self.tuned:
            w_s = max(0.3, min(0.85, 1.0 - self.comfort_target * 0.35))
            w_e = 1.0 - w_s
            comfort_weight = self.comfort_penalty * w_s
            energy_weight = w_e
            control_weight = 0.01 * w_e
        else:
            comfort_weight = self.comfort_penalty
            energy_weight = 1.0
            control_weight = 0.01

        energy_cost = 0.0
        comfort_cost = 0.0

        e_idx = STATE_VARS.index('EnergyConsumption')
        s_idx = STATE_VARS.index('Satisfaction')

        for t in range(1, len(trajectory)):
            energy_cost += energy_weight * trajectory[t][e_idx]
            sat_val = trajectory[t][s_idx]
            if sat_val < sat_floor:
                comfort_cost += comfort_weight * (sat_floor - sat_val) ** 2

        control_cost = np.sum(u_seq ** 2) * control_weight
        return energy_cost + comfort_cost + control_cost

    def _cem(self, x0: np.ndarray) -> np.ndarray:
        """Iterated CEM over H-step action sequences."""
        H = HORIZON
        mu = np.full(N_U * H, 0.5)
        sigma = np.full(N_U * H, 0.25)

        best_seq = mu.copy()
        best_cost = float('inf')

        for iteration in range(self.n_iter):
            # Sample candidates
            candidates = np.random.normal(
                mu[None, :], sigma[None, :], size=(self.pop, N_U * H))
            candidates = np.clip(candidates, 0.0, 1.0)

            # Evaluate each
            costs = np.zeros(self.pop)
            for k in range(self.pop):
                u_seq = candidates[k].reshape(H, N_U)
                traj = self._rollout(x0, u_seq)
                costs[k] = self._cost(traj, u_seq)

            # Select elite
            elite_idx = np.argsort(costs)[:self.elite]
            elite = candidates[elite_idx]

            if costs[elite_idx[0]] < best_cost:
                best_cost = costs[elite_idx[0]]
                best_seq = candidates[elite_idx[0]].copy()

            # Update distribution
            mu = elite.mean(axis=0)
            sigma = elite.std(axis=0) + 1e-4

        return best_seq

    def get_action(
        self, state: Dict[str, float], comfort_target: float
    ) -> Dict[str, float]:
        self.comfort_target = comfort_target

        x0 = np.array([state.get(v.lower(), state.get(v, 0.5))
                        for v in STATE_VARS])

        u_seq = self._cem(x0)
        u0 = u_seq[:N_U]
        return _unpack_action(u0)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Ensemble MPC — Bootstrap MLP ensemble + CEM (no TS-∞)
# ═══════════════════════════════════════════════════════════════════════════════

class EnsembleMPCController:
    """MPC with bootstrap ensemble of MLP dynamics models.

    Like PETS but WITHOUT Trajectory Sampling (TS-∞).  Uses the ensemble
    mean prediction for planning, capturing epistemic uncertainty only in
    the cost variance across ensemble members.  This isolates the value of
    PETS's TS-∞ uncertainty propagation.

    Reference: Kurutach et al., "Model-Ensemble TRPO", ICLR 2018.
    """

    def __init__(
        self,
        n_ensemble:      int   = 2,
        hidden:          Tuple[int, ...] = (64, 64),
        population:      int   = 120,
        elite_frac:      float = 0.2,
        n_iter:          int   = 8,
        comfort_penalty: float = 100.0,
    ):
        self.n_ensemble      = n_ensemble
        self.hidden          = hidden
        self.pop             = population
        self.elite           = max(5, int(population * elite_frac))
        self.n_iter          = n_iter
        self.comfort_penalty = comfort_penalty

        self.models: List[MLPRegressor] = []
        self.mean_x: np.ndarray = None
        self.comfort_target = 0.8

    def fit(self, df: pd.DataFrame) -> 'EnsembleMPCController':
        x_data, u_data, _ = _extract_data(df)
        T = len(x_data) - 1

        feat = np.hstack([x_data[:T], u_data[:T]])
        tgt = x_data[1:]
        self.mean_x = x_data.mean(axis=0)

        rng = np.random.default_rng(42)
        self.models = []
        r2s = []
        for i in range(self.n_ensemble):
            idx = rng.integers(0, T, size=T)  # bootstrap resample
            m = MLPRegressor(
                hidden_layer_sizes=self.hidden,
                activation='relu', solver='lbfgs',
                max_iter=1000, random_state=i)
            m.fit(feat[idx], tgt[idx])
            r2s.append(m.score(feat, tgt))
            self.models.append(m)

        logger.info(f"EnsembleMPC: {self.n_ensemble} MLPs, "
                    f"mean R²={np.mean(r2s):.3f}")
        return self

    def set_comfort_target(self, ct: float):
        self.comfort_target = ct

    def _rollout_mean(self, x0: np.ndarray, u_seq: np.ndarray) -> np.ndarray:
        """Roll out using ensemble mean prediction."""
        traj = [x0.copy()]
        x = x0.copy()
        for t in range(len(u_seq)):
            feat = np.concatenate([x, u_seq[t]]).reshape(1, -1)
            preds = np.array([m.predict(feat)[0] for m in self.models])
            x = preds.mean(axis=0)
            traj.append(x.copy())
        return np.array(traj)

    def _cost(self, trajectory: np.ndarray, u_seq: np.ndarray) -> float:
        sat_floor = max(0.3, 1.0 - self.comfort_target)
        e_idx = STATE_VARS.index('EnergyConsumption')
        s_idx = STATE_VARS.index('Satisfaction')

        energy_cost = sum(trajectory[t][e_idx] for t in range(1, len(trajectory)))
        comfort_cost = sum(
            self.comfort_penalty * max(0, sat_floor - trajectory[t][s_idx]) ** 2
            for t in range(1, len(trajectory)))
        control_cost = np.sum(u_seq ** 2) * 0.01
        return energy_cost + comfort_cost + control_cost

    def _cem(self, x0: np.ndarray) -> np.ndarray:
        H = HORIZON
        mu = np.full(N_U * H, 0.5)
        sigma = np.full(N_U * H, 0.3)

        best_seq = mu.copy()
        best_cost = float('inf')

        for iteration in range(self.n_iter):
            candidates = np.random.normal(
                mu[None, :], sigma[None, :], size=(self.pop, N_U * H))
            candidates = np.clip(candidates, 0.0, 1.0)

            costs = np.zeros(self.pop)
            for k in range(self.pop):
                u_seq = candidates[k].reshape(H, N_U)
                traj = self._rollout_mean(x0, u_seq)
                costs[k] = self._cost(traj, u_seq)

            elite_idx = np.argsort(costs)[:self.elite]
            elite = candidates[elite_idx]

            if costs[elite_idx[0]] < best_cost:
                best_cost = costs[elite_idx[0]]
                best_seq = candidates[elite_idx[0]].copy()

            mu = elite.mean(axis=0)
            sigma = elite.std(axis=0) + 1e-4

        return best_seq

    def get_action(
        self, state: Dict[str, float], comfort_target: float
    ) -> Dict[str, float]:
        self.comfort_target = comfort_target
        x0 = np.array([state.get(v.lower(), state.get(v, 0.5))
                        for v in STATE_VARS])
        u_seq = self._cem(x0)
        u0 = u_seq[:N_U]
        return _unpack_action(u0)


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_mpc_suite(df: pd.DataFrame) -> Dict[str, object]:
    """Create and fit both MPC variants."""
    linear = LinearMPCController().fit(df)
    ensemble = EnsembleMPCController().fit(df)
    return {
        'MPC_Linear': linear,
        'MPC_Ensemble': ensemble,
    }
