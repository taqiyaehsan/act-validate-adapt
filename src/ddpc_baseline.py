"""
DDPC Baseline Suite for PolicyGRID — Experiment B1
===================================================
Four Data-Driven Predictive Control variants, each using observational data
only (no causal graph, no interventions, no LLM).  Together they cover the
full spectrum from convex behavioral methods to probabilistic deep ensembles,
providing a rigorous multi-tier baseline for the Pareto comparison.

Variants
--------
DeePCController         Coulson, Lygeros & Dörfler, CDC 2019.
                        Willems' Fundamental Lemma: QP over Hankel-matrix
                        trajectory space, no parametric model identified.
                        Convex, data-efficient, academic SOTA.

SubspaceDDPCController  Lifted-ARX subspace identification (approx. N4SID)
                        + linear MPC QP.  Industry-standard for HVAC and
                        building energy management systems.

NeuralDDPCController    Two-layer MLP dynamics model + CEM trajectory
                        optimisation.  Represents modern deep-learning DDPC
                        with a single deterministic predictor.

PETSController          Probabilistic Ensemble Trajectory Sampling MPC.
                        Chua, Calandra, McAllister & Levine, NeurIPS 2018.
                        Bootstrap ensemble of MLPs + TS-∞ uncertainty
                        propagation + CEM.  Strongest nonlinear baseline:
                        captures epistemic uncertainty across the full
                        planning horizon.

All four report kWh / DH through the same _simulate_step infrastructure
as the existing policies, giving a fair apples-to-apples Pareto comparison.

Usage
-----
    from src.ddpc_baseline import create_ddpc_suite
    suite = create_ddpc_suite(dataset_df)
    action = suite['DDPC_PETS'].get_action(state, comfort_target=0.8)
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds
from sklearn.neural_network import MLPRegressor
from collections import deque
from typing import Dict, List, Tuple

import logging
logger = logging.getLogger(__name__)

# macOS Accelerate BLAS fires IEEE-754 edge-case flags (divide-by-zero,
# overflow, invalid) during matmul when the input data contains exact zeros.
# The computed results are mathematically correct and finite; these are
# spurious warnings from the BLAS C layer that bypass np.errstate.
# Suppress them module-wide so they don't flood the paper experiment logs.
for _m in ('divide by zero encountered in matmul',
           'overflow encountered in matmul',
           'invalid value encountered in matmul'):
    warnings.filterwarnings('ignore', category=RuntimeWarning, message=f'.*{_m}.*')

# ── column conventions (match ParetoOptimizer) ──────────────────────────────
STATE_VARS  = ['OutdoorTemp', 'SolarRadiation', 'Temperature', 'Humidity', 'CO2',
               'LightLevel', 'NoiseDB', 'AirQuality', 'PMV', 'HVACPower',
               'LightingPower', 'EnergyConsumption', 'Satisfaction']
ACTION_VARS = ['HVACPower', 'LightingPower']
OUTPUT_VARS = ['EnergyConsumption', 'Satisfaction']

N_U      = 2    # number of control inputs
N_Y      = 2    # number of performance outputs
HORIZON  = 12   # episode length (matches ParetoOptimizer.timesteps)


# ── shared helpers ───────────────────────────────────────────────────────────

def _build_hankel(seq: np.ndarray, depth: int) -> np.ndarray:
    """Block-Hankel matrix: shape (dim*depth, T-depth+1)."""
    T, dim = seq.shape
    n_col  = T - depth + 1
    if n_col <= 0:
        raise ValueError(f"Sequence length {T} < Hankel depth {depth}")
    H = np.zeros((dim * depth, n_col))
    for k in range(depth):
        H[k * dim : (k + 1) * dim, :] = seq[k : k + n_col].T
    return H


def _unpack_action(u: np.ndarray) -> Dict[str, float]:
    return {v: float(np.clip(u[i], 0.0, 1.0)) for i, v in enumerate(ACTION_VARS)}


def _normalize_state(state: Dict[str, float]) -> Dict[str, float]:
    """
    Case-insensitive state key resolution and optional percentage-to-[0,1]
    rescaling.  Maps incoming keys (any case) to the canonical STATE_VARS
    capitalisation so all downstream look-ups use a single convention.
    """
    lower_map = {k.lower(): v for k, v in state.items()}
    norm = {}
    for var in STATE_VARS:
        lk = var.lower()
        if lk in lower_map:
            norm[var] = lower_map[lk]
    # Keep extra keys not in STATE_VARS
    known = {v.lower() for v in STATE_VARS}
    for k, v in state.items():
        if k.lower() not in known:
            norm[k] = v
    # Rescale percentage-scale values
    for k in ('EnergyConsumption', 'Satisfaction'):
        if norm.get(k, 0.0) > 1.5:
            norm[k] = norm[k] / 100.0
    return norm


def _extract_arrays(data: pd.DataFrame):
    """Return (u_data, y_data, x_data) as numpy arrays from DataFrame.

    Uses case-insensitive column matching against STATE_VARS, ACTION_VARS,
    and OUTPUT_VARS so the caller does not need to worry about capitalisation.
    """
    col_map = {c.lower(): c for c in data.columns}
    state_cols = [col_map[v.lower()] for v in STATE_VARS if v.lower() in col_map]
    action_cols = [col_map[v.lower()] for v in ACTION_VARS if v.lower() in col_map]
    output_cols = [col_map[v.lower()] for v in OUTPUT_VARS if v.lower() in col_map]

    df = data[state_cols].dropna().reset_index(drop=True)
    u_data = df[action_cols].values    # (T, n_u)
    y_data = df[output_cols].values    # (T, n_y)
    x_data = df.values                  # (T, n_state)
    return u_data, y_data, x_data


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  DeePC — Behavioral / Willems' Fundamental Lemma
# ═══════════════════════════════════════════════════════════════════════════════

class DeePCController:
    """
    Data-enabled Predictive Control (Coulson, Lygeros & Dörfler, CDC 2019).

    Core idea
    ---------
    By Willems' Fundamental Lemma, any length-L trajectory of a controllable
    LTI system lies in the column span of the Hankel matrix H_L(u_d, y_d).
    We use this to predict future outputs given past data *directly*, without
    fitting a parametric model.

    QP formulation (reduced to u_f ∈ R^{N*n_u})
    ---------------------------------------------
    Pre-compute the "behavioral prediction map":
        y_f = Q3 @ u_f + Q12 @ [u_ini; y_ini]   (minimum-norm Hankel solution)

    Then at each step solve:
        min_{u_f ∈ [0,1]^{N*n_u}}
            Q_e * || energy rows of y_f - 0 ||²
          + λ_u * ||u_f||²
        s.t.  satisfaction rows of y_f ≥ sat_floor

    This is a 36-variable convex QP (fast SLSQP).
    """

    T_INI = 3    # initialisation window (past steps seen before planning)

    def __init__(
        self,
        lam_u:    float = 1e-1,
        Q_energy: float = 1.0,
    ):
        self.lam_u    = lam_u
        self.Q_energy = Q_energy

        # Precomputed prediction maps (set by fit)
        self.Q3:   np.ndarray = None   # (N*n_y, N*n_u)
        self.Q12:  np.ndarray = None   # (N*n_y, T_ini*(n_u+n_y))
        self.mean_u: np.ndarray = None
        self.mean_y: np.ndarray = None

        # Receding-horizon history queues
        self._u_hist: deque = deque(maxlen=self.T_INI)
        self._y_hist: deque = deque(maxlen=self.T_INI)

    # ── fit ────────────────────────────────────────────────────────────────────
    def fit(self, u_data: np.ndarray, y_data: np.ndarray) -> 'DeePCController':
        """
        Build Hankel matrices and precompute Q3, Q12.

        Parameters
        ----------
        u_data : (T, n_u)  input (action) trajectory
        y_data : (T, n_y)  output trajectory [EnergyConsumption, Satisfaction]
        """
        T_ini, H = self.T_INI, HORIZON
        L = T_ini + H

        H_u = _build_hankel(u_data, L)   # (n_u*L, T_col)
        H_y = _build_hankel(y_data, L)   # (n_y*L, T_col)
        
        # after building H_u, H_y
        idx = np.random.choice(H_u.shape[1], size=int(0.7 * H_u.shape[1]), replace=False)
        H_u = H_u[:, idx]
        H_y = H_y[:, idx]

        # Partition
        U_p = H_u[:N_U * T_ini, :]       # past  inputs   (n_u*T_ini, T_col)
        Y_p = H_y[:N_Y * T_ini, :]       # past  outputs  (n_y*T_ini, T_col)
        U_f = H_u[N_U * T_ini:, :]       # future inputs  (n_u*H, T_col)
        Y_f = H_y[N_Y * T_ini:, :]       # future outputs (n_y*H, T_col)

        # Consistency matrix: A g = [u_ini; y_ini; u_f]
        A = np.vstack([U_p, Y_p, U_f])   # (n_u*T_ini + n_y*T_ini + n_u*H, T_col)
        n_ini = N_U * T_ini + N_Y * T_ini

        # Behavioral prediction: y_f = Y_f @ pinv(A) @ [u_ini; y_ini; u_f]
        #                             = Q12 @ [u_ini; y_ini]  +  Q3 @ u_f
        # A is a fat matrix (51 rows, ~986 cols).  Use a Tikhonov-regularised
        # pseudoinverse to avoid amplifying near-zero singular values which
        # cause divide-by-zero in downstream matmul operations.
        # pinv_reg(A) = A.T @ inv(A @ A.T + λI)
        # np.errstate: macOS Accelerate BLAS fires IEEE-754 edge-case warnings
        # on exact-zero data even when results are finite — suppress them here.
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            AA_T    = A @ A.T                                     # (rows_A, rows_A)
            AA_T   += 1e-6 * np.eye(AA_T.shape[0])               # Tikhonov reg
            YfA     = Y_f @ A.T                                   # (n_y*H, rows_A)
            YfApinv = np.linalg.solve(AA_T.T, YfA.T).T           # (n_y*H, rows_A)
        self.Q12  = YfApinv[:, :n_ini]           # (n_y*H, n_ini)
        self.Q3   = YfApinv[:, n_ini:]           # (n_y*H, n_u*H)

        self.mean_u = u_data.mean(axis=0)
        self.mean_y = y_data.mean(axis=0)
        self._reset_history()
        logger.info(f"DeePCController fitted  T_col={H_u.shape[1]}, Q3={self.Q3.shape}")
        return self

    # ── episode bookkeeping ────────────────────────────────────────────────────
    def _reset_history(self):
        for _ in range(self.T_INI):
            self._u_hist.append(self.mean_u.copy())
            self._y_hist.append(self.mean_y.copy())

    def reset(self):
        """Call at the start of each new episode."""
        self._reset_history()

    # ── action selection ───────────────────────────────────────────────────────
    def get_action(
        self, state: Dict[str, float], comfort_target: float
    ) -> Dict[str, float]:
        """Solve QP; return first-step action; update internal window."""
        state = _normalize_state(state)

        u_ini = np.concatenate(list(self._u_hist))   # (n_u*T_ini,)
        y_ini = np.concatenate(list(self._y_hist))   # (n_y*T_ini,)
        base  = self.Q12 @ np.concatenate([u_ini, y_ini])  # (n_y*H,)

        Q3 = self.Q3  # (n_y*H, n_u*H)

        # Energy rows (index 0 of each n_y block)
        e_rows = [N_Y * k + 0 for k in range(HORIZON)]
        Q3_e   = Q3[e_rows, :]
        base_e = base[e_rows]

        # Quadratic objective: min Q_e*||Q3_e u_f + base_e||² + λ_u ||u_f||²
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            H_mat = self.Q_energy * (Q3_e.T @ Q3_e) + self.lam_u * np.eye(N_U * HORIZON)
            c_vec = self.Q_energy * (Q3_e.T @ base_e)

        def obj(uf):
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                return float(0.5 * uf @ H_mat @ uf + c_vec @ uf)

        def jac(uf):
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                return H_mat @ uf + c_vec

        # Satisfaction constraint: Q3_s u_f + base_s ≥ sat_floor
        s_rows    = [N_Y * k + 1 for k in range(HORIZON)]
        Q3_s      = Q3[s_rows, :]
        base_s    = base[s_rows]
        sat_floor = 1.0 - comfort_target

        constraints = [{
            'type': 'ineq',
            'fun':  lambda uf: Q3_s @ uf + base_s - sat_floor,
            'jac':  lambda uf: Q3_s,
        }]

        res = minimize(
            obj, np.full(N_U * HORIZON, 0.5), jac=jac,
            method='SLSQP',
            bounds=Bounds(lb=0.0, ub=1.0),
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-6},
        )

        u_f = np.clip(res.x, 0.0, 1.0)
        action = _unpack_action(u_f[:N_U])

        # Update history with current observation
        self._u_hist.append(np.array([action.get(v, 0.5) for v in ACTION_VARS]))
        self._y_hist.append(np.array([
            state.get('EnergyConsumption',   self.mean_y[0]),
            state.get('Satisfaction', self.mean_y[1]),
        ]))
        return action


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Subspace DDPC — N4SID-style identification + linear MPC
# ═══════════════════════════════════════════════════════════════════════════════

class SubspaceDDPCController:
    """
    Subspace Predictive Control (industry standard, approximates N4SID/MOESP).

    Step 1 – System Identification
        Lift state as x_t = [y_{t-p},...,y_{t-1}, u_{t-q},...,u_{t-1}].
        Fit x_{t+1} = A x_t + B u_t via least squares.
        Extract output matrix C (selects most-recent y from lifted state).

    Step 2 – Linear MPC
        Unroll N-step prediction:  Y = F_u @ U + F_x @ x_0
        Solve convex QP (SLSQP) in U ∈ R^{N*n_u}:
            min   ones' @ (F_u_energy @ U + base_e)  +  λ_u ||U||²
            s.t.  F_u_sat @ U + base_s ≥ sat_floor
                  0 ≤ U ≤ 1
    """

    def __init__(
        self,
        lag_y: int   = 2,
        lag_u: int   = 2,
        lam_u: float = 1e-1,
    ):
        self.lag_y = lag_y
        self.lag_u = lag_u
        self.lam_u = lam_u

        self.A: np.ndarray  = None
        self.B: np.ndarray  = None
        self.C: np.ndarray  = None
        self.n_x: int       = None
        self.mean_x: np.ndarray = None
        self._F_u: np.ndarray   = None
        self._F_x: np.ndarray   = None
        self._x_cur: np.ndarray = None

    # ── identification ─────────────────────────────────────────────────────────
    def fit(
        self, u_data: np.ndarray, y_data: np.ndarray
    ) -> 'SubspaceDDPCController':
        T    = len(u_data)
        p, q = self.lag_y, self.lag_u
        lag  = max(p, q)

        n_x   = N_Y * p + N_U * q
        self.n_x = n_x
        n_samples = T - lag - 1

        X_t    = np.zeros((n_samples, n_x))
        X_next = np.zeros((n_samples, n_x))

        for t in range(lag, T - 1):
            y_lag = y_data[t - p : t].flatten()
            u_lag = u_data[t - q : t].flatten()
            X_t[t - lag] = np.concatenate([y_lag, u_lag])

            y_lag_n = y_data[t - p + 1 : t + 1].flatten()
            u_lag_n = u_data[t - q + 1 : t + 1].flatten()
            X_next[t - lag] = np.concatenate([y_lag_n, u_lag_n])

        U_t = u_data[lag : T - 1]

        features = np.hstack([X_t, U_t])
        AB, _, _, _ = np.linalg.lstsq(features, X_next, rcond=1e-2)
        self.A = AB[:n_x, :].T     # (n_x, n_x)
        self.B = AB[n_x:, :].T     # (n_x, n_u)

        # Stabilize A: clamp spectral radius to < 0.99 so matrix_power(A, i)
        # stays bounded.  Unstable modes cause inf/nan in _build_prediction_matrices.
        eigvals, eigvecs = np.linalg.eig(self.A)
        scale = np.where(np.abs(eigvals) > 0.99,
                         0.99 / (np.abs(eigvals) + 1e-12),
                         np.ones_like(np.abs(eigvals)))
        eigvals_stable = eigvals * scale
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            self.A = (eigvecs @ np.diag(eigvals_stable) @ np.linalg.inv(eigvecs)).real

        # C: extract most-recent y from lifted state (last y-block in y_part)
        self.C = np.zeros((N_Y, n_x))
        self.C[:, N_Y * (p - 1) : N_Y * p] = np.eye(N_Y)

        self.mean_x = X_t.mean(axis=0)
        self._build_prediction_matrices()
        logger.info(
            f"SubspaceDDPC fitted  A{self.A.shape}  B{self.B.shape}  "
            f"C{self.C.shape}  n_x={n_x}"
        )
        return self

    def _build_prediction_matrices(self):
        """Cache F_u, F_x for the N-step prediction Y = F_u U + F_x x0."""
        A, B, C = self.A, self.B, self.C
        F_u = np.zeros((HORIZON * N_Y, HORIZON * N_U))
        F_x = np.zeros((HORIZON * N_Y, self.n_x))
        Ak  = np.eye(self.n_x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for i in range(HORIZON):
                Ak = A @ Ak if i > 0 else A
                F_x[i * N_Y : (i+1) * N_Y, :] = C @ Ak
                for j in range(i + 1):
                    exp = np.linalg.matrix_power(A, i - j)
                    F_u[i * N_Y : (i+1) * N_Y,
                        j * N_U : (j+1) * N_U] = C @ exp @ B
        self._F_u = F_u
        self._F_x = F_x

    # ── episode bookkeeping ────────────────────────────────────────────────────
    def reset(self):
        self._x_cur = self.mean_x.copy() if self.mean_x is not None else np.zeros(self.n_x)

    # ── action selection ───────────────────────────────────────────────────────
    def get_action(
        self, state: Dict[str, float], comfort_target: float
    ) -> Dict[str, float]:
        state = _normalize_state(state)
        if self._x_cur is None:
            self.reset()

        base = self._F_x @ self._x_cur   # (N*N_Y,)

        e_rows = [N_Y * k + 0 for k in range(HORIZON)]
        s_rows = [N_Y * k + 1 for k in range(HORIZON)]
        F_u_e  = self._F_u[e_rows, :]
        F_u_s  = self._F_u[s_rows, :]
        base_e = base[e_rows]
        base_s = base[s_rows]

        # min  sum(F_u_e U + base_e) + λ||U||²
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            lin   = F_u_e.sum(axis=0)
        H_mat = self.lam_u * np.eye(HORIZON * N_U)

        def obj(U):
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                return float(0.5 * U @ H_mat @ U + lin @ U)

        def jac(U):
            with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
                return H_mat @ U + lin

        sat_floor   = 1.0 - comfort_target
        constraints = [{
            'type': 'ineq',
            'fun':  lambda U: F_u_s @ U + base_s - sat_floor,
            'jac':  lambda U: F_u_s,
        }]

        res = minimize(
            obj, np.full(HORIZON * N_U, 0.5), jac=jac,
            method='SLSQP',
            bounds=Bounds(lb=0.0, ub=1.0),
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-6},
        )

        U_opt  = np.clip(res.x, 0.0, 1.0)
        action = _unpack_action(U_opt[:N_U])

        # Update lifted state
        y_obs = np.array([
            state.get('EnergyConsumption',   self.mean_x[0]),
            state.get('Satisfaction', self.mean_x[1]),
        ])
        u_act = np.array([action.get(v, 0.5) for v in ACTION_VARS])

        p, q = self.lag_y, self.lag_u
        y_part = self._x_cur[:N_Y * p].reshape(p, N_Y)
        u_part = self._x_cur[N_Y * p:].reshape(q, N_U)
        self._x_cur = np.concatenate([
            np.vstack([y_part[1:], y_obs]).flatten(),
            np.vstack([u_part[1:], u_act]).flatten(),
        ])
        return action


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Neural DDPC — MLP dynamics + CEM shooting
# ═══════════════════════════════════════════════════════════════════════════════

class NeuralDDPCController:
    """
    Neural Data-Driven Predictive Control.

    Trains a two-hidden-layer MLP on (x_t, u_t) → x_{t+1} pairs from
    observational data, then uses Cross-Entropy Method (CEM) to optimise
    action sequences over the N-step horizon.

    CEM planning cost (internal):
        J = Σ_t E_t  +  λ * Σ_t max(0, δ − S_t)
    E_t, S_t ∈ [0,1] from MLP predictions; δ = 1 − comfort_target.

    Reported kWh / DH come from the real simulator — fair comparison.
    """

    def __init__(
        self,
        hidden:          Tuple[int, ...] = (32, 16),
        population:      int   = 120,
        elite_frac:      float = 0.2,
        n_iter:          int   = 5,
        comfort_penalty: float = 100.0,
    ):
        self.hidden          = hidden
        self.pop             = population
        self.elite           = max(5, int(population * elite_frac))
        self.n_iter          = n_iter
        self.comfort_penalty = comfort_penalty

        self.model: MLPRegressor    = None
        self.mean_state: np.ndarray = None

    # ── fit ────────────────────────────────────────────────────────────────────
    def fit(
        self, x_data: np.ndarray, u_data: np.ndarray
    ) -> 'NeuralDDPCController':
        """
        Train on (x_t → x_{t+1}) pairs.

        Action variables (HVACPower, LightingPower) are already embedded in
        x_data at the correct column positions, so no feature injection is
        needed during training.  During rollout, proposed actions are injected
        at self._action_idx.
        """
        T = len(x_data) - 1
        self._action_idx = [STATE_VARS.index(v) for v in ACTION_VARS]
        feat = x_data[:T]
        tgt  = x_data[1:]

        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden,
            activation='relu',
            solver='lbfgs',       # more numerically stable than adam on small data
            max_iter=1000,
            random_state=42,
        )
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            warnings.simplefilter('ignore')   # also catches ConvergenceWarning
            self.model.fit(feat, tgt)
        self.mean_state = x_data.mean(axis=0)
        logger.info(f"NeuralDDPC fitted  MLP{self.hidden}  {T} transitions")
        return self

    # ── rollout ────────────────────────────────────────────────────────────────
    def _rollout(self, init_vec: np.ndarray, action_seq: np.ndarray) -> np.ndarray:
        """Multi-step rollout.  Injects proposed actions at the correct state
        column positions before calling the MLP."""
        traj, state = [], init_vec.copy()
        for t in range(HORIZON):
            feat = state.copy()
            for i, idx in enumerate(self._action_idx):
                feat[idx] = action_seq[t][i]
            state = np.clip(self.model.predict(feat.reshape(1, -1))[0], 0.0, 1.0)
            traj.append(state)
        return np.array(traj)   # (H, n_state)

    def _score(
        self, init_vec: np.ndarray, action_seq: np.ndarray, comfort_target: float
    ) -> float:
        traj  = self._rollout(init_vec, action_seq)
        e_idx = STATE_VARS.index('EnergyConsumption')
        s_idx = STATE_VARS.index('Satisfaction')
        energy_cost   = traj[:, e_idx].sum()
        comfort_viol  = np.maximum(0.0, (1.0 - comfort_target) - traj[:, s_idx]).sum()
        return energy_cost + self.comfort_penalty * comfort_viol

    # ── CEM optimiser ──────────────────────────────────────────────────────────
    def _cem(self, init_vec: np.ndarray, comfort_target: float) -> np.ndarray:
        rng   = np.random.default_rng()
        mu    = np.full((HORIZON, N_U), 0.5)
        sigma = np.full((HORIZON, N_U), 0.2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for _ in range(self.n_iter):
                samples = np.clip(rng.normal(mu, sigma, (self.pop, HORIZON, N_U)), 0.0, 1.0)
                scores  = np.array([self._score(init_vec, s, comfort_target) for s in samples])
                elite   = samples[np.argsort(scores)[:self.elite]]
                mu      = elite.mean(0)
                sigma   = elite.std(0) + 1e-6
        return mu

    # ── public ─────────────────────────────────────────────────────────────────
    def reset(self):
        pass   # stateless

    def get_action(
        self, state: Dict[str, float], comfort_target: float
    ) -> Dict[str, float]:
        state    = _normalize_state(state)
        init_vec = np.array([state.get(v, m)
                             for v, m in zip(STATE_VARS, self.mean_state)])
        seq      = self._cem(init_vec, comfort_target)
        return _unpack_action(seq[0])


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  PETS — Probabilistic Ensemble Trajectory Sampling MPC
# ═══════════════════════════════════════════════════════════════════════════════

class PETSController:
    """
    Probabilistic Ensemble Trajectory Sampling MPC (PETS-style).

    Reference
    ---------
    Chua, Calandra, McAllister & Levine, "Deep reinforcement learning in a
    handful of trials using probabilistic dynamics models", NeurIPS 2018.

    Why it is stronger than NeuralDDPC
    ------------------------------------
    NeuralDDPC uses a single MLP → deterministic, point-estimate predictions.
    PETS uses *n_ensemble* MLPs trained on bootstrap resamples, capturing
    epistemic uncertainty.  During planning it uses Trajectory Sampling-∞
    (TS-∞): each particle independently resamples an ensemble member at every
    timestep.  This propagates model uncertainty through the full horizon
    instead of collapsing to a mean after the first step.

    Implementation choices
    ----------------------
    * Bootstrap resampling per ensemble member (standard Bayesian deep ensemble
      approximation, Lakshminarayanan et al. 2017).
    * Same feature injection as NeuralDDPC (action injected into T/H/AQ slots)
      to avoid rank-deficient inputs on observational data.
    * TS-∞ rollout vectorised within each timestep: particles grouped by their
      sampled ensemble member → only n_ensemble batch predict() calls per step,
      not n_particles calls.
    * CEM planner scores each candidate by the *mean* cost across particles
      (expected cost under ensemble uncertainty).
    """

    def __init__(
        self,
        n_ensemble:      int   = 2,
        hidden:          Tuple[int, ...] = (64, 64),
        n_particles:     int   = 5,
        population:      int   = 150,
        elite_frac:      float = 0.1,
        n_iter:          int   = 10,
        comfort_penalty: float = 100.0,
    ):
        self.n_ensemble      = n_ensemble
        self.hidden          = hidden
        self.n_particles     = n_particles
        self.pop             = population
        self.elite           = max(5, int(population * elite_frac))
        self.n_iter          = n_iter
        self.comfort_penalty = comfort_penalty

        self.models:     List[MLPRegressor] = []
        self.mean_state: np.ndarray         = None

    # ── fit ────────────────────────────────────────────────────────────────────
    def fit(
        self, x_data: np.ndarray, u_data: np.ndarray
    ) -> 'PETSController':
        """Train each ensemble member on an independent bootstrap resample."""
        T = len(x_data) - 1
        self._action_idx = [STATE_VARS.index(v) for v in ACTION_VARS]
        feat_all = x_data[:T]   # actions already in state columns
        tgt_all  = x_data[1:]

        rng_boot   = np.random.default_rng(42)
        self.models = []
        for i in range(self.n_ensemble):
            idx = rng_boot.integers(0, T, size=T)   # bootstrap resample
            m = MLPRegressor(
                hidden_layer_sizes=self.hidden,
                activation='relu',
                solver='lbfgs',
                max_iter=1000,
                random_state=i,
            )
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                m.fit(feat_all[idx], tgt_all[idx])
            self.models.append(m)

        self.mean_state = x_data.mean(axis=0)
        logger.info(
            f"PETSController fitted  ensemble={self.n_ensemble}  "
            f"MLP{self.hidden}  {T} transitions"
        )
        return self

    # ── TS-∞ rollout ───────────────────────────────────────────────────────────
    def _rollout_ts(
        self, init_vec: np.ndarray, action_seq: np.ndarray
    ) -> np.ndarray:
        """
        Trajectory Sampling-∞ (TS-∞): n_particles particles, each sampling a
        fresh ensemble member at every timestep.

        Vectorised within each step: particles are grouped by their sampled
        ensemble member and batch-predicted, so the inner loop makes at most
        n_ensemble predict() calls per timestep (not n_particles).

        Returns the particle-mean trajectory: (H, n_state).
        """
        rng    = np.random.default_rng()
        states = np.tile(init_vec, (self.n_particles, 1))   # (P, n_state)
        traj   = np.zeros((HORIZON, len(STATE_VARS)))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for t in range(HORIZON):
                feats       = states.copy()
                for i, idx in enumerate(self._action_idx):
                    feats[:, idx] = action_seq[t][i]

                member_ids  = rng.integers(0, self.n_ensemble, size=self.n_particles)
                next_states = np.empty_like(states)

                for m_id in range(self.n_ensemble):
                    mask = member_ids == m_id
                    if mask.any():
                        next_states[mask] = np.clip(
                            self.models[m_id].predict(feats[mask]), 0.0, 1.0
                        )

                states  = next_states
                traj[t] = states.mean(axis=0)

        return traj   # (H, n_state)

    def _score(
        self, init_vec: np.ndarray, action_seq: np.ndarray, comfort_target: float
    ) -> float:
        traj         = self._rollout_ts(init_vec, action_seq)
        e_idx        = STATE_VARS.index('EnergyConsumption')
        s_idx        = STATE_VARS.index('Satisfaction')
        energy_cost  = traj[:, e_idx].sum()
        comfort_viol = np.maximum(0.0, (1.0 - comfort_target) - traj[:, s_idx]).sum()
        return energy_cost + self.comfort_penalty * comfort_viol

    # ── CEM optimiser ──────────────────────────────────────────────────────────
    def _cem(self, init_vec: np.ndarray, comfort_target: float) -> np.ndarray:
        rng   = np.random.default_rng()
        mu    = np.full((HORIZON, N_U), 0.5)
        sigma = np.full((HORIZON, N_U), 0.2)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            for _ in range(self.n_iter):
                samples = np.clip(rng.normal(mu, sigma, (self.pop, HORIZON, N_U)), 0.0, 1.0)
                scores  = np.array([self._score(init_vec, s, comfort_target) for s in samples])
                elite   = samples[np.argsort(scores)[:self.elite]]
                mu      = elite.mean(0)
                sigma   = elite.std(0) + 1e-6
        return mu

    # ── public ─────────────────────────────────────────────────────────────────
    def reset(self):
        pass   # stateless

    def get_action(
        self, state: Dict[str, float], comfort_target: float
    ) -> Dict[str, float]:
        state    = _normalize_state(state)
        init_vec = np.array([state.get(v, m)
                             for v, m in zip(STATE_VARS, self.mean_state)])
        seq      = self._cem(init_vec, comfort_target)
        return _unpack_action(seq[0])


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

def create_ddpc_suite(
    data:            pd.DataFrame,
    deepc_lam_u:     float = 1e-1,
    subspace_lag:    int   = 2,
    neural_hidden:   Tuple = (32, 16),
    pets_hidden:     Tuple = (32, 16),
) -> Dict[str, object]:
    """
    Fit all four DDPC controllers on *data* and return a name → controller dict.

    Keys
    ----
    'DDPC_Behavioral'  DeePCController
    'DDPC_Subspace'    SubspaceDDPCController
    'DDPC_Neural'      NeuralDDPCController
    'DDPC_PETS'        PETSController

    Parameters
    ----------
    data            : DataFrame with smart-room columns (normalized 0-1)
    deepc_lam_u     : DeePC input regularisation weight
    subspace_lag    : lag order p = q for subspace identification
    neural_hidden   : MLP hidden-layer sizes for NeuralDDPC
    pets_hidden     : MLP hidden-layer sizes per ensemble member (PETS)
    pets_ensemble   : number of bootstrap ensemble members (PETS)
    pets_particles  : TS-∞ particles per CEM candidate (PETS)
    cem_population  : CEM population per planning call
    cem_n_iter      : CEM iterations per planning call
    """
    u_data, y_data, x_data = _extract_arrays(data)

    suite: Dict[str, object] = {}

    logger.info("── Fitting DeePC (Behavioral) ──────────────────────────────")
    suite['DDPC_Behavioral'] = DeePCController(lam_u=deepc_lam_u).fit(u_data, y_data)

    logger.info("── Fitting Subspace DDPC ───────────────────────────────────")
    suite['DDPC_Subspace'] = SubspaceDDPCController(
        lag_y=subspace_lag, lag_u=subspace_lag
    ).fit(u_data, y_data)

    logger.info("── Fitting Neural DDPC ─────────────────────────────────────")
    suite['DDPC_Neural'] = NeuralDDPCController(
        hidden=neural_hidden,
    ).fit(x_data, u_data)

    logger.info("── Fitting PETS DDPC ───────────────────────────────────────")
    suite['DDPC_PETS'] = PETSController(
        hidden=pets_hidden,
    ).fit(x_data, u_data)

    logger.info("DDPC suite ready: " + ", ".join(suite.keys()))
    return suite
