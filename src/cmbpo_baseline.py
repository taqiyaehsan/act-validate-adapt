"""
C-MBPO Baseline for PolicyGRID — Causal Model-Based Policy Optimization
=========================================================================
Uses the SAME validated causal graph as PolicyGRID to build a factored
dynamics model, then trains a neural network policy via Dyna-style
model-based reinforcement learning.

This isolates PolicyGRID's contribution: if PolicyGRID beats C-MBPO,
the SEM gradient optimizer is better than RL on the same causal structure.
If C-MBPO beats PolicyGRID, the causal structure alone (without the
analytical optimizer) is sufficient.

Architecture:
  - Causal dynamics: per-variable Ridge regression using only DAG parents.
  - Policy: MLP mapping sensor observations to actions.
  - Training: Dyna-style — real simulator episodes + imagined rollouts
    on the causal model.
  - Policy update: REINFORCE with baseline (advantage-weighted regression).

Reference
---------
Janner, Fu, Zhang & Levine. "When to Trust Your Model: Model-Based
    Policy Optimization." NeurIPS 2019.
    https://arxiv.org/abs/1906.08253

Causal model-based RL:
Huang, Lu, Du, Lim, Zha & Zhang. "Action-Sufficient State Representation
    for Causal RL." NeurIPS 2022.
Lu, Huang, Zhang & Sch\"olkopf. "Causal RL: A Survey." 2024.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from typing import Dict, Set, List, Tuple
import logging

logger = logging.getLogger(__name__)

# Same interface as DDPC baselines
ACTION_VARS = ['HVACPower', 'LightingPower']
OUTPUT_VARS = ['EnergyConsumption', 'Satisfaction']
SENSOR_VARS = ['Temperature', 'Humidity', 'CO2', 'LightLevel', 'NoiseDB',
               'AirQuality', 'PMV', 'OutdoorTemp', 'SolarRadiation']
N_U = 2
N_Y = 2


def _unpack_action(u: np.ndarray) -> Dict[str, float]:
    return {v: float(np.clip(u[i], 0.0, 1.0)) for i, v in enumerate(ACTION_VARS)}


class CMBPOController:
    """Causal Model-Based Policy Optimization.

    Uses the validated causal DAG to build a FACTORED dynamics model
    (each variable predicted from its parents only), then trains an
    MLP policy via Dyna-style RL (real + imagined episodes).

    The key difference from PolicyGRID:
      - Same causal graph, same factored dynamics.
      - PolicyGRID: analytical SEM gradient optimizer.
      - C-MBPO: learned MLP policy via REINFORCE.

    The key difference from PETS/NeuralDDPC:
      - PETS: black-box dynamics (all variables predict all variables).
      - C-MBPO: factored dynamics (each variable predicted from DAG parents only).
    """

    def __init__(
        self,
        edges:              Set[Tuple[str, str]] = None,
        n_real_episodes:    int   = 30,
        n_imagined_per_real: int  = 1,
        episode_len:        int   = 60,
        policy_hidden:      Tuple[int, ...] = (64, 32),
        policy_lr:          float = 0.003,
        gamma:              float = 0.99,
    ):
        self.edges = edges or set()
        self.n_real_episodes = n_real_episodes
        self.n_imagined_per_real = n_imagined_per_real
        self.episode_len = episode_len
        self.policy_hidden = policy_hidden
        self.policy_lr = policy_lr
        self.gamma = gamma

        self.causal_models: Dict[str, Dict] = {}  # var → {model, parents}
        self.policy_mlp: MLPRegressor = None
        self.value_mlp: MLPRegressor = None
        self.comfort_target: float = 0.8
        self._trained: bool = False
        self._replay: list = []

    def set_comfort_target(self, ct: float):
        self.comfort_target = ct

    def _get_parents(self, var: str) -> List[str]:
        """Get causal parents of var from DAG, intersected with observables."""
        parents = set()
        for s, t in self.edges:
            if t.lower() == var.lower():
                parents.add(s)
        return sorted(parents)

    def _build_causal_dynamics(self, data: pd.DataFrame):
        """Build factored dynamics: each y_i ~ f(pa(y_i)).

        Unlike black-box MPC, this respects causal structure: only
        parents in the DAG are used as features for each variable.
        """
        col_map = {c.lower(): c for c in data.columns}
        all_targets = SENSOR_VARS + list(OUTPUT_VARS)

        for var in all_targets:
            parents = self._get_parents(var)
            if not parents:
                parents = list(SENSOR_VARS)  # fallback: use all sensors

            # Resolve column names
            parent_cols = []
            for p in parents:
                pcol = col_map.get(p.lower())
                if pcol:
                    parent_cols.append(pcol)

            target_col = col_map.get(var.lower())
            if target_col is None or not parent_cols:
                continue

            # Fit: y_{t+1} ~ f(parents_t)
            X = data[parent_cols].values[:-1]
            Y = data[target_col].values[1:]

            model = Ridge(alpha=2.0)
            model.fit(X, Y)
            self.causal_models[var.lower()] = {
                'model': model,
                'parents': [p.lower() for p in parents],
                'r2': model.score(X, Y),
            }

        avg_r2 = np.mean([m['r2'] for m in self.causal_models.values()])
        logger.info(f"C-MBPO: causal dynamics for {len(self.causal_models)} vars, "
                    f"mean R²={avg_r2:.3f}")

    def _causal_predict(self, state: Dict[str, float], action: Dict[str, float]) -> Dict[str, float]:
        """One-step prediction using causal dynamics model."""
        full = dict(state)
        full.update({k.lower(): v for k, v in action.items()})

        next_state = dict(state)
        for var, info in self.causal_models.items():
            features = np.array([full.get(p, 0.5) for p in info['parents']])
            pred = info['model'].predict(features.reshape(1, -1))[0]
            next_state[var] = float(np.clip(pred, 0.0, 1.0))
        return next_state

    def _state_vec(self, state: Dict[str, float]) -> np.ndarray:
        return np.array([state.get(s.lower(), 0.5) for s in SENSOR_VARS])

    def _init_policy(self):
        """Initialize MLP policy and value networks."""
        n_in = len(SENSOR_VARS)
        n_out = N_U

        self.policy_mlp = MLPRegressor(
            hidden_layer_sizes=self.policy_hidden,
            max_iter=1, solver='sgd', random_state=42,
            learning_rate_init=self.policy_lr,
            warm_start=True, alpha=0.001)
        self.value_mlp = MLPRegressor(
            hidden_layer_sizes=(32,),
            max_iter=1, solver='sgd', random_state=42,
            learning_rate_init=self.policy_lr * 2,
            warm_start=True, alpha=0.001)

        # Warm-start
        X_init = np.random.uniform(0.2, 0.8, size=(100, n_in))
        u_init = np.full((100, n_out), 0.4)
        v_init = np.full((100, 1), 0.5)
        self.policy_mlp.fit(X_init, u_init)
        self.value_mlp.fit(X_init, v_init.ravel())

    def _get_policy_action(self, state: Dict[str, float], explore: bool = False) -> Dict[str, float]:
        s_vec = self._state_vec(state).reshape(1, -1)
        raw = self.policy_mlp.predict(s_vec)[0]
        if explore:
            raw = raw + np.random.normal(0, 0.08, size=raw.shape)
        raw = np.clip(raw, 0.0, 0.8)
        return _unpack_action(raw)

    def _reward(self, state: Dict[str, float], action: Dict[str, float]) -> float:
        sat = state.get('overallsatisfaction', state.get('satisfaction', 0.5))
        eng_frac = sum(action.get(a.lower(), 0.3) for a in ACTION_VARS) / N_U
        sat_weight = max(1.0, min(3.0, 2.5 - self.comfort_target))
        return sat_weight * sat - eng_frac

    def _compute_advantages(self, trajectory: list) -> list:
        if not trajectory:
            return []
        states = np.array([self._state_vec(t['state']) for t in trajectory])
        values = self.value_mlp.predict(states)

        returns = np.zeros(len(trajectory))
        G = 0.0
        for i in reversed(range(len(trajectory))):
            G = trajectory[i]['reward'] + self.gamma * G
            returns[i] = G

        advantages = returns - values
        batch = []
        for i, t in enumerate(trajectory):
            batch.append({
                'state': t['state'], 'action': t['action'],
                'return': returns[i], 'advantage': advantages[i],
            })
        return batch

    def _update_policy(self, batch: list):
        if len(batch) < 10:
            return

        states = np.array([self._state_vec(t['state']) for t in batch])
        actions = np.array([[t['action'].get(a.lower(), 0.4) for a in ACTION_VARS]
                            for t in batch])
        advantages = np.array([t['advantage'] for t in batch])
        returns = np.array([t['return'] for t in batch])

        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Advantage-weighted regression: move toward good actions
        current = self.policy_mlp.predict(states)
        targets = current.copy()
        for i in range(len(batch)):
            if advantages[i] > 0:
                # targets[i] = 0.7 * current[i] + 0.3 * actions[i]
                targets[i] = 0.85 * current[i] + 0.1 * actions[i]
            else:
                targets[i] = current[i] + 0.1 * (current[i] - actions[i])
        targets = np.clip(targets, 0.0, 0.8)

        self.policy_mlp.partial_fit(states, targets)
        self.value_mlp.partial_fit(states, returns)

    def fit(
        self,
        data: pd.DataFrame,
        smap: Dict[str, Tuple[float, float]] = None,
        step_fn=None,
    ) -> 'CMBPOController':
        """Train C-MBPO.

        If step_fn is provided, uses it for real episodes (simulator access).
        Otherwise, trains offline on imagined rollouts only.

        step_fn signature: step_fn(state_phys, action_norm, smap, elapsed_ms, offset)
                          → (new_phys, new_norm, new_obs)
        """
        self._build_causal_dynamics(data)
        self._init_policy()

        if step_fn is None or smap is None:
            self._fit_offline(data)
            return self

        logger.info(f"C-MBPO: {self.n_real_episodes} real episodes + "
                    f"{self.n_imagined_per_real}x imagined rollouts")

        LATENT = {'occupancy', 'windowposition'}

        for ep_i in range(self.n_real_episodes):
            start_hour = 6.0 + (ep_i % 5) * 2
            offset = [-3.0, 0.0, 5.0, 10.0, 3.0][ep_i % 5]

            trajectory = []
            state_phys = None

            for t in range(self.episode_len):
                elapsed_ms = (start_hour * 3600 + t * 60) * 1000

                if state_phys is not None:
                    from neurips_final_experiments import phys_to_norm
                    state_norm = {k.lower(): phys_to_norm(v, k.lower(), smap)
                                  for k, v in state_phys.items()
                                  if isinstance(v, (int, float))}
                    state_obs = {k: v for k, v in state_norm.items()
                                 if k not in LATENT}
                else:
                    state_obs = {s.lower(): 0.5 for s in SENSOR_VARS}

                action = self._get_policy_action(state_obs, explore=True)
                new_phys, new_norm, new_obs = step_fn(
                    state_phys, {k.lower(): v for k, v in action.items()},
                    smap, elapsed_ms, offset)

                if new_phys is None:
                    continue

                next_obs = {k: v for k, v in new_norm.items()
                            if k not in LATENT} if new_norm else state_obs

                r = self._reward(next_obs, action)
                trajectory.append({
                    'state': state_obs, 'action': action,
                    'reward': r, 'next_state': next_obs,
                })
                state_phys = new_phys

            # Process real trajectory
            batch = self._compute_advantages(trajectory)
            self._replay.extend(batch)

            # Imagined rollouts on causal model
            for _ in range(self.n_imagined_per_real):
                if not trajectory:
                    break
                start_idx = np.random.randint(len(trajectory))
                s = dict(trajectory[start_idx]['state'])

                imag_traj = []
                for t_imag in range(min(20, self.episode_len)):
                    a = self._get_policy_action(s, explore=True)
                    s_next = self._causal_predict(s, a)
                    r = self._reward(s_next, a)
                    imag_traj.append({
                        'state': s, 'action': a,
                        'reward': r, 'next_state': s_next,
                    })
                    s = s_next

                imag_batch = self._compute_advantages(imag_traj)
                self._replay.extend(imag_batch)

            # Policy update
            if len(self._replay) >= 50:
                batch_size = min(256, len(self._replay))
                indices = np.random.choice(len(self._replay), batch_size,
                                           replace=False)
                mini_batch = [self._replay[i] for i in indices]
                self._update_policy(mini_batch)

                if len(self._replay) > 5000:
                    self._replay = self._replay[-3000:]

            if (ep_i + 1) % 10 == 0 and trajectory:
                avg_r = np.mean([s['reward'] for s in trajectory])
                logger.info(f"  C-MBPO episode {ep_i+1}/{self.n_real_episodes}: "
                            f"avg_reward={avg_r:.3f}, replay={len(self._replay)}")

        self._trained = True
        return self

    def _fit_offline(self, data: pd.DataFrame):
        """Fallback: imagined rollouts only (no simulator)."""
        logger.info("C-MBPO: offline training (imagined rollouts only)")
        col_map = {c.lower(): c for c in data.columns}

        for ep_i in range(self.n_real_episodes * 2):
            idx = np.random.randint(len(data))
            s = {}
            for var in SENSOR_VARS:
                col = col_map.get(var.lower())
                if col:
                    s[var.lower()] = float(data.iloc[idx][col])
                else:
                    s[var.lower()] = 0.5

            imag_traj = []
            for t in range(30):
                a = self._get_policy_action(s, explore=True)
                s_next = self._causal_predict(s, a)
                r = self._reward(s_next, a)
                imag_traj.append({
                    'state': s, 'action': a,
                    'reward': r, 'next_state': s_next,
                })
                s = s_next

            batch = self._compute_advantages(imag_traj)
            self._replay.extend(batch)

            if len(self._replay) >= 50:
                batch_size = min(256, len(self._replay))
                indices = np.random.choice(len(self._replay), batch_size,
                                           replace=False)
                self._update_policy([self._replay[i] for i in indices])

            if len(self._replay) > 5000:
                self._replay = self._replay[-3000:]

        self._trained = True

    def get_action(
        self, state: Dict[str, float], comfort_target: float
    ) -> Dict[str, float]:
        self.comfort_target = comfort_target
        if not self._trained:
            return {v.lower(): 0.4 for v in ACTION_VARS}
        return self._get_policy_action(state, explore=False)
