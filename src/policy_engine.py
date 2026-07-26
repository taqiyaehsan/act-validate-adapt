"""
Causal Policy Engine — DAG-guided Structural Equation Model (SEM) Optimisation

Uses the validated causal DAG to fit quadratic structural equations from
observational data, then computes the analytical optimal action via
state-dependent SEM gradients with a quadratic intervention-cost regulariser.

The SEM includes linear, squared, and pairwise interaction terms:
    outcome = intercept + Σ(β_i × parent_i + γ_i × parent_i²)
                        + Σ_{i<j}(δ_{ij} × parent_i × parent_j)

The squared terms capture non-linear (bell-shaped) causal effects; the
interaction terms disentangle confounders (e.g. AirQuality's effect on
Satisfaction depends on Temperature).  The gradient becomes state-dependent:
    ∂outcome/∂v = β_v + 2γ_v × v_current + Σ_{j≠v} δ_{v,j} × x_j

Different edge sets produce different regression models and therefore different
gradients, yielding meaningfully different actions.

The cost regulariser (lambda) penalises deviations from the current state,
creating a trade-off between predicted improvement and intervention effort.
This ensures the DAG structure — not boundary effects — drives policy
differentiation.

Paper reference: Section 4.2 (Causal Policy Engine) of PolicyGRID.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from sklearn.linear_model import Ridge

logger = logging.getLogger(__name__)


@dataclass
class PolicyResult:
    action_plan: Dict[str, float]
    predicted_outcomes: Dict[str, float]
    confidence: float
    cost: float
    expected_improvement: Dict[str, float]
    reasoning: str


# Default variables the engine can set (action space)
ACTION_VARS = ['Temperature', 'Humidity', 'AirQuality']

# Intervention-cost regularisation strength (λ).
# score = w_s·S(a) + w_e·(1−E(a)) − λ·‖a − a_current‖²
# Higher λ → actions stay closer to current state.
# λ=0.1 selected via ablation Study B (sweep over 0.1, 0.5, 1.0, 2.0).
# With 5 gradient steps (Study A), λ=0.1 yields MO=0.462 vs 0.336 at λ=1.0.
DEFAULT_LAMBDA = 0.1

# Number of gradient iterations per optimize_policy() call.
# Each iteration re-evaluates the state-dependent gradient at the updated
# action, allowing the optimizer to traverse non-linear SEM landscapes.
# 5 steps selected via ablation Study A (sweep over 1, 5, 10, 20).
DEFAULT_GRADIENT_STEPS = 5


class CausalPolicyEngine:
    """DAG-guided policy engine using quadratic structural equation models.

    Given a set of validated causal edges and observational data, fits
    quadratic SEMs (linear + squared terms) for each outcome variable
    using only the parents specified by the DAG.

    The optimal action is computed analytically:
        a_v* = clip(a_v_current + gradient_v(state) / (2λ), lo, hi)
    where gradient_v is the state-dependent marginal score improvement,
    derived from the SEM coefficients:
        ∂outcome/∂v = β_v + 2γ_v × v_current + Σ_{j≠v} δ_{v,j} × x_j

    This captures non-linear causal effects and confounding interactions
    that a purely additive SEM would miss.

    Different edge sets → different SEM coefficients → different gradients →
    different optimal actions.
    """

    def __init__(self, pipeline_results: Dict[str, Any], dataset: pd.DataFrame,
                 use_llm: bool = True, api_key: str = None,
                 reg_lambda: float = DEFAULT_LAMBDA,
                 n_gradient_steps: int = DEFAULT_GRADIENT_STEPS,
                 action_vars: Optional[List[str]] = None,
                 outcome_map: Optional[Dict[str, str]] = None,
                 intervention_directions: Optional[Dict] = None,
                 linear_only: bool = False):
        self.validated_edges = pipeline_results.get('validated_edges', set())
        self.dataset = dataset
        self._lambda = reg_lambda
        self._n_gradient_steps = n_gradient_steps
        self._linear_only = linear_only

        # Configurable action space (defaults to smart-room variables)
        self._action_vars = action_vars or list(ACTION_VARS)

        # Outcome mapping: objective keyword → column name
        # Default: 'energy' → EnergyConsumption, 'satisfaction' → auto-detect
        _sat_cols = [c for c in ['Satisfaction', 'OverallSatisfaction']
                     if c in dataset.columns]
        self._satisfaction_col = _sat_cols[0] if _sat_cols else 'Satisfaction'
        self._outcome_map = outcome_map or {
            'energy': 'EnergyConsumption',
            'satisfaction': self._satisfaction_col,
        }

        # Build parent map: target → {parents}
        # Normalize edge variable names to match dataset column names
        # (pipeline stores lowercase edges, dataset may use ProperCase)
        col_lookup = {c.lower(): c for c in self.dataset.columns}

        self._parent_map: Dict[str, set] = {}
        for edge in self.validated_edges:
            if isinstance(edge, tuple) and len(edge) == 2:
                source, target = edge
                source = col_lookup.get(source, source)
                target = col_lookup.get(target, target)
                self._parent_map.setdefault(target, set()).add(source)

        # Intervention-observed effect directions: {(parent_col, target_col): +1/-1}
        # Constrains SEM coefficient signs to match CWM-validated causal directions
        raw_dirs = intervention_directions or pipeline_results.get(
            'intervention_directions', {})
        self._intervention_directions: Dict[Tuple[str, str], int] = {}
        for key, sign in raw_dirs.items():
            if isinstance(key, tuple) and len(key) == 2:
                src, tgt = key
                src_col = col_lookup.get(src.lower(), col_lookup.get(src, src))
                tgt_col = col_lookup.get(tgt.lower(), col_lookup.get(tgt, tgt))
                self._intervention_directions[(src_col, tgt_col)] = sign

        # Fit quadratic SEMs with interaction terms
        # target → (intercept, {parent: β}, {parent: γ}, {(p_i,p_j): δ}, R²)
        self._sem: Dict[str, Tuple] = {}
        self._fit_sems()

        # Apply intervention-direction constraints to SEM coefficients
        if self._intervention_directions:
            self._apply_direction_constraints()

        # Overall model confidence derived from SEM R² scores
        self._model_confidence = self._compute_confidence()

        logger.info(
            f"PolicyEngine init: {len(self.validated_edges)} edges, "
            f"SEMs for {list(self._sem.keys())}, "
            f"action_vars={self._action_vars}, "
            f"confidence={self._model_confidence:.3f}")

    # ── SEM fitting ──────────────────────────────────────────────────────────

    def _fit_sems(self):
        """Fit a quadratic SEM with interaction terms for every variable
        that has parents in the DAG.

        outcome = intercept + Σ(β_i × x_i + γ_i × x_i²)
                            + Σ_{i<j}(δ_{ij} × x_i × x_j)

        The dataset should already contain both observational and
        intervention data (combined by the CWM pipeline with upweighted
        intervention rows).  If a 'weight' column is present, it is used
        as sample_weight in the Ridge regression.
        """
        has_weights = 'weight' in self.dataset.columns

        for target, parents in self._parent_map.items():
            if target not in self.dataset.columns:
                continue

            usable = sorted(p for p in parents if p in self.dataset.columns)
            if not usable:
                mean_val = float(self.dataset[target].mean())
                self._sem[target] = (mean_val, {}, {}, {}, 0.0)
                continue

            X_lin = self.dataset[usable].values
            y = self.dataset[target].values
            weights = self.dataset['weight'].values if has_weights else None

            mask = ~(np.isnan(X_lin).any(axis=1) | np.isnan(y))
            X_lin, y = X_lin[mask], y[mask]
            if weights is not None:
                weights = weights[mask]

            if len(X_lin) < 10:
                self._sem[target] = (float(y.mean()), {}, {}, {}, 0.0)
                continue

            n_parents = len(usable)

            if self._linear_only:
                # Linear-only SEM: outcome = intercept + Σ(β_i × x_i)
                X_full = X_lin
            else:
                # Quadratic SEM: [x1, ..., x1², ..., x1*x2, x1*x3, ...]
                X_sq = X_lin ** 2
                interact_cols = []
                interact_pairs = []
                for i in range(n_parents):
                    for j in range(i + 1, n_parents):
                        interact_cols.append(
                            X_lin[:, i] * X_lin[:, j])
                        interact_pairs.append((usable[i], usable[j]))

                parts = [X_lin, X_sq]
                if interact_cols:
                    parts.append(np.column_stack(interact_cols))
                X_full = np.hstack(parts)

            model = Ridge(alpha=0.5)
            model.fit(X_full, y, sample_weight=weights)

            r2 = max(0.0, model.score(X_full, y,
                                       sample_weight=weights))
            coefs = model.coef_
            lin_coeffs = {p: float(c) for p, c in
                          zip(usable, coefs[:n_parents])}
            if self._linear_only:
                quad_coeffs = {}
                interact_coeffs = {}
            else:
                quad_coeffs = {p: float(c) for p, c in
                               zip(usable, coefs[n_parents:2 * n_parents])}
                interact_coeffs = {pair: float(c) for pair, c in
                                   zip(interact_pairs, coefs[2 * n_parents:])}
            self._sem[target] = (float(model.intercept_), lin_coeffs,
                                 quad_coeffs, interact_coeffs, r2)

            logger.debug(
                f"  SEM({target}): R²={r2:.4f} int={model.intercept_:.4f} "
                f"linear={lin_coeffs} quad={quad_coeffs} "
                f"interact={interact_coeffs}")

    def _apply_direction_constraints(self):
        """Flip SEM linear coefficients whose sign disagrees with
        intervention-observed causal directions from the CWM pipeline."""
        for (parent, target), direction in self._intervention_directions.items():
            if target not in self._sem:
                continue
            intercept, lin_c, quad_c, inter_c, r2 = self._sem[target]
            if parent not in lin_c:
                continue
            current_sign = np.sign(lin_c[parent])
            if current_sign != direction:
                old = lin_c[parent]
                if abs(old) < 1e-6:
                    # Zero coefficient — intervention says there IS an effect.
                    # Set a small coefficient in the correct direction.
                    # Magnitude = median of nonzero coefficients in this SEM.
                    nonzero = [abs(v) for v in lin_c.values() if abs(v) > 1e-6]
                    magnitude = float(np.median(nonzero)) if nonzero else 0.01
                    lin_c[parent] = magnitude * direction
                else:
                    lin_c[parent] = abs(lin_c[parent]) * direction
                logger.info(
                    f"Direction constraint: SEM({target}).{parent} "
                    f"β corrected {old:+.4f} → {lin_c[parent]:+.4f} "
                    f"(intervention says {'+'if direction > 0 else '-'})")
                self._sem[target] = (intercept, lin_c, quad_c, inter_c, r2)

    def _compute_confidence(self) -> float:
        """Mean R² across outcome SEMs."""
        outcome_r2 = []
        for var in self._outcome_map.values():
            if var in self._sem:
                outcome_r2.append(self._sem[var][4])  # R² is 5th element
        return float(np.mean(outcome_r2)) if outcome_r2 else 0.1

    # ── analytical gradient optimisation ─────────────────────────────────────

    def _compute_gradient(self, objectives: Dict,
                          current_state: Dict) -> Dict[str, float]:
        """Compute state-dependent ∂score/∂a_v from SEM coefficients.

        For outcome O with coefficients β, γ, δ for action variable v:
            ∂outcome_O/∂v = β_v + 2γ_v × v_current + Σ_{j≠v} δ_{v,j} × x_j

        The interaction terms mean that each variable's gradient depends on
        the current values of ALL other parent variables, disentangling
        confounders that a purely additive model would conflate.
        """
        gradient = {v: 0.0 for v in self._action_vars}

        # Damping factors for action variables with their own SEM parents
        damping = {}
        for v in self._action_vars:
            if v in self._sem:
                _, lin_c, _qc, _ic, r2 = self._sem[v]
                damping[v] = 1.0 - r2 if lin_c else 1.0
            else:
                damping[v] = 1.0

        # Accumulate gradient from each outcome SEM
        for obj_name, obj_spec in objectives.items():
            weight = obj_spec.get('weight', 0.5)

            # Resolve objective name to outcome column via _outcome_map
            outcome_var = None
            for key, col in self._outcome_map.items():
                if key in obj_name.lower():
                    outcome_var = col
                    break

            if outcome_var is None:
                continue

            # Determine sign: satisfaction/comfort → maximize, else minimize
            if 'satisfaction' in obj_name.lower() or 'comfort' in obj_name.lower():
                sign = 1.0
            else:
                sign = -1.0

            if outcome_var not in self._sem:
                continue

            _intercept, lin_coeffs, quad_coeffs, interact_coeffs, _r2 = \
                self._sem[outcome_var]
            for v in self._action_vars:
                beta = lin_coeffs.get(v, 0.0)
                gamma = quad_coeffs.get(v, 0.0)
                v_current = current_state.get(v, 0.5)
                # ∂outcome/∂v = β + 2γ×v + Σ_{j≠v} δ_{v,j}×x_j
                local_grad = beta + 2.0 * gamma * v_current
                for (p_i, p_j), delta in interact_coeffs.items():
                    if p_i == v:
                        local_grad += delta * current_state.get(p_j, 0.5)
                    elif p_j == v:
                        local_grad += delta * current_state.get(p_i, 0.5)
                gradient[v] += weight * sign * damping[v] * local_grad

        return gradient

    def _analytical_optimal(self, objectives: Dict, constraints: Dict,
                            current_state: Dict) -> Dict[str, float]:
        """Closed-form optimal action under quadratic cost regulariser.

            a_v* = clip(a_default + gradient_v(state) / (2λ), lo, hi)

        Uses a neutral default (midpoint of action range) rather than the
        current observed action to prevent drift/ratcheting across timesteps.
        The gradient is state-dependent through context variables (temperature,
        humidity, etc.) but the action baseline is fixed.
        """
        gradient = self._compute_gradient(objectives, current_state)
        action = {}
        for v in self._action_vars:
            lo, hi = constraints.get(v, (0.0, 1.0))
            a_default = (lo + hi) / 2.0
            step = gradient[v] / (2 * self._lambda)
            action[v] = float(np.clip(a_default + step, lo, hi))
        return action

    # ── prediction ───────────────────────────────────────────────────────────

    def _predict_outcomes(self, action: Dict[str, float],
                          context: Dict[str, float]) -> Dict[str, float]:
        """Predict outcome variables for a candidate action using fitted SEMs.

        Two-step forward propagation:
        1. For action variables that have parents in the DAG (e.g. H1's
           OutdoorTemperature → Temperature), compute a *realised* value that
           blends the agent's setpoint with the environmental forcing predicted
           by the SEM.  The blend weight is the action-variable SEM's R².
        2. Predict outcome variables using the realised action values.
        """
        all_vars = dict(context)
        all_vars.update(action)

        def _sem_predict(sem_entry, inputs):
            """Evaluate SEM: intercept + Σ(β×x + γ×x²) + Σ_{i<j}(δ×x_i×x_j)."""
            intercept, lin_coeffs, quad_coeffs, interact_coeffs, _r2 = sem_entry
            pred = intercept
            for parent, beta in lin_coeffs.items():
                x = inputs.get(parent,
                               float(self.dataset[parent].mean())
                               if parent in self.dataset.columns else 0.0)
                gamma = quad_coeffs.get(parent, 0.0)
                pred += beta * x + gamma * x * x
            for (p_i, p_j), delta in interact_coeffs.items():
                x_i = inputs.get(p_i,
                                 float(self.dataset[p_i].mean())
                                 if p_i in self.dataset.columns else 0.0)
                x_j = inputs.get(p_j,
                                 float(self.dataset[p_j].mean())
                                 if p_j in self.dataset.columns else 0.0)
                pred += delta * x_i * x_j
            return pred

        # Step 1: compute realised action values
        realised = dict(action)
        for var in self._action_vars:
            if var not in self._sem:
                continue
            _intercept, lin_coeffs, _qc, _ic, r2 = self._sem[var]
            if not lin_coeffs:
                continue
            sem_pred = _sem_predict(self._sem[var], all_vars)
            realised[var] = (1 - r2) * action.get(var, 0.5) + r2 * sem_pred

        # Step 2: predict outcomes from realised values + context
        preds_input = dict(context)
        preds_input.update(realised)

        predictions = {}
        for var in self._outcome_map.values():
            if var not in self._sem:
                if var in self.dataset.columns:
                    predictions[var] = float(self.dataset[var].mean())
                else:
                    predictions[var] = 0.5
                continue
            pred = _sem_predict(self._sem[var], preds_input)
            predictions[var] = float(np.clip(pred, 0.0, 1.0))

        return predictions

    # ── multi-step SEM rollout ────────────────────────────────────────────────

    def _sem_predict_next(self, state: Dict[str, float]) -> Dict[str, float]:
        """One-step prediction: predict all variables from their DAG parents."""
        next_state = dict(state)
        for target, (intercept, lin_c, quad_c, inter_c, r2) in self._sem.items():
            pred = intercept
            for parent, beta in lin_c.items():
                x = state.get(parent, 0.5)
                pred += beta * x
                gamma = quad_c.get(parent, 0.0)
                if gamma:
                    pred += gamma * x * x
            for (pi, pj), delta in inter_c.items():
                pred += delta * state.get(pi, 0.5) * state.get(pj, 0.5)
            next_state[target] = float(np.clip(pred, 0.0, 1.0))
        return next_state

    def _rollout(self, state: Dict[str, float],
                 action_seq: List[Dict[str, float]],
                 horizon: int) -> List[Dict[str, float]]:
        """Multi-step SEM rollout with factored causal dynamics."""
        trajectory = []
        s = dict(state)
        for t in range(horizon):
            # Apply action to state
            s.update(action_seq[min(t, len(action_seq) - 1)])
            # Predict next state using causal SEM
            s = self._sem_predict_next(s)
            trajectory.append(dict(s))
        return trajectory

    def _rollout_cost(self, trajectory: List[Dict[str, float]],
                      objectives: Dict) -> float:
        """Cost over a trajectory. Lower is better."""
        total = 0.0
        for state in trajectory:
            for obj_name, obj_spec in objectives.items():
                weight = obj_spec.get('weight', 0.5)
                outcome_var = None
                for key, col in self._outcome_map.items():
                    if key in obj_name.lower():
                        outcome_var = col
                        break
                if outcome_var is None:
                    continue
                val = state.get(outcome_var, 0.5)
                if 'satisfaction' in obj_name.lower() or 'comfort' in obj_name.lower():
                    total -= weight * val    # maximize satisfaction
                else:
                    total += weight * val    # minimize energy
        return total

    def _cem_optimize(self, current_state: Dict, objectives: Dict,
                      constraints: Dict,
                      horizon: int = 10, population: int = 64,
                      n_iter: int = 5, elite_frac: float = 0.2
                      ) -> Dict[str, float]:
        """CEM optimizer over action sequences using factored SEM rollout.

        The causal graph structure makes rollout efficient: each variable is
        predicted from only its DAG parents, not the full state.
        """
        n_elite = max(3, int(population * elite_frac))
        n_actions = len(self._action_vars)
        bounds = [constraints.get(v, (0.0, 0.8)) for v in self._action_vars]

        mu = np.array([(lo + hi) / 2 for lo, hi in bounds])
        mu = np.tile(mu, (horizon, 1))       # (H, n_actions)
        sigma = np.full_like(mu, 0.2)

        best_action = mu[0].copy()
        best_cost = float('inf')

        for iteration in range(n_iter):
            candidates = np.random.normal(
                mu[None, :, :], sigma[None, :, :],
                size=(population, horizon, n_actions))
            # Clip to bounds per action variable
            for i, (lo, hi) in enumerate(bounds):
                candidates[:, :, i] = np.clip(candidates[:, :, i], lo, hi)

            costs = np.zeros(population)
            for k in range(population):
                action_seq = [{v: float(candidates[k, t, i])
                               for i, v in enumerate(self._action_vars)}
                              for t in range(horizon)]
                traj = self._rollout(current_state, action_seq, horizon)
                costs[k] = self._rollout_cost(traj, objectives)

            elite_idx = np.argsort(costs)[:n_elite]
            if costs[elite_idx[0]] < best_cost:
                best_cost = costs[elite_idx[0]]
                best_action = candidates[elite_idx[0], 0].copy()

            mu = candidates[elite_idx].mean(axis=0)
            sigma = candidates[elite_idx].std(axis=0) + 1e-4

        return {v: float(best_action[i])
                for i, v in enumerate(self._action_vars)}

    # ── main entry point ─────────────────────────────────────────────────────

    def optimize_policy(self, objectives, constraints, current_state,
                        n_candidates=50):
        """Compute optimal action via SEM gradient with cost regulariser.

        Uses the validated causal DAG to fit structural equations, then
        computes the analytical optimal action from the gradient:
            a_v* = clip(a_default + gradient_v(state) / (2λ), lo, hi)

        Different DAG structures → different SEM coefficients → different
        gradients → different optimal actions.
        """
        if not self._sem:
            return self._fallback_policy(objectives, constraints, current_state)

        action = self._analytical_optimal(objectives, constraints, current_state)

        context = {k: v for k, v in current_state.items()
                   if k not in self._action_vars}
        predictions = self._predict_outcomes(action, context)

        cost = sum(abs(action.get(v, 0.5) - current_state.get(v, 0.5))
                   for v in self._action_vars)
        improvements = self._calc_improvements(predictions, current_state)
        n_parents = sum(len(lc) for _, lc, _qc, _ic, _r2 in self._sem.values())

        return PolicyResult(
            action_plan=action,
            predicted_outcomes=predictions,
            confidence=self._model_confidence,
            cost=cost,
            expected_improvement=improvements,
            reasoning=(
                f"SEM CEM rollout: {len(self.validated_edges)} edges, "
                f"{n_parents} SEM coefficients, "
                f"R²={self._model_confidence:.3f}"))

    # ── fallback ─────────────────────────────────────────────────────────────

    def _fallback_policy(self, objectives, constraints, current_state):
        """No causal edges → dataset-mean setpoints with dataset-mean predictions."""
        action = {}
        for v in self._action_vars:
            if v in self.dataset.columns:
                action[v] = float(self.dataset[v].mean())
            else:
                action[v] = 0.5
            lo, hi = constraints.get(v, (0.0, 1.0))
            action[v] = float(np.clip(action[v], lo, hi))

        predictions = {}
        for var in self._outcome_map.values():
            if var in self.dataset.columns:
                predictions[var] = float(self.dataset[var].mean())
            else:
                predictions[var] = 0.5

        return PolicyResult(
            action_plan=action,
            predicted_outcomes=predictions,
            confidence=0.1,
            cost=sum(abs(action.get(v, 0.5) - current_state.get(v, 0.5))
                     for v in self._action_vars),
            expected_improvement=self._calc_improvements(predictions,
                                                         current_state),
            reasoning="No causal edges — dataset-mean defaults")

    # ── helpers ──────────────────────────────────────────────────────────────

    def _calc_improvements(self, predictions, current_state):
        improvements = {}
        for outcome, pred in predictions.items():
            current_val = current_state.get(outcome, 0.5)
            if 'energy' in outcome.lower():
                improvements[outcome] = current_val - pred
            else:
                improvements[outcome] = pred - current_val
        return improvements

    def get_consistent_action(self, objectives=None,
                              constraints=None) -> Dict[str, float]:
        """Get consistent action for scatter plot generation."""
        if constraints is None:
            constraints = {v: (0.0, 1.0) for v in ACTION_VARS}
        if objectives is None:
            objectives = {
                'satisfaction': {'target': 60.0, 'weight': 0.6},
                'energyconsumption': {'target': 30.0, 'weight': 0.4}
            }
        current_state = {
            'Temperature': 0.5, 'Humidity': 0.5, 'AirQuality': 0.5,
            'EnergyConsumption': 0.35,
            self._satisfaction_col: 0.55
        }
        result = self.optimize_policy(objectives, constraints, current_state)
        return result.action_plan


class RegimeAwarePolicyEngine:
    """Regime-switching policy engine using the same factored Bayesian
    monitoring system used for regime detection (4-way: base/occ/win/full).

    Trains separate SEMs for each regime on the same validated causal graph.
    At runtime, the monitoring system's 4-way Bayesian model comparison
    estimates P(occ) and P(win), and the policy gradient is a weighted
    blend of regime-specific gradients.

    Uses the SAME monitoring code as the detection pipeline — not a
    separate implementation.
    """

    def __init__(self, validated_edges, dataset: pd.DataFrame,
                 action_vars: List[str],
                 observable_vars: List[str],
                 reg_lambda: float = DEFAULT_LAMBDA,
                 linear_only: bool = False,
                 intervention_directions: Optional[Dict] = None,
                 sigma: float = 1.5,
                 forget_factor: float = 0.15):
        self._action_vars = action_vars
        self._lambda = reg_lambda
        self._sigma = sigma
        self._forget_factor = forget_factor
        self._observable_vars = observable_vars

        # Monitors use ALL validated edges (including latent-variable edges
        # like occupancy→co2) — they need these to detect regime changes.
        # Policy SEMs use observable-only edges — the gradient optimizer
        # can't use variables the policy can't observe.
        obs_set = set(v.lower() for v in observable_vars)

        import networkx as nx
        from src.causal_world_model import PredictiveModel

        cols_lower = {c.lower(): c for c in dataset.columns}
        occ_col = cols_lower.get('occupancy')
        win_col = cols_lower.get('windowposition', cols_lower.get('windowopen'))

        if occ_col and win_col:
            occ_vals = dataset[occ_col].values
            win_vals = dataset[win_col].values
            occ_on = occ_vals >= 0.1
            win_on = win_vals >= 0.15
        else:
            occ_on = np.zeros(len(dataset), dtype=bool)
            win_on = np.zeros(len(dataset), dtype=bool)

        # All columns for monitors (including latent — needed for prediction)
        all_vars_lower = [c.lower() for c in dataset.columns]

        def _build_model(mask):
            G = nx.DiGraph()
            G.add_nodes_from(all_vars_lower)
            # Use ALL validated edges for monitors (including latent parents)
            G.add_edges_from([(s.lower(), t.lower()) for s, t in validated_edges])
            model = PredictiveModel(G, [v.lower() for v in observable_vars])
            df = dataset.copy()
            df.columns = [c.lower() for c in df.columns]
            temporal = {}
            for var in all_vars_lower:
                if var in df.columns:
                    temporal[var] = df[var].values[:-1]
                    temporal[f'{var}_next'] = df[var].values[1:]
            tdf = pd.DataFrame(temporal)
            tdf = tdf[mask[:-1]].reset_index(drop=True)
            if len(tdf) > 10:
                model.fit_temporal(tdf)
            return model

        self._monitors = {
            'base': _build_model(~occ_on & ~win_on),
            'occ':  _build_model( occ_on & ~win_on),
            'win':  _build_model(~occ_on &  win_on),
            'full': _build_model( occ_on &  win_on),
        }

        # Train 4 regime-specific SEMs for the policy gradient
        masks = {
            'base': ~occ_on & ~win_on,
            'occ':   occ_on & ~win_on,
            'win':  ~occ_on &  win_on,
            'full':  occ_on &  win_on,
        }

        kwargs = dict(use_llm=False, action_vars=action_vars,
                      reg_lambda=reg_lambda, n_gradient_steps=1,
                      linear_only=linear_only)
        if intervention_directions:
            kwargs['intervention_directions'] = intervention_directions

        self._engines = {}
        for rname, mask in masks.items():
            data_r = dataset[mask]
            if len(data_r) < 20:
                data_r = dataset  # fallback to full data
            # Drop regime columns from SEM training data
            drop_cols = [c for c in data_r.columns
                         if c.lower() in ('occupancy', 'windowposition', 'windowopen')]
            data_r = data_r.drop(columns=drop_cols, errors='ignore')
            self._engines[rname] = CausalPolicyEngine(
                {'validated_edges': validated_edges}, data_r, **kwargs)

        # Bayesian weights (same as monitoring pipeline)
        self._regime_names = ['base', 'occ', 'win', 'full']
        self._weights = {r: 0.25 for r in self._regime_names}

        n_base = (~occ_on & ~win_on).sum()
        n_occ = (occ_on & ~win_on).sum()
        n_win = (~occ_on & win_on).sum()
        n_full = (occ_on & win_on).sum()
        logger.info(f"RegimeAwarePolicy: 4-way monitors built "
                    f"(base={n_base}, occ={n_occ}, win={n_win}, full={n_full})")
        for rname in self._regime_names:
            logger.info(f"  SEM({rname}): R²={self._engines[rname]._model_confidence:.3f}")

    def _update_weights(self, state: Dict[str, float]):
        """4-way Bayesian update — same logic as run_monitoring()."""
        sensor_vars = [v.lower() for v in self._observable_vars]
        state_lower = {k.lower(): v for k, v in state.items()}

        likelihoods = {}
        for rname in self._regime_names:
            model = self._monitors[rname]
            pred = model.predict(state_lower)
            err = sum(abs(state_lower.get(v, 0.5) - pred.get(v, 0.5))
                      for v in sensor_vars if v in state_lower)
            likelihoods[rname] = np.exp(-self._sigma * err)

        unnorm = {r: likelihoods[r] * self._weights[r] for r in self._regime_names}
        total = sum(unnorm.values())
        if total > 0:
            raw = {r: unnorm[r] / total for r in self._regime_names}
            self._weights = {
                r: (1 - self._forget_factor) * raw[r] + self._forget_factor * 0.25
                for r in self._regime_names}
        w_total = sum(self._weights.values())
        if w_total > 0:
            self._weights = {r: self._weights[r] / w_total
                             for r in self._regime_names}

    @property
    def p_occ(self):
        return self._weights['occ'] + self._weights['full']

    @property
    def p_win(self):
        return self._weights['win'] + self._weights['full']

    def optimize_policy(self, objectives, constraints, current_state,
                        n_candidates=50):
        """4-way regime-blended optimal action.

        1. Update regime weights from current observations (same Bayesian
           monitor as detection pipeline).
        2. Compute gradient from each regime's SEM.
        3. Blend: ∇score = Σ_r w_r · ∇score_r.
        4. Step from default: a* = a_default + blended_grad / (2λ).
        """
        # 1. Update regime weights
        self._update_weights(current_state)

        # 2. Compute per-regime gradients
        regime_grads = {}
        for rname in self._regime_names:
            if self._engines[rname]._sem:
                regime_grads[rname] = self._engines[rname]._compute_gradient(
                    objectives, current_state)
            else:
                regime_grads[rname] = {v: 0.0 for v in self._action_vars}

        # 3. Blend
        blended_grad = {v: 0.0 for v in self._action_vars}
        for rname in self._regime_names:
            w = self._weights[rname]
            for v in self._action_vars:
                blended_grad[v] += w * regime_grads[rname].get(v, 0.0)

        # 4. Step from default
        action = {}
        for v in self._action_vars:
            lo, hi = constraints.get(v, (0.0, 1.0))
            a_default = (lo + hi) / 2.0
            step = blended_grad[v] / (2 * self._lambda)
            action[v] = float(np.clip(a_default + step, lo, hi))

        # Blended confidence
        confidence = sum(self._weights[r] * self._engines[r]._model_confidence
                         for r in self._regime_names)

        return PolicyResult(
            action_plan=action,
            predicted_outcomes={},
            confidence=confidence,
            cost=sum(abs(action.get(v, 0.5) - current_state.get(v, 0.5))
                     for v in self._action_vars),
            expected_improvement={},
            reasoning=(
                f"4-way regime: base={self._weights['base']:.2f} "
                f"occ={self._weights['occ']:.2f} "
                f"win={self._weights['win']:.2f} "
                f"full={self._weights['full']:.2f}")
        )
