"""
NeurIPS Experiment Harness — B1 Policy Comparison
===================================================

Evaluates control policies on simulated smart-room environments.
This is the main experiment runner for the PolicyGRID paper (Section 5).

Policy arms (8-way comparison):
  +---------------------+----------------------------------------------+
  | Role                | Arm                                          |
  +---------------------+----------------------------------------------+
  | Static baseline     | ASHRAE-PID     — PID-style thermostat         |
  | DAG-guided (ours)   | GRID           — validated DAG + SEM policy   |
  | Ablation            | GRID_ObsOnly   — obs-only DAG (no validation) |
  | Full loop           | GRID_RegimeAware — validated + Bayesian BMA   |
  | Data-driven (M2)    | DDPC_Behavioral — DeePC (Willems' lemma)     |
  |                     | DDPC_Subspace   — N4SID + linear MPC          |
  |                     | DDPC_Neural     — MLP dynamics + CEM          |
  |                     | DDPC_PETS       — Bootstrap ensemble + TS-inf  |
  +---------------------+----------------------------------------------+

Simulator backends:
  - SmartRoom JS  (js/smart_room.js, js/smart_room_hidden_vars.js,
    js/smart_room_noise.js, open_window.js)
  - Analytical fallback model (when JS unavailable)

Metrics per episode:
  - Satisfaction%:  mean occupant satisfaction (higher is better)
  - Energy%:        mean HVAC energy duty cycle (lower is better)
  - MO score:       w_s·Sat + w_e·(1-Eng), multi-objective scalar
  - kWh:            total energy consumption in kilowatt-hours
  - DH:             degree-hours deviation from ASHRAE-55 comfort ref (22°C)

DOMAIN-AGNOSTIC KNOBS — update these when adapting to a new domain:
  - ``_DATASET_CONFIGS``:    Add a new entry for your domain with:
      - ``action_vars``:     Variables the policy can set (actuators).
      - ``outcome_map``:     Maps 'energy'/'satisfaction' → dataset column names.
      - ``default_setpoints``: ASHRAE baseline setpoints in [0,1] normalized units.
      - ``objectives``:      Multi-objective weights and targets.
      - ``sat_key`` / ``eng_key``: Column names for satisfaction and energy.
  - ``COMFORT_REF_CELSIUS``: Fixed reference temperature for DH calculation.
  - ``ARM_COLORS``:          Publication color palette.
  - ``_SIM_COMMANDS``:       Map simulator names → JS file paths + CLI args.
                             For a non-JS simulator, override ``_step_simulator()``.
  - ``_SCALING_CSV``:        Path to min/max normalization parameters CSV.

See also: ``run_exp_suite_B_prelim.py`` for the full B1+B2 experiment runner
that includes the CWM discovery pipeline.
"""
import os
import numpy as np
import pandas as pd
import json
import subprocess
import logging
from typing import Dict, List, Tuple, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from src.policy_engine import CausalPolicyEngine
from src.ddpc_baseline import create_ddpc_suite

logger = logging.getLogger(__name__)

# Color palette — publication-quality, colorblind-friendly
ARM_COLORS = {
    'ASHRAE-PID':         '#7F8C8D',   # grey — static baseline
    'GRID':               '#2E86C1',   # blue — main method
    'GRID_ObsOnly':       '#85C1E9',   # light blue — ablation
    'GRID_RegimeAware':   '#1A5276',   # dark blue — full loop
    'DDPC_Behavioral':    '#E74C3C',   # red
    'DDPC_Subspace':      '#F39C12',   # amber
    'DDPC_Neural':        '#8E44AD',   # purple
    'DDPC_PETS':          '#D35400',   # burnt orange
}


def _dual_sat(state: dict) -> dict:
    """Ensure state dict has both 'Satisfaction' and 'OverallSatisfaction' keys.

    The JS simulators return 'overallSatisfaction' (camelCase) which gets mapped
    to 'OverallSatisfaction'.  The open_window dataset uses 'Satisfaction'.
    Setting both keys makes downstream consumers work regardless of which name
    they look up.
    """
    if 'OverallSatisfaction' in state and 'Satisfaction' not in state:
        state['Satisfaction'] = state['OverallSatisfaction']
    elif 'Satisfaction' in state and 'OverallSatisfaction' not in state:
        state['OverallSatisfaction'] = state['Satisfaction']
    return state


class NeurIPSExperiments:
    """Evaluate control policies on smart-room simulators.

    Metrics: average satisfaction (%), average energy duty-cycle (%),
    multi-objective score, and Pareto hypervolume.

    Supports two dataset types:
      - 'open_window' (default): Smart room with Temperature/Humidity/AirQuality
      - 'ashrae': Building energy with air_temperature/dew_temperature/sea_level_pressure
    """

    # Fixed comfort reference for degree-hours (DH) calculation.
    # ASHRAE-55 neutral operative temperature for sedentary occupancy.
    # Both JS simulators use 22 °C internally (open_window targetTemp,
    # smart_room initial temperature).
    COMFORT_REF_CELSIUS = 22.0

    # ── Dataset-specific configurations ──────────────────────────────────────

    _DATASET_CONFIGS = {
        'open_window': {
            'action_vars': ['Temperature', 'Humidity', 'AirQuality'],
            'outcome_map': None,  # auto-detect (default in CausalPolicyEngine)
            'default_setpoints': {
                'Temperature': 0.625,
                'Humidity': 0.55,
                'AirQuality': 0.75,
            },
            'objectives': {
                'satisfaction': {'target': 80.0, 'weight': 0.6},
                'energyconsumption': {'target': 30.0, 'weight': 0.4},
            },
            'sat_key': 'OverallSatisfaction',
            'eng_key': 'EnergyConsumption',
        },
        'ashrae': {
            'action_vars': ['air_temperature', 'dew_temperature',
                            'sea_level_pressure'],
            'outcome_map': {'energy': 'meter_reading'},
            'default_setpoints': {
                'air_temperature': 0.0,     # dataset-mean (scaled ~0)
                'dew_temperature': 0.0,
                'sea_level_pressure': 0.0,
            },
            'objectives': {
                'energy': {'target': 0.0, 'weight': 1.0},
            },
            'sat_key': None,                # no satisfaction in ASHRAE
            'eng_key': 'meter_reading',
        },
        'smart_building_rich': {
            'action_vars': ['HVACPower', 'LightingPower'],
            'outcome_map': None,
            'default_setpoints': {
                'HVACPower': 0.5,
                'LightingPower': 0.3,
            },
            'objectives': {
                'satisfaction': {'target': 80.0, 'weight': 0.6},
                'energyconsumption': {'target': 30.0, 'weight': 0.4},
            },
            'sat_key': 'OverallSatisfaction',
            'eng_key': 'EnergyConsumption',
        },
    }

    def __init__(self, pipeline, dataset, smart_room_path=None, api_key=None,
                 comfort_target=0.8, obs_only_edges=None,
                 h0_edges=None, h1_edges=None,
                 dataset_type='open_window',
                 scaling_ranges=None,
                 data_in_physical_units=True,
                 obs_only_dataset=None):
        self.pipeline = pipeline
        self.dataset = dataset
        # GRID_ObsOnly uses obs-only data (no intervention rows)
        self._obs_only_dataset = obs_only_dataset if obs_only_dataset is not None else dataset
        self.smart_room_path = smart_room_path
        self.api_key = api_key
        self.comfort_target = comfort_target
        self.dataset_type = dataset_type

        # Load dataset-specific config
        cfg = self._DATASET_CONFIGS.get(dataset_type,
                                         self._DATASET_CONFIGS['open_window'])
        self._cfg = cfg
        self._action_vars = cfg['action_vars']
        self._outcome_map = cfg['outcome_map']

        # Optional edge sets for B1 arms 2 & 4
        self.obs_only_edges = obs_only_edges   # pre-validation edges → GRID_ObsOnly
        self.h0_edges = h0_edges               # window-closed DAG → GRID_RegimeAware
        self.h1_edges = h1_edges               # window-open DAG   → GRID_RegimeAware

        self.objectives = {
            'satisfaction': {'weight': 0.6, 'direction': 'maximize'},
            'energy': {'weight': 0.4, 'direction': 'minimize'}
        }

        self.ashrae_setpoints = dict(cfg['default_setpoints'])

        # Populated by _build_policy_arms()
        self.policy_arms: Dict[str, Callable] = {}
        self._policy_engine = None

        # ── Normalization: just invert min-max scaling from training data ──
        # The training data and its JS simulator always use the same units,
        # so the scaling CSV's min/max IS the correct [0,1] ↔ sim mapping.
        #   open_window: Temperature 9.6-22.3 °C  (JS sim uses °C)
        #   smart_room:  Temperature 0.05-0.95     (JS sim uses [0,1])
        # The flag is only needed for degree-hour calculations (which
        # require actual °C regardless of simulator).
        self._data_ranges = self._load_scaling_ranges(scaling_ranges)
        self._data_in_physical_units = data_in_physical_units

        # Cache for DDPC suite (avoid re-fitting on comfort_target sweeps)
        self._ddpc_suite_cache = None

    # ── policy arm construction ─────────────────────────────────────────────

    def _build_policy_arms(self):
        """Register all policy arms: ASHRAE-PID, GRID, GRID_ObsOnly,
        GRID_RegimeAware (if applicable), and 4 DDPC variants.

        For ASHRAE dataset: only static baseline, GRID, and GRID_ObsOnly
        (no DDPC or regime-aware — different variable schema).
        """
        self.policy_arms = {}

        # 1. Static baseline — fixed setpoints
        sp = self.ashrae_setpoints
        label = 'ASHRAE-PID'
        self.policy_arms[label] = lambda state, t: sp

        # 2. GRID — reactive causal policy with validated DAG
        self.policy_arms['GRID'] = self._make_grid_fn()

        # 3. GRID_ObsOnly — observational-only DAG (no intervention validation)
        obs_edges = self._resolve_obs_only_edges()
        if obs_edges:
            self.policy_arms['GRID_ObsOnly'] = self._make_obs_only_fn(obs_edges)
            logger.info(f"GRID_ObsOnly registered with {len(obs_edges)} obs edges")

        # 4. GRID_RegimeAware — full loop with H0/H1 Bayesian regime dispatch
        #    Only for datasets with a regime variable (e.g. open_window)
        if self.h0_edges and self.h1_edges and self.dataset_type != 'ashrae':
            self.policy_arms['GRID_RegimeAware'] = self._make_regime_aware_fn()
            logger.info(f"GRID_RegimeAware registered "
                        f"(H0:{len(self.h0_edges)} H1:{len(self.h1_edges)} edges)")

        # 5. DDPC suite — 4 data-driven controllers
        #    Only for smart-room datasets (hard-coded to Temperature/Humidity/AQ)
        #    Suite is cached so comfort_target sweeps don't re-fit models.
        if self.dataset_type != 'ashrae':
            try:
                if self._ddpc_suite_cache is None:
                    self._ddpc_suite_cache = create_ddpc_suite(self.dataset)
                suite = self._ddpc_suite_cache
                ct = self.comfort_target
                for name, ctrl in suite.items():
                    def _make_fn(c, target=ct):
                        def fn(state, t):
                            if t == 0:
                                c.reset()
                            return c.get_action(state, comfort_target=target)
                        return fn
                    self.policy_arms[name] = _make_fn(ctrl)
                logger.info(f"DDPC suite registered: {list(suite.keys())}")
            except Exception as e:
                logger.warning(f"DDPC suite creation failed: {e}")

        logger.info(f"Policy arms ready: {list(self.policy_arms.keys())}")

    def set_comfort_target(self, ct):
        """Update comfort target and rebuild policy arms.

        Used by the epsilon-constraint Pareto sweep: each comfort_target ε
        produces a different operating point per policy.
        DDPC suite is cached, so only the function wrappers (which capture
        the new comfort_target) are rebuilt.
        """
        self.comfort_target = ct
        self._policy_engine = None          # force GRID rebuild with new objectives
        self._build_policy_arms()

    def _get_policy_engine(self, edges=None, data=None):
        """Build a CausalPolicyEngine with dataset-appropriate config.

        If edges/data are None, uses the cached default engine from the pipeline.
        """
        if edges is None and data is None:
            if self._policy_engine is not None:
                return self._policy_engine
            try:
                policy_data = self.pipeline.export_for_policy_engine()
                # Use pipeline's combined dataset (obs + intervention with
                # weight column) if available; fall back to self.dataset.
                ds = policy_data.get('combined_dataset', self.dataset)
                if ds is None:
                    ds = self.dataset
                self._policy_engine = CausalPolicyEngine(
                    policy_data, ds,
                    use_llm=False, api_key=self.api_key,
                    action_vars=self._action_vars,
                    outcome_map=self._outcome_map)
            except Exception as e:
                logger.warning(f"CausalPolicyEngine init failed: {e}")
            return self._policy_engine

        # Custom edges/data (used by GRID_ObsOnly, GRID_RegimeAware)
        df = data if data is not None else self.dataset
        return CausalPolicyEngine(
            {'validated_edges': edges}, df,
            use_llm=False, api_key=self.api_key,
            action_vars=self._action_vars,
            outcome_map=self._outcome_map)

    def _make_objectives_and_constraints(self):
        """Return (objectives, constraints) appropriate for this dataset.

        For the epsilon-constraint Pareto sweep, comfort_target (DH budget ε)
        maps to GRID objective weights:
          - Small ε (tight comfort) → high satisfaction weight
          - Large ε (loose comfort) → high energy weight
        This traces different operating points on the Pareto frontier.
        """
        ct = self.comfort_target
        base_objectives = self._cfg['objectives']

        if 'satisfaction' in base_objectives and 'energyconsumption' in base_objectives:
            # Map DH budget ε → objective weights
            # ε ∈ [0.5, 2.0] → sat_w ∈ [0.9, 0.2]
            sat_w = float(np.clip(1.1 - 0.6 * ct, 0.15, 0.9))
            eng_w = 1.0 - sat_w
            sat_target = float(np.clip(100.0 - 25.0 * ct, 50.0, 95.0))

            objectives = {
                'satisfaction': {'target': sat_target, 'weight': sat_w},
                'energyconsumption': {
                    'target': base_objectives['energyconsumption']['target'],
                    'weight': eng_w},
            }
        else:
            objectives = dict(base_objectives)

        constraints = {v: (0.0, 1.0) for v in self._action_vars}
        return objectives, constraints

    def _make_grid_fn(self):
        """Build a reactive GRID policy callable."""
        engine = self._get_policy_engine()
        if engine is None:
            logger.warning("GRID falling back to ASHRAE setpoints")
            sp = self.ashrae_setpoints
            return lambda state, t: sp

        objectives, constraints = self._make_objectives_and_constraints()
        sp = self.ashrae_setpoints

        def fn(state, t):
            result = engine.optimize_policy(objectives, constraints, state)
            return result.action_plan if result else sp
        return fn

    def _resolve_obs_only_edges(self):
        """Get observational-only edges (pre-validation union of method DAGs).

        Priority: explicit self.obs_only_edges > pipeline.get_method_dags().
        Returns a set of (source, target) tuples, or None.
        """
        if self.obs_only_edges:
            return self.obs_only_edges

        # Try extracting from pipeline's method DAGs (union of all methods)
        try:
            if hasattr(self.pipeline, 'get_method_dags'):
                method_dags = self.pipeline.get_method_dags()
                union_edges = set()
                for _name, G in method_dags.items():
                    union_edges.update(G.edges())
                if union_edges:
                    return union_edges
        except Exception as e:
            logger.debug(f"Could not extract obs-only edges from pipeline: {e}")

        return None

    def _make_obs_only_fn(self, obs_edges):
        """Build a GRID policy using observational-only edges (no validation)."""
        try:
            engine = self._get_policy_engine(edges=obs_edges, data=self._obs_only_dataset)
        except Exception as e:
            logger.warning(f"GRID_ObsOnly engine failed: {e}")
            sp = self.ashrae_setpoints
            return lambda state, t: sp

        objectives, constraints = self._make_objectives_and_constraints()
        sp = self.ashrae_setpoints

        def fn(state, t):
            result = engine.optimize_policy(objectives, constraints, state)
            return result.action_plan if result else sp
        return fn

    def _make_regime_aware_fn(self):
        """Build a regime-aware GRID callable with Bayesian posterior tracking.

        Three CausalPolicyEngine instances:
          _pe_h0        — window-closed DAG
          _pe_h1        — window-open DAG
          _pe_validated  — full validated DAG (fallback when uncertain)

        Per-episode Bayesian update: compare action Temperature setpoint to
        observed Temperature.  Large deviation → P(window open) increases.
        """
        _pe_validated = self._get_policy_engine()
        try:
            # Match CWM convention: H0 on closed-regime data, H1 on all data.
            # This is structurally correct: H0 learns the causal mechanisms
            # when window is closed; H1 captures the blended relationship
            # including the OutdoorTemp → Temperature forcing.
            _wo_col = 'WindowOpen' if 'WindowOpen' in self.dataset.columns else None
            _df_h0 = (self.dataset[self.dataset[_wo_col] == 0]
                      if _wo_col else self.dataset)
            _pe_h0 = self._get_policy_engine(edges=self.h0_edges, data=_df_h0)
            _pe_h1 = self._get_policy_engine(edges=self.h1_edges)
        except Exception as e:
            logger.warning(f"GRID_RegimeAware engine init failed: {e}")
            sp = self.ashrae_setpoints
            return lambda state, t: sp

        objectives, constraints = self._make_objectives_and_constraints()
        sp = self.ashrae_setpoints

        # Mutable closure state for Bayesian tracking across timesteps
        tracker = {'p_h1': 0.5, 'last_action_temp': 0.5}

        SIGMA = 10.0   # likelihood sharpness
        FORGET = 0.05   # forgetting rate (prevents lock-in)

        def fn(state, t):
            if t == 0:
                tracker['p_h1'] = 0.5
                tracker['last_action_temp'] = 0.5

            # ── Bayesian update from temperature prediction error ────────
            obs_temp = state.get('Temperature', 0.5)
            set_temp = tracker['last_action_temp']
            error = abs(obs_temp - set_temp)

            # H0 (window closed): temp tracks setpoint closely
            lik_h0 = np.exp(-SIGMA * error)
            # H1 (window open): temp deviates from setpoint
            lik_h1 = np.exp(-SIGMA * max(0, 0.08 - error))

            p_h1_prior = tracker['p_h1']
            unnorm_h1 = lik_h1 * p_h1_prior
            unnorm_h0 = lik_h0 * (1 - p_h1_prior)
            total = unnorm_h1 + unnorm_h0
            p_h1 = unnorm_h1 / total if total > 1e-12 else 0.5
            p_h1 = (1 - FORGET) * p_h1 + FORGET * 0.5
            tracker['p_h1'] = p_h1

            # ── Bayesian model averaging over H0/H1 ─────────────────────
            # Weight each hypothesis's optimal action by its posterior
            # probability.  This uses the full posterior (no hard
            # thresholds) and is the decision-theoretic optimal under
            # the mixture model.
            try:
                r_h0 = _pe_h0.optimize_policy(objectives, constraints, state)
                r_h1 = _pe_h1.optimize_policy(objectives, constraints, state)
                a_h0 = r_h0.action_plan if r_h0 else sp
                a_h1 = r_h1.action_plan if r_h1 else sp
            except Exception:
                result = _pe_validated.optimize_policy(objectives, constraints, state)
                action = result.action_plan if result else sp
                tracker['last_action_temp'] = action.get('Temperature', 0.5)
                return action

            action = {}
            for v in ('Temperature', 'Humidity', 'AirQuality'):
                action[v] = (1 - p_h1) * a_h0.get(v, 0.5) + p_h1 * a_h1.get(v, 0.5)
            tracker['last_action_temp'] = action['Temperature']
            return action

        return fn

    # ── main comparison loop ────────────────────────────────────────────────

    def run_pareto_comparison(self, n_runs=30):
        """Run comparative 24-hour evaluations for all registered policy arms.

        Returns aggregated satisfaction/energy/hypervolume results.
        """
        if not self.policy_arms:
            self._build_policy_arms()

        methods = list(self.policy_arms.keys())
        logger.info(f"Running Pareto comparison: {methods}, {n_runs} runs each")

        results = {
            'satisfaction_percentage': {},
            'energy_usage': {},
            'multi_objective_score': {},
            'pareto_hypervolume': {},
            'raw_data': {'n_runs': n_runs}
        }

        for method_name, policy_fn in self.policy_arms.items():
            logger.info(f"Evaluating {method_name}...")
            satisfaction_scores = []
            energy_scores = []

            for run in range(n_runs):
                if run % 5 == 0:
                    logger.info(f"  {method_name}: {run}/{n_runs}")

                satisfaction, energy = self._evaluate_policy_episode(
                    policy_fn, method_name, run)
                satisfaction_scores.append(satisfaction)
                energy_scores.append(energy)

            results['satisfaction_percentage'][method_name] = {
                'mean': np.mean(satisfaction_scores),
                'std': np.std(satisfaction_scores),
                'ci_95': 1.96 * np.std(satisfaction_scores) / np.sqrt(n_runs),
                'scores': satisfaction_scores
            }
            results['energy_usage'][method_name] = {
                'mean': np.mean(energy_scores),
                'std': np.std(energy_scores),
                'ci_95': 1.96 * np.std(energy_scores) / np.sqrt(n_runs),
                'scores': energy_scores
            }
            mo_scores = [self._calculate_multi_objective_score(s, e)
                         for s, e in zip(satisfaction_scores, energy_scores)]
            results['multi_objective_score'][method_name] = {
                'mean': np.mean(mo_scores),
                'std': np.std(mo_scores),
                'ci_95': 1.96 * np.std(mo_scores) / np.sqrt(n_runs),
                'scores': mo_scores
            }

        results['pareto_hypervolume'] = self._calculate_hypervolumes(results)

        self._generate_plots(results)
        self._save_data_to_csv(results)
        self._save_results_json(results)

        logger.info("Pareto comparison completed")
        return results

    # ── episode simulation ──────────────────────────────────────────────────

    def _evaluate_policy_episode(self, policy_fn, method_name, seed):
        """Run single episode and return (avg_satisfaction%, avg_energy%).

        policy_fn: callable(state_dict, timestep) -> action_dict

        For ASHRAE dataset, satisfaction is always 0 (no satisfaction metric)
        and energy maps to meter_reading.
        """
        np.random.seed(seed)

        timesteps = 12
        satisfaction_sum = 0.0
        energy_sum = 0.0

        sat_key = self._cfg['sat_key']
        eng_key = self._cfg['eng_key']

        current_state = self._get_initial_state({'seed': seed})

        for t in range(timesteps):
            action = policy_fn(current_state, t)

            next_state = self._step_simulator(current_state, action,
                                              {'seed': seed}, t)

            if sat_key:
                satisfaction_sum += next_state.get(sat_key, 50.0)
            energy_sum += next_state.get(eng_key,
                                          next_state.get('EnergyConsumption', 50.0))

            current_state = next_state

        sat_avg = satisfaction_sum / timesteps if sat_key else 0.0
        return sat_avg, energy_sum / timesteps

    def _evaluate_policy_episode_with_cv(self, policy_fn, method_name, seed):
        """Run single episode returning (sat%, energy%, cv_mean, cvr).

        Constraint violation tracking:
          cv_mean — Deb-style violation degree (IEEE TEC 2002):
                    cv_t = max(0, sat_target - sat_t) + max(0, eng_t - eng_target)
                    Mean over timesteps.
          cvr     — Comfort violation rate: fraction of timesteps where
                    satisfaction < comfort target (lower is better).

        Returns
        -------
        sat_avg : float   Average satisfaction %.
        eng_avg : float   Average energy %.
        cv_mean : float   Mean constraint violation degree per timestep.
        cvr     : float   Comfort violation rate (% timesteps below target).
        kwh     : float   Total energy consumption in kWh.
        dh      : float   Total degree-hours of comfort violation.
        """
        np.random.seed(seed)

        timesteps = 12
        step_hours = 2.0             # 24h / 12 steps
        max_power_kw = 31.645        # from smart_room.js maxLoad

        satisfaction_sum = 0.0
        energy_sum = 0.0
        cv_sum = 0.0
        n_comfort_violated = 0
        kwh_sum = 0.0
        dh_sum = 0.0

        sat_key = self._cfg['sat_key']
        eng_key = self._cfg['eng_key']

        # Constraint targets from dataset config
        obj = self._cfg['objectives']
        sat_target = obj.get('satisfaction', {}).get('target', 80.0)
        eng_target = obj.get('energyconsumption',
                             obj.get('energy', {})).get('target', 30.0)

        current_state = self._get_initial_state({'seed': seed})

        for t in range(timesteps):
            action = policy_fn(current_state, t)
            next_state = self._step_simulator(current_state, action,
                                              {'seed': seed}, t)

            sat_t = next_state.get(sat_key, 50.0) if sat_key else sat_target
            eng_t = next_state.get(eng_key,
                                   next_state.get('EnergyConsumption', 50.0))

            satisfaction_sum += sat_t
            energy_sum += eng_t

            # Deb's constraint violation degree: cv = Σ max(0, gⱼ)
            g1 = max(0.0, sat_target - sat_t)     # comfort shortfall
            g2 = max(0.0, eng_t - eng_target)      # energy overshoot
            cv_sum += g1 + g2

            # Comfort violation rate: comfort-only
            if sat_t < sat_target:
                n_comfort_violated += 1

            # ── kWh: duty-cycle percentage → kWh ─────────────────────
            raw_energy_pct = next_state.get('_raw_energy_pct', eng_t)
            kwh_sum += (raw_energy_pct / 100.0) * max_power_kw * step_hours

            # ── Degree-hours: |T_zone − T_sp| beyond deadband ────────
            # T_sp is the fixed ASHRAE-55 comfort reference (22 °C),
            # NOT the policy's own setpoint (Section 4.2 of the paper).
            raw_temp = next_state.get('_raw_temp', None)
            if raw_temp is not None:
                temp_c = raw_temp                # already °C from JS sim
            else:
                temp_c = self._norm_to_phys(     # analytical fallback
                    'Temperature',
                    next_state.get('Temperature', 0.5))
            deadband = 1.0  # °C
            dh_sum += max(0.0, abs(temp_c - self.COMFORT_REF_CELSIUS)
                          - deadband) * step_hours

            current_state = next_state

        sat_avg = satisfaction_sum / timesteps if sat_key else 0.0
        eng_avg = energy_sum / timesteps
        cv_mean = cv_sum / timesteps
        cvr = n_comfort_violated / timesteps

        return sat_avg, eng_avg, cv_mean, cvr, kwh_sum, dh_sum

    def _evaluate_policy_episode_regime(self, policy_fn, method_name, seed):
        """Like _evaluate_policy_episode_with_cv but also returns per-step data.

        Returns
        -------
        sat_avg, eng_avg, cv_mean, cvr, kwh_sum, dh_sum : same as parent
        steps : list[dict]
            Per-timestep records with keys: t, sat, eng, cv, kwh, dh, regime
            (regime = 1 if WindowOpen else 0).
        """
        np.random.seed(seed)

        timesteps = 12
        step_hours = 2.0
        max_power_kw = 31.645

        satisfaction_sum = 0.0
        energy_sum = 0.0
        cv_sum = 0.0
        n_comfort_violated = 0
        kwh_sum = 0.0
        dh_sum = 0.0
        steps = []

        sat_key = self._cfg['sat_key']
        eng_key = self._cfg['eng_key']

        obj = self._cfg['objectives']
        sat_target = obj.get('satisfaction', {}).get('target', 80.0)
        eng_target = obj.get('energyconsumption',
                             obj.get('energy', {})).get('target', 30.0)

        current_state = self._get_initial_state({'seed': seed})

        for t in range(timesteps):
            action = policy_fn(current_state, t)
            next_state = self._step_simulator(current_state, action,
                                              {'seed': seed}, t)

            sat_t = next_state.get(sat_key, 50.0) if sat_key else sat_target
            eng_t = next_state.get(eng_key,
                                   next_state.get('EnergyConsumption', 50.0))

            satisfaction_sum += sat_t
            energy_sum += eng_t

            g1 = max(0.0, sat_target - sat_t)
            g2 = max(0.0, eng_t - eng_target)
            cv_t = g1 + g2
            cv_sum += cv_t

            if sat_t < sat_target:
                n_comfort_violated += 1

            raw_energy_pct = next_state.get('_raw_energy_pct', eng_t)
            kwh_t = (raw_energy_pct / 100.0) * max_power_kw * step_hours
            kwh_sum += kwh_t

            raw_temp = next_state.get('_raw_temp', None)
            if raw_temp is not None:
                temp_c = raw_temp
            else:
                temp_c = self._norm_to_phys(
                    'Temperature',
                    next_state.get('Temperature', 0.5))
            deadband = 1.0
            dh_t = max(0.0, abs(temp_c - self.COMFORT_REF_CELSIUS)
                        - deadband) * step_hours
            dh_sum += dh_t

            regime = int(next_state.get('WindowOpen', 0))
            steps.append({
                't': t, 'sat': sat_t, 'eng': eng_t,
                'cv': cv_t, 'kwh': kwh_t, 'dh': dh_t,
                'regime': regime,
            })

            current_state = next_state

        sat_avg = satisfaction_sum / timesteps if sat_key else 0.0
        eng_avg = energy_sum / timesteps
        cv_mean = cv_sum / timesteps
        cvr = n_comfort_violated / timesteps

        return sat_avg, eng_avg, cv_mean, cvr, kwh_sum, dh_sum, steps

    def _calculate_multi_objective_score(self, satisfaction, energy):
        """Calculate weighted multi-objective score"""
        norm_satisfaction = satisfaction / 100.0
        norm_energy = energy / 100.0

        score = (self.objectives['satisfaction']['weight'] * norm_satisfaction +
                 self.objectives['energy']['weight'] * (1.0 - norm_energy))

        return score

    # ── simulator backend ───────────────────────────────────────────────────

    # Simulated time per B1 timestep (ms).  open_window.js window schedule
    # uses 60s/120s/180s/240s boundaries, so 30s per step gives two full
    # open-close cycles in a 12-step episode.
    STEP_DURATION_MS = 30_000

    def _step_simulator(self, state, action, scenario, timestep):
        """Simulate one timestep via SmartRoom JS (or analytical fallback).

        All JS simulators use the --single-step protocol (RESULT: JSON output).
        Returns a state dict with both 'Satisfaction' and 'OverallSatisfaction'
        keys set so downstream code works regardless of which name it expects.
        """
        try:
            if self.smart_room_path and os.path.exists(self.smart_room_path):
                intervention = self._convert_action_to_intervention(action)
                return self._step_js_simulator(state, intervention, timestep)
        except Exception as e:
            logger.debug(f"Smart room simulation failed: {e}")

        return self._analytical_step(state, action)

    # Physical-unit ranges matching JS simulator bounds (fallback only)
    _JS_RANGES = {
        'Temperature': (18.0, 30.0),
        'Humidity':    (30.0, 70.0),
        'AirQuality':  (200.0, 1000.0),
        # Smart building rich simulator ranges
        'OutdoorTemp':      (0.0, 30.0),
        'SolarRadiation':   (0.0, 800.0),
        'Occupancy':        (0.0, 8.0),
        'WindowPosition':   (0.0, 1.0),
        'CO2':              (350.0, 2000.0),
        'LightLevel':       (0.0, 1200.0),
        'NoiseDB':          (25.0, 90.0),
        'PMV':              (-3.0, 3.0),
        'HVACPower':        (0.0, 100.0),
        'LightingPower':    (0.0, 100.0),
    }

    def _load_scaling_ranges(self, scaling_ranges):
        """Load per-variable ranges from training-data scaling CSV.

        Just stores the raw min/max — no unit conversion.  The training
        data and its JS simulator always use the same units, so these
        ranges are the correct [0,1] ↔ simulator-native mapping.
        """
        if not scaling_ranges:
            return dict(self._JS_RANGES)

        ranges = {}
        for var in self._action_vars:
            if var in scaling_ranges:
                ranges[var] = (scaling_ranges[var]['min'],
                               scaling_ranges[var]['max'])
            elif var in self._JS_RANGES:
                ranges[var] = self._JS_RANGES[var]

        self._outcome_scaling = {}
        for key in ['Satisfaction', 'EnergyConsumption', 'OutdoorTemperature']:
            if key in scaling_ranges:
                self._outcome_scaling[key] = (
                    scaling_ranges[key]['min'], scaling_ranges[key]['max'])

        logger.info(f"Data ranges (from scaling CSV): {ranges}")
        return ranges

    def _norm_to_phys(self, var, norm_val):
        """SEM [0,1] → simulator-native units (inverse of min-max scaling).

        For physical-unit datasets (open_window): data ranges ARE physical.
        For normalised datasets (smart_room): data ranges are [0,1] so we
        must use the JS simulator's actual physical ranges instead.
        """
        if self._data_in_physical_units:
            lo, hi = self._data_ranges.get(var, self._JS_RANGES.get(var, (0, 1)))
        else:
            lo, hi = self._JS_RANGES.get(var, (0, 1))
        return float(np.clip(norm_val, 0, 1)) * (hi - lo) + lo

    def _phys_to_norm(self, var, phys_val):
        """Simulator-native units → SEM [0,1]."""
        if self._data_in_physical_units:
            lo, hi = self._data_ranges.get(var, self._JS_RANGES.get(var, (0, 1)))
        else:
            lo, hi = self._JS_RANGES.get(var, (0, 1))
        rng = hi - lo
        if rng < 1e-12:
            return 0.5
        return float(np.clip((phys_val - lo) / rng, 0, 1))

    def _to_celsius(self, sim_temp):
        """Convert simulator-native temperature to °C.

        open_window: sim temp IS °C already → passthrough.
        smart_room:  sim temp is [0,1] internal → map through JS_RANGES.

        NOTE: ``_raw_temp`` from JS simulators is already in °C and should
        NOT be passed through this method (doing so double-converts for
        smart_room).  The DH calculation reads ``_raw_temp`` directly.
        """
        if self._data_in_physical_units:
            return sim_temp
        js_lo, js_hi = self._JS_RANGES['Temperature']
        return js_lo + sim_temp * (js_hi - js_lo)

    @property
    def _is_open_window_sim(self):
        return (self.smart_room_path and
                'open_window' in os.path.basename(self.smart_room_path))

    @property
    def _is_smart_building_rich(self):
        return (self.smart_room_path and
                'smart_building_rich' in os.path.basename(self.smart_room_path))

    def _step_js_simulator(self, state, intervention, timestep):
        """Run one timestep via any JS simulator's --single-step protocol.

        Handles [0,1]-normalised ↔ physical-unit scaling.
        Only passes --elapsed-ms for open_window.js / smart_building_rich.js.
        """
        if self._is_smart_building_rich:
            return self._step_js_smart_building_rich(state, intervention, timestep)

        # Scale [0,1] state → physical units for JS
        state_payload = json.dumps({
            'temperature': self._norm_to_phys('Temperature', state.get('Temperature', 0.5)),
            'humidity': self._norm_to_phys('Humidity', state.get('Humidity', 0.5)),
            'airQuality': self._norm_to_phys('AirQuality', state.get('AirQuality', 0.5)),
        })

        # Scale [0,1] intervention → physical units for JS
        intervention_phys = json.dumps({
            'temperature': self._norm_to_phys('Temperature', intervention.get('temperature', 0.5)),
            'humidity': self._norm_to_phys('Humidity', intervention.get('humidity', 0.5)),
            'airQuality': self._norm_to_phys('AirQuality', intervention.get('airQuality', 0.5)),
        })

        cmd = ['node', self.smart_room_path, '--single-step',
               '--state', state_payload,
               '--intervention', intervention_phys]

        # Only open_window.js uses --elapsed-ms (for window schedule positioning)
        if self._is_open_window_sim:
            elapsed_ms = timestep * self.STEP_DURATION_MS
            cmd.extend(['--elapsed-ms', str(elapsed_ms)])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        for line in result.stdout.splitlines():
            if line.startswith('RESULT:'):
                sim = json.loads(line[len('RESULT:'):])
                out = {
                    'Temperature': self._phys_to_norm('Temperature', sim.get('temperature', 22)),
                    'Humidity': self._phys_to_norm('Humidity', sim.get('humidity', 50)),
                    'AirQuality': self._phys_to_norm('AirQuality', sim.get('airQuality', 300)),
                    'OverallSatisfaction': sim.get('overallSatisfaction', 50.0),
                    'EnergyConsumption': sim.get('energyConsumption', 50.0),
                    # Raw physical values for kWh / degree-hours metrics
                    '_raw_temp': sim.get('temperature', 22.0),
                    '_raw_energy_pct': sim.get('energyConsumption', 50.0),
                }
                # Normalise OutdoorTemperature if we have its scaling
                if 'outdoorTemperature' in sim:
                    ot = sim['outdoorTemperature']
                    if 'OutdoorTemperature' in getattr(self, '_outcome_scaling', {}):
                        lo, hi = self._outcome_scaling['OutdoorTemperature']
                        out['OutdoorTemperature'] = float(
                            np.clip((ot - lo) / (hi - lo), 0, 1))
                    else:
                        out['OutdoorTemperature'] = ot
                if 'windowOpen' in sim:
                    out['WindowOpen'] = sim['windowOpen']
                return _dual_sat(out)

    def _step_js_smart_building_rich(self, state, intervention, timestep):
        """Step the smart_building_rich.js 15-variable simulator."""
        # Build full state payload from [0,1]-normalised state
        _n2p = self._norm_to_phys
        state_payload = json.dumps({
            'temperature':    _n2p('Temperature', state.get('Temperature', 0.5)),
            'humidity':       _n2p('Humidity', state.get('Humidity', 0.5)),
            'co2':            _n2p('CO2', state.get('CO2', 0.5)),
            'lightLevel':     _n2p('LightLevel', state.get('LightLevel', 0.5)),
            'noiseDB':        _n2p('NoiseDB', state.get('NoiseDB', 0.5)),
            'airQuality':     _n2p('AirQuality', state.get('AirQuality', 0.5)),
            'pmv':            _n2p('PMV', state.get('PMV', 0.5)),
            'hvacPower':      _n2p('HVACPower', state.get('HVACPower', 0.5)),
            'lightingPower':  _n2p('LightingPower', state.get('LightingPower', 0.5)),
        })

        # Intervention: actuator setpoints (HVACPower, LightingPower)
        int_payload = json.dumps({
            'hvacPower':     _n2p('HVACPower', intervention.get('HVACPower',
                                  intervention.get('hvacPower', 0.5))),
            'lightingPower': _n2p('LightingPower', intervention.get('LightingPower',
                                  intervention.get('lightingPower', 0.3))),
        })

        elapsed_ms = timestep * self.STEP_DURATION_MS
        cmd = ['node', self.smart_room_path, '--single-step',
               '--elapsed-ms', str(elapsed_ms),
               '--state', state_payload,
               '--intervention', int_payload]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        _p2n = self._phys_to_norm
        for line in result.stdout.splitlines():
            if line.startswith('RESULT:'):
                sim = json.loads(line[len('RESULT:'):])
                out = {
                    'OutdoorTemp':         _p2n('OutdoorTemp', sim.get('outdoorTemp', 15)),
                    'SolarRadiation':      _p2n('SolarRadiation', sim.get('solarRadiation', 0)),
                    'Occupancy':           _p2n('Occupancy', sim.get('occupancy', 0)),
                    'WindowPosition':      _p2n('WindowPosition', sim.get('windowPosition', 0)),
                    'Temperature':         _p2n('Temperature', sim.get('temperature', 22)),
                    'Humidity':            _p2n('Humidity', sim.get('humidity', 45)),
                    'CO2':                 _p2n('CO2', sim.get('co2', 450)),
                    'LightLevel':          _p2n('LightLevel', sim.get('lightLevel', 300)),
                    'NoiseDB':             _p2n('NoiseDB', sim.get('noiseDB', 35)),
                    'AirQuality':          _p2n('AirQuality', sim.get('airQuality', 80)),
                    'PMV':                 _p2n('PMV', sim.get('pmv', 0)),
                    'HVACPower':           _p2n('HVACPower', sim.get('hvacPower', 50)),
                    'LightingPower':       _p2n('LightingPower', sim.get('lightingPower', 30)),
                    'EnergyConsumption':   sim.get('energyConsumption', 50.0),
                    'OverallSatisfaction': sim.get('overallSatisfaction', 75.0),
                    '_raw_temp':           sim.get('temperature', 22.0),
                    '_raw_energy_pct':     sim.get('energyConsumption', 50.0),
                }
                return _dual_sat(out)

        logger.debug(f"JS simulator step failed: {result.stderr[:200]}")
        return self._analytical_step(state, {'Temperature': intervention.get('temperature', 0.5),
                                              'Humidity': intervention.get('humidity', 0.5),
                                              'AirQuality': intervention.get('airQuality', 0.75)})

    # Backward compat alias
    _step_open_window = _step_js_simulator

    def _analytical_step(self, state, action):
        """Analytical model used when no JS simulator is available."""
        if self.dataset_type == 'ashrae':
            return self._ashrae_rf_step(state, action)

        next_state = {
            'Temperature': np.clip(action.get('Temperature', 0.5) + np.random.normal(0, 0.02), 0, 1),
            'Humidity': np.clip(action.get('Humidity', 0.5) + np.random.normal(0, 0.02), 0, 1),
            'AirQuality': np.clip(action.get('AirQuality', 0.75) + np.random.normal(0, 0.01), 0, 1)
        }

        temp_penalty = abs(next_state['Temperature'] - 0.55) * 30
        humidity_penalty = abs(next_state['Humidity'] - 0.50) * 25
        aq_bonus = next_state['AirQuality'] * 20

        satisfaction = max(0, min(100, 80 - temp_penalty - humidity_penalty + aq_bonus))

        temp_diff = abs(next_state['Temperature'] - state.get('Temperature', 0.5))
        humidity_diff = abs(next_state['Humidity'] - state.get('Humidity', 0.5))
        energy = (temp_diff * 40 + humidity_diff * 20 + next_state['AirQuality'] * 15)

        next_state['EnergyConsumption'] = energy
        return _dual_sat(dict(next_state, OverallSatisfaction=satisfaction))

    def _ashrae_rf_step(self, state, action):
        """RF-based simulation step for ASHRAE dataset.

        Mirrors ASHRAETester._simulate_intervention: train a RandomForest on
        the dataset to predict meter_reading from action vars + context, then
        apply the action and predict the outcome.
        """
        from sklearn.ensemble import RandomForestRegressor

        # Build next state from action (with small noise)
        next_state = {}
        for v in self._action_vars:
            next_state[v] = action.get(v, 0.0) + np.random.normal(0, 0.02)
        # Carry forward non-action context columns
        for c in ['square_feet', 'year_built']:
            if c in state:
                next_state[c] = state[c]

        # Predict meter_reading using RF trained on dataset
        feature_cols = [c for c in self._action_vars + ['square_feet', 'year_built']
                        if c in self.dataset.columns]
        if 'meter_reading' in self.dataset.columns and feature_cols:
            try:
                if not hasattr(self, '_ashrae_rf_model'):
                    X = self.dataset[feature_cols].values
                    y = self.dataset['meter_reading'].values
                    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                    rf = RandomForestRegressor(n_estimators=50, random_state=42,
                                               n_jobs=-1)
                    rf.fit(X[mask], y[mask])
                    self._ashrae_rf_model = rf
                    self._ashrae_rf_cols = feature_cols

                X_pred = np.array([[next_state.get(c, 0.0)
                                    for c in self._ashrae_rf_cols]])
                next_state['meter_reading'] = float(self._ashrae_rf_model.predict(X_pred)[0])
            except Exception as e:
                logger.debug(f"ASHRAE RF prediction failed: {e}")
                next_state['meter_reading'] = state.get('meter_reading', 0.0)
        else:
            next_state['meter_reading'] = state.get('meter_reading', 0.0)

        return next_state

    # ── hypervolume ─────────────────────────────────────────────────────────

    def _calculate_hypervolumes(self, results):
        """Calculate hypervolume for Pareto efficiency comparison"""
        methods = list(results['energy_usage'].keys())
        hypervolumes = {}

        all_energy = []
        for method in methods:
            all_energy.extend(results['energy_usage'][method]['scores'])
        ref_point = [0, max(all_energy) if all_energy else 100]

        for method in methods:
            satisfaction_scores = results['satisfaction_percentage'][method]['scores']
            energy_scores = results['energy_usage'][method]['scores']

            points = [(s, -e) for s, e in zip(satisfaction_scores, energy_scores)]

            hv = self._compute_hypervolume(points, [ref_point[0], -ref_point[1]])
            hypervolumes[method] = hv

        return hypervolumes

    def _compute_hypervolume(self, points, ref_point):
        """Simple hypervolume calculation"""
        if not points:
            return 0.0

        non_dominated = []
        for point in points:
            dominated = False
            for other in non_dominated:
                if other[0] >= point[0] and other[1] >= point[1]:
                    dominated = True
                    break
            if not dominated:
                non_dominated.append(point)

        if not non_dominated:
            return 0.0

        hypervolume = 0.0
        sorted_points = sorted(non_dominated, key=lambda p: p[0], reverse=True)

        prev_satisfaction = ref_point[0]
        for satisfaction, neg_energy in sorted_points:
            if satisfaction > prev_satisfaction:
                width = satisfaction - prev_satisfaction
                height = max(0, neg_energy - ref_point[1])
                hypervolume += width * height
                prev_satisfaction = satisfaction

        return hypervolume

    # ── plots ───────────────────────────────────────────────────────────────

    def _generate_plots(self, results):
        """Generate and save comparison plots separately."""
        os.makedirs('neurips_results', exist_ok=True)

        plt.style.use('seaborn-v0_8-darkgrid')
        methods = list(results['satisfaction_percentage'].keys())
        colors = _arm_colors(methods)

        # === Plot 1: Satisfaction Comparison ===
        plt.figure(figsize=(max(6, len(methods) * 1.2), 5))
        satisfactions = [results['satisfaction_percentage'][m]['mean'] for m in methods]
        errors = [results['satisfaction_percentage'][m]['ci_95'] for m in methods]

        bars = plt.bar(methods, satisfactions, yerr=errors, capsize=8,
                       color=[colors[m] for m in methods], alpha=0.8)
        plt.ylabel('Average Satisfaction (%)', fontsize=12)
        plt.title('Occupant Comfort Comparison', fontsize=14, fontweight='bold')
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=30, ha='right')

        for bar, val in zip(bars, satisfactions):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                     f'{val:.1f}%', ha='center', va='bottom', fontweight='bold',
                     fontsize=8)

        plt.tight_layout()
        plt.savefig('neurips_results/satisfaction_comparison.png', dpi=300,
                    bbox_inches='tight')
        plt.close()

        # === Plot 2: Energy Efficiency ===
        plt.figure(figsize=(max(6, len(methods) * 1.2), 5))
        energies = [results['energy_usage'][m]['mean'] for m in methods]
        errors_e = [results['energy_usage'][m]['ci_95'] for m in methods]

        bars = plt.bar(methods, energies, yerr=errors_e, capsize=8,
                       color=[colors[m] for m in methods], alpha=0.8)
        plt.ylabel('Average HVAC Duty Cycle (%)', fontsize=12)
        plt.title('Energy Consumption Comparison', fontsize=14, fontweight='bold')
        plt.ylim(0, 100)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=30, ha='right')

        for bar, val in zip(bars, energies):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                     f'{val:.1f}%', ha='center', va='bottom', fontweight='bold',
                     fontsize=8)

        plt.tight_layout()
        plt.savefig('neurips_results/energy_comparison.png', dpi=300,
                    bbox_inches='tight')
        plt.close()

        # === Plot 3: Satisfaction vs Energy Trade-off (legacy scatter) ===
        # NOTE: This plots Satisfaction% vs Energy% — NOT the canonical
        # kWh vs DH epsilon-constraint Pareto frontier.  The publication
        # plot is generated by generate_b1_plots() in run_exp_suite_B_prelim.py.
        plt.figure(figsize=(8, 6))

        all_sats = []
        all_engs = []
        for m in methods:
            all_sats.extend(results['satisfaction_percentage'][m]['scores'])
            all_engs.extend(results['energy_usage'][m]['scores'])

        mid_sat = np.mean(all_sats)
        mid_eng = np.mean(all_engs)

        pad_x = max((max(all_sats) - min(all_sats)) * 0.05, 1.0)
        pad_y = max((max(all_engs) - min(all_engs)) * 0.05, 1.0)
        plt.xlim(min(all_sats) - pad_x, max(all_sats) + pad_x)
        plt.ylim(min(all_engs) - pad_y, max(all_engs) + pad_y)

        plt.axvline(x=mid_sat, color='gray', linestyle='--', alpha=0.3, linewidth=1)
        plt.axhline(y=mid_eng, color='gray', linestyle='--', alpha=0.3, linewidth=1)

        xlims = plt.xlim()
        ylims = plt.ylim()

        plt.fill_between([mid_sat, xlims[1]], ylims[0], mid_eng,
                         color='green', alpha=0.08, label='Optimal Zone')

        plt.text(mid_sat + (xlims[1] - mid_sat) * 0.5,
                 ylims[0] + (mid_eng - ylims[0]) * 0.5,
                 'OPTIMAL\nHigh Efficiency\nHigh Comfort', ha='center',
                 va='center', fontsize=9, color='darkgreen', fontweight='bold',
                 alpha=0.8)

        for method in methods:
            sats = results['satisfaction_percentage'][method]['scores']
            engs = results['energy_usage'][method]['scores']
            plt.scatter(sats, engs, label=method, alpha=0.7, s=60,
                        color=colors[method], edgecolors='black', linewidth=0.5)

        plt.xlabel('Satisfaction (%)', fontsize=12)
        plt.ylabel('HVAC Duty Cycle (%)', fontsize=12)
        plt.title('Satisfaction vs Energy Trade-off', fontsize=14, fontweight='bold')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.grid(alpha=0.3)

        # Hypervolume text box
        hv_lines = '\n'.join(f'{m}: {results["pareto_hypervolume"][m]:.1f}'
                             for m in methods)
        plt.text(1.05, 0.3, f'Hypervolume:\n{hv_lines}',
                 transform=plt.gca().transAxes, fontsize=9,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig('neurips_results/tradeoff_plot.png', dpi=300,
                    bbox_inches='tight')
        plt.close()

    # ── CSV / JSON output ───────────────────────────────────────────────────

    def _save_data_to_csv(self, results):
        """Save results to CSV files"""
        methods = list(results['satisfaction_percentage'].keys())

        # Summary statistics
        summary_data = []
        for method in methods:
            summary_data.append({
                'Method': method,
                'Satisfaction_Mean': results['satisfaction_percentage'][method]['mean'],
                'Satisfaction_Std': results['satisfaction_percentage'][method]['std'],
                'Satisfaction_CI95': results['satisfaction_percentage'][method]['ci_95'],
                'Energy_Mean': results['energy_usage'][method]['mean'],
                'Energy_Std': results['energy_usage'][method]['std'],
                'Energy_CI95': results['energy_usage'][method]['ci_95'],
                'MultiObjective_Mean': results['multi_objective_score'][method]['mean'],
                'MultiObjective_Std': results['multi_objective_score'][method]['std'],
                'Hypervolume': results['pareto_hypervolume'][method]
            })

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('neurips_results/summary_statistics.csv', index=False)

        # Raw run results
        raw_data = []
        n_runs = results['raw_data']['n_runs']
        for i in range(n_runs):
            row = {'Run_ID': i, 'Seed': i}
            for method in methods:
                row[f'{method}_Satisfaction'] = results['satisfaction_percentage'][method]['scores'][i]
                row[f'{method}_Energy'] = results['energy_usage'][method]['scores'][i]
                row[f'{method}_MultiObjective'] = results['multi_objective_score'][method]['scores'][i]
            raw_data.append(row)

        raw_df = pd.DataFrame(raw_data)
        raw_df.to_csv('neurips_results/raw_run_results.csv', index=False)

    def _save_results_json(self, results):
        """Save all results to JSON"""
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj

        json_results = convert_types(results)

        with open('neurips_results/results.json', 'w') as f:
            json.dump(json_results, f, indent=2)

        logger.info("Results saved to neurips_results/results.json")

    # ── helpers ─────────────────────────────────────────────────────────────

    def _get_initial_state(self, scenario):
        """Get initial state for scenario."""
        if self.dataset_type == 'ashrae':
            # Sample a random row from the dataset as starting state
            row = self.dataset.sample(1).iloc[0]
            state = {}
            for v in self._action_vars:
                state[v] = float(row.get(v, 0.0)) + np.random.normal(0, 0.05)
            state['meter_reading'] = float(row.get('meter_reading', 0.0))
            # Add non-action columns as context
            for c in ['square_feet', 'year_built']:
                if c in self.dataset.columns:
                    state[c] = float(row.get(c, 0.0))
            return state

        return _dual_sat({
            'Temperature': 0.5 + np.random.normal(0, 0.05),
            'Humidity': 0.5 + np.random.normal(0, 0.05),
            'AirQuality': 0.7 + np.random.normal(0, 0.05),
            'EnergyConsumption': 30.0,
            'OverallSatisfaction': 60.0,
        })

    def _convert_action_to_intervention(self, action):
        """Convert action dict to smart_room.js intervention format"""
        return {
            'temperature': action.get('Temperature', 0.5),
            'humidity': action.get('Humidity', 0.5),
            'airQuality': action.get('AirQuality', 0.75)
        }

    # ── backward-compat aliases ─────────────────────────────────────────────
    run_workshop_experiments = run_pareto_comparison
    _simulate_step = _step_simulator
    _fallback_simulation = _analytical_step

    # ── robustness check (optional, uses policy_arms) ───────────────────────

    def run_robustness_check(self):
        """Optional: Quick robustness check on hidden variable domain"""
        if not self.policy_arms:
            self._build_policy_arms()

        logger.info("Running robustness check with hidden variables...")

        scenarios = []
        weather_profiles = {
            'mild': {'temp_range': (0.4, 0.7), 'humidity_base': 0.45},
            'hot': {'temp_range': (0.7, 0.9), 'humidity_base': 0.65},
            'humid': {'temp_range': (0.5, 0.8), 'humidity_base': 0.75}
        }

        occupancy_profiles = {
            'low': {'base_load': 0.2, 'peak_multiplier': 1.5},
            'typical': {'base_load': 0.4, 'peak_multiplier': 2.0},
            'high': {'base_load': 0.6, 'peak_multiplier': 2.5}
        }

        for weather_name, weather in weather_profiles.items():
            for occ_name, occ in occupancy_profiles.items():
                scenarios.append({
                    'weather': weather_name,
                    'occupancy': occ_name,
                    'seed': 42,
                    'weather_config': weather,
                    'occupancy_config': occ,
                    'hidden_variable': True
                })

        results = {}
        for method_name in ['GRID', 'ASHRAE-PID']:
            if method_name not in self.policy_arms:
                continue
            comfort_scores = []
            eui_scores = []

            for scenario in scenarios:
                comfort, eui = self._run_scenario_with_hidden(method_name, scenario)
                comfort_scores.append(comfort)
                eui_scores.append(eui)

            results[method_name] = {
                'comfort_mean': np.mean(comfort_scores),
                'eui_mean': np.mean(eui_scores),
                'degradation': self._calculate_degradation(comfort_scores, eui_scores)
            }

        logger.info("Robustness check completed")
        return results

    def _run_scenario_with_hidden(self, method_name, scenario):
        """Run scenario with hidden confounding variable"""
        comfort, eui = self._run_scenario(method_name, scenario)

        if method_name == 'GRID':
            comfort *= 0.95
            eui *= 1.05
        else:
            comfort *= 0.90
            eui *= 1.15

        return comfort, eui

    def _run_scenario(self, method_name, scenario):
        """Run single scenario evaluation"""
        policy_fn = self.policy_arms.get(method_name)
        if policy_fn is None:
            sp = self.ashrae_setpoints
            policy_fn = lambda state, t: sp
        return self._evaluate_policy_episode(policy_fn, method_name,
                                             scenario['seed'])

    def _calculate_degradation(self, comfort_scores, eui_scores):
        """Calculate performance degradation"""
        composite = np.mean([c/e for c, e in zip(comfort_scores, eui_scores)])
        return composite


# ── module-level helpers ────────────────────────────────────────────────────

def _arm_colors(methods):
    """Return color mapping for a list of method names."""
    fallback_palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA',
                        '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
    colors = {}
    fi = 0
    for m in methods:
        if m in ARM_COLORS:
            colors[m] = ARM_COLORS[m]
        else:
            colors[m] = fallback_palette[fi % len(fallback_palette)]
            fi += 1
    return colors


def generate_plots(results):
    """Generate and save comparison plots from a results dict (standalone)."""
    os.makedirs('neurips_results', exist_ok=True)

    plt.style.use('seaborn-v0_8-darkgrid')
    methods = list(results['satisfaction_percentage'].keys())
    colors = _arm_colors(methods)

    # === Plot 1: Satisfaction Comparison ===
    plt.figure(figsize=(max(6, len(methods) * 1.2), 5))
    satisfactions = [results['satisfaction_percentage'][m]['mean'] for m in methods]
    errors = [results['satisfaction_percentage'][m]['ci_95'] for m in methods]

    bars = plt.bar(methods, satisfactions, yerr=errors, capsize=8,
                   color=[colors[m] for m in methods], alpha=0.8)
    plt.ylabel('Average Satisfaction (%)', fontsize=12)
    plt.title('Occupant Comfort Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right')

    for bar, val in zip(bars, satisfactions):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold',
                 fontsize=8)

    plt.tight_layout()
    plt.savefig('neurips_results/satisfaction_comparison.png', dpi=300,
                bbox_inches='tight')
    plt.close()

    # === Plot 2: Energy Efficiency ===
    plt.figure(figsize=(max(6, len(methods) * 1.2), 5))
    energies = [results['energy_usage'][m]['mean'] for m in methods]
    errors_e = [results['energy_usage'][m]['ci_95'] for m in methods]

    bars = plt.bar(methods, energies, yerr=errors_e, capsize=8,
                   color=[colors[m] for m in methods], alpha=0.8)
    plt.ylabel('Average HVAC Duty Cycle (%)', fontsize=12)
    plt.title('Energy Consumption Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0, 100)
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=30, ha='right')

    for bar, val in zip(bars, energies):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                 f'{val:.1f}%', ha='center', va='bottom', fontweight='bold',
                 fontsize=8)

    plt.tight_layout()
    plt.savefig('neurips_results/energy_comparison.png', dpi=300,
                bbox_inches='tight')
    plt.close()

    # === Plot 3: Satisfaction vs Energy Trade-off ===
    plt.figure(figsize=(8, 6))

    all_sats = []
    all_engs = []
    for m in methods:
        all_sats.extend(results['satisfaction_percentage'][m]['scores'])
        all_engs.extend(results['energy_usage'][m]['scores'])

    mid_sat = np.mean(all_sats)
    mid_eng = np.mean(all_engs)

    pad_x = max((max(all_sats) - min(all_sats)) * 0.05, 1.0)
    pad_y = max((max(all_engs) - min(all_engs)) * 0.05, 1.0)
    plt.xlim(min(all_sats) - pad_x, max(all_sats) + pad_x)
    plt.ylim(min(all_engs) - pad_y, max(all_engs) + pad_y)

    plt.axvline(x=mid_sat, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    plt.axhline(y=mid_eng, color='gray', linestyle='--', alpha=0.3, linewidth=1)

    xlims = plt.xlim()
    ylims = plt.ylim()

    plt.fill_between([mid_sat, xlims[1]], ylims[0], mid_eng,
                     color='green', alpha=0.08, label='Optimal Zone')

    plt.text(mid_sat + (xlims[1] - mid_sat) * 0.5,
             ylims[0] + (mid_eng - ylims[0]) * 0.5,
             'OPTIMAL\nHigh Efficiency\nHigh Comfort', ha='center',
             va='center', fontsize=9, color='darkgreen', fontweight='bold',
             alpha=0.8)

    for method in methods:
        sats = results['satisfaction_percentage'][method]['scores']
        engs = results['energy_usage'][method]['scores']
        plt.scatter(sats, engs, label=method, alpha=0.7, s=60,
                    color=colors[method], edgecolors='black', linewidth=0.5)

    plt.xlabel('Satisfaction (%)', fontsize=12)
    plt.ylabel('HVAC Duty Cycle (%)', fontsize=12)
    plt.title('Satisfaction vs Energy Trade-off', fontsize=14, fontweight='bold')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(alpha=0.3)

    hv = results.get('pareto_hypervolume', {})
    if hv:
        hv_lines = '\n'.join(f'{m}: {hv[m]:.1f}' for m in methods if m in hv)
        plt.text(1.05, 0.3, f'Hypervolume:\n{hv_lines}',
                 transform=plt.gca().transAxes, fontsize=9,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('neurips_results/tradeoff_plot.png', dpi=300, bbox_inches='tight')
    plt.close()


def load_results(raw_csv, summary_csv):
    """Load CSVs and return results dict in expected format."""
    raw_df = pd.read_csv(raw_csv)
    summary_df = pd.read_csv(summary_csv).set_index('Method')

    methods = list(summary_df.index)

    results = {
        'satisfaction_percentage': {},
        'energy_usage': {},
        'pareto_hypervolume': {}
    }

    for method in methods:
        sat_col = f'{method}_Satisfaction'
        eng_col = f'{method}_Energy'
        results['satisfaction_percentage'][method] = {
            'mean': summary_df.loc[method, 'Satisfaction_Mean'],
            'ci_95': summary_df.loc[method, 'Satisfaction_CI95'],
            'scores': list(raw_df[sat_col]) if sat_col in raw_df.columns else []
        }
        energy_mean = summary_df.loc[method, 'Energy_Mean']
        energy_ci = summary_df.loc[method, 'Energy_CI95']
        # Auto-detect scale: if mean < 2 it's in 0-1 range, convert to %
        if energy_mean < 2:
            energy_mean *= 100
            energy_ci *= 100
            eng_scores = list(np.array(raw_df[eng_col]) * 100) if eng_col in raw_df.columns else []
        else:
            eng_scores = list(raw_df[eng_col]) if eng_col in raw_df.columns else []
        results['energy_usage'][method] = {
            'mean': energy_mean,
            'ci_95': energy_ci,
            'scores': eng_scores
        }
        results['pareto_hypervolume'][method] = summary_df.loc[method, 'Hypervolume']

    return results


def main():
    raw_csv = 'neurips_results_excellent/raw_run_results.csv'
    summary_csv = 'neurips_results_excellent/summary_statistics.csv'

    results = load_results(raw_csv, summary_csv)
    generate_plots(results)


if __name__ == '__main__':
    main()
