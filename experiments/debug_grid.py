#!/usr/bin/env python
"""Compare GRID vs GRID_ObsOnly actions step-by-step."""
import os, sys, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(__file__))

from quick_smoke import VALIDATED_EDGES, INTERVENTION_DIRECTIONS, FakePipeline, load_scaling
from experiments_neurips import NeurIPSExperiments
from src.policy_engine import CausalPolicyEngine

cfg = {
    'sim_path': 'js/smart_room.js',
    'obs_data_path': 'data_regen/smart_room_processed.csv',
    'scaling_path': 'data_regen/smart_room_processed_scaling.csv',
    'dataset_type': 'open_window',
    'data_in_physical_units': False,
}

obs_df = pd.read_csv(cfg['obs_data_path'])
scaling = load_scaling(cfg['scaling_path'])

exp = NeurIPSExperiments(
    pipeline=FakePipeline(VALIDATED_EDGES),
    dataset=obs_df,
    smart_room_path=cfg['sim_path'],
    comfort_target=1.0,
    obs_only_edges=VALIDATED_EDGES,
    h0_edges=None, h1_edges=None,
    dataset_type=cfg['dataset_type'],
    scaling_ranges=scaling,
    data_in_physical_units=cfg['data_in_physical_units'],
)

# Build both engines
engine_grid = exp._get_policy_engine()  # has intervention_directions
engine_obs = CausalPolicyEngine(
    {'validated_edges': list(VALIDATED_EDGES)}, obs_df,
    use_llm=False, action_vars=['Temperature', 'Humidity', 'AirQuality'],
    outcome_map={'energy': 'EnergyConsumption', 'satisfaction': 'Satisfaction'})

objectives, constraints = exp._make_objectives_and_constraints()

# Compare SEM coefficients for Satisfaction
print("=== SEM(Satisfaction) comparison ===")
print(f"{'':30s} {'GRID (dir-constrained)':>22s}  {'ObsOnly (raw)':>22s}")
for p in ['AirQuality', 'Temperature', 'Humidity']:
    _, lG, qG, _, _ = engine_grid._sem['Satisfaction']
    _, lO, qO, _, _ = engine_obs._sem['Satisfaction']
    gG = lG.get(p, 0) + 2 * qG.get(p, 0) * 0.5
    gO = lO.get(p, 0) + 2 * qO.get(p, 0) * 0.5
    print(f"  {p:28s}  β={lG.get(p,0):+.4f} ∂|0.5={gG:+.4f}   β={lO.get(p,0):+.4f} ∂|0.5={gO:+.4f}")

# Trace first 3 timesteps side by side
np.random.seed(0)
state_grid = exp._get_initial_state({'seed': 0})
state_obs = dict(state_grid)

print(f"\nInitial: T={state_grid['Temperature']:.4f} H={state_grid['Humidity']:.4f} AQ={state_grid['AirQuality']:.4f}")
print(f"Objectives: {objectives}")

for t in range(4):
    print(f"\n--- Timestep {t} ---")

    # GRID gradients and action
    grad_g = engine_grid._compute_gradient(objectives, state_grid)
    res_g = engine_grid.optimize_policy(objectives, constraints, state_grid)
    act_g = res_g.action_plan

    # ObsOnly gradients and action
    grad_o = engine_obs._compute_gradient(objectives, state_obs)
    res_o = engine_obs.optimize_policy(objectives, constraints, state_obs)
    act_o = res_o.action_plan

    print(f"  {'':20s} {'GRID':>12s} {'ObsOnly':>12s}")
    for v in ['Temperature', 'Humidity', 'AirQuality']:
        print(f"  grad({v:14s})  {grad_g[v]:+10.4f}  {grad_o[v]:+10.4f}")
    for v in ['Temperature', 'Humidity', 'AirQuality']:
        phys_g = exp._norm_to_phys(v, act_g[v])
        phys_o = exp._norm_to_phys(v, act_o[v])
        print(f"  action({v:11s})  {act_g[v]:8.4f}({phys_g:6.1f})  {act_o[v]:8.4f}({phys_o:6.1f})")

    # Step simulator
    ns_g = exp._step_simulator(state_grid, act_g, {'seed': 0}, t)
    ns_o = exp._step_simulator(state_obs, act_o, {'seed': 0}, t)
    sat_g = ns_g.get('Satisfaction', ns_g.get('OverallSatisfaction', 0))
    sat_o = ns_o.get('Satisfaction', ns_o.get('OverallSatisfaction', 0))
    eng_g = ns_g.get('EnergyConsumption', 0)
    eng_o = ns_o.get('EnergyConsumption', 0)
    print(f"  sat                    {sat_g:8.1f}%      {sat_o:8.1f}%")
    print(f"  eng                    {eng_g:8.1f}%      {eng_o:8.1f}%")
    print(f"  next T/H/AQ         {ns_g['Temperature']:.3f}/{ns_g['Humidity']:.3f}/{ns_g['AirQuality']:.3f}"
          f"   {ns_o['Temperature']:.3f}/{ns_o['Humidity']:.3f}/{ns_o['AirQuality']:.3f}")

    state_grid = ns_g
    state_obs = ns_o
