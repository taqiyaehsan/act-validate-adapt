"""Build cross-environment edge_sets.json for Step 2c closedloop.

Reads per-seed edges.json from results/full_pipeline/<sim>/seed_<n>/ (which
must include the obs_only_union_edges key from the June 1 patch), unions
both validated and obs-only sets across all seeds, samples a size-matched
random edge set, and writes results/closedloop_xenv/<sim>/edge_sets.json
in the same format run_closedloop_server.py:99 consumes.

Usage:
    python scripts/build_xenv_edge_sets.py --sim smart_room
    python scripts/build_xenv_edge_sets.py --sim smart_room smart_room_hidden_vars
"""
import argparse
import json
import os
import random
import sys
from itertools import permutations

# Variable lists per sim, mirroring run_full_pipeline.py:state_vars (preserve casing).
SIM_STATE_VARS = {
    'smart_room': ['Temperature', 'Humidity', 'AirQuality',
                   'EnergyConsumption', 'Satisfaction'],
    'smart_room_hidden_vars': ['Temperature', 'Humidity', 'AirQuality',
                               'EnergyConsumption', 'Satisfaction'],
}

DEFAULT_SEEDS = [42, 123, 456, 789, 999]
RANDOM_SET_SEED = 0


def build_one_sim(sim, seeds, out_root, repo_root):
    seed_dir_root = os.path.join(repo_root, 'results', 'full_pipeline', sim)
    if not os.path.isdir(seed_dir_root):
        print(f"[{sim}] FAIL: no directory at {seed_dir_root}")
        return None

    validated_union = set()
    obs_union = set()
    missing = []
    for seed in seeds:
        edges_path = os.path.join(seed_dir_root, f'seed_{seed}', 'edges.json')
        if not os.path.exists(edges_path):
            missing.append(seed)
            continue
        with open(edges_path) as f:
            data = json.load(f)
        # Normalize edge casing: validated_edges carry mixed-case names
        # (e.g. 'AirQuality') from intervention testing while
        # obs_only_union_edges are lowercase. Lowercasing both prevents
        # duplicate edges and keeps SEM variable lookups consistent.
        for e in data.get('validated_edges', []):
            validated_union.add((e[0].lower(), e[1].lower()))
        # The June 1 patch persists obs_only_union_edges; fall back to [] for
        # pre-patch files (those won't carry the cross-env story).
        for e in data.get('obs_only_union_edges', []):
            obs_union.add((e[0].lower(), e[1].lower()))

    if missing:
        print(f"[{sim}] WARN: edges.json missing for seeds {missing}")

    if not obs_union:
        print(f"[{sim}] FAIL: obs_only_union_edges empty across all seeds — "
              f"either the patch didn't write or the runs predate the patch.")
        return None

    state_vars = SIM_STATE_VARS[sim]
    all_pairs = list(permutations(state_vars, 2))
    rng = random.Random(RANDOM_SET_SEED)
    n_random = len(validated_union)
    pool = [p for p in all_pairs if p not in validated_union]
    random_edges = set(rng.sample(pool, min(n_random, len(pool))))

    edge_sets = {
        'validated': sorted([list(e) for e in validated_union]),
        'obs_only':  sorted([list(e) for e in obs_union]),
        'random':    sorted([list(e) for e in random_edges]),
    }

    out_dir = os.path.join(out_root, sim)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'edge_sets.json')
    with open(out_path, 'w') as f:
        json.dump(edge_sets, f, indent=2)

    print(f"[{sim}] wrote {out_path}: "
          f"validated={len(validated_union)} "
          f"obs_only={len(obs_union)} "
          f"random={len(random_edges)}")
    return edge_sets


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--sim', nargs='+', required=True,
                   choices=list(SIM_STATE_VARS.keys()))
    p.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    p.add_argument('--out-root', default='results/closedloop_xenv')
    args = p.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    out_root = os.path.join(repo_root, args.out_root)

    n_ok = 0
    for sim in args.sim:
        if build_one_sim(sim, args.seeds, out_root, repo_root) is not None:
            n_ok += 1

    if n_ok != len(args.sim):
        sys.exit(1)


if __name__ == '__main__':
    main()
