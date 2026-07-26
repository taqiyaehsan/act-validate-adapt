"""Step 6 — Policy gradient-sign test (sbr only).

Defensive sanity check: does the CausalPolicyEngine, built on the validated
graph, move actuators in physically-correct directions?

The engine's single-objective gradient is ∂score/∂a with the objective sign
already applied (satisfaction → maximize, +; energy → minimize, −). So:

  * energy-only objective  → gradient should be NEGATIVE for both actuators
    (more HVAC/lighting raises energy, so to minimize energy the policy turns
    them down). HVAC→Energy and Lighting→Energy are direct validated edges, so
    this sign is physically unambiguous → asserted PASS/FAIL.
  * satisfaction-only      → reported descriptively. Actuators affect comfort
    mostly INDIRECTLY (HVAC→Temperature→Satisfaction), so the *direct* SEM
    gradient may be ~0 unless an actuator→satisfaction edge survived validation.

We evaluate across many real states (sampled from the training data) and across
several sampling seeds, and report sign-consistency. We run the same test on the
obs-only graph for contrast — spurious actuator edges there can flip signs or
add noise, which is exactly the failure the validated graph guards against.

Output: results/gradient_sign/gradient_sign.csv  (+ printed summary)
"""
import sys, os, json, argparse, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd

from src.policy_engine import CausalPolicyEngine

DATA_PATH = 'data_regen/smart_building_rich_processed.csv'
EDGE_SETS = 'results/closedloop_server/edge_sets.json'
ACTION_VARS = ['HVACPower', 'LightingPower']
OUT_DIR = 'results/gradient_sign'

# Physical priors for the *direct* actuator→outcome gradient sign, expressed as
# the sign of the single-objective ∂score/∂a the engine returns.
#   energy objective: score = -energy, ∂energy/∂actuator > 0  → expect grad < 0
EXPECTED = {
    ('energy', 'HVACPower'):     'neg',
    ('energy', 'LightingPower'): 'neg',
}

parser = argparse.ArgumentParser()
parser.add_argument('--n-states', type=int, default=300,
                    help='states sampled per seed')
parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 999])
parser.add_argument('--edge-sets', nargs='+', default=['validated', 'obs_only'],
                    help='which edge sets from edge_sets.json to test')
parser.add_argument('--smoke', action='store_true')
args = parser.parse_args()

if args.smoke:
    args.n_states = 25
    args.seeds = [42]
    args.edge_sets = ['validated']

data = pd.read_csv(DATA_PATH)
cols = [c for c in data.columns if c.lower() not in {'timestamp', 'elapsedtime'}]
data = data[cols]
edge_sets = json.load(open(EDGE_SETS))


def grad_for_objective(engine, state, objective):
    """Return {actuator: ∂score/∂a} for a single-objective spec."""
    return engine._compute_gradient({objective: {'weight': 1.0}}, state)


def sign_label(x, tol=1e-9):
    if x > tol:
        return 'pos'
    if x < -tol:
        return 'neg'
    return 'zero'


rows = []
print(f"\nGradient-sign test — {args.n_states} states × {len(args.seeds)} seeds "
      f"per edge set\n" + "=" * 70)

for es_name in args.edge_sets:
    if es_name not in edge_sets:
        print(f"[skip] '{es_name}' not in {EDGE_SETS}")
        continue
    edges = {tuple(e) for e in edge_sets[es_name]}
    engine = CausalPolicyEngine(
        {'validated_edges': edges}, data,
        use_llm=False, action_vars=ACTION_VARS,
        reg_lambda=0.1, n_gradient_steps=5,
    )

    for objective in ['energy', 'satisfaction']:
        # accumulate per-actuator gradient samples across seeds/states
        samples = {a: [] for a in ACTION_VARS}
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(data), size=min(args.n_states, len(data)),
                             replace=False)
            for i in idx:
                state = data.iloc[int(i)].to_dict()
                g = grad_for_objective(engine, state, objective)
                for a in ACTION_VARS:
                    samples[a].append(float(g.get(a, 0.0)))

        for a in ACTION_VARS:
            arr = np.array(samples[a])
            mean_g = float(arr.mean())
            # dominant sign + consistency among non-zero samples
            nz = arr[np.abs(arr) > 1e-9]
            if len(nz) == 0:
                dom, consist = 'zero', 1.0
            else:
                pos_frac = float((nz > 0).mean())
                dom = 'pos' if pos_frac >= 0.5 else 'neg'
                consist = max(pos_frac, 1 - pos_frac)
            exp = EXPECTED.get((objective, a))
            verdict = ''
            if exp is not None:
                if dom == 'zero':
                    # actuator is not a direct parent of this outcome in the
                    # graph — structural fact, not a sign error
                    verdict = 'no-direct-edge'
                elif dom == exp:
                    # correct sign; flag if consistency is marginal (typically a
                    # near-zero coefficient whose state-dependent sign wobbles)
                    verdict = 'PASS' if consist >= 0.95 else 'PASS-weak'
                else:
                    # dominant sign is the OPPOSITE of the physical prior
                    verdict = 'SIGN-VIOLATION'
            rows.append({
                'edge_set': es_name, 'objective': objective, 'actuator': a,
                'mean_grad': round(mean_g, 6), 'dominant_sign': dom,
                'sign_consistency': round(consist, 4),
                'expected_sign': exp or '', 'verdict': verdict,
                'n_samples': len(arr),
            })
            tag = f"  [{verdict}]" if verdict else ""
            print(f"{es_name:<10} {objective:<13} {a:<14} "
                  f"mean={mean_g:+.5f}  sign={dom:<4} "
                  f"consistency={consist*100:4.1f}%{tag}")
    print("-" * 70)

os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR, 'gradient_sign.csv')
pd.DataFrame(rows).to_csv(out, index=False)
print(f"\nWrote {out}")

fails = [r for r in rows if r['verdict'] == 'SIGN-VIOLATION']
weak = [r for r in rows if r['verdict'] == 'PASS-weak']
absent = [r for r in rows if r['verdict'] == 'no-direct-edge']
if fails:
    print(f"\n⚠️  {len(fails)} TRUE sign violation(s) (opposite of physical prior):")
    for r in fails:
        print(f"    {r['edge_set']} / {r['objective']} / {r['actuator']}: "
              f"got {r['dominant_sign']} (expected {r['expected_sign']}), "
              f"consistency {r['sign_consistency']*100:.1f}%")
else:
    print("\n✅ No sign violations — every direct actuator→outcome gradient that "
          "exists has the physically-correct sign.")
if weak:
    print(f"\nℹ️  {len(weak)} correct-sign pair(s) with marginal consistency "
          "(<95%, near-zero coefficient):")
    for r in weak:
        print(f"    {r['edge_set']} / {r['objective']} / {r['actuator']}: "
              f"{r['dominant_sign']} at {r['sign_consistency']*100:.1f}% "
              f"(mean {r['mean_grad']:+.5f})")
if absent:
    print(f"\nℹ️  {len(absent)} actuator→outcome pair(s) have no direct edge "
          "(zero gradient) — reported for graph-coverage comparison:")
    for r in absent:
        print(f"    {r['edge_set']} / {r['objective']} / {r['actuator']}")
