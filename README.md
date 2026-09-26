# Act, Validate, Adapt: Closing the Causal Discovery–Control Loop

Code and results for our NeurIPS 2026 paper.

Taqiya Ehsan, Shuren Xia, Jorge Ortiz (Rutgers University)

The framework (called **PolicyGRID** in the paper) treats control actions as experiments: it generates candidate causal edges from observational data (PC + SAM + LLM + VARLiNGAM), validates each edge through do-operator interventions in the environment, rescues non-intervenable edges via back-door-adjusted partial correlation, monitors latent regime changes with factored Bayesian monitors (the monitor distinguishes among known coefficient regimes on the fixed validated graph; it does not detect structural/topology change), and drives multi-objective Pareto control from the validated structural equation model. The same pipeline is deployed on a physical sensor/actuator testbed.

| Paper variant | Graph | Policy |
|---|---|---|
| PolicyGRID-O | Observational union of the 4 generators (no validation) | see note |
| PolicyGRID | Intervention-validated | see note |

*Policy note: two evaluation configurations are used. The open-loop Pareto sweep (`run_full_pipeline.py --stages policy`) fits one static engine per method. The closed-loop evaluation (Table 1; `runners/`) runs monitor-in-the-loop: at each step a Bayesian regime posterior selects among four per-regime coefficient sets fitted on the same graph, with that machinery identical across compared rows.*

This README covers: [installation](#1-installation), [artifact-based replay of the paper's results](#3-reproducing-the-papers-results) (bit-exact for phase-1 discovery and monitoring-given-graph; near-exact for validated graphs — see §6), [additional analyses](#4-additional-analyses), and the [repository layout](#5-repository-layout).

---

## 1. Installation

Requirements: **Python 3.12+**, **Node.js 18+** (the simulators are JavaScript), ~4 GB disk.

```bash
git clone https://github.com/taqiyaehsan/act-validate-adapt.git
cd act-validate-adapt
python3 -m venv causal_env && source causal_env/bin/activate
pip install -r requirements.txt
npm install
```

**OpenAI API key** (optional): replace the `YOUR_OPENAI_API_KEY` placeholder in `run_full_pipeline.py`
(module-level `API_KEY`) with your key. The LLM is used in two places — hypothesis generation, and the
third (domain-aware intermediate) intervention probe during validation; the first is served from the
shipped response cache when present (`results/llm_edge_cache/`, see [§4](#4-additional-analyses)),
and the second falls back to a deterministic midpoint intervention when no key is set, so the full
pipeline runs API-free. Model and decoding: `gpt-3.5-turbo`; edge proposals are called at
`temperature=1.0`, `top_p=0.8` (`src/generators.py`), and intervention-design calls use the
client defaults (`top_p=0.9`, max 1000 tokens) in `src/gpt_client.py`.

Sanity-check the environment before any long run:

```bash
python preflight_check.py   # 68 diagnostic checks (simulators, data, dependencies)
```

One known warning: preflight flags `data/ashrae_data_processed_scaling_params.csv` as missing. That file is read only by the monitoring/closed-loop stages, which do not apply to the ASHRAE environment (discovery-only, no actuator); the ASHRAE replication commands below run without it.

---

## 2. Environments

Six simulated environments plus one physical testbed. Ground-truth DAGs for all simulators are in `ground_truth_graphs.json`.

| Environment | Variables | GT edges | Hidden regime variables | Regime monitoring |
|---|---|---|---|---|
| `smart_building_rich` (**primary**) | 15 | 29 | Occupancy, WindowPosition | Yes |
| `open_window` | 8 | 10 | WindowOpen | Yes |
| `smart_room` | 5 | 6 | — | No |
| `smart_room_noise` | 5 | 6 | — | No |
| `smart_room_hidden_vars` | 5 | 6 | — | No |
| `ashrae` | 6 | 7 | — | No |
| Physical testbed | see `smart_room_physical/` | — | — | Yes |

The exact training datasets used for the reported runs (10k rows per environment, with their scaling parameters) ship in `data_regen/` and `data/`; `generate_varied_data.py` regenerates them from the JS simulators if desired. Monitoring and policy are always evaluated on **fresh simulator episodes**, never on training data.

---

## 3. Reproducing the paper's results

The single entry point is `run_full_pipeline.py`. One invocation runs, per seed: discovery (4 generators → union → interventional validation → rescue → pruning) → 9 discovery benchmarks → regime monitoring → the policy Pareto sweep, and writes everything under `results/full_pipeline/<sim>/`.

The paper's headline seeds are **42, 123, 456**.

```bash
# Primary environment, full pipeline, paper seeds
python run_full_pipeline.py --sim smart_building_rich --seed-list 42,123,456

# All six simulators
python run_full_pipeline.py --sim all --seed-list 42,123,456

# Individual stages (discovery: all methods; benchmarks: PC, SAM, GIES, ICP,
# NOTEARS-I, ABCD, JCI, Causal Bandits, IID)
python run_full_pipeline.py --sim smart_building_rich --seed-list 42,123,456 --stages discovery,benchmarks

# Targeted experiments: m3a (graph attribution), m3b (intervention ordering),
# a1 (sample efficiency), a2 (regime recovery), n2 (LLM wrong-prior stress test),
# m4 (lag ablation), ablation (sensor dropout)
python run_full_pipeline.py --sim smart_building_rich --seed-list 42,123,456 --experiments m3a,m3b,a1,a2,n2
```

Per-seed outputs: `results/full_pipeline/<sim>/seed_<n>/{edges.json, monitoring_metrics.json, policy_metrics.csv}`, plus `benchmarks.csv`, `consensus_analysis.json`, and `summary.json` per environment. `edges.json` includes `phase1_method_edges` — per-generator proposal sets feeding validation. Note the field is overwritten at each discovery regeneration, so it records each generator's **final-iteration** proposals (fit on intervention-augmented data), not the initial observational-only Phase-1 pool.

| Paper result | Command | Pre-computed output in this repo |
|---|---|---|
| Discovery F1/SHD vs. 9 benchmarks (§5.1) | `run_full_pipeline.py --stages discovery,benchmarks` | `results/full_pipeline/`, `results/5seed_results/`, `results/benchmarks_clean/` |
| Regime monitoring accuracy, Brier, discriminability (§5.2) | `run_full_pipeline.py --stages monitoring` | `results/full_pipeline/<sim>/seed_*/monitoring_metrics.json` |
| Policy Pareto sweep, 5 comfort targets, 10 arms (§5.3) | `run_full_pipeline.py --stages policy` | `results/full_pipeline/<sim>/seed_*/policy_metrics.csv`, `results/policy_comparison/` |
| Closed-loop paired statistics (§5.3) | `python reproduce_paired_stats.py` | prints paired t, Cohen's d, CI from `server_runs/closedloop_server/` |
| Graph-attribution ablation (validated / obs-only / random / empty) | `--experiments m3a` | `results/m3a_attribution_server/`, `results/rowA_ablation/` |
| Ranked vs. random intervention ordering | `--experiments m3b` | per-seed `m3b_ordering/` dirs |
| LLM wrong-prior stress test | `--experiments n2` | per-seed outputs |
| Sensor-dropout ablation | `--experiments ablation` | per-seed `exp2_ablation/` dirs |
| Post-hoc analyses (10 analyses: heatmaps, calibration, confusion, stability) | `python scripts/posthoc_analysis.py` | `results/neurips_final/posthoc/` |
| Physical testbed (§5.4) | see `smart_room_physical/deployed/` | `smart_room_physical/deployed/results/` |

Gold-standard reference results used for the paper's tables are preserved unmodified in `results/neurips_final_good_result_3seeds/` and `results/5seed_results/`.

---

## 4. Additional analyses

Where to find the ablations and robustness checks that go beyond the main tables.

**Leave-one-generator-out ablation.**
Rerun with the `--generators` flag, which restricts Phase 1 to a subset of `pc,sam,llm,varlingam` and writes to a suffixed directory so primary results are never overwritten:

```bash
python run_full_pipeline.py --sim smart_building_rich --seed-list 42,123,456 \
    --generators pc,sam,varlingam --out-suffix no_llm
python scripts/c2_aggregate.py    # aggregates all configs into one summary table
```

Raw per-seed results for every configuration (`abl_full`, `no_pc`, `no_sam`, `no_llm`, `no_varlingam`, `pc_only`): `results/server_pull_jul23/smart_building_rich_*/` and `results/full_pipeline/open_window_*/`. Aggregated tables: `results/c2_analysis/`.

**Regime-aware vs. static control on the same validated graph.**
Isolates regime inference from graph quality: all four conditions (RegimeAware, Static, RandomSwitch, Oracle) share the identical validated edge set, engine hyperparameters, objectives, and — via common random numbers — identical exogenous simulator disturbances (a per-(seed, timestep) simulator seed is passed to every condition; see `step_simulator` in the runner).

```bash
# reproduce the released grid layout (one directory per seed, as shipped):
for s in 42 123 456; do
  python runners/run_regime_vs_static_policy.py --seeds $s --out results/regime_vs_static_crn/seed$s
done
python scripts/regime_paired_stats.py                    # paired statistics over the seed dirs
python runners/run_regime_vs_static_policy.py --smoke    # 1-seed quick check (writes to the default results/regime_vs_static/)
```

Full-grid per-episode traces, per-condition results, and paired statistics: `results/regime_vs_static_crn/` (the runner's default output directory `regime_vs_static/` holds only a smoke-test run — pass `--out` as above to reproduce the released grid).

**Classifier baseline for regime detection.**
Logistic regression and a small decision tree against the factored Bayesian monitor, with identical labels and evaluation episodes, in two feature configurations — all 13 observables, and feature-matched to the monitor's six sensor variables:

```bash
python run_classifier_baseline.py --seeds 42,123,456                 # 13 observables (26 dims)
python run_classifier_baseline.py --seeds 42,123,456 --sensor-only   # monitor's 6 sensors (12 dims)
```

Per-seed outputs and summaries: `results/classifier_baseline/` (13-obs) and `results/classifier_baseline_sensor6/` (feature-matched).

**LLM replay artifacts** (determinism and reproducibility of the LLM generator).
The complete prompt templates, including the domain-context system prompts, are code: `src/generators.py` (`DOMAIN_CONTEXT`, `LLMGenerator`). Model: `gpt-3.5-turbo`, `temperature=1.0` throughout; decoding differs by call site — edge proposals pass `top_p=0.8` explicitly (`src/generators.py`), while intervention-design calls use the client defaults (`top_p=0.9`, `max_tokens=1000`) in `src/gpt_client.py`. Verbatim raw LLM responses are provided at two levels: `results/llm_edge_cache/` (cached responses for `smart_room` and `smart_room_hidden_vars`) and `results/llm_raw_responses/` — the unparsed API responses for `smart_building_rich` and `open_window`, one JSONL record per (configuration, seed) across the generator-ablation grids, extracted verbatim from the run logs of those runs. For those grids, `edges.json` additionally stores the parsed per-generator candidate sets consumed by the pipeline (`phase1_method_edges`), together with the pre-validation union and the validated set. Dropping any raw response into `results/llm_edge_cache/<env>.json` reruns the full pipeline API-free (see `src/generators.py`); a ready-made cache file for `open_window` is included. Note that the validation stage itself uses an LLM-designed intervention as one of its probes (deterministic fallback without a key; see §1), so graph-level replay is near-exact rather than bit-exact — quantified in §6 — while everything downstream of a validated graph is fully deterministic. Runs predating the ablation grids store the validated edge set per seed rather than the raw candidate sets; their monitoring and policy results are re-derivable exactly from those stored graphs, and rerunning discovery end-to-end with fresh LLM draws reproduces the paper's discovery numbers within seed variance (the `abl_full` grid configuration is exactly this rerun). The wrong-prior stress test (`--experiments n2`) quantifies how interventional validation prunes deliberately injected bad LLM edges.

**Physical validation gate, as deployed.**
The gate that ran on hardware is `smart_room_physical/deployed/run_policygrid_discovery.py`: 3 screening trials with a per-variable effect-size screen, then 4 confirmatory trials and a one-sample t-test on all 7 deltas at α = 0.10, followed by BH-FDR (α = 0.10) across edges when ≥10 reach the t-test — with no sign-consistency gate (this differs from the simulation gate; see the operating-characteristic analysis below). The realized run's per-edge evidence, including the FDR marks and the separately-labeled observational-rescue edges, is in `smart_room_physical/deployed/results/discovery/seed_42/edge_evidence.json`. The full deployed code path — drivers, calibration, data collection, discovery, monitoring, and closed-loop runs, together with the collected sensor data and results — is `smart_room_physical/deployed/`.

```bash
python scripts/physical_gate_oc.py   # Type-I / power of the deployed gate's per-edge stage (pre-FDR, conservative)
```

**Paired closed-loop statistics** (significance of the validated vs. obs-only gap).

```bash
python reproduce_paired_stats.py
# n = 15 paired (seed, weather) conditions: paired t = 77.88, df = 14, Cohen's d = 20.11
```

**Per-generator candidate sets and per-seed variability** (VARLiNGAM behavior; paired per-seed reporting).
`results/server_pull_jul23/smart_building_rich_*/seed_*/edges.json` (`phase1_method_edges`) records what each generator contributed in each seed of each configuration; `scripts/c2_aggregate.py` computes the per-generator unique-contribution and coverage statistics from these files.

---

## 5. Repository layout

```
act-validate-adapt/
├── run_full_pipeline.py           # Primary entry point (discovery → benchmarks → monitoring → policy)
├── run_classifier_baseline.py     # Classifier regime-detection baseline
├── reproduce_paired_stats.py      # §5.3 paired statistics
├── preflight_check.py             # 68 pre-run diagnostics
├── generate_varied_data.py        # Training-data generator
├── ground_truth_graphs.json       # GT DAGs, all simulators
├── regime_aware_validation.py     # WindowStatePredictor (imported by src/)
│
├── src/                           # Core framework
│   ├── pipeline.py                #   Base pipeline (PC, SAM, LLM, VARLiNGAM)
│   ├── pipeline_cwm.py            #   Validation, rescue, pruning, monitoring
│   ├── causal_world_model.py      #   Edge confidence + predictive models
│   ├── generators.py              #   Generators + full LLM prompts
│   ├── gpt_client.py              #   Pinned LLM model/decoding + token tracking
│   ├── tester.py                  #   Interventional hypothesis testing
│   ├── policy_engine.py           #   Ridge-SEM gradient policy engine
│   ├── ddpc_baseline.py           #   DDPC baselines   ├── mpc_baseline.py   ├── cmbpo_baseline.py
│   └── metrics.py                 #   SHD / F1 / precision / recall
│
├── js/                            # Simulators (smart_building_rich.js is primary)
├── benchmarks_new/                # 9 benchmark discovery methods
├── runners/                       # Standalone experiment runners (regime-vs-static, ablations, attribution)
├── scripts/                       # Analysis: posthoc_analysis.py, c2_aggregate.py, physical_gate_oc.py
├── experiments/                   # Historical one-off experiments and notebooks
├── tests/                         # Test suite
│
├── smart_room_physical/           # Deployed testbed code
│   └── deployed/                #   Deployed drivers, calibration, data, results
│
├── data/ · data_regen/            # Datasets + scaling parameters
├── server_runs/closedloop_server/ # Closed-loop CSVs consumed by reproduce_paired_stats.py
└── results/                       # Pre-computed results (see §3 and §4 tables)
```

---

## 6. Reproducibility notes

- **Determinism, measured.** We verified the replay path end-to-end from a clean checkout of this repository, API-free, against a shipped run (`open_window` `no_sam`, seed 42). Phase 1 discovery is bit-exact (identical per-method candidate sets and scores). The validated graph reproduces 9 of 10 edges with no spurious additions; the single divergence is the one edge whose confirmatory interventions were LLM-designed in the original run and fall back to a deterministic probe without an API key (the LLM probes are stochastic at temperature 1.0, so validation is near-exact across any rerun — the shipped per-seed validated graphs are the exact artifacts of record). Monitoring rerun on the identical shipped graph (`--skip-discovery`) matches every reported metric exactly.
- **Compute.** A full `smart_building_rich` seed (discovery + benchmarks + monitoring + policy) takes several hours on a laptop-class CPU; SAM is the dominant cost. `--stages` and `--experiments` subset the work. No GPU is required.
- **Ground truth is never used for online decisions.** GT metrics are logged for evaluation only and GT-based early stopping is disabled. Regime labels (occupancy/window thresholds) are used at **commissioning** — they partition the training data used to fit the four regime-specific predictive models — and in evaluation metrics, but never in the online Bayesian updates, which see only prediction errors.
- **Hardware credentials** in the physical code are placeholders (`YOUR_KASA_PASSWORD`, `YOUR_GOVEE_API_KEY`); the physical scripts require the corresponding devices and cannot run in simulation, but all collected data and results are included.

---

## License

MIT. See [LICENSE](LICENSE).
