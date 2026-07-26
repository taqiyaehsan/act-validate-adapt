#!/usr/bin/env python3
"""
Quick test: re-run Phase 3 (SEM fitting) + Phase 4 (monitoring) only,
using previously discovered edges. Validates the AR(1) + one-step-ahead fix.
"""
import os, sys, json, time, logging
import numpy as np
import pandas as pd
import networkx as nx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("test_monitoring_fix.log", mode='w'),
                              logging.StreamHandler()])
for _q in ['matplotlib', 'PIL', 'urllib3']:
    logging.getLogger(_q).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    from src.pipeline_cwm import create_cwm_pipeline, plot_calibration_analysis
    from src.causal_world_model import PredictiveModel

    output_dir = os.path.join('neurips_results', 'monitoring_fix_test_'
                              + time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data_path = 'data/challenge1_data_10k_processed.csv'
    sim_path = os.path.abspath('open_window.js')
    dataset = pd.read_csv(data_path)
    logger.info(f"Loaded {len(dataset)} rows")

    # Load previously discovered edges
    prev_results = 'neurips_results/discovery_monitoring_20260306_185604/discovered_graphs.json'
    with open(prev_results) as f:
        graphs = json.load(f)

    h0_edges = [tuple(e) for e in graphs['h0_edges']]
    h1_edges = [tuple(e) for e in graphs['h1_edges']]
    validated_edges = set(tuple(e) for e in graphs['validated_edges'])
    logger.info(f"Loaded edges: H0={len(h0_edges)}, H1={len(h1_edges)}, validated={len(validated_edges)}")

    # Create pipeline (will initialize but we skip discovery)
    api_key = os.environ.get('OPENAI_API_KEY', '')
    pipeline = create_cwm_pipeline(
        csv_data=dataset, api_key=api_key,
        smart_room_path=sim_path, dataset_type='open_window',
        max_iterations=1, alpha=0.5, beta=0.5, effect_threshold=0.1,
    )
    pipeline.has_regime = True

    # Inject discovered edges (skip Phase 1-2)
    pipeline.pipeline.validated_edges = validated_edges

    # Build H0/H1 graphs
    G_H0 = nx.DiGraph()
    for e in h0_edges:
        G_H0.add_edge(*e)
    G_H1 = nx.DiGraph()
    for e in h1_edges:
        G_H1.add_edge(*e)

    pipeline.H0_graph = G_H0
    pipeline.H1_graph = G_H1

    # Initialize CWM (normally done in Phase 1)
    all_edges = list(set(h0_edges + h1_edges))
    # CausalWorldModel expects method_dags as {name: {'edges': [...]}}
    method_dags = {name: {'edges': [tuple(e) for e in edges]}
                   for name, edges in graphs['method_dags'].items()}
    pipeline._initialize_cwm(all_edges, method_dags)

    variables = pipeline.cwm.variables
    logger.info(f"Variables: {sorted(variables)}")

    # Phase 3: Train SEMs with AR(1)
    logger.info("\n=== Phase 3: Training H0/H1 SEMs (with AR(1) self-lag) ===")
    pipeline.H0_model = PredictiveModel(G_H0, variables)
    pipeline.H1_model = PredictiveModel(G_H1, variables)

    temporal_df_closed = pipeline._prepare_temporal_data(regime_filter='closed')
    temporal_df_all = pipeline._prepare_temporal_data(regime_filter=None)
    logger.info(f"Temporal data: closed={len(temporal_df_closed)}, all={len(temporal_df_all)}")

    n_closed = len(temporal_df_closed)
    n_train = max(int(0.8 * n_closed), min(n_closed, 10))
    temporal_df_closed_train = temporal_df_closed.iloc[:n_train]
    temporal_df_closed_val = temporal_df_closed.iloc[n_train:]

    pipeline.H0_model.fit_temporal(temporal_df_closed_train)
    pipeline.H1_model.fit_temporal(temporal_df_all)

    # Sigma calibration
    SIGMA_MAX = 30.0
    _h0_val_errs = []
    for _, row in temporal_df_closed_val.iterrows():
        row_dict = {k: v for k, v in dict(row).items() if not k.endswith('_next')}
        pred = pipeline.H0_model.predict(row_dict)
        t_next = row.get('temperature_next', row_dict.get('temperature', 0))
        err = abs(pred.get('temperature', 0) - t_next)
        _h0_val_errs.append(err)
    _mean_err = float(np.mean(_h0_val_errs)) if _h0_val_errs else 0.1
    pipeline.cwm.likelihood_sigma = min(1.0 / max(_mean_err, 1e-3), SIGMA_MAX)
    logger.info(f"Sigma: mean_val_err={_mean_err:.4f}, sigma={pipeline.cwm.likelihood_sigma:.2f}")

    # Log SEM parameters
    h0_temp = pipeline.H0_model.parameters.get('temperature', {})
    h1_temp = pipeline.H1_model.parameters.get('temperature', {})
    logger.info(f"H0 temp: self_lag={h0_temp.get('self_lag', 'N/A')}, "
                f"intercept={h0_temp.get('intercept', 'N/A')}, "
                f"weights={h0_temp.get('weights', {})}")
    logger.info(f"H1 temp: self_lag={h1_temp.get('self_lag', 'N/A')}, "
                f"intercept={h1_temp.get('intercept', 'N/A')}, "
                f"weights={h1_temp.get('weights', {})}")

    # Quick sanity check: predict on training data
    sample = temporal_df_closed.iloc[0]
    sample_dict = {k: v for k, v in dict(sample).items() if not k.endswith('_next')}
    pred_h0 = pipeline.H0_model.predict(sample_dict)
    logger.info(f"Sanity: T_current={sample_dict.get('temperature', '?'):.4f}, "
                f"T_next_actual={sample.get('temperature_next', '?'):.4f}, "
                f"T_pred_H0={pred_h0.get('temperature', '?'):.4f}")

    # Phase 4: Monitoring (shorter for testing: 10 min instead of 30)
    MONITORING_DURATION = 600  # 10 minutes
    logger.info(f"\n=== Phase 4: Monitoring ({MONITORING_DURATION//60} min) ===")

    # Initialize CWM hypotheses for the pool
    from src.causal_world_model import GraphHypothesis
    pipeline.cwm.hypotheses = []
    for method_name, method_edges in graphs['method_dags'].items():
        G_method = nx.DiGraph()
        for e in method_edges:
            G_method.add_edge(*e)
        pm = PredictiveModel(G_method, variables)
        pm.fit_temporal(temporal_df_all)
        hyp = GraphHypothesis(name=method_name, graph=G_method,
                              predictive_model=pm, weight=1.0 / len(graphs['method_dags']))
        hyp.predictive_model._parent_cwm = pipeline.cwm
        pipeline.cwm.hypotheses.append(hyp)

    pipeline.P_H0 = 0.5
    pipeline.P_H1 = 0.5

    obs_df = pipeline.monitor_regime_changes(duration=MONITORING_DURATION, sample_interval=5)
    logger.info(f"Monitoring: {len(obs_df)} observations")

    # Save results
    obs_path = os.path.join(output_dir, 'monitoring_observations.csv')
    obs_df.to_csv(obs_path, index=False)

    # Generate plots
    from run_discovery_monitoring import generate_monitoring_plots
    generate_monitoring_plots(obs_df, output_dir)

    cal_path = os.path.join(output_dir, 'monitoring_calibration.png')
    plot_calibration_analysis(obs_df, cal_path)

    # Quick analysis
    if 'actual_window' in obs_df.columns and 'err_H0' in obs_df.columns:
        closed = obs_df[obs_df['actual_window'] == 0]
        opened = obs_df[obs_df['actual_window'] == 1]
        logger.info(f"\n=== RESULTS ===")
        logger.info(f"Closed window: err_H0={closed['err_H0'].mean():.4f}, err_H1={closed['err_H1'].mean():.4f}")
        if len(opened) > 0:
            logger.info(f"Open window:   err_H0={opened['err_H0'].mean():.4f}, err_H1={opened['err_H1'].mean():.4f}")
        logger.info(f"Mean P(H0)={obs_df['P_H0_binary'].mean():.3f}, P(H1)={obs_df['P_H1_binary'].mean():.3f}")

    logger.info(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()
