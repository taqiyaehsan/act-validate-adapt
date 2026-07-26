# main.py
import os
import pandas as pd
import numpy as np
from src.pipeline import CausalPipeline
from src.metrics_viz import MetricsVisualizer, export_metrics_to_csv
from energyplus_interface import EnergyPlusInterface
from src.evaluator import EdgeRanker
import matplotlib.pyplot as plt
import networkx as nx
import warnings
import logging
import atexit
import json 

from src.policy_engine import CausalPolicyEngine
# from policy_integration import SmartRoomSimulator, run_policy_experiments
from experiments_neurips import NeurIPSExperiments
from pareto_integration import EnhancedNeurIPSExperiments as NeurIPSExperiments

warnings.simplefilter('ignore')

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

package_name = "machine_causation"  
for name in logging.root.manager.loggerDict:
    if not name.startswith(package_name):
        logging.getLogger(name).setLevel(logging.INFO)

eplus = EnergyPlusInterface(
    idf_template_path="templates/smart_room.idf",
    weather_file="weather/USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw"
)

def log_edge_metrics(pipeline):
    """Log detailed edge testing metrics"""
    logger.info("\n=== Edge Testing Metrics ===")
    
    for edge, confidences in pipeline.tester.edge_history.items():
        mean_conf = sum(confidences) / len(confidences) if confidences else 0
        logger.info(f"\nEdge: {edge[0]} -> {edge[1]}")
        logger.info(f"Mean confidence: {mean_conf:.3f}")
        logger.info(f"Number of tests: {len(confidences)}")
        logger.info(f"Agreement level: {pipeline.edge_ranker.edge_confidence.get(edge, 0):.2f}")

def log_method_comparison(pipeline):
    logger.info("\n=== Method Comparison ===")
    
    for method, result in pipeline._results.items():
        edges = set(map(tuple, result['edges']))
        method_support = pipeline.edge_ranker.edge_confidence
        metrics = pipeline.metrics_calculator.calculate_metrics(
            method,
            edges,
            method_support,
            pipeline.tester.intervention_results
        )
        
        logger.info(f"\n{method.upper()}:")
        logger.info(f"Cost (Wrong Edge Penalty): {metrics.cost:.4f}")
        logger.info(f"Risk (Confidence * Cost): {metrics.risk:.4f}")
        logger.info(f"Performance (F1): {metrics.f1_score:.4f}")
        logger.info(f"Wrong Edges: {len(edges - set(pipeline.ground_truth.edges))}")

def log_detailed_metrics(metrics, iteration=None, logger=logging.getLogger(__name__)):
    """Log comprehensive metrics with optional iteration tracking"""
    prefix = f"Iteration {iteration}: " if iteration is not None else ""
    
    for method, metric_values in metrics.items():
        logger.info(f"\n{prefix}Method: {method.upper()}")
        logger.info(f"{'='*50}")
        
        # Core metrics
        logger.info("Core Performance Metrics:")
        logger.info(f"Precision: {metric_values.precision:.4f}")
        logger.info(f"Recall: {metric_values.recall:.4f}")
        logger.info(f"F1 Score: {metric_values.f1_score:.4f}")
        
        # Cost and efficiency metrics
        logger.info("\nEfficiency Metrics:")
        logger.info(f"Cost: {metric_values.cost:.4f}")
        logger.info(f"Risk: {metric_values.risk:.4f}")
        
        # Structural metrics
        logger.info("\nStructural Metrics:")
        logger.info(f"SHD (Structural Hamming Distance): {metric_values.shd}")
        
        logger.info(f"{'='*50}\n")

def log_shd_history(shd_history, logger=logging.getLogger(__name__)):
    """Log SHD trends for each method"""
    logger.info("\nStructural Hamming Distance (SHD) History:")
    logger.info(f"{'='*50}")
    
    for method, values in shd_history.items():
        logger.info(f"\nMethod: {method.upper()}")
        for iteration, shd in enumerate(values, 1):
            logger.info(f"Iteration {iteration}: SHD = {shd}")
    
    logger.info(f"{'='*50}\n")

def log_edge_statistics(edge_stats, logger=logging.getLogger(__name__)):
    """Log detailed edge statistics"""
    logger.info("\nEdge Statistics:")
    logger.info(f"{'='*50}")
    
    for edge, stats in edge_stats.items():
        logger.info(f"\nEdge: {edge[0]} → {edge[1]}")
        for metric, value in stats.items():
            if isinstance(value, (float, np.floating)):
                logger.info(f"{metric}: {value:.4f}")
            else:
                logger.info(f"{metric}: {value}")
    
    logger.info(f"{'='*50}\n")

def export_detailed_metrics(pipeline, output_dir):
    # Create list of dictionaries for metrics
    metrics_data = []
    
    for method, result in pipeline._results.items():
        edges = set(map(tuple, result['edges']))
        wrong_edges = edges - set(pipeline.ground_truth.edges)
        
        for edge in wrong_edges:
            confidence = pipeline.edge_ranker.edge_confidence.get(edge, 0)
            cost = pipeline.metrics_calculator._calculate_method_cost(method, {edge}, pipeline.tester.intervention_results)
            risk = pipeline.metrics_calculator._calculate_method_risk(method, {edge}, {edge: confidence}, pipeline.tester.intervention_results)
            
            metrics_data.append({
                'Method': method,
                'Edge': f"{edge[0]}->{edge[1]}",
                'Confidence': confidence,
                'Cost': cost,
                'Risk': risk
            })
    
    # Create DataFrame from list of dictionaries
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(os.path.join(output_dir, 'detailed_metrics.csv'), index=False)

def log_validation_progress(pipeline):
    """Log validation progress across iterations"""
    logger.info("\n=== Edge Validation Progress ===")
    
    if not hasattr(pipeline, 'validated_edges') or not pipeline.validated_edges:
        logger.info("No edges have been validated yet.")
        return
    
    validated_edges = pipeline.validated_edges
    edge_ranker = pipeline.edge_ranker
    
    logger.info(f"Total validated edges: {len(validated_edges)}")
    
    # Sort validated edges by confidence
    sorted_edges = sorted(
        validated_edges,
        key=lambda e: edge_ranker.edge_confidence.get(e, 0),
        reverse=True
    )
    
    for edge in sorted_edges:
        confidence = edge_ranker.edge_confidence.get(edge, 0)
        intervention_count = len(pipeline.tester.intervention_results.get(edge, []))
        methods_supporting = []
        
        # Check which methods include this edge
        for method, result in pipeline._results.items():
            edges = pipeline._extract_edges(result)
            if edge in edges:
                methods_supporting.append(method)
        
        logger.info(f"Edge: {edge[0]} -> {edge[1]}")
        logger.info(f"  Confidence: {confidence:.3f}")
        logger.info(f"  Interventions: {intervention_count}")
        logger.info(f"  Methods supporting: {', '.join(methods_supporting)}")
    
    logger.info(f"{'='*50}\n")

def cleanup_processes():
    """Cleanup function to ensure all processes are terminated"""
    import psutil
    current_process = psutil.Process()
    children = current_process.children(recursive=True)
    for child in children:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass

# def create_sample_robot_arm_data(output_path):
#     """Create a simple sample dataset for the robot arm"""
#     # Generate 100 sample data points
#     np.random.seed(42)
#     data = []
    
#     for i in range(100):
#         # Generate random hand and ball positions
#         hand_x = np.random.uniform(50, 350)
#         hand_y = np.random.uniform(50, 250)
#         ball_x = np.random.uniform(50, 350)
#         ball_y = np.random.uniform(50, 250)
        
#         # Determine if there's a collision
#         distance = math.sqrt((hand_x - ball_x)**2 + (hand_y - ball_y)**2)
#         collision = distance < 20
        
#         # Ball is caught if there's a collision
#         ball_caught = collision
        
#         # Add random hand velocities
#         hand_vx = np.random.uniform(-5, 5)
#         hand_vy = np.random.uniform(-5, 5)
        
#         # Add data point
#         data.append({
#             'handX': hand_x,
#             'handY': hand_y,
#             'ballX': ball_x,
#             'ballY': ball_y,
#             'handVx': hand_vx,
#             'handVy': hand_vy,
#             'collision': collision
#         })
    
#     # Create DataFrame and save to CSV
#     df = pd.DataFrame(data)
#     df.to_csv(output_path, index=False)
#     return df

def main():
    try:
        # Initialize logging
        logger.info("=== Starting Causal Discovery Pipeline ===")

        # Register cleanup
        atexit.register(cleanup_processes)

        api_key = "YOUR_OPENAI_API_KEY"
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # smart_room_path = "./smart_room.js"
        # robot_arm_path = os.path.join(current_dir, "robot_arm.js")
        # smart_room_path = "./smart_room_noise.js"
        smart_room_path = "./smart_room_hidden_vars.js"
        # smart_bldg_path = "./smart_building.js"
        
        output_dir = os.path.join(current_dir, "results")
        os.makedirs(output_dir, exist_ok=True)

        # Initialize EnergyPlus interface with absolute paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        eplus = EnergyPlusInterface(
            idf_template_path=os.path.join(current_dir, "templates", "smart_room.idf"),
            weather_file=os.path.join(current_dir, "weather", "USA_IL_Chicago-OHare.Intl.AP.725300_TMY3.epw")
        )

        # Load data
        # data_path = "data/simulation_log_2025-03-04T16-42-08-821Z_preprocessed.csv"
        # data_path = "data/smart_room_noisy_preprocessed.csv"
        # data_path = "data/smart-room-hidden-vars_preprocessed.csv"
        # data_path = "data/building_simulation_2025-06-09T01-07-57-790Z_processed.csv"
        data_path = "data/ashrae_data_processed.csv"
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            raise FileNotFoundError(f"Data file not found: {data_path}")
        # data_path = "data/robot_arm_dag_log_preprocessed.csv"
        # if not os.path.exists(data_path):
        #     logger.error(f"Data file not found: {data_path}")
        #     # If no data file exists, create a simple simulation data file
        #     create_sample_robot_arm_data(data_path)
        #     logger.info(f"Created sample robot arm data at {data_path}")
            
        df = pd.read_csv(data_path)
        
        # Initialize and run pipeline with max_iterations
        # New parameter allows early termination when SHD=0
        pipeline = CausalPipeline(
            df, 
            api_key, 
            smart_room_path=None, 
            dataset_type='ashrae',
            max_iterations=60,
            alpha=0.6,
            beta=0.6,
            effect_threshold=0.1
        )
        final_dag, final_metrics = pipeline.run()
        try:
            # Export pipeline results
            policy_data = pipeline.export_for_policy_engine()
            
            # Initialize policy engine
            policy_engine = CausalPolicyEngine(policy_data, df)
            
            # Run NeurIPS experiments
            experiments = NeurIPSExperiments(pipeline, df, smart_room_path, api_key)  
            workshop_results = experiments.run_workshop_experiments(n_runs=20, use_pareto=True)
            
            # Save results
            json_results = {}
            for policy, points in workshop_results['pareto_points'].items():
                json_results[policy] = [{
                    'kwh': p.kwh,
                    'dh': p.dh,
                    'episode': p.episode,
                    'constraint_target': p.constraint_target,
                    'feasible': p.feasible
                } for p in points]

            with open('results/pareto_results_noisy.json', 'w') as f:
                json.dump(json_results, f, indent=2)
                
        except Exception as e:
            import traceback
            logger.error(f"Policy experiments failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            print(f"ERROR: {e}")
            print(f"TRACEBACK: {traceback.format_exc()}")

        # Calculate final SHD
        final_edges = set(tuple(edge) if isinstance(edge, list) else edge for edge in final_dag['edges'])
        final_shd = pipeline.ground_truth.get_shd(final_edges)
        logger.info(f"Final DAG achieved SHD: {final_shd}")

        # Export metrics
        export_metrics_to_csv(final_metrics, output_dir)
        
        # Generate visualizations
        visualizer = MetricsVisualizer(output_dir)
        figs = visualizer.plot_metrics(final_metrics)
        
        # Create subplot for each method's final DAG and the final validated DAG
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
        
        # Plot PC DAG
        G_pc = nx.DiGraph()
        pc_edges = pipeline._results['pc']['edges']
        G_pc.add_edges_from(pc_edges)
        pos_pc = nx.spring_layout(G_pc)
        nx.draw(G_pc, pos_pc, ax=ax1, with_labels=True, node_color='lightblue', node_size=1000)
        pc_shd = pipeline.shd_history.get('pc', [0])
        ax1.set_title(f"Final PC DAG (SHD: {pc_shd[-1] if pc_shd else 'N/A'})")
        
        # Plot SAM DAG
        G_sam = nx.DiGraph()
        sam_edges = pipeline._results['sam']['edges']
        G_sam.add_edges_from(sam_edges)
        pos_sam = nx.spring_layout(G_sam)
        nx.draw(G_sam, pos_sam, ax=ax2, with_labels=True, node_color='lightgreen', node_size=1000)
        sam_shd = pipeline.shd_history.get('sam', [0])
        ax2.set_title(f"Final SAM DAG (SHD: {sam_shd[-1] if sam_shd else 'N/A'})")
        
        # Plot LLM DAG
        G_llm = nx.DiGraph()
        llm_edges = pipeline._results['llm']['edges']
        G_llm.add_edges_from(llm_edges)
        pos_llm = nx.spring_layout(G_llm)
        nx.draw(G_llm, pos_llm, ax=ax3, with_labels=True, node_color='lightpink', node_size=1000)
        llm_shd = pipeline.shd_history.get('llm', [0])
        ax3.set_title(f"Final LLM DAG (SHD: {llm_shd[-1] if llm_shd else 'N/A'})")
        
        # Plot Final Validated DAG
        G_final = nx.DiGraph()
        G_final.add_edges_from(final_dag['edges'])
        pos_final = nx.spring_layout(G_final)
        nx.draw(G_final, pos_final, ax=ax4, with_labels=True, node_color='gold', node_size=1000)
        
        # Add edge labels with confidence scores for final DAG
        edge_labels = {(u, v): f"{final_dag['confidence_scores'].get(f'{u}->{v}', 0):.2f}" for (u, v) in G_final.edges()}
        nx.draw_networkx_edge_labels(G_final, pos_final, edge_labels, ax=ax4)
        
        # Add SHD to title
        final_title = f"Final Validated DAG (SHD: {final_shd})"
        if final_shd == 0:
            final_title += " - PERFECT MATCH!"
        ax4.set_title(final_title)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'final_dags_comparison.png'))
        plt.close()

        # Save detailed metrics
        metrics_df = pd.DataFrame({
            'Edge': [f"{e[0]}->{e[1]}" for e in final_dag['edges']],
            'Confidence': [pipeline.edge_ranker.edge_confidence.get(e, 0) for e in final_dag['edges']],
            'Num_Interventions': [len(pipeline.tester.intervention_results.get(e, []))
                                for e in final_dag['edges']]
        })
        metrics_df.to_csv(os.path.join(output_dir, 'edge_metrics.csv'), index=False)
        export_detailed_metrics(pipeline, output_dir)

        # Add edge confidence comparison
        confidence_fig = visualizer.plot_edge_confidence_comparison(final_dag, pipeline._results, pipeline.tester.intervention_results)
        confidence_fig.savefig(os.path.join(output_dir, 'edge_confidence_comparison.png'))
        plt.close()

        # Plot SHD history with termination point highlighted
        plt.figure(figsize=(10, 6))
        for method, shd_values in pipeline.shd_history.items():
            plt.plot(range(1, len(shd_values) + 1), shd_values, 
                    marker='o', label=method)
        
        # Add a line for the final DAG SHD if it's different from method SHDs
        if final_shd not in [pipeline.shd_history[m][-1] for m in pipeline.shd_history]:
            iterations = list(range(1, pipeline.current_iteration + 1))
            # Check if we have a final SHD history
            if 'final' in pipeline.shd_history:
                plt.plot(iterations, pipeline.shd_history['final'], 
                       marker='s', label='final DAG', color='red', linewidth=2)
            else:
                # Just show the final point
                plt.plot(pipeline.current_iteration, final_shd, 
                       marker='s', label='final DAG', color='red', markersize=10)
        
        # Highlight the point where SHD=0 was achieved (if it was)
        if final_shd == 0:
            # Find first iteration where SHD=0
            termination_point = None
            for method, values in pipeline.shd_history.items():
                if 0 in values:
                    term_iter = values.index(0) + 1  # +1 because iterations start at 1
                    if termination_point is None or term_iter < termination_point:
                        termination_point = term_iter
            
            if termination_point:
                plt.axvline(x=termination_point, color='green', linestyle='--', 
                          label=f'SHD=0 achieved at iteration {termination_point}')
                plt.plot(termination_point, 0, 'go', markersize=12)
        
        plt.xlabel('Iteration')
        plt.ylabel('SHD')
        plt.title('Structural Hamming Distance Over Iterations')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'shd_history.png'))
        plt.close()
        
        # Save detailed metrics
        metrics_df = pd.DataFrame({
            'Edge': [f"{e[0]}->{e[1]}" for e in final_dag['edges']],
            'Confidence': [pipeline.edge_ranker.edge_confidence.get(e, 0) for e in final_dag['edges']],
            'Num_Interventions': [len(pipeline.tester.intervention_results.get(e, []))
                                for e in final_dag['edges']]
        })
        metrics_df.to_csv(os.path.join(output_dir, 'edge_metrics.csv'), index=False)
        
        # Log final summary
        logger.info("\n=== Final Summary ===")
        logger.info(f"Total iterations: {pipeline.current_iteration}")
        logger.info(f"Total edges validated: {len(final_dag['edges'])}")
        logger.info(f"Average edge confidence: {metrics_df['Confidence'].mean():.3f}")
        logger.info(f"Total interventions performed: {metrics_df['Num_Interventions'].sum()}")
        
        # Add terminal output for SHD=0 achievement
        if final_shd == 0:
            logger.info("\n" + "="*60)
            logger.info("PERFECT MATCH ACHIEVED! Final DAG has SHD = 0")
            logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Error in pipeline execution: {str(e)}", exc_info=True)
        raise

    finally:
        cleanup_processes()
        if 'pipeline' in locals():
            pipeline.cleanup()
        if 'eplus' in locals():
            eplus.cleanup()

if __name__ == "__main__":
    main()