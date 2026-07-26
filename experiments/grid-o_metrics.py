#!/usr/bin/env python3
"""
DAG Evaluation Metrics Script

This script computes Structural Hamming Distance (SHD), F1 score, precision, 
and recall for comparing a predicted DAG against a ground truth DAG, following 
the methodology used in the project workflow.

Author: Based on project methodology
"""

import numpy as np
import pandas as pd
from typing import Set, List, Tuple, Dict, Union
import networkx as nx
from dataclasses import dataclass
import json
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


@dataclass
class DAGMetrics:
    """Container for DAG evaluation metrics"""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    shd: int
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int


class DAGEvaluator:
    """
    Evaluates DAG quality metrics following the project methodology.
    
    The evaluation computes:
    - SHD (Structural Hamming Distance): Missing + Extra edges
    - Precision: TP / (TP + FP) 
    - Recall: TP / (TP + FN)
    - F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
    - Accuracy: (TP + TN) / (TP + TN + FP + FN)
    """
    
    def __init__(self, ground_truth_edges: List[Tuple[str, str]]):
        """
        Initialize evaluator with ground truth DAG.
        
        Args:
            ground_truth_edges: List of directed edges as (source, target) tuples
        """
        self.ground_truth_edges = self._normalize_edges(ground_truth_edges)
        self.all_possible_edges = self._get_all_possible_edges()
        
        # Default node positions and colors for visualization (Smart Room setup)
        self.default_node_positions = {
            'temperature': (0, 2),
            'humidity': (2, 2),
            'airquality': (1, 1.5),
            'energyconsumption': (0, 0),
            'overallsatisfaction': (2, 0)
        }
        
        self.default_node_colors = {
            'temperature': '#FF9966',     # Orange
            'humidity': '#66B2FF',        # Blue  
            'airquality': '#99FF66',      # Green
            'energyconsumption': '#FF6666', # Red
            'overallsatisfaction': '#FFFF66' # Yellow
        }
        
    def _normalize_edges(self, edges: List[Union[Tuple[str, str], List[str]]]) -> Set[Tuple[str, str]]:
        """
        Normalize edges exactly like utils.py normalize_edge function.
        
        Args:
            edges: List of edges as tuples or lists
            
        Returns:
            Set of normalized edge tuples
        """
        normalized = set()
        for edge in edges:
            if isinstance(edge, (list, tuple)):
                # Match utils.py: tuple(str(x).strip() for x in edge)
                normalized_edge = tuple(str(x).strip() for x in edge)
            else:
                normalized_edge = edge
            normalized.add(normalized_edge)
        return normalized
    
    def _get_all_possible_edges(self) -> Set[Tuple[str, str]]:
        """
        Get all possible directed edges between nodes in the ground truth.
        
        Returns:
            Set of all possible directed edges
        """
        # Extract unique nodes from ground truth edges
        nodes = set()
        for source, target in self.ground_truth_edges:
            nodes.add(source)
            nodes.add(target)
        
        # Generate all possible directed edges (excluding self-loops)
        all_edges = set()
        for source in nodes:
            for target in nodes:
                if source != target:
                    all_edges.add((source, target))
        
        return all_edges
    
    def compute_metrics(self, predicted_edges: List[Union[Tuple[str, str], List[str]]]) -> DAGMetrics:
        """
        Compute metrics exactly following the project's utils.py implementation.
        
        Args:
            predicted_edges: List of predicted edges as tuples or lists
            
        Returns:
            DAGMetrics object containing all computed metrics
        """
        predicted_edges_set = self._normalize_edges(predicted_edges)
        
        # Calculate true positives, false positives, and false negatives
        # Following utils.py: predicted & true, predicted - true, true - predicted
        true_positives = len(predicted_edges_set & self.ground_truth_edges)
        false_positives = len(predicted_edges_set - self.ground_truth_edges)
        false_negatives = len(self.ground_truth_edges - predicted_edges_set)
        
        # Calculate precision, recall, and F1 score exactly like utils.py
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1_score = 2 * precision * recall / max(precision + recall, 1e-10)
        
        # Calculate SHD exactly like utils.py: missing + extra edges
        shd = false_negatives + false_positives
        
        # Calculate accuracy exactly like utils.py
        # true_negatives = (len(true) * len(true)) - len(true) - false_positives - false_negatives
        true_negatives = (len(self.ground_truth_edges) * len(self.ground_truth_edges)) - len(self.ground_truth_edges) - false_positives - false_negatives
        accuracy = (true_positives + true_negatives) / max(true_positives + true_negatives + false_positives + false_negatives, 1)
        
        return DAGMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            shd=shd,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            true_negatives=true_negatives
        )
    
    def visualize_dag_comparison(self, predicted_edges: List[Union[Tuple[str, str], List[str]]], 
                               method_name: str = "Predicted",
                               node_positions: Dict[str, Tuple[float, float]] = None,
                               node_colors: Dict[str, str] = None,
                               output_dir: str = "dag_visualizations") -> None:
        """Create DAG visualizations with numbered nodes."""
        if node_positions is None:
            node_positions = self.default_node_positions
        if node_colors is None:
            node_colors = self.default_node_colors
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot ground truth
        self._plot_single_dag(
            list(self.ground_truth_edges), 
            node_positions, 
            node_colors,
            self.ground_truth_edges,
            "Ground Truth DAG",
            os.path.join(output_dir, "ground_truth_dag.png")
        )
        
        # Plot predicted DAG
        metrics = self.compute_metrics(predicted_edges)
        pred_title = f"{method_name} DAG (SHD: {metrics.shd})"
        
        self._plot_single_dag(
            predicted_edges,
            node_positions,
            node_colors, 
            self.ground_truth_edges,
            pred_title,
            os.path.join(output_dir, f"{method_name.lower().replace(' ', '_')}_dag.png")
        )
    
    def _plot_single_dag(self, edges: List[Union[Tuple[str, str], List[str]]], 
                        node_positions: Dict[str, Tuple[float, float]],
                        node_colors: Dict[str, str],
                        ground_truth_edges: Set[Tuple[str, str]],
                        title: str,
                        filename: str) -> None:
        """Plot a single DAG with numbered nodes following improved_plots.py style."""
        import matplotlib.pyplot as plt
        import networkx as nx
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize edges
        normalized_edges = self._normalize_edges(edges)
        
        # Create graph
        G = nx.DiGraph()
        G.add_edges_from(normalized_edges)
        
        # Add all nodes to ensure consistent positioning
        all_nodes = list(node_positions.keys())
        G.add_nodes_from(all_nodes)
        
        # Create case-insensitive position mapping
        case_insensitive_positions = {}
        for node in G.nodes():
            for pos_key, pos_val in node_positions.items():
                if node.lower() == pos_key.lower():
                    case_insensitive_positions[node] = pos_val
                    break
            if node not in case_insensitive_positions:
                case_insensitive_positions[node] = (0, 0)
        
        # Draw nodes with NetworkX
        nx.draw_networkx_nodes(G, case_insensitive_positions,
                               node_color=[node_colors.get(node.lower(), '#CCCCCC') for node in G.nodes()],
                               node_size=5000,
                               edgecolors='black',
                               linewidths=3,
                               alpha=1.0,
                               ax=ax)
        
        # Add node numbering
        ordered_nodes = self._get_consistent_node_ordering(node_colors)
        
        # Create mapping from node name to number
        node_numbers = {}
        for i, ordered_node in enumerate(ordered_nodes):
            for node in G.nodes():
                if node.lower() == ordered_node.lower():
                    node_numbers[node] = i + 1
                    break
        
        # Draw the numbers
        for node in G.nodes():
            if node in node_numbers:
                pos = case_insensitive_positions[node]
                ax.text(pos[0], pos[1], str(node_numbers[node]), ha='center', va='center',
                       fontsize=50, fontweight='bold', color='white',
                       bbox=dict(boxstyle="circle,pad=0.15", facecolor='black', alpha=0.5))
        
        # Determine edge colors and styles
        edge_colors = []
        edge_styles = []
        for edge in normalized_edges:
            if title == 'Ground Truth DAG':
                edge_colors.append('#000000')  # Black for ground truth
                edge_styles.append('-')        # Solid
            else:
                norm_edge = (edge[0].lower(), edge[1].lower())
                norm_gt_edges = {(e[0].lower(), e[1].lower()) for e in ground_truth_edges}
                
                if norm_edge in norm_gt_edges:
                    edge_colors.append('#2ECC71')  # Green for correct edges
                    edge_styles.append('-')        # Solid
                else:
                    edge_colors.append('#E74C3C')  # Red for wrong edges
                    edge_styles.append('--')       # Dashed
        
        # Draw edges
        for i, edge in enumerate(normalized_edges):
            nx.draw_networkx_edges(G, case_insensitive_positions,
                                   edgelist=[edge],
                                   edge_color=edge_colors[i],
                                   style=edge_styles[i],
                                   width=4,
                                   alpha=1.0,
                                   arrowsize=70,
                                   arrowstyle='-|>',
                                   connectionstyle='arc3,rad=0.1',
                                   node_size=5000,
                                   min_source_margin=20,
                                   min_target_margin=20,
                                   ax=ax)
        
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=600, bbox_inches='tight')
        plt.close()
    
    def _get_consistent_node_ordering(self, node_colors: Dict[str, str]) -> List[str]:
        """Get consistent node ordering for numbering."""
        # Default smart room ordering
        default_order = ['temperature', 'humidity', 'airquality', 'energyconsumption', 'overallsatisfaction']
        
        # Filter to only nodes that exist in the color mapping
        existing_nodes = [node for node in default_order if node in node_colors]
        
        # Add any additional nodes not in default order
        for node in node_colors:
            if node not in existing_nodes:
                existing_nodes.append(node)
        
        return existing_nodes
    
    def print_detailed_report(self, predicted_edges: List[Union[Tuple[str, str], List[str]]], 
                             method_name: str = "Predicted") -> None:
        """
        Print detailed evaluation report.
        
        Args:
            predicted_edges: List of predicted edges
            method_name: Name of the method being evaluated
        """
        metrics = self.compute_metrics(predicted_edges)
        predicted_edges_set = self._normalize_edges(predicted_edges)
        
        print(f"\n{'='*60}")
        print(f"DAG EVALUATION REPORT: {method_name.upper()}")
        print(f"{'='*60}")
        
        print(f"\nGround Truth Edges ({len(self.ground_truth_edges)}):")
        for edge in sorted(self.ground_truth_edges):
            print(f"  {edge[0]} -> {edge[1]}")
        
        print(f"\nPredicted Edges ({len(predicted_edges_set)}):")
        for edge in sorted(predicted_edges_set):
            print(f"  {edge[0]} -> {edge[1]}")
        
        print(f"\nCONFUSION MATRIX:")
        print(f"  True Positives (TP):  {metrics.true_positives}")
        print(f"  False Positives (FP): {metrics.false_positives}")
        print(f"  False Negatives (FN): {metrics.false_negatives}")
        print(f"  True Negatives (TN):  {metrics.true_negatives}")
        
        print(f"\nCORE METRICS:")
        print(f"  Precision: {metrics.precision:.4f}")
        print(f"  Recall:    {metrics.recall:.4f}")
        print(f"  F1 Score:  {metrics.f1_score:.4f}")
        print(f"  Accuracy:  {metrics.accuracy:.4f}")
        print(f"  SHD:       {metrics.shd}")
        
        # Edge analysis
        correct_edges = predicted_edges_set.intersection(self.ground_truth_edges)
        wrong_edges = predicted_edges_set - self.ground_truth_edges
        missing_edges = self.ground_truth_edges - predicted_edges_set
        
        if correct_edges:
            print(f"\nCORRECT EDGES ({len(correct_edges)}):")
            for edge in sorted(correct_edges):
                print(f"  ✓ {edge[0]} -> {edge[1]}")
        
        if wrong_edges:
            print(f"\nINCORRECT EDGES ({len(wrong_edges)}):")
            for edge in sorted(wrong_edges):
                print(f"  ✗ {edge[0]} -> {edge[1]}")
        
        if missing_edges:
            print(f"\nMISSING EDGES ({len(missing_edges)}):")
            for edge in sorted(missing_edges):
                print(f"  ○ {edge[0]} -> {edge[1]}")
        
        print(f"\n{'='*60}")


def get_hardcoded_test_cases() -> Dict[str, Dict[str, List[Tuple[str, str]]]]:
    """
    Get hardcoded ground truth and learned DAG pairs for testing.
    
    Returns:
        Dictionary with test cases containing ground truth and GRID-O learned DAGs
    """
    return {
        "ashrae": {
            "ground_truth": [
                ("square_feet", "meter_reading"),
                ("year_built", "meter_reading"), 
                ("air_temperature", "dew_temperature"),
                ("air_temperature", "meter_reading"),
                ("dew_temperature", "meter_reading")
            ],
            "grid_o": [
                ("square_feet", "meter_reading"),
                ("sea_level_pressure", "meter_reading"),
                ("dew_temperature", "air_temperature"),
                ("air_temperature", "dew_temperature"),
                ("year_built", "square_feet"),
                ("air_temperature", "meter_reading"),
                ("dew_temperature", "meter_reading")
            ]
        },
        "base": {
            "ground_truth": [
                ("temperature", "energyconsumption"),
                ("temperature", "overallsatisfaction"),
                ("humidity", "energyconsumption"),
                ("humidity", "overallsatisfaction"),
                ("airquality", "energyconsumption"),
                ("airquality", "overallsatisfaction")
            ],
            "grid_o": [
                ("humidity", "energyconsumption"),
                ("airquality", "overallsatisfaction"),
                ("temperature", "humidity"),
                ("temperature", "airquality"),
                ("temperature", "overallsatisfaction"),
                ("humidity", "airquality"),
                ("temperature", "energyconsumption")
            ]
        },
        "noisy": {
            "ground_truth": [
                ("temperature", "energyconsumption"),
                ("temperature", "overallsatisfaction"),
                ("humidity", "energyconsumption"),
                ("humidity", "overallsatisfaction"),
                ("airquality", "energyconsumption"),
                ("airquality", "overallsatisfaction")
            ],
            "grid_o": [
                ("humidity", "overallsatisfaction"),
                ("temperature", "energyconsumption"),
                ("temperature", "humidity"),
                ("airquality", "energyconsumption"),
                ("humidity", "temperature"),
                ("temperature", "overallsatisfaction"),
                ("humidity", "energyconsumption"),
                ("airquality", "overallsatisfaction")
            ]
        },
        "hidden_vars": {
            "ground_truth": [
                ("temperature", "energyconsumption"),
                ("temperature", "overallsatisfaction"),
                ("humidity", "energyconsumption"),
                ("humidity", "overallsatisfaction"),
                ("airquality", "energyconsumption"),
                ("airquality", "overallsatisfaction")
            ],
            "grid_o": [
                ("airquality", "overallsatisfaction"),
                ("temperature", "overallsatisfaction"),
                ("airquality", "humidity"),
                ("temperature", "humidity"),
                ("humidity", "overallsatisfaction"),
                ("airquality", "energyconsumption"),
                ("humidity", "airquality"),
                ("temperature", "energyconsumption"),
                ("humidity", "energyconsumption")
            ]
        },
        "smart_building": {
            "ground_truth": [
                ("temperature", "hvacpower"),
            ("humidity", "hvacpower"),
            ("hvacsetpoint", "hvacpower"),
            ("occupantcount", "hvacpower"),

            # Lighting dynamics
            ("lightinglevel", "lightingpower"),
            ("occupant_count", "lightingpower"),

            # Comfort metrics
            ("temperature", "thermal_comfort"),
            ("humidity", "thermal_comfort"),
            ("lightinglevel", "visualcomfort"),

            # Air quality index
            ("airquality", "airqualityindex"),

            # Energy consumption
            ("temperature", "energyconsumption"),
            ("humidity", "energyconsumption"),
            ("hvacsetpoint", "energyconsumption"),
            ("lightinglevel", "energyconsumption"),
            ("occupantcount", "energyconsumption"),
            ("airquality", "energyconsumption"),

            # Overall satisfaction
            ("temperature", "overallsatisfaction"),
            ("humidity", "overallsatisfaction"),
            ("hvacsetpoint", "overallsatisfaction"),
            ("lightinglevel", "overallsatisfaction"),
            ("occupantcount", "overallsatisfaction"),
            ("airquality", "overallsatisfaction")
            ],
            "grid_o": [
                ("lightinglevel", "energyconsumption"),
                ("airquality", "visualcomfort"),
                ("humidity", "hvacsetpoint"),
                ("occupantcount", "overallsatisfaction"),
                ("temperature", "hvacpower"),
                ("hvacsetpoint", "hvacpower"),
                ("temperature", "lightinglevel"),
                ("occupantcount", "lightingpower"),
                ("humidity", "visualcomfort"),
                ("lightinglevel", "hvacpower"),
                ("hvacsetpoint", "overallsatisfaction"),
                ("temperature", "thermalcomfort"),
                ("hvacsetpoint", "thermalcomfort"),
                ("lightinglevel", "overallsatisfaction"),
                ("temperature", "lightingpower"),
                ("hvacsetpoint", "lightingpower"),
                ("lightinglevel", "thermalcomfort"),
                ("airquality", "lightinglevel"),
                ("temperature", "hvacsetpoint"),
                ("airquality", "overallsatisfaction"),
                ("lightinglevel", "lightingpower"),
                ("occupantcount", "visualcomfort"),
                ("hvacsetpoint", "occupantcount"),
                ("hvacsetpoint", "airqualityindex"),
                ("lightinglevel", "hvacsetpoint"),
                ("airquality", "thermalcomfort"),
                ("airquality", "lightingpower"),
                ("humidity", "lightinglevel"),
                ("temperature", "visualcomfort"),
                ("hvacsetpoint", "visualcomfort"),
                ("airquality", "hvacsetpoint"),
                ("temperature", "humidity"),
                ("lightinglevel", "occupantcount"),
                ("occupantcount", "energyconsumption"),
                ("lightinglevel", "airqualityindex"),
                ("humidity", "overallsatisfaction"),
                ("lightinglevel", "visualcomfort"),
                ("humidity", "airquality"),
                ("humidity", "thermalcomfort"),
                ("hvacsetpoint", "energyconsumption"),
                ("airquality", "airqualityindex"),
                ("temperature", "energyconsumption"),
                ("humidity", "lightingpower")
            ]
        },
        "physical": {
            "ground_truth": [
                ("temperature", "energyconsumption"),
                ("humidity", "energyconsumption"),
                ("airquality", "energyconsumption"),
                ("temperature", "overallsatisfaction"),
                ("humidity", "overallsatisfaction"),
                ("airquality", "overallsatisfaction"),
                ("temperature", "humidity"),
                ("humidity", "airquality")
            ],
            "grid_o": [
                ("humidity", "energyconsumption"),
                ("temperature", "energyconsumption"),
                ("airquality", "overallsatisfaction"),
                ("humidity", "temperature"),
                ("humidity", "overallsatisfaction"),
                ("temperature", "overallsatisfaction"),
                ("temperature", "humidity"),
                ("airquality", "energyconsumption")
            ]
        }
    }


def run_all_hardcoded_evaluations(output_base_dir: str = "hardcoded_evaluations") -> pd.DataFrame:
    """
    Run evaluations on all hardcoded test cases.
    
    Args:
        output_base_dir: Base directory for all evaluation outputs
        
    Returns:
        DataFrame with summary results for all test cases
    """
    test_cases = get_hardcoded_test_cases()
    all_results = []
    
    print("Running evaluations on all hardcoded test cases...")
    print("=" * 60)
    
    for scenario_name, data in test_cases.items():
        print(f"\nEvaluating {scenario_name.upper()} scenario...")
        
        # Create evaluator
        evaluator = DAGEvaluator(data["ground_truth"])
        
        # Set up output directory
        scenario_output_dir = os.path.join(output_base_dir, scenario_name)
        
        # Custom node positions and colors for different scenarios
        if scenario_name == "ashrae":
            node_positions = {
                'square_feet': (0, 2),
                'year_built': (2, 2),
                'air_temperature': (0, 1),
                'dew_temperature': (1, 1),
                'sea_level_pressure': (2, 1),
                'meter_reading': (1, 0)
            }
            node_colors = {
                'square_feet': '#FF9966',
                'year_built': '#66B2FF',
                'air_temperature': '#99FF66',
                'dew_temperature': '#FF6666',
                'sea_level_pressure': '#FFFF66',
                'meter_reading': '#FF66FF'
            }
        elif scenario_name == "smart_building":
            node_positions = {
                'temperature': (0, 3),
                'humidity': (1.5, 3),
                'airquality': (3, 3),
                'hvacsetpoint': (4.5, 3),
                'lightinglevel': (6, 3),
                'occupantcount': (7.5, 3),
                'thermalcomfort': (0, 2),
                'visualcomfort': (2, 2),
                'airqualityindex': (4, 2),
                'hvacpower': (6, 2),
                'lightingpower': (7.5, 2),
                'energyconsumption': (2, 1),
                'overallsatisfaction': (5, 1)
            }
            node_colors = {
                'temperature': '#FF9966',
                'humidity': '#66B2FF',
                'airquality': '#99FF66',
                'hvacsetpoint': '#FF6666',
                'lightinglevel': '#FFFF66',
                'occupantcount': '#FF66FF',
                'thermalcomfort': '#66FFFF',
                'visualcomfort': '#FFB366',
                'airqualityindex': '#B366FF',
                'hvacpower': '#66FFB3',
                'lightingpower': '#FFD966',
                'energyconsumption': '#FF6B6B',
                'overallsatisfaction': '#6BFF6B'
            }
        else:
            # Default smart room layout
            node_positions = evaluator.default_node_positions
            node_colors = evaluator.default_node_colors
        
        # Run evaluation with visualization
        evaluator.visualize_dag_comparison(
            data["grid_o"],
            f"GRID-O ({scenario_name})",
            node_positions,
            node_colors,
            scenario_output_dir
        )
        
        # Compute metrics
        metrics = evaluator.compute_metrics(data["grid_o"])
        evaluator.print_detailed_report(data["grid_o"], f"GRID-O ({scenario_name})")
        
        # Store results
        result = {
            'Scenario': scenario_name,
            'Method': 'GRID-O',
            'Precision': metrics.precision,
            'Recall': metrics.recall,
            'F1_Score': metrics.f1_score,
            'Accuracy': metrics.accuracy,
            'SHD': metrics.shd,
            'True_Positives': metrics.true_positives,
            'False_Positives': metrics.false_positives,
            'False_Negatives': metrics.false_negatives,
            'GT_Edges': len(data["ground_truth"]),
            'Predicted_Edges': len(data["grid_o"])
        }
        all_results.append(result)
        
        print(f"✓ {scenario_name}: SHD={metrics.shd}, F1={metrics.f1_score:.3f}")
    
    # Create summary DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save summary results
    os.makedirs(output_base_dir, exist_ok=True)
    summary_file = os.path.join(output_base_dir, "evaluation_summary.csv")
    results_df.to_csv(summary_file, index=False)
    
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(results_df[['Scenario', 'SHD', 'F1_Score', 'Precision', 'Recall']].round(3))
    print(f"\nDetailed results saved to: {output_base_dir}/")
    print(f"Summary saved to: {summary_file}")
    
    return results_df


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Evaluate DAG metrics with visualization')
    parser.add_argument('--ground_truth', type=str, 
                       help='JSON file with ground truth edges')
    parser.add_argument('--predicted', type=str,
                       help='JSON file with predicted edges')
    parser.add_argument('--method', type=str, default='ours',
                       help='Method key in the JSON file')
    parser.add_argument('--output', type=str, default='dag_evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--no-viz', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--run-hardcoded', action='store_true',
                       help='Run all hardcoded test cases')
    parser.add_argument('--scenario', type=str, choices=['ashrae', 'base', 'noisy', 'hidden_vars', 'smart_building', 'physical'],
                       help='Run specific hardcoded scenario')
    
    args = parser.parse_args()
    
    # Run hardcoded evaluations
    if args.run_hardcoded:
        run_all_hardcoded_evaluations(args.output)
        return
    
    # Run specific scenario
    if args.scenario:
        test_cases = get_hardcoded_test_cases()
        if args.scenario not in test_cases:
            print(f"Error: Scenario '{args.scenario}' not found")
            return
        
        data = test_cases[args.scenario]
        evaluator = DAGEvaluator(data["ground_truth"])
        
        metrics = evaluator.compute_metrics(data["grid_o"])
        evaluator.print_detailed_report(data["grid_o"], f"GRID-O ({args.scenario})")
        return
    
    print("Error: Must specify --run-hardcoded, --scenario, or provide --predicted file")


if __name__ == "__main__":
    main()