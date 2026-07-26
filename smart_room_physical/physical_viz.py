#!/usr/bin/env python3
"""
Causal Discovery Visualization and Metrics Analysis Script

This script computes and visualizes:
- Structural Hamming Distance (SHD)
- Core metrics (precision, recall, F1)
- Risk and cost metrics
- Individual DAG visualizations with correct/incorrect edges highlighted

Input: JSON file containing ground truth and method DAGs
Output: Separate visualization files for each analysis component
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from typing import Dict, List, Tuple, Set
import argparse
import os

class CausalMetricsAnalyzer:
    def __init__(self, ground_truth_edges: List[Tuple], method_dags: Dict):
        """
        Initialize the analyzer with ground truth and method DAGs
        
        Args:
            ground_truth_edges: List of ground truth edges as tuples
            method_dags: Dictionary mapping method names to their edge lists
        """
        self.ground_truth = set(self._normalize_edges(ground_truth_edges))
        self.method_dags = {
            method: set(self._normalize_edges(edges)) 
            for method, edges in method_dags.items()
        }
        
        # Color scheme consistent with MetricsVisualizer
        self.colors = {
            'correct': '#2ecc71',      # Green for correct edges
            'incorrect': '#e74c3c',    # Red for incorrect edges
            'precision': '#4C72B0',    # Blue
            'recall': '#55A868',       # Green
            'f1_score': '#C44E52',     # Red
            'shd': '#F39C12',          # Orange 
            'runtime': '#8172B3',      # Purple
            'node_colors': {
                'Temperature': '#FF9966',           # Orange
                'temperature': '#FF9966',           # Orange
                'Humidity': '#66B2FF',              # Blue
                'humidity': '#66B2FF',              # Blue
                'AirQuality': '#66CC66',            # Green
                'airquality': '#66CC66',            # Green
                'EnergyConsumption': '#FF6666',     # Red
                'energyconsumption': '#FF6666',     # Red
                'OverallSatisfaction': '#B266FF',   # Purple
                'overallsatisfaction': '#B266FF',   # Purple
                'weight': '#FFCC66',                # Yellow
            }
        }
    
    def _normalize_edges(self, edges: List) -> List[Tuple]:
        """Normalize edges to consistent tuple format"""
        return [tuple(edge) if isinstance(edge, list) else edge for edge in edges]
    
    def calculate_shd(self, predicted_edges: Set[Tuple]) -> int:
        """Calculate Structural Hamming Distance"""
        missing = len(self.ground_truth - predicted_edges)
        extra = len(predicted_edges - self.ground_truth)
        return missing + extra
    
    def calculate_core_metrics(self, predicted_edges: Set[Tuple]) -> Dict[str, float]:
        """Calculate precision, recall, and F1 score"""
        if not predicted_edges and not self.ground_truth:
            return {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0}
        
        if not predicted_edges:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}
        
        if not self.ground_truth:
            return {'precision': 0.0, 'recall': 1.0, 'f1_score': 0.0}
        
        true_positives = len(predicted_edges & self.ground_truth)
        false_positives = len(predicted_edges - self.ground_truth)
        false_negatives = len(self.ground_truth - predicted_edges)
        
        precision = true_positives / len(predicted_edges) if predicted_edges else 0.0
        recall = true_positives / len(self.ground_truth) if self.ground_truth else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def calculate_risk_cost(self, predicted_edges: Set[Tuple]) -> Dict[str, float]:
        """
        Calculate risk and cost metrics aligned with the original framework:
        - Cost: penalty for wrong edges normalized by ground truth size
        - Risk: proportion of wrong edges weighted by their confidence
        """
        wrong_edges = predicted_edges - self.ground_truth
        total_predicted = len(predicted_edges)
        total_ground_truth = len(self.ground_truth)
        
        # Cost: Normalized penalty for wrong edges (from original framework)
        # Cost represents the "expense" of including wrong edges relative to the true structure
        cost = len(wrong_edges) / max(total_ground_truth, 1)
        
        # Risk: Proportion of wrong edges among predicted edges (uncertainty measure)
        # Risk represents the likelihood that a randomly selected predicted edge is wrong
        risk = len(wrong_edges) / max(total_predicted, 1)
        
        return {'risk': risk, 'cost': cost}
    
    def analyze_all_methods(self) -> pd.DataFrame:
        """Analyze all methods and return comprehensive metrics"""
        results = []
        
        for method, edges in self.method_dags.items():
            shd = self.calculate_shd(edges)
            core_metrics = self.calculate_core_metrics(edges)
            risk_cost = self.calculate_risk_cost(edges)
            
            results.append({
                'method': method,
                'shd': shd,
                'precision': core_metrics['precision'],
                'recall': core_metrics['recall'],
                'f1_score': core_metrics['f1_score'],
                'risk': risk_cost['risk'],
                'cost': risk_cost['cost'],
                'num_edges': len(edges),
                'correct_edges': len(edges & self.ground_truth),
                'wrong_edges': len(edges - self.ground_truth)
            })
        
        return pd.DataFrame(results)
    
    def create_all_visualizations(self, output_dir: str = "analysis_output"):
        """Create all visualizations and save as separate files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Individual DAG visualizations (separate files)
        self.create_individual_dags(output_dir)
        
        # 2. Core metrics visualization
        self.create_core_metrics_chart(os.path.join(output_dir, "core_metrics.png"))
        
        # 3. SHD comparison
        self.create_shd_chart(os.path.join(output_dir, "shd_comparison.png"))
        
        # 4. Risk and cost analysis
        self.create_risk_cost_chart(os.path.join(output_dir, "risk_cost_analysis.png"))
        
        # 5. Summary table
        self.create_summary_table(os.path.join(output_dir, "summary_table.png"))
        
        # 6. Comprehensive dashboard (all in one)
        self.create_comprehensive_dashboard(os.path.join(output_dir, "comprehensive_dashboard.png"))
        
        print(f"All visualizations saved to: {output_dir}")
        
        return output_dir
    
    def create_individual_dags(self, output_dir: str):
        """Create individual DAG visualizations with curved arrows"""
        # Ground Truth DAG
        fig, ax = plt.subplots(figsize=(10, 8))
        self._plot_dag(ax, self.ground_truth, "Ground Truth DAG", is_ground_truth=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ground_truth_dag.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Ground truth DAG saved to: {os.path.join(output_dir, 'ground_truth_dag.png')}")
        
        # Method DAGs
        for method, edges in self.method_dags.items():
            fig, ax = plt.subplots(figsize=(10, 8))
            
            correct_count = len(edges & self.ground_truth)
            total_count = len(edges)
            title = f"{method.upper()} DAG: {correct_count}/{total_count} correct edges"
            
            self._plot_dag(ax, edges, title)
            plt.tight_layout()
            
            filename = f"{method}_dag.png"
            save_path = os.path.join(output_dir, filename)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"{method.upper()} DAG saved to: {save_path}")
    
    def create_core_metrics_chart(self, save_path: str):
        """Create core metrics bar chart"""
        df = self.analyze_all_methods()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metrics = ['precision', 'recall', 'f1_score']
        x = np.arange(len(df))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            bars = ax.bar(x + i*width, df[metric], width, 
                         label=metric.capitalize(), 
                         color=self.colors[metric])
            
            # Add value labels on bars
            for bar, value in zip(bars, df[metric]):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Core Performance Metrics (Precision, Recall, F1)', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(df['method'], rotation=45)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Core metrics chart saved to: {save_path}")
    
    def create_shd_chart(self, save_path: str):
        """Create SHD comparison chart"""
        df = self.analyze_all_methods()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars = ax.bar(df['method'], df['shd'], color=self.colors['shd'], alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, df['shd']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(value)}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('Structural Hamming Distance (SHD)', fontsize=12)
        ax.set_title('Structural Hamming Distance Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Highlight best performing method
        min_shd_idx = df['shd'].idxmin()
        bars[min_shd_idx].set_color('#2ecc71')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SHD comparison chart saved to: {save_path}")
    
    def create_risk_cost_chart(self, save_path: str):
        """Create risk and cost analysis chart"""
        df = self.analyze_all_methods()
        
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(df))
        width = 0.35
        
        # Risk and Cost bars
        bars1 = ax1.bar(x - width/2, df['risk'], width, label='Risk', 
                       color='salmon', alpha=0.7)
        bars2 = ax1.bar(x + width/2, df['cost'], width, label='Cost', 
                       color='skyblue', alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars1, df['risk']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        for bar, value in zip(bars2, df['cost']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax1.set_xlabel('Method', fontsize=12)
        ax1.set_ylabel('Risk/Cost Score', fontsize=12)
        ax1.set_title('Risk and Cost Analysis by Method\n(Risk = Wrong Edges / Total Predicted, Cost = Wrong Edges / Ground Truth Size)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(df['method'], rotation=45)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Risk/cost analysis saved to: {save_path}")
    
    def create_summary_table(self, save_path: str):
        """Create summary statistics table"""
        df = self.analyze_all_methods()
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Round values for display
        display_df = df[['method', 'shd', 'precision', 'recall', 'f1_score', 'risk', 'cost', 'num_edges']].copy()
        display_df = display_df.round(3)
        
        # Create table
        table = ax.table(cellText=display_df.values,
                        colLabels=['Method', 'SHD', 'Precision', 'Recall', 'F1 Score', 'Risk', 'Cost', '# Edges'],
                        cellLoc='center',
                        loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.3, 2)
        
        # Color code best performers
        for i, row in display_df.iterrows():
            # Highlight best F1 score
            if row['f1_score'] == display_df['f1_score'].max():
                table[(i+1, 4)].set_facecolor('#90EE90')  # Light green
            # Highlight best SHD
            if row['shd'] == display_df['shd'].min():
                table[(i+1, 1)].set_facecolor('#87CEEB')  # Light blue
        
        ax.set_title('Comprehensive Performance Summary', fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        legend_text = "Green = Best F1 Score, Blue = Best SHD (Lowest)"
        ax.text(0.5, 0.02, legend_text, ha='center', va='bottom', transform=ax.transAxes, 
               fontsize=10, style='italic')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Summary table saved to: {save_path}")
    
    def create_comprehensive_dashboard(self, save_path: str):
        """Create comprehensive dashboard with all metrics"""
        fig = plt.figure(figsize=(20, 16))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 4, height_ratios=[2, 1, 1], hspace=0.3, wspace=0.3)
        
        # DAGs (top row)
        dag_positions = [(0, 0), (0, 1), (0, 2), (0, 3)]
        methods = ['Ground Truth'] + list(self.method_dags.keys())
        
        for i, (method, pos) in enumerate(zip(methods, dag_positions)):
            ax = fig.add_subplot(gs[pos])
            
            if method == 'Ground Truth':
                self._plot_dag(ax, self.ground_truth, method, is_ground_truth=True)
            else:
                edges = self.method_dags[method]
                correct_count = len(edges & self.ground_truth)
                total_count = len(edges)
                title = f"{method}: {correct_count}/{total_count} correct"
                self._plot_dag(ax, edges, title)
        
        # Core metrics (middle left)
        ax_core = fig.add_subplot(gs[1, :2])
        self._plot_core_metrics_subplot(ax_core)
        
        # SHD (middle right)
        ax_shd = fig.add_subplot(gs[1, 2:])
        self._plot_shd_subplot(ax_shd)
        
        # Risk/Cost (bottom)
        ax_risk = fig.add_subplot(gs[2, :])
        self._plot_risk_cost_subplot(ax_risk)
        
        plt.suptitle('Comprehensive Causal Discovery Analysis Dashboard', 
                    fontsize=18, fontweight='bold')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comprehensive dashboard saved to: {save_path}")
    
    def _plot_core_metrics_subplot(self, ax):
        """Plot core metrics in subplot"""
        df = self.analyze_all_methods()
        metrics = ['precision', 'recall', 'f1_score']
        x = np.arange(len(df))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            ax.bar(x + i*width, df[metric], width, label=metric.capitalize(), 
                  color=self.colors[metric])
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Score')
        ax.set_title('Core Performance Metrics')
        ax.set_xticks(x + width)
        ax.set_xticklabels(df['method'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_shd_subplot(self, ax):
        """Plot SHD in subplot"""
        df = self.analyze_all_methods()
        bars = ax.bar(df['method'], df['shd'], color=self.colors['shd'])
        
        for bar, value in zip(bars, df['shd']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{int(value)}', ha='center', va='bottom')
        
        ax.set_xlabel('Method')
        ax.set_ylabel('SHD')
        ax.set_title('Structural Hamming Distance')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_risk_cost_subplot(self, ax):
        """Plot risk/cost in subplot"""
        df = self.analyze_all_methods()
        x = np.arange(len(df))
        width = 0.35
        
        ax.bar(x - width/2, df['risk'], width, label='Risk', color='salmon')
        ax.bar(x + width/2, df['cost'], width, label='Cost', color='skyblue')
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Score')
        ax.set_title('Risk and Cost Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(df['method'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_dag(self, ax, edges: Set[Tuple], title: str, is_ground_truth: bool = False):
        """Plot a single DAG with curved arrows and correct/incorrect edge coloring"""
        G = nx.DiGraph()
        
        # Add all nodes from ground truth to ensure consistent layout
        all_nodes = set()
        for edge in self.ground_truth:
            all_nodes.update(edge)
        for edge in edges:
            all_nodes.update(edge)
        
        G.add_nodes_from(all_nodes)
        G.add_edges_from(edges)
        
        # Create layout
        pos = self._get_node_positions(all_nodes)
        
        # Draw nodes
        for node in all_nodes:
            color = self.colors['node_colors'].get(node, '#CCCCCC')
            nx.draw_networkx_nodes(G, pos, nodelist=[node], 
                                 node_color=color, node_size=1500, 
                                 edgecolors='black', linewidths=1, ax=ax)
        
        # Draw edges with curved arrows and color coding
        if is_ground_truth:
            # All edges are correct for ground truth
            if edges:
                for edge in edges:
                    nx.draw_networkx_edges(G, pos, edgelist=[edge],
                                         edge_color=self.colors['correct'],
                                         width=3, arrowsize=25, 
                                         connectionstyle="arc3,rad=0.1",
                                         arrowstyle='-|>', ax=ax)
        else:
            # Color edges based on correctness with curved arrows
            correct_edges = list(edges & self.ground_truth)
            incorrect_edges = list(edges - self.ground_truth)
            
            if correct_edges:
                for edge in correct_edges:
                    nx.draw_networkx_edges(G, pos, edgelist=[edge],
                                         edge_color=self.colors['correct'],
                                         width=3, arrowsize=25,
                                         connectionstyle="arc3,rad=0.1",
                                         arrowstyle='-|>', ax=ax)
            
            if incorrect_edges:
                for edge in incorrect_edges:
                    nx.draw_networkx_edges(G, pos, edgelist=[edge],
                                         edge_color=self.colors['incorrect'],
                                         width=3, arrowsize=25,
                                         connectionstyle="arc3,rad=0.1",
                                         arrowstyle='-|>', ax=ax)
        
        # Draw labels with better styling
        labels = {node: node for node in all_nodes}
        nx.draw_networkx_labels(G, pos, labels, font_size=10, 
                              font_weight='bold', font_color='black', ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.axis('off')
        
        # Add legend for edge colors (only for non-ground truth)
        if not is_ground_truth and len(edges) > 0:
            from matplotlib.lines import Line2D
            legend_elements = []
            if len(edges & self.ground_truth) > 0:
                legend_elements.append(Line2D([0], [0], color=self.colors['correct'], 
                                            lw=3, label='Correct edges'))
            if len(edges - self.ground_truth) > 0:
                legend_elements.append(Line2D([0], [0], color=self.colors['incorrect'], 
                                            lw=3, label='Incorrect edges'))
            if legend_elements:
                ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    def _get_node_positions(self, nodes):
        """Get consistent node positions for layout"""
        # Predefined positions for smart room variables
        predefined_pos = {
            'Temperature': (0, 1),
            'temperature': (0, 1),
            'Humidity': (0, 0), 
            'humidity': (0, 0),
            'AirQuality': (0, -1),
            'airquality': (0, -1),
            'EnergyConsumption': (2, 0.5),
            'energyconsumption': (2, 0.5),
            'OverallSatisfaction': (2, -0.5),
            'overallsatisfaction': (2, -0.5),
            'weight': (1, 1)
        }
        
        pos = {}
        for node in nodes:
            if node in predefined_pos:
                pos[node] = predefined_pos[node]
            else:
                # Use spring layout for unknown nodes
                G_temp = nx.Graph()
                G_temp.add_nodes_from(nodes)
                spring_pos = nx.spring_layout(G_temp, seed=42)
                pos.update(spring_pos)
                break
        
        return pos

def load_data_from_json(json_path: str) -> Tuple[List, Dict]:
    """Load ground truth and method DAGs from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    ground_truth = data.get('ground_truth', [])
    method_dags = {k: v for k, v in data.items() if k != 'ground_truth'}
    
    return ground_truth, method_dags

def main():
    parser = argparse.ArgumentParser(description='Analyze and visualize causal discovery results')
    parser.add_argument('--json_file', type=str, required=True,
                       help='Path to JSON file containing ground truth and method DAGs')
    parser.add_argument('--output_dir', type=str, default='analysis_output',
                       help='Output directory for all visualizations')
    parser.add_argument('--csv_output', type=str, default='metrics_summary.csv',
                       help='Output path for CSV metrics')
    
    args = parser.parse_args()
    
    # Load data
    ground_truth, method_dags = load_data_from_json(args.json_file)
    
    # Initialize analyzer
    analyzer = CausalMetricsAnalyzer(ground_truth, method_dags)
    
    # Generate comprehensive analysis
    df_results = analyzer.analyze_all_methods()
    
    # Print summary
    print("Causal Discovery Analysis Summary")
    print("=" * 50)
    print(df_results.to_string(index=False))
    print()
    
    # Find best performing method
    best_method = df_results.loc[df_results['f1_score'].idxmax(), 'method']
    best_shd = df_results.loc[df_results['shd'].idxmin(), 'shd']
    print(f"Best F1 Score: {best_method}")
    print(f"Lowest SHD: {best_shd}")
    
    # Save CSV
    df_results.to_csv(args.csv_output, index=False)
    print(f"Metrics saved to: {args.csv_output}")
    
    # Create all visualizations
    analyzer.create_all_visualizations(args.output_dir)

if __name__ == "__main__":
    # Example usage if run directly
    if len(os.sys.argv) == 1:
        # Demo data for testing
        demo_data = {
            "ground_truth": [
                ["Temperature", "EnergyConsumption"],
                ["Humidity", "EnergyConsumption"], 
                ["AirQuality", "EnergyConsumption"],
                ["Temperature", "OverallSatisfaction"],
                ["Humidity", "OverallSatisfaction"],
                ["AirQuality", "OverallSatisfaction"],
                ["Temperature", "Humidity"],
                ["Humidity", "AirQuality"]
            ],
            "final": [
                ["Humidity", "EnergyConsumption"],
                ["Temperature", "EnergyConsumption"], 
                ["AirQuality", "OverallSatisfaction"],
                ["Humidity", "AirQuality"],
                ["Humidity", "OverallSatisfaction"],
                ["Temperature", "OverallSatisfaction"],
                ["Temperature", "Humidity"],
                ["AirQuality", "EnergyConsumption"]
            ],
            "pc": [
                ["Humidity", "AirQuality"],
                ["Humidity", "OverallSatisfaction"],
                ["Temperature", "OverallSatisfaction"],
                ["Humidity", "EnergyConsumption"],
                ["Temperature", "EnergyConsumption"],
                ["Temperature", "Humidity"]
            ],
            "sam": [
                ["Temperature", "OverallSatisfaction"],
                ["Humidity", "OverallSatisfaction"],
                ["Humidity", "Temperature"]
            ],
            "llm": [
                ["Humidity", "OverallSatisfaction"],
                ["Temperature", "OverallSatisfaction"],
                ["AirQuality", "OverallSatisfaction"],
                ["Humidity", "EnergyConsumption"],
                ["Temperature", "EnergyConsumption"],
                ["AirQuality", "EnergyConsumption"]
            ]
        }
        
        # Save demo data
        with open('demo_results.json', 'w') as f:
            json.dump(demo_data, f, indent=2)
        
        # Analyze demo data
        analyzer = CausalMetricsAnalyzer(demo_data['ground_truth'], {k: v for k, v in demo_data.items() if k != 'ground_truth'})
        analyzer.create_all_visualizations('physical_results_15iters')
        
        print("Demo analysis completed! Check 'physical_results_15iters' folder")
    else:
        main()