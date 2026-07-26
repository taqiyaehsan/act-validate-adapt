#!/usr/bin/env python3
"""
Smart Building Causal Discovery Visualization Script
Computes basic metrics (SHD, precision, recall, F1) and visualizes DAGs
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Tuple, Set
import argparse
import os

class SmartBuildingAnalyzer:
    def __init__(self, ground_truth_edges: List[Tuple], method_dags: Dict):
        self.ground_truth = set(self._normalize_edges(ground_truth_edges))
        self.method_dags = {
            method: set(self._normalize_edges(edges)) 
            for method, edges in method_dags.items()
        }
        
        # Smart building node colors
        self.node_colors = {
            'temperature': '#FF6B35', 'humidity': '#4ECDC4', 'airquality': '#45B7D1',
            'hvacsetpoint': '#FF9FF3', 'lightinglevel': '#FECA57', 'occupantcount': '#54A0FF',
            'energyconsumption': '#5F27CD', 'overallsatisfaction': '#A55EEA', 
            'thermalcomfort': '#00D2D3', 'visualcomfort': '#96CEB4', 'airqualityindex': '#FD79A8',
            'hvacpower': '#E67E22', 'lightingpower': '#9B59B6'
        }
    
    def _normalize_edges(self, edges: List) -> List[Tuple]:
        return [tuple(edge) if isinstance(edge, list) else edge for edge in edges]
    
    def calculate_metrics(self, predicted_edges: Set[Tuple]) -> Dict[str, float]:
        """Calculate SHD, precision, recall, F1"""
        if not predicted_edges and not self.ground_truth:
            return {'precision': 1.0, 'recall': 1.0, 'f1_score': 1.0, 'shd': 0}
        
        if not predicted_edges:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'shd': len(self.ground_truth)}
        
        if not self.ground_truth:
            return {'precision': 0.0, 'recall': 1.0, 'f1_score': 0.0, 'shd': len(predicted_edges)}
        
        true_positives = len(predicted_edges & self.ground_truth)
        false_positives = len(predicted_edges - self.ground_truth)
        false_negatives = len(self.ground_truth - predicted_edges)
        
        precision = true_positives / len(predicted_edges)
        recall = true_positives / len(self.ground_truth)
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        shd = false_positives + false_negatives
        
        # Risk and cost calculations
        risk = len(predicted_edges - self.ground_truth) / len(predicted_edges) if predicted_edges else 0.0
        cost = len(predicted_edges - self.ground_truth) / len(self.ground_truth) if self.ground_truth else 0.0
        
        return {
            'precision': precision,
            'recall': recall, 
            'f1_score': f1_score,
            'shd': shd,
            'risk': risk,
            'cost': cost,
            'num_edges': len(predicted_edges),
            'correct_edges': true_positives
        }
    
    def analyze_all_methods(self) -> pd.DataFrame:
        """Analyze all methods and return metrics"""
        results = []
        for method, edges in self.method_dags.items():
            metrics = self.calculate_metrics(edges)
            metrics['method'] = method
            results.append(metrics)
        return pd.DataFrame(results)
    
    def create_core_metrics_chart(self, save_path: str):
        """Create core metrics bar chart"""
        df = self.analyze_all_methods()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metrics = ['precision', 'recall', 'f1_score']
        x = np.arange(len(df))
        width = 0.25
        colors = ['#4C72B0', '#55A868', '#C44E52']
        
        for i, metric in enumerate(metrics):
            bars = ax.bar(x + i*width, df[metric], width, label=metric.capitalize(), color=colors[i])
            for bar, value in zip(bars, df[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Score')
        ax.set_title('Core Performance Metrics')
        ax.set_xticks(x + width)
        ax.set_xticklabels(df['method'])
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Core metrics chart saved to: {save_path}")
    
    def create_shd_chart(self, save_path: str):
        """Create SHD comparison chart"""
        df = self.analyze_all_methods()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        bars = ax.bar(df['method'], df['shd'], color='#F39C12')
        for bar, value in zip(bars, df['shd']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{int(value)}', ha='center', va='bottom')
        
        ax.set_xlabel('Method')
        ax.set_ylabel('SHD')
        ax.set_title('Structural Hamming Distance Comparison')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"SHD chart saved to: {save_path}")
    
    def create_risk_cost_chart(self, save_path: str):
        """Create risk and cost analysis chart"""
        df = self.analyze_all_methods()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(df))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, df['risk'], width, label='Risk', color='salmon', alpha=0.7)
        bars2 = ax.bar(x + width/2, df['cost'], width, label='Cost', color='skyblue', alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars1, df['risk']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        for bar, value in zip(bars2, df['cost']):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Score')
        ax.set_title('Risk and Cost Analysis')
        ax.set_xticks(x)
        ax.set_xticklabels(df['method'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Risk/cost chart saved to: {save_path}")
    
    def create_summary_table(self, save_path: str):
        """Create summary metrics table"""
        df = self.analyze_all_methods()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = df[['method', 'precision', 'recall', 'f1_score', 'shd', 'risk', 'cost', 'num_edges', 'correct_edges']].round(3)
        table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.set_title('Comprehensive Metrics Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Summary table saved to: {save_path}")
        """Create summary metrics table"""
        df = self.analyze_all_methods()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')
        
        table_data = df[['method', 'precision', 'recall', 'f1_score', 'shd', 'risk', 'cost', 'num_edges', 'correct_edges']].round(3)
        table = ax.table(cellText=table_data.values, colLabels=table_data.columns,
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.set_title('Comprehensive Metrics Summary', fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Summary table saved to: {save_path}")
    
    def create_individual_dag(self, method: str, edges: Set[Tuple], save_path: str, is_ground_truth: bool = False):
        """Create individual DAG visualization with legend"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        G = nx.DiGraph()
        G.add_edges_from(edges)
        
        # Add isolated nodes from ground truth
        all_nodes = set()
        for edge in self.ground_truth:
            all_nodes.update(edge)
        for edge in edges:
            all_nodes.update(edge)
        G.add_nodes_from(all_nodes)
        
        # Layout - use spring layout for better spacing
        pos = nx.spring_layout(G, seed=42, k=2.5, iterations=100)
        
        # Node colors
        node_colors = [self.node_colors.get(node.lower(), '#CCCCCC') for node in G.nodes()]
        
        # Draw nodes with larger size and borders
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=2000, 
                              node_color=node_colors, edgecolors='black', linewidths=3)
        
        # Draw edges with color coding
        if is_ground_truth:
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', 
                                  width=3, arrowsize=30, arrowstyle='-|>',
                                  connectionstyle='arc3,rad=0.1')
            title = "Ground Truth DAG"
        else:
            correct_edges = [e for e in edges if e in self.ground_truth]
            wrong_edges = [e for e in edges if e not in self.ground_truth]
            
            if correct_edges:
                nx.draw_networkx_edges(G, pos, edgelist=correct_edges, ax=ax,
                                      edge_color='#2ecc71', width=3, arrowsize=30, 
                                      arrowstyle='-|>', connectionstyle='arc3,rad=0.1')
            if wrong_edges:
                nx.draw_networkx_edges(G, pos, edgelist=wrong_edges, ax=ax,
                                      edge_color='#e74c3c', width=3, arrowsize=30, 
                                      arrowstyle='-|>', connectionstyle='arc3,rad=0.1')
            
            correct = len(correct_edges)
            total = len(edges)
            title = f"{method.upper()} DAG: {correct}/{total} correct edges"
            
            # Add legend for non-ground truth DAGs
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], color='#2ecc71', lw=3, label='Correct edges'),
                Line2D([0], [0], color='#e74c3c', lw=3, label='Incorrect edges')
            ]
            ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        # Labels with larger font
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight='bold')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"{method} DAG saved to: {save_path}")
    
    def _plot_dag(self, ax, edges: Set[Tuple], title: str, is_ground_truth: bool = False):
        """Plot a single DAG"""
        G = nx.DiGraph()
        G.add_edges_from(edges)
        
        # Add isolated nodes from ground truth
        all_nodes = set()
        for edge in self.ground_truth:
            all_nodes.update(edge)
        for edge in edges:
            all_nodes.update(edge)
        G.add_nodes_from(all_nodes)
        
        # Layout
        pos = nx.spring_layout(G, seed=42, k=1, iterations=50)
        
        # Node colors
        node_colors = [self.node_colors.get(node.lower(), '#CCCCCC') for node in G.nodes()]
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=800, 
                              node_color=node_colors, edgecolors='black')
        
        # Draw edges with color coding
        if is_ground_truth:
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='black', 
                                  width=2, arrowsize=20)
        else:
            correct_edges = [e for e in edges if e in self.ground_truth]
            wrong_edges = [e for e in edges if e not in self.ground_truth]
            
            nx.draw_networkx_edges(G, pos, edgelist=correct_edges, ax=ax,
                                  edge_color='green', width=2, arrowsize=20)
            nx.draw_networkx_edges(G, pos, edgelist=wrong_edges, ax=ax,
                                  edge_color='red', width=2, arrowsize=20)
        
        # Labels
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_weight='bold')
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
    
    def create_all_visualizations(self, output_dir: str = "benchmark_results"):
        """Create all visualizations separately"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Core metrics chart
        self.create_core_metrics_chart(os.path.join(output_dir, "core_metrics.png"))
        
        # 2. SHD comparison
        self.create_shd_chart(os.path.join(output_dir, "shd_comparison.png"))
        
        # 3. Risk and cost analysis
        self.create_risk_cost_chart(os.path.join(output_dir, "risk_cost_analysis.png"))
        
        # 4. Summary table
        self.create_summary_table(os.path.join(output_dir, "summary_table.png"))
        
        # 5. Ground truth DAG
        self.create_individual_dag("ground_truth", self.ground_truth, 
                                  os.path.join(output_dir, "ground_truth_dag.png"), 
                                  is_ground_truth=True)
        
        # 5. Individual method DAGs
        for method, edges in self.method_dags.items():
            self.create_individual_dag(method, edges, 
                                      os.path.join(output_dir, f"{method}_dag.png"))
        
        # 6. Export metrics to CSV
        df = self.analyze_all_methods()
        csv_path = os.path.join(output_dir, "metrics_summary.csv")
        df.to_csv(csv_path, index=False)
        print(f"Metrics exported to: {csv_path}")
        
        print(f"All outputs saved to: {output_dir}")
        return output_dir

def load_data_from_json(json_path: str = 'benchmark_results/smart_dags.json') -> Tuple[List, Dict]:
    """Load ground truth and method DAGs from JSON file"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    ground_truth = data.get('ground_truth', [])
    method_dags = {k: v for k, v in data.items() if k != 'ground_truth'}
    
    return ground_truth, method_dags

def main():
    # Hardcoded paths
    json_file = 'benchmark_results/smart_dags.json'
    output_dir = 'benchmark_results'
    
    # Load data
    ground_truth, method_dags = load_data_from_json(json_file)
    
    # Initialize analyzer
    analyzer = SmartBuildingAnalyzer(ground_truth, method_dags)
    
    # Generate analysis
    df_results = analyzer.analyze_all_methods()
    print("Metrics Summary:")
    print(df_results.to_string(index=False))
    
    # Create visualizations
    analyzer.create_all_visualizations(output_dir)

if __name__ == "__main__":
    main()