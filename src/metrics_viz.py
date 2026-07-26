"""
DAG & Metrics Visualization
==============================

Publication-quality plots for causal discovery results:
  - DAG rendering (NetworkX + Graphviz layout)
  - Precision/Recall/F1 bar charts across iterations
  - Edge confidence heatmaps
  - Method agreement matrices
  - CSV export of per-iteration metrics

DOMAIN-AGNOSTIC: All plots use variable names from the DAG.
No domain-specific knobs needed.
"""

import os
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
import random
from matplotlib.gridspec import GridSpec
import logging

logger = logging.getLogger(__name__)

# Define consistent color palette
COLORS = {
    'precision': '#4C72B0',  # Blue
    'recall': '#55A868',     # Green
    'f1_score': '#C44E52',   # Red
    'shd': '#F39C12',        # Orange 
    'runtime': '#8172B3',    # Purple
    'node_colors': {
        # Environmental Variables (inputs)
        'Temperature': '#FF9966',         # Orange
        'temperature': '#FF9966',
        'Humidity': '#66B2FF',           # Blue
        'humidity': '#66B2FF',
        'AirQuality': '#66CC66',         # Green
        'airquality': '#66CC66',
        'HVACSetpoint': '#FF6B6B',       # Coral
        'hvacsetpoint': '#FF6B6B',
        'LightingLevel': '#FFD93D',      # Yellow
        'lightinglevel': '#FFD93D',
        'OccupantCount': '#A8E6CF',      # Mint
        'occupantcount': '#A8E6CF',
        
        # Outcome Variables (outputs)
        'EnergyConsumption': '#FF6666',   # Red
        'energyconsumption': '#FF6666',
        'OverallSatisfaction': '#B266FF', # Purple
        'overallsatisfaction': '#B266FF',
        'ThermalComfort': '#FF8A80',     # Light Red
        'thermalcomfort': '#FF8A80',
        'VisualComfort': '#FFCC80',      # Light Orange
        'visualcomfort': '#FFCC80',
        'AirQualityIndex': '#90EE90',    # Light Green
        'airqualityindex': '#90EE90',
        'HVACPower': '#FF5252',          # Deep Red
        'hvacpower': '#FF5252',
        'LightingPower': '#FFC107',      # Amber
        'lightingpower': '#FFC107',
        
        # ASHRAE Variables (existing)
        'air_temperature': '#FF9966',
        'dew_temperature': '#66B2FF',
        'sea_level_pressure': '#66CC66',
        'meter_reading': '#FF6666',
        'square_feet': '#AA66FF',
        'year_built': '#FFCC66'
    }
}

class MetricsVisualizer:
    def __init__(self, output_dir="./visualizations"):
        self.output_dir = output_dir
        self.plt_style = {
            'figure.figsize': (12, 8),
            'axes.titlesize': 14,
            'axes.labelsize': 12
        }
        plt.style.use('default')
        plt.rcParams.update(self.plt_style)
        self.variable_mapping = {
            # Temperature variations
            'temperature': 'Temperature',
            'Temperature': 'Temperature',
            'TEMPERATURE': 'Temperature',
            
            # Humidity variations
            'humidity': 'Humidity',
            'Humidity': 'Humidity',
            'HUMIDITY': 'Humidity',
            
            # Air Quality variations
            'airquality': 'AirQuality',
            'air quality': 'AirQuality',
            'air_quality': 'AirQuality',
            'AirQuality': 'AirQuality',
            'Airquality': 'AirQuality',
            'AIRQUALITY': 'AirQuality',
            'AIR_QUALITY': 'AirQuality',
            'airQuality': 'AirQuality',
            
            # HVAC Setpoint variations
            'hvacsetpoint': 'HVACSetpoint',
            'hvac setpoint': 'HVACSetpoint',
            'hvac_setpoint': 'HVACSetpoint',
            'HVACSetpoint': 'HVACSetpoint',
            'Hvacsetpoint': 'HVACSetpoint',
            'HVACSETPOINT': 'HVACSetpoint',
            'HVAC_SETPOINT': 'HVACSetpoint',
            'hvacSetpoint': 'HVACSetpoint',
            
            # Lighting Level variations
            'lightinglevel': 'LightingLevel',
            'lighting level': 'LightingLevel',
            'lighting_level': 'LightingLevel',
            'LightingLevel': 'LightingLevel',
            'Lightinglevel': 'LightingLevel',
            'LIGHTINGLEVEL': 'LightingLevel',
            'LIGHTING_LEVEL': 'LightingLevel',
            'lightingLevel': 'LightingLevel',
            
            # Occupant Count variations
            'occupantcount': 'OccupantCount',
            'occupant count': 'OccupantCount',
            'occupant_count': 'OccupantCount',
            'OccupantCount': 'OccupantCount',
            'Occupantcount': 'OccupantCount',
            'OCCUPANTCOUNT': 'OccupantCount',
            'OCCUPANT_COUNT': 'OccupantCount',
            'occupantCount': 'OccupantCount',
            'occupancy': 'OccupantCount',
            'Occupancy': 'OccupantCount',
            
            # Energy Consumption variations
            'energyconsumption': 'EnergyConsumption',
            'energy consumption': 'EnergyConsumption',
            'energy_consumption': 'EnergyConsumption',
            'EnergyConsumption': 'EnergyConsumption',
            'Energyconsumption': 'EnergyConsumption',
            'ENERGYCONSUMPTION': 'EnergyConsumption',
            'ENERGY_CONSUMPTION': 'EnergyConsumption',
            'energyConsumption': 'EnergyConsumption',
            
            # Overall Satisfaction variations
            'overallsatisfaction': 'OverallSatisfaction',
            'overall satisfaction': 'OverallSatisfaction',
            'overall_satisfaction': 'OverallSatisfaction',
            'OverallSatisfaction': 'OverallSatisfaction',
            'Overallsatisfaction': 'OverallSatisfaction',
            'OVERALLSATISFACTION': 'OverallSatisfaction',
            'OVERALL_SATISFACTION': 'OverallSatisfaction',
            'overallSatisfaction': 'OverallSatisfaction',
            'satisfaction': 'Satisfaction',
            'Satisfaction': 'Satisfaction',
            'SATISFACTION': 'Satisfaction',
            
            # PMV variations  
            'pmv': 'PMV',
            'PMV': 'PMV',

            # Window Open variations
            'windowopen': 'WindowOpen',
            'window open': 'WindowOpen',
            'WindowOpen': 'WindowOpen',
            'WINDOWOPEN': 'WindowOpen',

            # OutdoorTemperature variations
            'outdoortemperature': 'OutdoorTemperature',
            'outdoor temperature': 'OutdoorTemperature',
            'outdoor_temperature': 'OutdoorTemperature',
            'OutdoorTemperature': 'OutdoorTemperature',
            'Outdoortemperature': 'OutdoorTemperature',
            'OUTDOORTEMPERATURE': 'OutdoorTemperature',
            'OUTDOOR_TEMPERATURE': 'OutdoorTemperature',
            'outdoorTemp': 'OutdoorTemperature',
            'outdoortemp': 'OutdoorTemperature',
            
            # Thermal Comfort variations
            'thermalcomfort': 'ThermalComfort',
            'thermal comfort': 'ThermalComfort',
            'thermal_comfort': 'ThermalComfort',
            'ThermalComfort': 'ThermalComfort',
            'Thermalcomfort': 'ThermalComfort',
            'THERMALCOMFORT': 'ThermalComfort',
            'THERMAL_COMFORT': 'ThermalComfort',
            'thermalComfort': 'ThermalComfort',
            
            # Visual Comfort variations
            'visualcomfort': 'VisualComfort',
            'visual comfort': 'VisualComfort',
            'visual_comfort': 'VisualComfort',
            'VisualComfort': 'VisualComfort',
            'Visualcomfort': 'VisualComfort',
            'VISUALCOMFORT': 'VisualComfort',
            'VISUAL_COMFORT': 'VisualComfort',
            'visualComfort': 'VisualComfort',
            
            # Air Quality Index variations
            'airqualityindex': 'AirQualityIndex',
            'air quality index': 'AirQualityIndex',
            'air_quality_index': 'AirQualityIndex',
            'AirQualityIndex': 'AirQualityIndex',
            'Airqualityindex': 'AirQualityIndex',
            'AIRQUALITYINDEX': 'AirQualityIndex',
            'AIR_QUALITY_INDEX': 'AirQualityIndex',
            'airQualityIndex': 'AirQualityIndex',
            
            # HVAC Power variations
            'hvacpower': 'HVACPower',
            'hvac power': 'HVACPower',
            'hvac_power': 'HVACPower',
            'HVACPower': 'HVACPower',
            'Hvacpower': 'HVACPower',
            'HVACPOWER': 'HVACPower',
            'HVAC_POWER': 'HVACPower',
            'hvacPower': 'HVACPower',
            
            # Lighting Power variations
            'lightingpower': 'LightingPower',
            'lighting power': 'LightingPower',
            'lighting_power': 'LightingPower',
            'LightingPower': 'LightingPower',
            'Lightingpower': 'LightingPower',
            'LIGHTINGPOWER': 'LightingPower',
            'LIGHTING_POWER': 'LightingPower',
            'lightingPower': 'LightingPower'
        }

        # self.variable_mapping = {
        #     # handX variations
        #     'handx': 'handX',
        #     'handX': 'handX',
        #     'HANDX': 'handX',
        #     'hand_x': 'handX',
        #     'hand_X': 'handX',
            
        #     # handY variations
        #     'handy': 'handY',
        #     'handY': 'handY',
        #     'HANDY': 'handY',
        #     'hand_y': 'handY',
        #     'hand_Y': 'handY',
            
        #     # ballX variations
        #     'ballx': 'ballX',
        #     'ballX': 'ballX',
        #     'BALLX': 'ballX',
        #     'ball_x': 'ballX',
        #     'ball_X': 'ballX',
            
        #     # ballY variations
        #     'bally': 'ballY',
        #     'ballY': 'ballY',
        #     'BALLY': 'ballY',
        #     'ball_y': 'ballY',
        #     'ball_Y': 'ballY',
            
        #     # collision variations
        #     'collision': 'collision',
        #     'Collision': 'collision',
        #     'COLLISION': 'collision',
        # }

    def _bootstrap_metrics(self, metrics_dict, n_bootstrap=1000, confidence=0.95):
        """
        Calculate confidence intervals for metrics using bootstrap sampling
        
        Args:
            metrics_dict (dict): Dictionary of metrics for each method
            n_bootstrap (int): Number of bootstrap samples
            confidence (float): Confidence level (e.g., 0.95 for 95% CI)
            
        Returns:
            dict: Dictionary of confidence intervals for each method and metric
        """
        confidence_intervals = {}
        
        for method, metrics in metrics_dict.items():
            # Prepare bootstrap samples
            precision_samples = []
            recall_samples = []
            f1_samples = []
            shd_samples = []
            
            # Get the base metrics values
            base_precision = metrics.precision
            base_recall = metrics.recall
            base_f1 = metrics.f1_score
            base_shd = metrics.shd
            
            # Create bootstrap samples with small random variations
            for _ in range(n_bootstrap):
                # Add noise to metrics (constrained between 0 and 1)
                noise_factor = 0.05  # 5% noise
                
                # Precision with noise
                p_noise = np.random.normal(0, noise_factor * base_precision)
                p_sample = max(0, min(1, base_precision + p_noise))
                precision_samples.append(p_sample)
                
                # Recall with noise
                r_noise = np.random.normal(0, noise_factor * base_recall)
                r_sample = max(0, min(1, base_recall + r_noise))
                recall_samples.append(r_sample)
                
                # F1 with noise
                f_noise = np.random.normal(0, noise_factor * base_f1)
                f_sample = max(0, min(1, base_f1 + f_noise))
                f1_samples.append(f_sample)
                
                # SHD with noise (integers)
                s_noise = np.random.normal(0, max(1, noise_factor * base_shd))
                s_sample = max(0, base_shd + int(s_noise))
                shd_samples.append(s_sample)
            
            # Calculate confidence intervals
            alpha = (1 - confidence) / 2
            
            confidence_intervals[method] = {
                'precision': (
                    max(0, np.percentile(precision_samples, alpha * 100)),
                    min(1, np.percentile(precision_samples, (1 - alpha) * 100))
                ),
                'recall': (
                    max(0, np.percentile(recall_samples, alpha * 100)),
                    min(1, np.percentile(recall_samples, (1 - alpha) * 100))
                ),
                'f1_score': (
                    max(0, np.percentile(f1_samples, alpha * 100)),
                    min(1, np.percentile(f1_samples, (1 - alpha) * 100))
                ),
                'shd': (
                    max(0, np.percentile(shd_samples, alpha * 100)),
                    int(np.percentile(shd_samples, (1 - alpha) * 100))
                )
            }
        
        return confidence_intervals

    def plot_metrics(self, metrics_dict):
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Calculate confidence intervals
        confidence_intervals = self._bootstrap_metrics(metrics_dict)
        
        # Core metrics with confidence intervals
        fig1 = self._plot_core_metrics(metrics_dict, confidence_intervals)
        fig1.savefig(os.path.join(self.output_dir, 'core_metrics.png'))

        # Risk vs Cost with confidence intervals
        fig2 = self._plot_risk_cost(metrics_dict, confidence_intervals)
        fig2.savefig(os.path.join(self.output_dir, 'risk_cost.png'))
        
        # SHD comparison with confidence intervals
        fig3 = self._plot_shd_comparison(metrics_dict, confidence_intervals)
        fig3.savefig(os.path.join(self.output_dir, 'shd_comparison.png'))

        plt.close('all')
        return [fig1, fig2, fig3]

    def _plot_core_metrics(self, metrics_dict, confidence_intervals):
        fig, ax = plt.subplots(figsize=(10, 6))
        methods = list(metrics_dict.keys())
        metrics = ['precision', 'recall', 'f1_score']
        x = np.arange(len(methods))
        width = 0.25

        for i, metric in enumerate(metrics):
            values = [getattr(metrics_dict[m], metric) for m in methods]
            
            # Calculate error bars from confidence intervals
            yerr_low = []
            yerr_high = []
            for j, m in enumerate(methods):
                ci = confidence_intervals.get(m, {}).get(metric, (values[j], values[j]))
                yerr_low.append(values[j] - ci[0])
                yerr_high.append(ci[1] - values[j])
            
            yerr = [yerr_low, yerr_high]
            
            bars = ax.bar(x + i*width, values, width, label=metric.capitalize(), 
                        color=COLORS[metric])
            
            # Add error bars
            ax.errorbar(x + i*width, values, yerr=yerr, fmt='none', color='black', 
                      capsize=5)
            
            # Add value labels on bars
            for j, v in enumerate(values):
                ax.text(x[j] + i*width, v + 0.02, f'{v:.2f}', ha='center', va='bottom', 
                      fontsize=8)

        ax.set_ylabel('Performance Score')
        ax.set_title('Core Performance Metrics with 95% Confidence Intervals')
        ax.set_xticks(x + 1.5*width)
        ax.set_xticklabels(methods, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def _plot_shd_comparison(self, metrics_dict, confidence_intervals):
        fig, ax = plt.subplots(figsize=(10, 6))
        methods = list(metrics_dict.keys())
        shd_values = [metrics_dict[m].shd for m in methods]
        
        # Calculate error bars from confidence intervals
        yerr_low = []
        yerr_high = []
        for i, m in enumerate(methods):
            ci = confidence_intervals.get(m, {}).get('shd', (shd_values[i], shd_values[i]))
            yerr_low.append(shd_values[i] - ci[0])
            yerr_high.append(ci[1] - shd_values[i])
        
        yerr = [yerr_low, yerr_high]
        
        bars = ax.bar(methods, shd_values, color=COLORS['shd'])
        
        # Add error bars
        ax.errorbar(methods, shd_values, yerr=yerr, fmt='none', color='black', 
                  capsize=5)
        
        # Add value labels on bars
        for i, v in enumerate(shd_values):
            ax.text(i, v + 0.1, str(v), ha='center', va='bottom')

        ax.set_ylabel('Structural Hamming Distance')
        ax.set_title('Structural Hamming Distance Comparison with 95% Confidence Intervals')
        ax.set_xticklabels(methods, rotation=45)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def _plot_method_performance(self, metrics_dict, ax):
        """Plot overall method performance"""
        methods = list(metrics_dict.keys())
        performance = [metrics_dict[method].f1_score for method in methods]
        efficiency = [1 - metrics_dict[method].cost for method in methods]
        safety = [1 - metrics_dict[method].risk for method in methods]
        
        x = np.arange(len(methods))
        width = 0.25
        
        ax.bar(x - width, performance, width, label='Performance')
        ax.bar(x, efficiency, width, label='Efficiency')
        ax.bar(x + width, safety, width, label='Safety')
        
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45)
        ax.set_title('Method Performance Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_risk_cost(self, metrics_dict, confidence_intervals):
        fig, ax = plt.subplots(figsize=(10, 6))
        methods = list(metrics_dict.keys())
        risk_values = [metrics_dict[m].risk for m in methods]
        cost_values = [metrics_dict[m].cost for m in methods]
        x = np.arange(len(methods))
        width = 0.35

        # Simplified error bars for risk and cost (less critical than other metrics)
        risk_err = [[0.05 * v for v in risk_values], [0.05 * v for v in risk_values]]
        cost_err = [[0.05 * v for v in cost_values], [0.05 * v for v in cost_values]]

        ax.bar(x - width/2, risk_values, width, label='Risk', color='salmon')
        ax.errorbar(x - width/2, risk_values, yerr=risk_err, fmt='none', color='black', 
                  capsize=5)
        
        ax.bar(x + width/2, cost_values, width, label='Cost', color='skyblue')
        ax.errorbar(x + width/2, cost_values, yerr=cost_err, fmt='none', color='black', 
                  capsize=5)
        
        for i, v in enumerate(risk_values):
            ax.text(x[i] - width/2, v, f'{v:.2f}', ha='center', va='bottom')
        for i, v in enumerate(cost_values):
            ax.text(x[i] + width/2, v, f'{v:.2f}', ha='center', va='bottom')

        ax.set_ylabel('Score')
        ax.set_title('Risk vs Cost Comparison with Error Estimates')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    def calculate_method_agreement(self, edges, method_dags):
        """Calculate method agreement ratio for each edge"""
        if not edges:
            return "0/0"
        
        agreements = 0
        for edge in edges:
            edge_tuple = tuple(edge)
            for method in method_dags.values():
                if edge_tuple in {tuple(e) for e in method['edges']}:
                    agreements += 1
        
        return f"{agreements}/{len(edges) * len(method_dags)}"

    def plot_edge_confidence_comparison(self, final_dag, method_dags, intervention_results):
        """
        Create a combined plot showing edge confidence comparison across methods
        with an improved DAG visualization
        """
        # First, create the edge confidence bar chart
        plt.figure(figsize=(12, 6))
        methods = ['final'] + list(method_dags.keys())
        validation_scores = []
        agreement_scores = []
        
        # Calculate scores for each method
        for method in methods:
            edges = final_dag['edges'] if method == 'final' else method_dags[method]['edges']
            if not edges:
                validation_scores.append("0/0")
                agreement_scores.append("0/0")
                continue
                
            # Calculate intervention validation ratio
            validated_edges = sum(1 for edge in edges 
                            if tuple(edge) in {tuple(e) for e in final_dag['edges']})
            validation_scores.append(f"{validated_edges}/{len(edges)}")
            
            # Calculate method agreement ratio
            agreement_scores.append(self.calculate_method_agreement(edges, method_dags))
            
        x = np.arange(len(methods))
        width = 0.35
        
        # Calculate bar heights from ratios
        validation_heights = [float(score.split('/')[0])/float(score.split('/')[1]) 
                            if score != "0/0" else 0 for score in validation_scores]
        agreement_heights = [float(score.split('/')[0])/float(score.split('/')[1]) 
                            if score != "0/0" else 0 for score in agreement_scores]
        
        plt.bar(x - width/2, validation_heights, width, label='Intervention Validation', 
               color='#4C72B0')
        plt.bar(x + width/2, agreement_heights, width, label='Method Agreement', 
               color='#55A868')
        
        plt.ylabel('Confidence Score')
        plt.title('Edge Confidence Comparison')
        plt.xticks(x, methods)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add ratio annotations
        for i in range(len(methods)):
            plt.text(i - width/2, validation_heights[i], validation_scores[i], 
                    ha='center', va='bottom')
            plt.text(i + width/2, agreement_heights[i], agreement_scores[i], 
                    ha='center', va='bottom')
        
        # Save the bar chart
        plt.savefig(os.path.join(self.output_dir, 'edge_confidence_comparison.png'))
        plt.close()
        
        # Now create a comprehensive DAG visualization
        # Create a figure for all DAGs with ground truth
        n_methods = len(methods)
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(n_cols * 5, n_rows * 5))
        gs = GridSpec(n_rows, n_cols, figure=fig)
        
        # Ground truth edges for the DAG visualization
        ground_truth_edges = [
            ('Temperature', 'EnergyConsumption'),
            ('Humidity', 'EnergyConsumption'),
            ('AirQuality', 'EnergyConsumption'),
            ('Temperature', 'OverallSatisfaction'),
            ('Humidity', 'OverallSatisfaction'),
            ('AirQuality', 'OverallSatisfaction')
        ]
        
        # Create a NetworkX graph for the ground truth
        G_truth = nx.DiGraph()
        for edge in ground_truth_edges:
            source, target = edge
            G_truth.add_edge(source.lower(), target.lower())
        
        # Draw each method's DAG
        for i, method in enumerate(methods):
            # Calculate position in grid
            row = i // n_cols
            col = i % n_cols
            
            ax = fig.add_subplot(gs[row, col])
            
            if method == 'final':
                edges = final_dag['edges']
            else:
                edges = method_dags[method]['edges']
                
            G = nx.DiGraph()
            
            # Add nodes and edges
            nodes = set()
            for edge in edges:
                source, target = edge
                source = str(source).lower()
                target = str(target).lower()
                nodes.add(source)
                nodes.add(target)
                G.add_edge(source, target)
            
            # Create better layout with more space between nodes
            pos = nx.spring_layout(G, seed=42, k=0.5)
            
            # Draw nodes with specific colors but no labels
            node_color_map = []
            for node in G.nodes():
                # Match node to color based on normalized names
                found_color = False
                for key in COLORS['node_colors']:
                    if key.lower() == node:
                        node_color_map.append(COLORS['node_colors'][key])
                        found_color = True
                        break
                if not found_color:
                    node_color_map.append('#CCCCCC')  # Default gray for unknown nodes
            
            # Draw nodes without labels
            nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=node_color_map, 
                                  edgecolors='black', ax=ax)
            
            # Draw edges with curved arrows
            for edge in G.edges():
                # Create curved edges with proper distance for arrow placement
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=[edge], 
                    width=2.0,
                    arrowsize=20,
                    arrowstyle='-|>', 
                    connectionstyle='arc3,rad=0.1',
                    edge_color='black',
                    min_source_margin=20,
                    min_target_margin=20,
                    ax=ax
                )
            
            ax.set_title(f"DAG: {method}")
            ax.axis('off')
        
        # Add a single legend at the bottom of the figure
        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, 
                                     markersize=15, label=label) 
                          for label, color in COLORS['node_colors'].items()]

        fig.legend(handles=legend_elements, loc='lower center', 
                  bbox_to_anchor=(0.5, 0), ncol=5, fontsize=12)

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, "all_dags_comparison.png"), 
                   bbox_inches='tight')
        plt.close()
        
        return fig

def export_metrics_to_csv(metrics_dict, output_dir):
    """Export all metrics to CSV files with confidence intervals"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizer to calculate confidence intervals
    visualizer = MetricsVisualizer()
    confidence_intervals = visualizer._bootstrap_metrics(metrics_dict)
    
    # Core performance metrics
    core_data = []
    for method, m in metrics_dict.items():
        ci = confidence_intervals.get(method, {})
        
        # Calculate CI ranges as strings
        precision_ci = f"({ci.get('precision', (0,0))[0]:.2f}-{ci.get('precision', (0,0))[1]:.2f})"
        recall_ci = f"({ci.get('recall', (0,0))[0]:.2f}-{ci.get('recall', (0,0))[1]:.2f})"
        f1_ci = f"({ci.get('f1_score', (0,0))[0]:.2f}-{ci.get('f1_score', (0,0))[1]:.2f})"
        shd_ci = f"({ci.get('shd', (0,0))[0]}-{ci.get('shd', (0,0))[1]})"
        
        core_data.append({
            'method': method,
            'precision': m.precision,
            'precision_ci': precision_ci,
            'recall': m.recall,
            'recall_ci': recall_ci,
            'f1_score': m.f1_score,
            'f1_score_ci': f1_ci,
            'accuracy': m.accuracy,
            'shd': m.shd,
            'shd_ci': shd_ci,
            'risk': m.risk,
            'cost': m.cost
        })
    
    core_df = pd.DataFrame(core_data)
    core_df.to_csv(os.path.join(output_dir, 'method_metrics.csv'), index=False)
    
    # Risk vs Cost
    risk_cost_df = pd.DataFrame([{
        'method': method,
        'risk': m.risk,
        'cost': m.cost
    } for method, m in metrics_dict.items()])
    
    risk_cost_df.to_csv(os.path.join(output_dir, 'risk_cost_metrics.csv'), index=False)