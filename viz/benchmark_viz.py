import json
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
import numpy as np
import random
from matplotlib.gridspec import GridSpec

# File paths
csv_path = "benchmark_results/benchmark_results.csv"
json_path = "benchmark_results/raw_results.json"
output_dir = "benchmark_results"

# Create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Define consistent colors - matches benchmark_framework.py
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

# Load data
csv_data = pd.read_csv(csv_path)
with open(json_path, 'r') as f:
    json_data = json.load(f)

# Extract metric info
methods = list(json_data.keys())
metrics = ['precision', 'recall', 'f1_score']
runtime = []
shd = []

performance_data = {metric: [] for metric in metrics}
for method in methods:
    for metric in metrics:
        performance_data[metric].append(json_data[method]['metrics'][metric])
    runtime.append(json_data[method]['runtime'])
    shd.append(json_data[method]['metrics']['shd'])

# Add bootstrap confidence interval calculation
def bootstrap_metrics(json_data, n_bootstrap=1000, confidence=0.95):
    """Calculate confidence intervals using bootstrap sampling"""
    confidence_intervals = {}
    
    for method in json_data:
        # Skip methods with no edges
        if not json_data[method].get("normalized_edges", []):
            confidence_intervals[method] = {
                'precision': (0, 0),
                'recall': (0, 0),
                'f1_score': (0, 0),
                'shd': (0, 0)
            }
            continue
        
        # Get normalized edges
        normalized_edges = json_data[method].get("normalized_edges", [])
        
        # Get ground truth edges
        ground_truth_edges = [
            ["temperature", "energyconsumption"],
            ["humidity", "energyconsumption"],
            ["airquality", "energyconsumption"],
            ["temperature", "overallsatisfaction"],
            ["humidity", "overallsatisfaction"],
            ["airquality", "overallsatisfaction"]
        ]
        ground_truth_edges = set(tuple(edge) for edge in ground_truth_edges)
        
        # Bootstrap samples
        precision_samples = []
        recall_samples = []
        f1_samples = []
        shd_samples = []
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            if len(normalized_edges) > 0:
                bootstrap_edges = random.choices(normalized_edges, k=len(normalized_edges))
                bootstrap_edges = set(tuple(edge) for edge in bootstrap_edges)
                
                # Calculate metrics
                true_pos = len(bootstrap_edges.intersection(ground_truth_edges))
                precision = true_pos / max(len(bootstrap_edges), 1)
                recall = true_pos / len(ground_truth_edges)
                f1 = 2 * (precision * recall) / max(precision + recall, 1e-10)
                
                # Calculate SHD
                missing = len(ground_truth_edges - bootstrap_edges)
                extra = len(bootstrap_edges - ground_truth_edges)
                shd = missing + extra
                
                precision_samples.append(precision)
                recall_samples.append(recall)
                f1_samples.append(f1)
                shd_samples.append(shd)
            else:
                precision_samples.append(0)
                recall_samples.append(0)
                f1_samples.append(0)
                shd_samples.append(len(ground_truth_edges))
        
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
                min(len(ground_truth_edges) * 2, np.percentile(shd_samples, (1 - alpha) * 100))
            )
        }
    
    return confidence_intervals

# Calculate confidence intervals
confidence_intervals = bootstrap_metrics(json_data)

# 1. Performance Metrics Plot with confidence intervals
x = range(len(methods))
bar_width = 0.2
plt.figure(figsize=(10, 6))

for i, metric in enumerate(metrics):
    values = performance_data[metric]
    
    # Calculate error bars from confidence intervals
    yerr_low = []
    yerr_high = []
    for j, m in enumerate(methods):
        ci = confidence_intervals.get(m, {}).get(metric, (values[j], values[j]))
        yerr_low.append(max(0, values[j] - ci[0]))  # Ensure non-negative
        yerr_high.append(max(0, ci[1] - values[j]))  # Ensure non-negative
    
    yerr = [yerr_low, yerr_high]
    
    plt.bar([p + bar_width*i for p in x], values, width=bar_width, label=metric, color=COLORS[metric])
    plt.errorbar([p + bar_width*i for p in x], values, yerr=yerr, fmt='none', capsize=5, color='black')

plt.xticks([p + bar_width for p in x], methods, rotation=45, ha="right")
plt.ylabel("Performance Score")
plt.title("Performance Metrics with 95% Confidence Intervals")
plt.legend()
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig(os.path.join(output_dir, "performance_metrics.png"))
plt.close()

# 2. Runtime Plot with proper y-axis label
plt.figure(figsize=(10, 6))
plt.bar(methods, runtime, color=COLORS['runtime'])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Runtime (seconds)")
plt.title("Runtime by Method")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "runtime.png"))
plt.close()

# 3. SHD Plot with confidence intervals
plt.figure(figsize=(10, 6))

# Calculate error bars for SHD
yerr_low = []
yerr_high = []
for j, m in enumerate(methods):
    ci = confidence_intervals.get(m, {}).get('shd', (shd[j], shd[j]))
    yerr_low.append(max(0, shd[j] - ci[0]))  # Ensure non-negative
    yerr_high.append(max(0, ci[1] - shd[j]))  # Ensure non-negative

yerr = [yerr_low, yerr_high]

plt.bar(methods, shd, color=COLORS['shd'])
plt.errorbar(methods, shd, yerr=yerr, fmt='none', capsize=5, color='black')

plt.xticks(rotation=45, ha="right")
plt.ylabel("Structural Hamming Distance")
plt.title("Structural Hamming Distance (SHD) with 95% Confidence Intervals")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shd.png"))
plt.close()

# 4. Save individual DAGs for each method
for method in methods:
    plt.figure(figsize=(8, 6))
    
    edges = json_data[method]["normalized_edges"]
    G = nx.DiGraph()
    
    # Add nodes and edges
    nodes = set()
    for edge in edges:
        source, target = edge
        nodes.add(source.lower())
        nodes.add(target.lower())
        G.add_edge(source.lower(), target.lower())
    
    # Create layout
    pos = nx.spring_layout(G, seed=42, k=0.5)
    
    # Draw nodes with specific colors
    node_color_map = []
    for node in G.nodes():
        found_color = False
        for key in COLORS['node_colors']:
            if key.lower() == node:
                node_color_map.append(COLORS['node_colors'][key])
                found_color = True
                break
        if not found_color:
            node_color_map.append('#CCCCCC')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=node_color_map, edgecolors='black')
    
    # Draw edges with curved arrows
    for edge in G.edges():
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[edge], 
            width=2.0,
            arrowsize=20,
            arrowstyle='-|>', 
            connectionstyle='arc3,rad=0.1',
            edge_color='black',
            min_source_margin=20,
            min_target_margin=20
        )
    
    # Draw labels
    # nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
    plt.title(f"DAG: {method}")
    plt.axis('off')
    
    # Add metrics in the title
    metrics_text = f"Precision: {json_data[method]['metrics']['precision']:.2f}, Recall: {json_data[method]['metrics']['recall']:.2f}, SHD: {json_data[method]['metrics']['shd']}"
    plt.figtext(0.5, 0.01, metrics_text, ha="center", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"dag_{method}.png"), bbox_inches='tight')
    plt.close()

# 5. Combined DAG Visualization with ground truth
n_methods = len(methods) + 1  # +1 for ground truth
n_cols = min(3, n_methods)
n_rows = (n_methods + n_cols - 1) // n_cols

fig = plt.figure(figsize=(n_cols * 5, n_rows * 5))
gs = GridSpec(n_rows, n_cols, figure=fig)

# First, create the ground truth DAG
ground_truth_edges = [
    ["temperature", "energyconsumption"],
    ["humidity", "energyconsumption"],
    ["airquality", "energyconsumption"],
    ["temperature", "overallsatisfaction"],
    ["humidity", "overallsatisfaction"],
    ["airquality", "overallsatisfaction"]
]

G_truth = nx.DiGraph()
for edge in ground_truth_edges:
    source, target = edge
    G_truth.add_edge(source.lower(), target.lower())

# Draw ground truth DAG
ax_truth = fig.add_subplot(gs[0, 0])
pos_truth = nx.spring_layout(G_truth, seed=42, k=0.5)

# Draw ground truth nodes
node_color_map_truth = []
for node in G_truth.nodes():
    for key in COLORS['node_colors']:
        if key.lower() == node:
            node_color_map_truth.append(COLORS['node_colors'][key])
            break
    else:
        node_color_map_truth.append('#CCCCCC')

nx.draw_networkx_nodes(G_truth, pos_truth, ax=ax_truth, node_size=1000, 
                     node_color=node_color_map_truth, edgecolors='black')

# Draw ground truth edges
for edge in G_truth.edges():
    nx.draw_networkx_edges(
        G_truth, pos_truth,
        edgelist=[edge], 
        width=2.0,
        arrowsize=20,
        arrowstyle='-|>', 
        connectionstyle='arc3,rad=0.1',
        edge_color='black',
        min_source_margin=20,
        min_target_margin=20,
        ax=ax_truth
    )

# Draw labels
nx.draw_networkx_labels(G_truth, pos_truth, ax=ax_truth, font_size=10, font_weight='bold')

ax_truth.set_title("Ground Truth DAG")
ax_truth.axis('off')

# Now draw each method's DAG
for i, method in enumerate(methods):
    # Calculate position in grid
    row = (i + 1) // n_cols
    col = (i + 1) % n_cols
    
    ax = fig.add_subplot(gs[row, col])
    
    edges = json_data[method]["normalized_edges"]
    G = nx.DiGraph()
    
    # Add nodes and edges
    nodes = set()
    for edge in edges:
        source, target = edge
        nodes.add(source.lower())
        nodes.add(target.lower())
        G.add_edge(source.lower(), target.lower())
    
    # Create layout
    pos = nx.spring_layout(G, seed=42, k=0.5)
    
    # Draw nodes with specific colors
    node_color_map = []
    for node in G.nodes():
        for key in COLORS['node_colors']:
            if key.lower() == node:
                node_color_map.append(COLORS['node_colors'][key])
                break
        else:
            node_color_map.append('#CCCCCC')
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=1000, node_color=node_color_map, 
                          edgecolors='black', ax=ax)
    
    # Draw edges with curved arrows
    for edge in G.edges():
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
    
    # Draw labels
    # nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')
    
    ax.set_title(f"Learned DAG: {method}")
    ax.axis('off')

# Add a single legend at the bottom of the figure
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=color, 
                             markersize=15, label=label) 
                  for label, color in COLORS['node_colors'].items()]

fig.legend(handles=legend_elements, loc='lower center', 
          bbox_to_anchor=(0.5, 0), ncol=5, fontsize=12)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout for legend
plt.savefig(os.path.join(output_dir, "all_dags_comparison.png"), bbox_inches='tight')
plt.close()