import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import seaborn as sns
import json
import os

# Set the style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Increase font sizes globally
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.titlesize'] = 20

# Custom color palettes
bar_colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']  # Blue, Red, Green, Orange
method_colors = ['#3498db', '#2ecc71', '#e74c3c', '#f1c40f']  # Blue, Green, Red, Yellow

# Load the data
method_metrics = pd.read_csv('results+logs/results_60iters/method_metrics.csv')
risk_cost_metrics = pd.read_csv('results+logs/results_60iters/risk_cost_metrics.csv')

# Color palettes for each dataset
ASHRAE_COLORS = {
    'square_feet': '#E74C3C',        # Red
    'year_built': '#3498DB',         # Blue
    'air_temperature': '#F39C12',    # Orange
    'dew_temperature': '#27AE60',    # Green
    # 'sea_level_pressure': '#9B59B6', # Purple
    'meter_reading': '#34495E'  # Dark Slate Blue / Charcoal
}

SMART_ROOM_COLORS = {
    'temperature': '#FF6B6B',         # Coral
    'humidity': '#4ECDC4',            # Teal
    'airquality': '#45B7D1',          # Sky blue
    'energyconsumption': '#96CEB4',   # Mint
    'overallsatisfaction': '#FFEAA7'  # Light yellow
}

PHYSICAL_ROOM_COLORS = {
    'temperature': '#FF7675',         # Light red
    'humidity': '#74B9FF',            # Light blue
    'airquality': '#00B894',          # Emerald
    'energyconsumption': '#FDCB6E',   # Yellow
    'overallsatisfaction': '#E17055'  # Terracotta
}

SMART_BUILDING_COLORS = {
    # Environmental inputs
    'temperature': '#2D3436',         # Dark gray
    'humidity': '#636E72',            # Gray
    'airquality': '#00CEC9',          # Cyan
    'hvacsetpoint': '#0984E3',        # Blue
    'lightinglevel': '#FDCB6E',       # Yellow
    'occupantcount': '#6C5CE7',       # Purple
    
    # Intermediate comfort metrics
    'thermalcomfort': '#A29BFE',      # Light purple
    'visualcomfort': '#FD79A8',       # Pink
    'airqualityindex': '#55EFC4',     # Light green
    'hvacpower': '#81ECEC',           # Light cyan
    'lightingpower': '#FFD93D',       # Bright yellow
    
    # Final outcomes
    'energyconsumption': '#00B894',   # Green
    'overallsatisfaction': '#E84393'  # Magenta
}

# Function to create core metrics bar plot
def plot_core_metrics(output_dir='improved_viz'):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    methods = method_metrics['method'].tolist()
    x = np.arange(len(methods))
    width = 0.2
    
    metrics = ['precision', 'recall', 'f1_score', 'accuracy']
    colors = bar_colors
    patterns = ['/', '\\', 'x', 'o']
    
    for i, metric in enumerate(metrics):
        values = method_metrics[metric].tolist()
        bars = ax.bar(x + (i - 1.5) * width, values, width, 
                      label=metric.capitalize(),
                      color=colors[i], 
                      edgecolor='black',
                      linewidth=1.5,
                      alpha=0.8,
                      hatch=patterns[i] if i > 0 else '')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Core Performance Metrics', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05),
              ncol=4, fancybox=True, shadow=True)
    
    # Add grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 1.15)  # Increase y-axis limit to accommodate labels
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improved_core_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Function to create edge confidence comparison
def plot_edge_confidence(output_dir='improved_viz'):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Example data, replace with your actual data
    methods = ['final', 'pc', 'sam', 'llm']
    intervention_values = [1.0, 0.71, 0.33, 0.4]  # Replace with actual values
    agreement_values = [0.5, 0.52, 0.44, 0.47]    # Replace with actual values
    
    # Labels for the bars (from your original image)
    intervention_labels = ['6/6', '5/7', '2/6', '2/5']
    agreement_labels = ['9/18', '11/21', '8/18', '7/15']
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, intervention_values, width, label='Intervention Validation',
                  color=bar_colors[0], edgecolor='black', linewidth=1.5, alpha=0.8, hatch='+')
    bars2 = ax.bar(x + width/2, agreement_values, width, label='Method Agreement',
                  color=bar_colors[2], edgecolor='black', linewidth=1.5, alpha=0.8, hatch='o')
    
    # Add the fraction labels
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        ax.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height() + 0.02,
               intervention_labels[i], ha='center', va='bottom', fontweight='bold')
        ax.text(bar2.get_x() + bar2.get_width()/2., bar2.get_height() + 0.02,
               agreement_labels[i], ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Confidence Score', fontweight='bold')
    ax.set_title('Edge Confidence Comparison', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontweight='bold')
    ax.legend(loc='upper right', fancybox=True, shadow=True)
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 1.2)  # Increase y-axis limit
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improved_edge_confidence.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Function to create risk vs cost comparison
def plot_risk_cost(output_dir='improved_viz'):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    methods = risk_cost_metrics['method'].tolist()
    risk_values = risk_cost_metrics['risk'].tolist()
    cost_values = risk_cost_metrics['cost'].tolist()
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, risk_values, width, label='Risk',
                  color=bar_colors[1], edgecolor='black', linewidth=1.5, alpha=0.8, hatch='o')
    bars2 = ax.bar(x + width/2, cost_values, width, label='Cost',
                  color=bar_colors[3], edgecolor='black', linewidth=1.5, alpha=0.8, hatch='*')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Risk vs Cost Comparison', fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontweight='bold')
    ax.legend(loc='upper right', fancybox=True, shadow=True)
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    ax.set_ylim(0, 0.6)  # Adjust based on your data
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improved_risk_cost.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Function to create SHD comparison
def plot_shd(output_dir='improved_viz'):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    methods = method_metrics['method'].tolist()
    shd_values = method_metrics['shd'].tolist()
    
    # Create bars with custom colors and patterns
    bars = ax.bar(methods, shd_values, color='#2ecc71', edgecolor='black', 
                  linewidth=1.0, alpha=1.0, width=0.6)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
               f'{int(height)}', ha='center', va='bottom', 
               fontweight='bold', fontsize=14)
    
    ax.set_ylabel('Structural Hamming Distance', fontweight='bold')
    ax.set_title('SHD Comparison', fontweight='bold', pad=20)
    ax.set_xticklabels(methods, fontweight='bold')
    
    ax.grid(axis='y', linestyle='--', alpha=1.0)
    
    # Set y-axis to start from 0
    ax.set_ylim(0, max(shd_values) * 1.2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'improved_shd.png'), dpi=600, bbox_inches='tight')
    plt.close()

def plot_shd_comparison(shd_data, selected_setups=None, selected_methods=None, 
                       output_dir='improved_viz', plot_individual=True, plot_combined=True,
                       iteration_data=None, confidence_data=None, show_termination=True):
    """
    Enhanced SHD comparison plotting with true scale, line patterns, and inflection highlighting.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup configuration with complexity ordering
    all_setups = ['base', 'noisy', 'hidden_vars', 'ashrae', 'physical', 'large_scale']
    all_methods = ['pc', 'sam', 'llm', 'abcd', 'gies', 'icp', 'jci', 'notears', 'iid', 'causal_bandits', 'policygrid']
    
    if selected_setups is None:
        selected_setups = [s for s in all_setups if s in shd_data]
    if selected_methods is None:
        selected_methods = all_methods
    
    # Get SHD values (final or latest iteration)
    def get_value(setup, method):
        if setup in shd_data and method in shd_data[setup]:
            value = shd_data[setup][method]
            return value[-1] if isinstance(value, list) else value
        return 0

    # Enhanced colors and line patterns for B&W differentiation
    method_styles = {
        'pc': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'linewidth': 6, 'alpha': 1, 'markersize': 2},
        'sam': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'linewidth': 6, 'alpha': 1, 'markersize': 2}, 
        'llm': {'color': '#2ca02c', 'linestyle': '-.', 'marker': '^', 'linewidth': 6, 'alpha': 1, 'markersize': 2},
        'abcd': {'color': '#d62728', 'linestyle': '-', 'marker': 'v', 'linewidth': 6, 'alpha': 1, 'markersize': 2},
        'gies': {'color': '#9467bd', 'linestyle': '--', 'marker': '<', 'linewidth': 6, 'alpha': 1, 'markersize': 2},
        'icp': {'color': '#8c564b', 'linestyle': '-.', 'marker': '>', 'linewidth': 6, 'alpha': 1, 'markersize': 2},
        'jci': {'color': '#e377c2', 'linestyle': ':', 'marker': 'p', 'linewidth': 6, 'alpha': 1, 'markersize': 2},
        'notears': {'color': '#7f7f7f', 'linestyle': '-', 'marker': '*', 'linewidth': 6, 'alpha': 1, 'markersize': 2},
        'iid': {'color': '#bcbd22', 'linestyle': '--', 'marker': 'h', 'linewidth': 6, 'alpha': 0.6, 'markersize': 2},
        'causal_bandits': {'color': '#17becf', 'linestyle': '-.', 'marker': 'x', 'linewidth': 6, 'alpha': 1, 'markersize': 2},
        'policygrid': {'color': '#ff0000', 'linestyle': ':', 'marker': 'D', 'linewidth': 8, 'alpha': 1.0, 'markersize': 3}
    }
    
    # Create combined comparison plot
    if plot_combined:
        fig, ax = plt.subplots(figsize=(14, 10))
        
        x_positions = np.arange(len(selected_setups))
        
        # Create complexity labels with mathematical notation
        complexity_map = {
            'base': r'$\mathcal{O}(n)$',
            'noisy': r'$\mathcal{O}(n + \beta)$',
            'hidden_vars': r'$\mathcal{O}(n + \gamma)$',
            'ashrae': r'$\mathcal{O}(n + m)$',
            'physical': r'$\mathcal{O}(n + m + z)$',
            'large_scale': r'$\mathcal{O}(n z + m)$'
        }
        
        setup_names = []
        for setup in selected_setups:
            if setup == 'large_scale':
                title = 'Large Sim'
            elif setup == 'ashrae':
                title = 'ASHRAE'
            else:
                title = setup.replace('_', ' ').title()
            
            if setup in complexity_map:
                setup_names.append(f"{title}\n{complexity_map[setup]}")
            else:
                setup_names.append(title)
        
        # Plot each method with distinct patterns
        for method in selected_methods:
            if method in method_styles:
                y_values = [get_value(setup, method) for setup in selected_setups]
                style = method_styles[method]
                
                ax.plot(x_positions, y_values, 
                        color=style['color'],
                        linestyle=style['linestyle'],
                        marker=style['marker'],
                        linewidth=style['linewidth'],
                        markersize=style['markersize'],
                        markeredgewidth=1.5,
                        markeredgecolor='white',  # White edge for better separation
                        label=method.upper(),
                        alpha=style['alpha'],
                        markevery=1,  # Show every marker
                        zorder=3 if method == 'policygrid' else 2)
        
        # Find ASHRAE and Large-Scale positions for inflection highlighting
        try:
            ashrae_idx = selected_setups.index('ashrae')
            large_scale_idx = selected_setups.index('large_scale')
            
            # Create box around inflection segment - thinner, more distinctive
            box_x = ashrae_idx - 0.1  # Much thinner
            box_width = (large_scale_idx - ashrae_idx) + 0.2
            
            # Get max y-value in this range for box height
            max_y_in_range = 0
            for method in selected_methods:
                if method in method_styles:
                    y_vals = [get_value(setup, method) for setup in selected_setups]
                    range_vals = y_vals[ashrae_idx:large_scale_idx+1]
                    max_y_in_range = max(max_y_in_range, max(range_vals))
            
            # Add highlighting box with thick purple border, distinct from lines
            box_height = max_y_in_range * 1.07
            rect = patches.Rectangle((box_x, -1.5), box_width, box_height,
                                   linewidth=3, edgecolor='purple', facecolor='none',
                                   linestyle='-', alpha=1.0)  # Solid thick purple line
            ax.add_patch(rect)
            
            # Add inflection arrow
            arrow_x = (ashrae_idx + large_scale_idx) / 2
            arrow_y = max_y_in_range * 0.6
            
            ax.annotate('Complexity\nInflection', 
                       xy=(arrow_x, arrow_y), 
                       xytext=(arrow_x, arrow_y + max_y_in_range * 0.3),
                       arrowprops=dict(arrowstyle='->', color='red', lw=3),
                       fontsize=20, fontweight='bold', color='red',
                       ha='center', va='bottom',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                               edgecolor='red', alpha=0.9))
            
        except (ValueError, IndexError):
            pass  # Skip if ashrae or large_scale not in selected_setups
        
        # Enhanced formatting with TRUE SCALE (no log)
        ax.set_ylabel('Structural Hamming Distance', fontweight='bold', fontsize=20)
        ax.set_xlabel('Problem Complexity', fontweight='bold', fontsize=20)
        # ax.set_title('SHD Comparison: Iterative DAG Construction Performance', 
        #             fontweight='bold', fontsize=16, pad=25)
        
        ax.set_xticks(x_positions)
        ax.set_xticklabels(setup_names, fontsize=20, fontweight='bold', rotation=45, ha='right')
        ax.grid(True, linestyle='--', alpha=0.3, zorder=1)  # Lighter grid
        ax.set_facecolor('#fafafa')  # Very light gray background
        
        # TRUE SCALE - Linear y-axis starting from -3, ticks from 0
        max_val = max([get_value(setup, method) 
                      for setup in selected_setups 
                      for method in selected_methods])
        ax.set_ylim(-3, max_val * 1.1)  # Start from -3 with reduced headroom
        
        # Set ticks to start from 0
        y_ticks = np.arange(0, max_val * 1.1 + 1, max(1, int(max_val * 0.1)))
        ax.set_yticks(y_ticks)
        
        # Enhanced legend with better positioning
        legend = ax.legend(loc='upper left', fontsize=15, frameon=True, 
                  fancybox=True, shadow=True, ncol=2,  # 2 columns
                  columnspacing=1.5, handlelength=2.5)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.95)
                
        plt.subplots_adjust(bottom=0.15)
        filename = 'shd_comparison_enhanced_true_scale_policygrid.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Enhanced SHD comparison plot saved to {output_dir}/{filename}")
    
    # Individual method plots (if requested)
    if plot_individual:
        for method in selected_methods:
            if method in method_styles:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                y_values = [get_value(setup, method) for setup in selected_setups]
                style = method_styles[method]
                
                ax.plot(x_positions, y_values, 
                       color=style['color'],
                       linestyle=style['linestyle'],
                       marker=style['marker'],
                       linewidth=style['linewidth'],
                       markersize=12,
                       markeredgewidth=2,
                       markeredgecolor='black')
                
                ax.set_ylabel('Structural Hamming Distance', fontweight='bold')
                ax.set_xlabel('Problem Complexity', fontweight='bold')
                ax.set_title(f'{method.upper()} Method - SHD Performance', 
                           fontweight='bold', pad=20)
                ax.set_xticks(x_positions)
                ax.set_xticklabels(setup_names, fontweight='bold', rotation=45, ha='right')
                ax.grid(True, linestyle='--', alpha=0.3)
                
                # True scale for individual plots too
                ax.set_ylim(0, max(y_values) * 1.2)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'shd_{method}_true_scale.png'), 
                          dpi=300, bbox_inches='tight')
                plt.close()

def load_shd_data_from_results(result_files):
    """Load SHD data from DAG result JSON files."""
    import json
    
    shd_data = {}
    
    for setup, file_path in result_files.items():
        try:
            with open(file_path, 'r') as f:
                results = json.load(f)
            
            setup_data = {}
            for method, data in results.items():
                method_name = method.lower()
                
                # Handle different JSON structures
                if isinstance(data, dict):
                    if 'metrics' in data and 'shd' in data['metrics']:
                        setup_data[method_name] = data['metrics']['shd']
                    elif 'shd' in data:
                        setup_data[method_name] = data['shd']
                    else:
                        print(f"Warning: No SHD found for {method} in {setup}")
            
            if setup_data:
                shd_data[setup] = setup_data
            
        except Exception as e:
            print(f"Error loading {setup} from {file_path}: {e}")
    
    return shd_data

# Function to create scientific SHD plots from your JSON files
def plot_shd_from_files(result_files, output_dir='improved_viz'):
    """
    Create scientific SHD plots directly from JSON result files.
    
    Parameters:
    - result_files: dict with {setup_name: json_file_path}
    
    Example:
    result_files = {
        'base': 'results/base_results.json',
        'noisy': 'results/noisy_results.json',
        'hidden_vars': 'results/hidden_vars_results.json',
        'ashrae': 'results/ashrae_results.json', 
        'physical': 'results/physical_results.json',
        'large_scale': 'results/building_results.json'
    }
    """
    # Load SHD data from files
    shd_data = load_shd_data_from_results(result_files)
    
    if not shd_data:
        print("No SHD data loaded. Check file paths and JSON structure.")
        return
    
    print(f"Loaded data for setups: {list(shd_data.keys())}")
    for setup, methods in shd_data.items():
        print(f"  {setup}: {methods}")
    
    # Create raw SHD plots only
    plot_shd_comparison(shd_data, output_dir=output_dir)
    
    print(f"Scientific SHD plots saved to {output_dir}/")

# Function to create individual DAG plots
def plot_single_dag(edges, node_positions, node_colors, ground_truth_edges, title, filename, output_dir, use_patterns=False, number_nodes=True):
    """Plot a single DAG with optional patterns and node numbering."""
    import matplotlib.pyplot as plt
    import networkx as nx
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create graph
    G = nx.DiGraph()
    G.add_edges_from(edges)
    
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
    
    if use_patterns:
        # Pattern-based approach for B&W compatibility
        node_styles = {
            'temperature': {'hatch': ''},
            'humidity': {'hatch': '///'},
            'airquality': {'hatch': '...'},
            'air_quality': {'hatch': '...'},
            'energyconsumption': {'hatch': 'xxx'},
            'energy_consumption': {'hatch': 'xxx'},
            'overallsatisfaction': {'hatch': '|||'},
            'overall_satisfaction': {'hatch': '|||'},
            'temp': {'hatch': ''},
            'humid': {'hatch': '///'},
            'air': {'hatch': '...'},
            'energy': {'hatch': 'xxx'},
            'satisfaction': {'hatch': '|||'},
            'overall': {'hatch': '|||'}
        }
        
        # Draw nodes manually with patterns
        for node in G.nodes():
            node_key = node.lower().replace(' ', '').replace('_', '')
            possible_keys = [node.lower(), node_key, node.lower().replace('_', ''), node.lower().replace(' ', '')]
            
            style = {'hatch': ''}
            for key in possible_keys:
                if key in node_styles:
                    style = node_styles[key]
                    break
            
            pos = case_insensitive_positions[node]
            color = node_colors.get(node.lower(), '#CCCCCC')
            radius = 0.08
            
            circle = plt.Circle(pos, radius, facecolor=color, edgecolor='black', 
                              linewidth=3, alpha=0.9, hatch=style['hatch'])
            ax.add_patch(circle)
    else:
        # Original NetworkX approach - clean and simple
        nx.draw_networkx_nodes(G, case_insensitive_positions,
                               node_color=[node_colors.get(node.lower(), '#CCCCCC') for node in G.nodes()],
                               node_size=5000,
                               edgecolors='black',
                               linewidths=3,
                               alpha=1.0,
                               ax=ax)
    
    # Add node numbering if requested
    if number_nodes:
        # Use the SAME ordering function as the legend
        ordered_nodes = get_consistent_node_ordering(node_colors)
        
        # Create mapping from node name to number based on consistent ordering
        node_numbers = {}
        for i, ordered_node in enumerate(ordered_nodes):
            # Find the actual node in the graph that matches this ordered node
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
    
    # Determine edge colors and styles (matching original logic)
    edge_colors = []
    edge_styles = []
    for edge in edges:
        if title == 'Ground Truth DAG':
            edge_colors.append('#000000')  # Black for ground truth
            edge_styles.append('-')        # Solid
        else:
            norm_edge = (edge[0].lower(), edge[1].lower())
            rev_edge = (edge[1].lower(), edge[0].lower())
            
            if norm_edge in ground_truth_edges or rev_edge in ground_truth_edges:
                edge_colors.append('#2ECC71')  # Green for correct edges
                edge_styles.append('-')        # Solid
            else:
                edge_colors.append('#E74C3C')  # Red for wrong edges
                edge_styles.append('--')       # Dashed
    
    # Draw edges with proper margins (matching original)
    for i, edge in enumerate(edges):
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
    
    # ax.set_title(title, fontsize=20, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=600, bbox_inches='tight')
    plt.close()

def create_enhanced_legend(node_colors, output_dir, filename='enhanced_legend.png'):
    """Create legend showing node patterns and edge styles with actual visual patterns."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Node pattern legend with actual visual patterns
    node_styles = {
        'Temperature': {'hatch': '', 'color': node_colors.get('temperature', '#FF9966')},
        'Humidity': {'hatch': '///', 'color': node_colors.get('humidity', '#66B2FF')},
        'Air Quality': {'hatch': '...', 'color': node_colors.get('airquality', '#66CC66')},
        'Energy Consumption': {'hatch': 'xxx', 'color': node_colors.get('energyconsumption', '#FF6666')},
        'Overall Satisfaction': {'hatch': '|||', 'color': node_colors.get('overallsatisfaction', '#B266FF')}
    }
    
    # Create custom legend elements with actual circles showing patterns
    legend_elements = []
    
    # Add node elements with actual pattern visualization
    for i, (node, style) in enumerate(node_styles.items()):
        # Create a custom legend element using a circle patch
        circle_patch = patches.Circle((0, 0), 0.5, facecolor=style['color'], 
                                    edgecolor='black', linewidth=2, 
                                    hatch=style['hatch'])
        legend_elements.append((circle_patch, node))
    
    # Position legend elements manually
    y_start = 0.8
    x_node = 0.1
    
    # Draw node legend
    ax.text(0.05, 0.9, 'Node Types:', fontsize=14, fontweight='bold', transform=ax.transAxes)
    
    for i, (patch, label) in enumerate(legend_elements):
        y_pos = y_start - i * 0.12
        
        # Create and add the circle patch
        circle = patches.Circle((x_node, y_pos), 0.03, facecolor=patch.get_facecolor(), 
                              edgecolor=patch.get_edgecolor(), linewidth=patch.get_linewidth(),
                              hatch=patch.get_hatch(), transform=ax.transAxes)
        ax.add_patch(circle)
        
        # Add label
        ax.text(x_node + 0.08, y_pos, label, fontsize=12, va='center', transform=ax.transAxes)
    
    # Add edge legend
    ax.text(0.55, 0.9, 'Edge Types:', fontsize=14, fontweight='bold', transform=ax.transAxes)
    
    # Correct edge (solid green)
    ax.plot([0.6, 0.75], [0.8, 0.8], color='#2ECC71', linewidth=4, linestyle='-', transform=ax.transAxes)
    ax.text(0.78, 0.8, 'Correct Edge', fontsize=12, va='center', transform=ax.transAxes)
    
    # Incorrect edge (dashed red)
    ax.plot([0.6, 0.75], [0.68, 0.68], color='#E74C3C', linewidth=4, linestyle='--', transform=ax.transAxes)
    ax.text(0.78, 0.68, 'Incorrect Edge', fontsize=12, va='center', transform=ax.transAxes)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

# Updated main plotting function with options
def plot_dags(json_file_path, ground_truth_edges, node_positions, node_colors, 
                      selected_methods=None, output_dir='improved_viz', use_patterns=False, number_nodes=True):
    """Plot DAGs with enhanced options for patterns and numbering."""
    import json
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load DAG results from JSON
    with open(json_file_path, 'r') as f:
        dag_results = json.load(f)
    
    # Filter and select methods to display
    if selected_methods is None:
        method_scores = [(method, data['metrics']['f1_score']) 
                        for method, data in dag_results.items() 
                        if data['metrics']['f1_score'] > 0]
        method_scores.sort(key=lambda x: x[1], reverse=True)
        selected_methods = [method for method, _ in method_scores[:4]]
    
    # Plot ground truth
    plot_single_dag(list(ground_truth_edges), node_positions, node_colors, 
                   ground_truth_edges, 'Ground Truth DAG', 'ground_truth.png', 
                   output_dir, use_patterns=use_patterns, number_nodes=number_nodes)
    
    # Plot each method
    for method in selected_methods:
        if method in dag_results:
            data = dag_results[method]
            edges = [tuple(edge) for edge in data['edges']]
            shd = data['metrics']['shd']
            title = f"{method.upper().replace('_', ' ')}\nSHD: {shd}"
            filename = f"{method}_dag.png"
            
            plot_single_dag(edges, node_positions, node_colors, 
                           ground_truth_edges, title, filename, output_dir,
                           use_patterns=use_patterns, number_nodes=number_nodes)
    
    # Create appropriate legend
    if use_patterns:
        create_enhanced_legend(node_colors, output_dir)
    else:
        create_legend(node_colors, output_dir, number_nodes)

def get_consistent_node_ordering(node_colors):
    """Get consistent node ordering for both DAG and legend numbering"""
    # Define preferred ordering for different scenarios
    preferred_order = [
        # Smart room order
        'temperature', 'humidity', 'airquality', 'energyconsumption', 'overallsatisfaction',
        # Smart building additional nodes (in logical order)
        'hvacsetpoint', 'lightinglevel', 'occupantcount', 
        'thermalcomfort', 'visualcomfort', 'airqualityindex',
        'hvacpower', 'lightingpower',
        # ASHRAE nodes
        'air_temperature', 'dew_temperature', 'sea_level_pressure', 
        'meter_reading', 'square_feet', 'year_built'
    ]
    
    # Sort nodes by preferred order, then alphabetically for any not in preferred list
    available_nodes = []
    # First add nodes in preferred order if they exist
    for preferred_node in preferred_order:
        if preferred_node in node_colors:
            available_nodes.append(preferred_node)
    
    # Then add any remaining nodes alphabetically
    remaining_nodes = sorted([node for node in node_colors.keys() 
                            if node not in available_nodes])
    available_nodes.extend(remaining_nodes)
    
    return available_nodes

def create_legend(node_colors, output_dir, number_nodes=False, filename='legend.png'):
    """Create a separate legend file (matching original style)."""
    import matplotlib.pyplot as plt
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Node legend elements
    node_legend_elements = []
    
    if number_nodes:
        # Use consistent ordering function
        available_nodes = get_consistent_node_ordering(node_colors)
        
        # Define mapping from internal keys to display names for different scenarios
        node_display_mapping = {
            # Smart Room nodes
            'temperature': 'Temperature',
            'humidity': 'Humidity', 
            'airquality': 'Air Quality',
            'energyconsumption': 'Energy Consumption',
            'overallsatisfaction': 'Overall Satisfaction',
            
            # Smart Building nodes (additional)
            'hvacsetpoint': 'HVAC Setpoint',
            'lightinglevel': 'Lighting Level',
            'occupantcount': 'Occupant Count',
            'thermalcomfort': 'Thermal Comfort',
            'visualcomfort': 'Visual Comfort',
            'airqualityindex': 'Air Quality Index',
            'hvacpower': 'HVAC Power',
            'lightingpower': 'Lighting Power',
            
            # ASHRAE nodes
            'air_temperature': 'Air Temperature',
            'dew_temperature': 'Dew Temperature',
            'sea_level_pressure': 'Sea Level Pressure',
            'meter_reading': 'Meter Reading',
            'square_feet': 'Square Feet',
            'year_built': 'Year Built'
        }
        
        for i, key in enumerate(available_nodes):
            color = node_colors.get(key, '#CCCCCC')
            # Get display name, fallback to formatted key if not in mapping
            display_name = node_display_mapping.get(key, key.replace('_', ' ').title())
            label = f"{i+1}. {display_name}"
            
            node_legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                   markerfacecolor=color, markersize=15, 
                                                   markeredgecolor='black', markeredgewidth=2,
                                                   label=label))
    else:
        # Original style legend
        for node, color in node_colors.items():
            node_legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                   markerfacecolor=color, markersize=15, 
                                                   markeredgecolor='black', markeredgewidth=2,
                                                   label=node.replace('_', ' ').title()))
    
    # Edge legend elements (matching original)
    edge_legend_elements = [
        plt.Line2D([0], [0], color='#2ECC71', linewidth=4, label='Correct Edge'),
        plt.Line2D([0], [0], color='#E74C3C', linewidth=4, linestyle='--', label='Wrong Edge')
    ]
    
    # Create legend (matching original style)
    legend = ax.legend(handles=node_legend_elements + edge_legend_elements, 
                      loc='center', fontsize=30, title='Legend', title_fontsize=40)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=600, bbox_inches='tight')
    plt.close()

# Convenience functions with dataset-specific colors and positions

def plot_ashrae_dags(json_file_path='ashrae_results.json', selected_methods=None):
    """Plot ASHRAE dataset DAGs."""
    output_dir = 'improved_viz/ashrae'
    
    ground_truth_edges = {
        ('square_feet', 'meter_reading'),
        ('year_built', 'meter_reading'), 
        ('air_temperature', 'dew_temperature'),
        ('air_temperature', 'meter_reading'),
        ('dew_temperature', 'meter_reading')
    }
    
    node_positions = {
        'square_feet': (0, 2),
        'year_built': (2, 2),
        'air_temperature': (0, 1),
        'dew_temperature': (1, 1),
        # 'sea_level_pressure': (2, 1),
        'meter_reading': (1, 0)
    }
    
    if selected_methods is None:
        selected_methods = ['policygrid', 'sam', 'pc', 'llm']
    
    plot_dags(json_file_path, ground_truth_edges, node_positions, ASHRAE_COLORS, selected_methods, output_dir)

def plot_smart_room_dags(json_file_path='smart_room_results.json', selected_methods=None):
    """Plot Smart Room simulation DAGs."""
    output_dir = 'improved_viz/base'
    
    ground_truth_edges = {
        ('temperature', 'energyconsumption'),
        ('temperature', 'overallsatisfaction'),
        ('humidity', 'energyconsumption'),
        ('humidity', 'overallsatisfaction'),
        ('airquality', 'energyconsumption'),
        ('airquality', 'overallsatisfaction')
    }
    
    node_positions = {
        'temperature': (0, 2),
        'humidity': (2, 2),
        'airquality': (1, 1.5),
        'energyconsumption': (0, 0),
        'overallsatisfaction': (2, 0)
    }
    
    if selected_methods is None:
        selected_methods = ['pc', 'sam', 'llm', 'policygrid', 'abcd', 'iid', 'gies', 'jci', 'causal_bandits', 'icp', 'notears_i']
    
    plot_dags(json_file_path, ground_truth_edges, node_positions, SMART_ROOM_COLORS, selected_methods, output_dir)

def plot_physical_smart_room_dags(json_file_path='physical_smart_room_results.json', selected_methods=None):
    """Plot Physical Smart Room DAGs."""
    output_dir = 'improved_viz/physical'
    
    ground_truth_edges = {
        ('temperature', 'energyconsumption'),
        ('humidity', 'energyconsumption'), 
        ('airquality', 'energyconsumption'),
        ('temperature', 'overallsatisfaction'),
        ('humidity', 'overallsatisfaction'),
        ('airquality', 'overallsatisfaction'),
        ('temperature', 'humidity'),
        ('humidity', 'airquality')
    }
    
    node_positions = {
        'temperature': (0, 2),
        'humidity': (2, 2),
        'airquality': (1, 1.5),
        'energyconsumption': (0, 0),
        'overallsatisfaction': (2, 0)
    }
    
    if selected_methods is None:
        selected_methods = ['pc', 'jci', 'notears_i', 'policygrid']
    
    plot_dags(json_file_path, ground_truth_edges, node_positions, PHYSICAL_ROOM_COLORS, selected_methods, output_dir)

def plot_smart_building_dags(json_file_path='smart_building_results.json', selected_methods=None):
    """Plot Smart Building DAGs."""
    output_dir = 'improved_viz/building'
    
    ground_truth_edges = {
        ('temperature', 'energyconsumption'),
        ('humidity', 'energyconsumption'),
        ('airquality', 'energyconsumption'),
        ('hvacsetpoint', 'energyconsumption'),
        ('occupantcount', 'energyconsumption'),
        ('temperature', 'overallsatisfaction'),
        ('humidity', 'overallsatisfaction'),
        ('airquality', 'overallsatisfaction'),
        ('lightinglevel', 'overallsatisfaction'),
        ('hvacsetpoint', 'overallsatisfaction'),
        ('occupantcount', 'overallsatisfaction'),
        ('temperature', 'thermalcomfort'),
        ('humidity', 'thermalcomfort'),
        ('lightinglevel', 'visualcomfort'),
        ('airquality', 'airqualityindex'),
        ('energyconsumption', 'hvacpower'),
        ('lightinglevel', 'lightingpower'),
        ('hvacsetpoint', 'hvacpower')
    }
    
    node_positions = {
        # Input variables (top row)
        'temperature': (0, 3),
        'humidity': (1.5, 3),
        'airquality': (3, 3),
        'hvacsetpoint': (4.5, 3),
        'lightinglevel': (6, 3),
        'occupantcount': (7.5, 3),
        
        # Intermediate variables (middle rows)
        'thermalcomfort': (0, 2),
        'visualcomfort': (2, 2),
        'airqualityindex': (4, 2),
        'hvacpower': (6, 2),
        'lightingpower': (7.5, 2),
        
        # Output variables (bottom row)
        'energyconsumption': (2, 1),
        'overallsatisfaction': (5, 1)
    }
    
    if selected_methods is None:
        selected_methods = ['policygrid', 'gies', 'jci', 'sam']
    
    plot_dags(json_file_path, ground_truth_edges, node_positions, SMART_BUILDING_COLORS, selected_methods, output_dir)

# Run all plotting functions
if __name__ == "__main__":
    # plot_core_metrics()
    # plot_edge_confidence()
    # plot_risk_cost()
    # plot_shd()
    
    # plot_ashrae_dags('results+logs/ashrae_benchmark_60iters_pt2/ashrae_raw_results.json')
    # plot_smart_room_dags('results+logs/results_60iters/method_dags.json')
    # plot_physical_smart_room_dags('results+logs/physical_results_15iters/methods_dags.json')
    # plot_smart_building_dags('results+logs/benchmark_results_60iters_bldg/method_dags.json')
    # print("All plots generated successfully!")

    # result_files = {
    #     'base': 'results/base/method_dags.json',
    #     'noisy': 'results/noisy/method_dags.json', 
    #     'hidden_vars': 'results/hidden_vars/method_dags.json',
    #     'ashrae': 'results/ashrae/method_dags.json',
    #     'physical': 'results/physical/method_dags.json',
    #     'large_scale': 'results/building/method_dags.json'
    # }
    
    # # Load from JSON files and create plots
    # plot_shd_from_files(result_files)

    manual_shd_data = {
    'base': {'pc': 4, 'sam': 8, 'llm': 7, 'abcd': 5, 'gies': 6, 'icp': 5, 'jci': 8, 'notears': 8, 'iid': 4, 'causal_bandits': 8, 'policygrid': 0},
    'noisy': {'pc': 4, 'sam': 6, 'llm': 7, 'abcd': 8, 'gies': 10, 'icp': 5, 'jci': 3, 'notears': 9, 'iid': 4, 'causal_bandits': 5, 'policygrid': 2},
    'hidden_vars': {'pc': 4, 'sam': 8, 'llm': 7, 'abcd': 6, 'gies': 8, 'icp': 4, 'jci': 8, 'notears': 3, 'iid': 4, 'causal_bandits': 9, 'policygrid': 0},
    'ashrae': {'pc': 4, 'sam': 4, 'llm': 4, 'abcd': 4, 'gies': 4, 'icp': 2, 'jci': 13, 'notears': 6, 'iid': 12, 'causal_bandits': 8, 'policygrid': 1},
    'physical': {'pc': 2, 'sam': 7, 'llm': 2, 'abcd': 8, 'gies': 2, 'icp': 6, 'jci': 6, 'notears': 4, 'iid': 2, 'causal_bandits': 7, 'policygrid': 0},
    'large_scale': {'pc': 49, 'sam': 21, 'llm': 17, 'abcd': 23, 'gies': 22, 'icp': 39, 'jci': 28, 'notears': 26, 'iid': 56, 'causal_bandits': 43, 'policygrid': 13}
    }
    
    plot_shd_comparison(manual_shd_data)
    print("Scientific SHD comparison plots generated!")