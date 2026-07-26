import json
import matplotlib.pyplot as plt
import numpy as np

# ===== GLOBAL FONT SIZE CONFIGURATION =====
# Set up matplotlib parameters for better readability
plt.rcParams.update({
    'font.size': 16,              # Base font size (increased from default 10)
    'axes.titlesize': 20,         # Title font size
    'axes.labelsize': 18,         # Axis label font size  
    'xtick.labelsize': 16,        # X-axis tick label font size
    'ytick.labelsize': 16,        # Y-axis tick label font size
    'legend.fontsize': 16,        # Legend font size
    'figure.titlesize': 22,       # Figure title font size
    'axes.titleweight': 'bold',   # Make titles bold
    'axes.labelweight': 'bold',   # Make axis labels bold
})

# Load data - UPDATE THIS PATH TO YOUR JSON FILE LOCATION
json_file_path = 'neurips_pareto_results_hidden_vars/pareto_points.json'  # <-- CHANGE THIS PATH
try:
    with open(json_file_path, 'r') as f:
        raw_data = json.load(f)
    print(f"✓ Successfully loaded data from: {json_file_path}")
except FileNotFoundError:
    print(f"❌ Error: Could not find JSON file at: {json_file_path}")
    print("Please either:")
    print("1. Create the folder 'neurips_pareto_results_hidden_vars/' and put your JSON file there")
    print("2. Or change the 'json_file_path' variable above to point to your actual JSON file")
    exit(1)
except json.JSONDecodeError as e:
    print(f"❌ Error: Invalid JSON format in {json_file_path}")
    print(f"JSON Error: {e}")
    exit(1)

# Map old keys to new labels
key_mapping = {
    'GRID_Causal': 'PolicyGRID',
    'WorldModel_Only': 'PolicyGRID (w/o DAG)',
    'ASHRAE_PID': 'ASHRAE',
    'Correlation': 'Correlation'
}

pareto_data = {key_mapping.get(k, k): v for k, v in raw_data.items()}

# Aesthetic color palette - modern, sophisticated, accessible
colors = {
    'PolicyGRID': '#3498DB',      # Beautiful blue
    'PolicyGRID (w/o DAG)': '#9B59B6',   # Rich purple 
    'ASHRAE': '#1ABC9C',       # Elegant teal-green
    'Correlation': '#E67E22'       # Warm orange (better than red for accessibility)
}

# Alternative patterns for accessibility (different markers)
markers = {
    'PolicyGRID': 'o',      # Circle
    'PolicyGRID (w/o DAG)': 's',   # Square
    'ASHRAE': '^',       # Triangle up
    'Correlation': 'D'       # Diamond
}

# Collect all data points for axis limits
all_energies = []
all_comforts = []
for points in pareto_data.values():
    for point in points:
        all_energies.append(point['kwh'])
        all_comforts.append(point['dh'])

# Dynamic axis limits
pad_x = (max(all_energies) - min(all_energies)) * 0.05
pad_y = (max(all_comforts) - min(all_comforts)) * 0.05
xlims = (min(all_energies) - pad_x, max(all_energies) + pad_x)
ylims = (min(all_comforts) - pad_y, max(all_comforts) + pad_y)

# Define boundaries
mid_eng = 85
mid_comfort = 0.4

# ===== ENHANCED INDIVIDUAL CONSTRAINT POINTS PLOT =====
plt.figure(figsize=(16, 10))  # Increased figure size for better readability
plt.style.use('seaborn-v0_8-whitegrid')

# Plot individual constraint points with enhanced visibility
for policy, points in pareto_data.items():
    if not points:
        continue
    
    # Extract data
    energy_vals = [point['kwh'] for point in points]
    comfort_vals = [point['dh'] for point in points]
    episodes = [point['episode'] for point in points]
    
    # Plot individual points with enhanced visibility
    plt.scatter(energy_vals, comfort_vals, 
                color=colors[policy], 
                marker=markers[policy],  # Different shapes for accessibility
                label=f"{policy}",
                alpha=0.85,              # Increased opacity
                s=120,                   # LARGER marker size (increased from 80)
                edgecolor='white',       # White border for contrast
                linewidth=2,             # Thicker border (increased from 1.5)
                zorder=5)                # Bring to front

# Styling with enhanced contrast
plt.xlim(xlims)
plt.ylim(ylims)

# Enhanced grid lines with better visibility
plt.axvline(x=mid_eng, color='gray', linestyle='-', alpha=0.7, linewidth=1.5)
plt.axhline(y=mid_comfort, color='gray', linestyle='-', alpha=0.7, linewidth=1.5)

# Enhanced optimal zone with better contrast
plt.fill_between([xlims[0], mid_eng], ylims[0], mid_comfort, 
                color='lightgreen', alpha=0.15, zorder=1)
plt.text(xlims[0] + (mid_eng - xlims[0]) * 0.15, 
         ylims[0] + (mid_comfort - ylims[0]) * 0.15,
         'OPTIMAL ZONE', ha='center', va='center',
         fontsize=14, color='darkgreen', fontweight='bold', alpha=0.9,
         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# Enhanced styling for better accessibility
plt.gca().grid(True, alpha=0.3, linewidth=1.2)
plt.gca().set_facecolor('white')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_linewidth(1.2)

# Enhanced labels with better contrast and LARGER font sizes
plt.xlabel('Energy Consumption (kWh)', fontsize=18, color='black', fontweight='bold')
plt.ylabel('Comfort Violations (Degree-Hours)', fontsize=18, color='black', fontweight='bold')
plt.title('Individual Constraint Pareto Frontier: Energy vs Comfort Trade-offs', 
          fontsize=20, fontweight='bold', pad=20, color='black')

# Enhanced legend with better accessibility and larger font
legend = plt.legend(loc='lower right', frameon=True, fancybox=False, shadow=True,
                   facecolor='white', edgecolor='black', framealpha=0.95,
                   fontsize=16, markerscale=1.5)  # Increased fontsize and markerscale
legend.get_frame().set_linewidth(1.5)

# Enhance tick labels with LARGER font sizes
plt.xticks(fontsize=16, color='black', fontweight='normal')  # Increased from 12
plt.yticks(fontsize=16, color='black', fontweight='normal')  # Increased from 12

plt.tight_layout()
plt.savefig("neurips_pareto_results_hidden_vars/individual_constraint_points_enhanced.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Enhanced plot saved as 'individual_constraint_points_enhanced.png'")
print("\nAccessibility and readability features added:")
print("- SIGNIFICANTLY larger font sizes throughout:")
print("  * Axis labels: 18pt (was 14pt)")
print("  * Tick labels: 16pt (was 12pt)")
print("  * Title: 20pt (was 16pt)")
print("  * Legend: 16pt (was 11pt)")
print("- Larger marker sizes (s=120)")
print("- Thicker marker borders (linewidth=2)")
print("- Enhanced grid visibility")
print("- Better overall contrast and readability")

# ===== ADDITIONAL IMPROVEMENTS FOR OTHER PLOT TYPES =====

# Example of how to apply these settings to other plots in the file
def create_enhanced_plot_template():
    """Template function showing how to create plots with enhanced readability"""
    plt.figure(figsize=(16, 10))
    
    # The rcParams set above will automatically apply to new plots
    # But you can also manually specify larger sizes for specific elements:
    
    # For scatter plots:
    # plt.scatter(..., s=120, linewidth=2)
    
    # For labels:
    # plt.xlabel('X Label', fontsize=18, fontweight='bold')
    # plt.ylabel('Y Label', fontsize=18, fontweight='bold')
    # plt.title('Title', fontsize=20, fontweight='bold')
    
    # For ticks:
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    
    # For legends:
    # plt.legend(fontsize=16, markerscale=1.5)
    
    pass

# Print configuration summary
print("\n=== FONT SIZE CONFIGURATION SUMMARY ===")
print(f"Base font size: {plt.rcParams['font.size']}pt")
print(f"Axis labels: {plt.rcParams['axes.labelsize']}pt") 
print(f"Tick labels: {plt.rcParams['xtick.labelsize']}pt")
print(f"Title: {plt.rcParams['axes.titlesize']}pt")
print(f"Legend: {plt.rcParams['legend.fontsize']}pt")
print("All fonts are now much more readable!")