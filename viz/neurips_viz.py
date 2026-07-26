import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
with open('neurips_pareto_results_20runs_3dhs_base/pareto_points.json', 'r') as f:
    raw_data = json.load(f)

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
    'PolicyGRID (w/o DAG)': '#6A1B9A',   # Darker purple for better contrast
    'ASHRAE': '#1ABC9C',       # Elegant teal-green
    'Correlation': '#D84315'       # Darker orange for better contrast
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
plt.figure(figsize=(14, 8))
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
                s=80,                    # Larger marker size  
                edgecolor='white',       # White border for contrast
                linewidth=1.5,           # Thicker border
                zorder=5)                # Bring to front

# Styling with enhanced contrast
plt.xlim(xlims)
plt.ylim(ylims)

# Enhanced grid lines
plt.axvline(x=mid_eng, color='gray', linestyle='-', alpha=0.6, linewidth=1.2)
plt.axhline(y=mid_comfort, color='gray', linestyle='-', alpha=0.6, linewidth=1.2)

# OPTIMAL ZONE (bottom-left: low energy, low comfort violations)
plt.fill_between([xlims[0], mid_eng], ylims[0], mid_comfort, 
                color='#E8F5E8', alpha=0.6, zorder=1)  # Soft mint green
plt.text(xlims[0] + (mid_eng - xlims[0]) * 0.15, 
         ylims[0] + (mid_comfort - ylims[0]) * 0.15,
         'OPTIMAL ZONE', ha='center', va='center',
         fontsize=11, color='#2E7D32', fontweight='bold', alpha=0.9,
         bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9, edgecolor='#4CAF50', linewidth=1.5))
# plt.text(xlims[0] + (mid_eng - xlims[0]) * 0.5, 
#          ylims[0] + (mid_comfort - ylims[0]) * 0.5,
#          'Low Energy\nLow Discomfort', ha='center', va='center',
#          fontsize=10, color='#2E7D32', fontweight='bold', alpha=0.9, style='italic')

# WORST CASE ZONE (top-right: high energy, high comfort violations)
plt.fill_between([mid_eng, xlims[1]], mid_comfort, ylims[1], 
                color='#FFF3E0', alpha=0.6, zorder=1)  # Soft peach/cream
plt.text(mid_eng + (xlims[1] - mid_eng) * 0.85, 
         mid_comfort + (ylims[1] - mid_comfort) * 0.85,
         'WORST CASE', ha='center', va='center',
         fontsize=11, color='#BF360C', fontweight='bold', alpha=0.9,
         bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9, edgecolor='#FF7043', linewidth=1.5))
# plt.text(mid_eng + (xlims[1] - mid_eng) * 0.5, 
#          mid_comfort + (ylims[1] - mid_comfort) * 0.5,
#          'High Energy\nHigh Discomfort', ha='center', va='center',
#          fontsize=10, color='#BF360C', fontweight='bold', alpha=0.9, style='italic')

# NEUTRAL ZONES (no background color, just labels)
# Top-left: High comfort violations, low energy
plt.text(xlims[0] + (mid_eng - xlims[0]) * 0.15, 
         mid_comfort + (ylims[1] - mid_comfort) * 0.85,
         'Low Energy\nHigh Discomfort', ha='left', va='top',
         fontsize=9, color='#666666', alpha=0.8, style='italic')

# Bottom-right: Low comfort violations, high energy
plt.text(mid_eng + (xlims[1] - mid_eng) * 0.85, 
         ylims[0] + (mid_comfort - ylims[0]) * 0.15,
         'High Energy\nLow Discomfort', ha='right', va='bottom',
         fontsize=9, color='#666666', alpha=0.8, style='italic')

# Enhanced styling for better accessibility
plt.gca().grid(True, alpha=0.3, linewidth=0.8)
plt.gca().set_facecolor('white')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_color('black')
plt.gca().spines['bottom'].set_color('black')
plt.gca().spines['left'].set_linewidth(1.2)
plt.gca().spines['bottom'].set_linewidth(1.2)

# Enhanced labels with better contrast
plt.xlabel('Energy Consumption (kWh)', fontsize=14, color='black', fontweight='bold')
plt.ylabel('Comfort Violations (Degree-Hours)', fontsize=14, color='black', fontweight='bold')
plt.title('Pareto Frontier Analysis: Energy vs Comfort Trade-offs', 
          fontsize=16, fontweight='bold', pad=20, color='black')

# Enhanced legend with better accessibility (COMMENTED OUT)
# legend = plt.legend(loc='upper left', frameon=True, fancybox=False, shadow=True,
#                    facecolor='white', edgecolor='black', framealpha=0.95,
#                    fontsize=11, markerscale=1.2)
# legend.get_frame().set_linewidth(1.5)

# Enhance tick labels
plt.xticks(fontsize=12, color='black')
plt.yticks(fontsize=12, color='black')

plt.tight_layout()
plt.savefig("neurips_pareto_results_20runs_3dhs_base/individual_constraint_points_imrpoved_base.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Enhanced quadrant analysis plot saved as 'individual_constraint_points_imrpoved.png'")
print("\nQuadrant Analysis Features:")
print("- OPTIMAL ZONE: Low energy + Low comfort violations (soft mint green)")
print("- WORST CASE: High energy + High comfort violations (soft peach)")
print("- Compromise zones with subtle backgrounds and descriptive labels")
print("- Soothing, non-clashing color palette with intuitive understanding")
print("- Enhanced accessibility with different marker shapes and clear contrast")