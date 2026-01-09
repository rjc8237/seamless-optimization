from glob import glob
import json
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
import numpy as np

OUTPUT_DIR = Path("./output/Lp_shifted")

def plot_scatter_distribution_combined(output_base, lp_values, solver="Ch_LLT", metric_x="total_time", metric_y="E_worst"):
    """Plot scatter distribution for a solver across multiple Lp values on one plot"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    
    all_values, all_times = [], []  # Collect all values for y-axis limits

    for lp_idx, lp_value in enumerate(lp_values):
        base_path = output_base / ("Lp_" + str(lp_value))
        
        # Get all mesh folders
        mesh_folders = sorted([f for f in base_path.iterdir() if f.is_dir()])
        
        if not mesh_folders:
            print(f"No mesh folders found in {base_path}")
            continue
        
        print(f"Found {len(mesh_folders)} mesh folders for Lp={lp_value}")
        
        # Collect data for all meshes
        times = []
        values = []
        
        for mesh_folder in mesh_folders:
            mesh_name = mesh_folder.name
            
            # Look for solver json file
            solver_folder = mesh_folder / solver
            if solver_folder.exists():
                json_files = list(solver_folder.glob("*.json"))
                if json_files:
                    try:
                        with open(json_files[0], 'r') as f:
                            data = json.load(f)
                            total_time = data.get(metric_x, 0)
                            # Extract E_worst from opt_log
                            e_worst = data.get("E_worst", 1e-10)
                            
                            # Ensure positive values for log scale
                            e_worst = max(e_worst, 1e-10)
                            e_worst = np.log10(e_worst)
                            
                            times.append(total_time)
                            values.append(e_worst)
                            all_values.append(e_worst)
                            all_times.append(total_time)
                    except Exception as e:
                        print(f"Error loading {solver} JSON for {mesh_name}: {e}")
        print("The min values for Lp =", lp_value, "is", min(times), end = ';')
        print("The max values for Lp =", lp_value, "is", max(times))
        # Plot scatter for this Lp value
        color = colors[lp_idx % len(colors)]
        marker = markers[lp_idx % len(markers)]
        
        ax.scatter(times, values, s=120, color=color, 
                  alpha=0.7, edgecolor='black', linewidth=1.2, 
                  marker=marker, label=f'Lp={lp_value}')
    
    # Customize plot
    ax.set_xlabel(metric_x.replace("_", " ").title(), fontsize=10, fontweight='bold')
    ax.set_ylabel(f'log10(||{metric_y.title()}||_2)', fontsize=10, fontweight='bold')
    ax.set_title(f'{solver} - {metric_x} vs log10({metric_y})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Adjust y-axis limits based on all collected values
    if all_values:
        y_min = min(all_values)
        y_max = max(all_values)
        y_margin = (y_max - y_min) * 0.15
        ax.set_ylim(y_min - 1, y_max + y_margin)
    
    # Adjust x-axis limits based on all collected times
    if all_times:
        x_min = min(all_times)
        x_max = 180
        x_margin = (x_max - x_min) * 0.1
        ax.set_xlim(x_min - x_margin, x_max + x_margin)

    plt.tight_layout()
    
    graphs_dir = output_base / "histograms"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    output_path = graphs_dir / f"scatter_{solver}_{metric_x}_vs_log_{metric_y}_combined.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot to {output_path}")
    plt.close()

def plot_all_histograms_grid(output_base, lp_value, metric="total_time"):
    """Plot histograms in a grid, each subplot has up to 10 meshes"""
    
    base_path = output_base / ("Lp_" + str(lp_value))
    
    # Get all mesh folders
    mesh_folders = sorted([f for f in base_path.iterdir() if f.is_dir()])
    
    if not mesh_folders:
        print(f"No mesh folders found in {base_path}")
        return
    
    print(f"Found {len(mesh_folders)} mesh folders")
    
    # Collect data for all meshes
    mesh_data = {}
    
    for mesh_folder in mesh_folders:
        mesh_name = mesh_folder.name
        
        cg_time = 0
        cg_label = "CG"
        ch_llt_time = 0
        ch_llt_label = "Ch_LLT"
        
        # Look for CG json file
        cg_folder = mesh_folder / "CG"
        if cg_folder.exists():
            json_files = list(cg_folder.glob("*.json"))
            if json_files:
                try:
                    with open(json_files[0], 'r') as f:
                        data = json.load(f)
                        cg_time = data.get(metric, 0)
                        # Extract cg_rel_err if it exists
                        cg_rel_err = data.get("args", {}).get("cg_rel_err")
                        if cg_rel_err is not None:
                            # Format as scientific notation
                            if isinstance(cg_rel_err, str):
                                cg_label = f"CG_{cg_rel_err}"
                            else:
                                cg_label = f"CG_{cg_rel_err:.0e}"
                except Exception as e:
                    print(f"Error loading CG JSON for {mesh_name}: {e}")
        
        # Look for Ch_LLT json file
        ch_llt_folder = mesh_folder / "Ch_LLT"
        if ch_llt_folder.exists():
            json_files = list(ch_llt_folder.glob("*.json"))
            if json_files:
                try:
                    with open(json_files[0], 'r') as f:
                        data = json.load(f)
                        ch_llt_time = data.get(metric, 0)
                except Exception as e:
                    print(f"Error loading Ch_LLT JSON for {mesh_name}: {e}")
        
        mesh_data[mesh_name] = (cg_time, ch_llt_time, cg_label, ch_llt_label)
    
    # Split meshes into groups of 10
    meshes_list = list(mesh_data.keys())
    meshes_per_plot = 10
    num_plots = (len(meshes_list) + meshes_per_plot - 1) // meshes_per_plot
    
    # Create grid: 3 columns
    cols = 3
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16 * cols, 7 * rows))
    axes = axes.flatten()
    
    # Plot each group of meshes
    for plot_idx in range(num_plots):
        start_idx = plot_idx * meshes_per_plot
        end_idx = min(start_idx + meshes_per_plot, len(meshes_list))
        meshes_group = meshes_list[start_idx:end_idx]
        
        ax = axes[plot_idx]
        plot_combined_histogram(meshes_group, mesh_data, lp_value, metric, ax)
        
        ax.set_title(f'Meshes {start_idx + 1}-{end_idx} (Lp={lp_value})', 
                    fontsize=11, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(num_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f'Total Time Comparison - All {len(meshes_list)} Meshes (Lp={lp_value})', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    graphs_dir = output_base / "histograms"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    output_path = graphs_dir / f"histogram_grid_{metric}_Lp{lp_value}.png"
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved histogram grid to {output_path}")
    plt.close()


def plot_combined_histogram(meshes, mesh_data, lp_value, metric="total_time", ax=None):
    """Plot one histogram with up to 10 meshes, comparing CG and Ch_LLT"""
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(16, 7))
    
    cg_times = [mesh_data[m][0] for m in meshes]
    ch_llt_times = [mesh_data[m][1] for m in meshes]
    cg_labels = [mesh_data[m][2] for m in meshes]
    ch_llt_labels = [mesh_data[m][3] for m in meshes]
    
    # Set up bar positions
    x = np.arange(len(meshes))
    bar_width = 0.35
    
    # Create bars
    # Use the first label as legend (they should all be the same)
    bars1 = ax.bar(x - bar_width/2, cg_times, bar_width, label=cg_labels[0] if cg_labels else "CG", 
                   color='blue', edgecolor='black', alpha=0.7)
    bars2 = ax.bar(x + bar_width/2, ch_llt_times, bar_width, label=ch_llt_labels[0] if ch_llt_labels else "Ch_LLT",
                   color='red', edgecolor='black', alpha=0.7)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=8)
    
    # Customize plot
    ax.set_xlabel('Mesh Name', fontsize=10, fontweight='bold')
    ax.set_ylabel('Total Time (s)', fontsize=10, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(meshes, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    if cg_times and ch_llt_times:
        max_time = max(max(cg_times), max(ch_llt_times))
        ax.set_ylim(0, 180)
    else:
        ax.set_ylim(0, 1)

    

def main():
    parser = argparse.ArgumentParser(
        description="Plot histogram grid and scatter distribution for all meshes."
    )
    parser.add_argument("--metric", default="total_time", help="Which metric to plot")
    parser.add_argument("--Lp", default="1", help="Lp value")
    parser.add_argument("--scatter", action='store_true', help="Plot scatter distribution")
    parser.add_argument("--scatter-multi", action='store_true', help="Plot scatter for multiple Lp values")

    args = parser.parse_args()
    
    if args.scatter_multi:
        lp_values = [1, 2, 3]
        # Plot Ch_LLT - all Lp on one plot
        plot_scatter_distribution_combined(OUTPUT_DIR, lp_values, solver="Ch_LLT", 
                                           metric_x="total_time", metric_y="E_worst")
        # Plot CG - all Lp on one plot
        plot_scatter_distribution_combined(OUTPUT_DIR, lp_values, solver="CG", 
                                           metric_x="total_time", metric_y="E_worst")
    else:
        plot_all_histograms_grid(OUTPUT_DIR, args.Lp, metric=args.metric)

if __name__ == "__main__":
    main()