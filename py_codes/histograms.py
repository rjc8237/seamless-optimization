# Script to generate histograms for per-vertex colormaps used for rendering meshes

import igl
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import sys
import argparse
import numpy as np
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import symdir
import json
from pathlib import Path
from collections import defaultdict

def calculate_weighted_energy_score(X, args):
    """Calculate weighted energy score: sum(count_i * threshold_i)"""
    
    bin_min = args.get('bin_min') or 0
    bin_max = args.get('bin_max') or int(np.max(X)) + 1
    
    thresholds = np.arange(bin_min, bin_max + 1, 0.5)
    weighted_score = 0
    
    for threshold in thresholds:
        count = np.sum(X >= threshold)
        weighted_score += count * threshold
    
    return weighted_score

def find_best_solver_pair(xs_dict, args):
    """Find best (n, m) pair for CG and Ch_LLT solvers"""
    
    cg_scores = {}  # {n: weighted_score}
    chllт_scores = {}  # {n: weighted_score}
    
    # Group by solver and calculate scores
    for mesh_name, data_list in xs_dict.items():
        for X, mesh_params, Lp in data_list:
            # Parse mesh_params: "mesh=name, L{Lp}, p={p}, err={d}"
            parts = mesh_params.split(", ")
            p_val = int(parts[2].split("=")[1])  # Get p value
            d_val = float(parts[3].split("=")[1])  # Get err value
            
            score = calculate_weighted_energy_score(X, args)
            
            # Store score with (p, err) key
            key = (p_val, d_val)
            
            # Determine solver from mesh_name or add parameter
            # For now, assume we process CG and Ch_LLT separately
            if "CG" in mesh_params:
                if key not in cg_scores:
                    cg_scores[key] = []
                cg_scores[key].append(score)
            else:  # Ch_LLT
                if key not in chllт_scores:
                    chllт_scores[key] = []
                chllт_scores[key].append(score)
    
    # Average scores for each (p, err) pair
    cg_avg = {k: np.mean(v) for k, v in cg_scores.items()}
    chllт_avg = {k: np.mean(v) for k, v in chllт_scores.items()}
    
    # Find best pairs (lowest score is better)
    best_cg = min(cg_avg.items(), key=lambda x: x[1]) if cg_avg else None
    best_chllт = min(chllт_avg.items(), key=lambda x: x[1]) if chllт_avg else None
    
    return best_cg, best_chllт, cg_avg, chllт_avg

def histogram_energy_threshold_combined(xs, mesh_name, args):
    """Create single plot with 9 overlaid bar histograms for a mesh/solver combination"""
    
    if not xs or len(xs) == 0:
        print(f"Warning: No data for {mesh_name}")
        return
    
    # Get histogram colors
    color_dict = {
        'red': "#b90f29",
        'blue': "#3c4ac8",
        'green': "#2ecc71",
        'orange': "#e74c3c",
        'purple': "#9b59b6",
        'cyan': "#1abc9c",
        'yellow': "#f39c12",
        'pink': "#e91e63",
        'brown': "#795548"
    }
    
    colors = list(color_dict.values())[:len(xs)]

    try:
        lp_value = xs[0][2]
    except (IndexError, TypeError):
        lp_value = 2
    
    output_dir = Path(args['output_dir']) / f"Lp_{lp_value}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / f"{mesh_name}_histograms.png"
    
    matplotlib.rcParams['figure.figsize'] = (18, 10)
    
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # Get bin range
    bin_min = args.get('bin_min') or 0
    try:
        bin_max = args.get('bin_max') or max(int(np.max(X)) for X, _, _ in xs)
    except (ValueError, TypeError):
        bin_max = 5
    
    thresholds = np.arange(bin_min, bin_max + 1, 1)
    bar_width = 0.08  # Width of each bar (for 9 datasets: 0.08 * 9 ≈ 0.72)
    
    # Plot each dataset as grouped bars
    for idx, (X, dataset_name, Lp) in enumerate(xs):
        if len(X) == 0:
            print(f"  Skipping {dataset_name} - empty data")
            continue
        
        percentages = []
        total_triangles = len(X)
        
        for threshold in thresholds:
            count = np.sum(X >= threshold)
            percent = (count / total_triangles) * 100.0
            percentages.append(percent)
        
        # Offset bar positions for grouped bars
        x_positions = thresholds + (idx - 4) * bar_width  # Center around threshold
        color = colors[idx]
        ax.bar(x_positions, percentages, width=bar_width, 
               label=dataset_name, color=color, alpha=0.8)
    
    # Set axes labels and formatting
    ax.set_xlabel('Energy Threshold', fontsize=40)
    ax.set_ylabel('Percentage (%)', fontsize=40)
    ax.set_title(f"Energy Threshold Comparison - {mesh_name}", fontsize=45, fontweight='bold')
    ax.set_xticks(thresholds)
    ax.tick_params(labelsize=28)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.legend(fontsize=14, loc='upper right', framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    fig.savefig(str(output_path), bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    print(f"Saved: {output_path}")

def histogram_energy_threshold(X, args, m):
    """Create histogram showing triangle counts for energy >= threshold"""
    
    # Get bin range with proper defaults
    bin_min = args.get('bin_min') or 0
    bin_max = args.get('bin_max') or int(np.max(X)) + 1
    
    # Get histogram color
    color_dict = {
        'red': "#b90f29",
        'blue': "#3c4ac8"
    }
    color = args.get('color', 'red')
    if color in color_dict:
        color = color_dict[color]
    colors = [color]
    sns.set_palette(colors)
    
    os.makedirs(args['output_dir'], exist_ok=True)
    output_path = os.path.join(args['output_dir'], m + "_energy_threshold.png")
    
    matplotlib.rcParams['figure.figsize'] = (args.get('width', 12.80), args.get('height', 8.00))
    
    # Create thresholds data for histogram
    thresholds = np.arange(bin_min, bin_max + 1, 1)
    counts = []
    percentages = []
    total_triangles = len(X)
    
    for threshold in thresholds:
        count = np.sum(X >= threshold)
        percent = (count / total_triangles) * 100.0
        counts.append(count)
        percentages.append(percent)
    
    # Plot using sns.histplot style with percentage data
    fig, ax = plt.subplots(1)
    bars = ax.bar(thresholds, percentages, color=color, width=0.8)
    
    # Add count and percentage labels on top of each bar
    for bar, count, percent in zip(bars, counts, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}\n({percent:.1f}%)',
                ha='center', va='bottom', fontsize=15)
    
    # Set axes labels
    ax.set_xlabel(args.get('label', 'Energy Threshold'), fontsize=50)
    ax.tick_params(labelsize=30)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylim(0, max(percentages) * 1.1)
    
    # Save figure to file
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)
    
    # Print summary
    print(f"\nEnergy Threshold Summary:")
    for t, c, p in zip(thresholds, counts, percentages):
        print(f"  Triangles with energy >= {t}: {c} ({p:.2f}%)")

def histogram_data(X, args, m):
    # Get histogram color
    color_dict = {
        'red': "#b90f29",
        'blue': "#3c4ac8"
    }
    color = args['color']
    if color in color:
        color = color_dict[color]
    colors = [color,]
    sns.set_palette(colors)

    # Get bin range (or None if no range values provided)
    if (args['bin_min'] and args['bin_max']):
        binrange = (args['bin_min'], args['bin_max'])
    elif args['bin_min']:
        binrange = (args['bin_min'], np.max(X))
    elif args['bin_max']:
        binrange = (np.min(X), args['bin_max'])
    else:
        binrange = None

    # Generate histogram and save to file
    os.makedirs(args['output_dir'], exist_ok=True)
    output_path = os.path.join(
        args['output_dir'],
        m+"_sym_dir.png"
    )

    matplotlib.rcParams['figure.figsize'] = (args['width'], args['height'])

    # Set percentage or absolute scale for y axis
    fig, ax = plt.subplots(1)
    bins=21
    hist = sns.histplot(X, bins = bins, stat='percent', binrange=binrange, ax=ax)
    hist.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylim(0, args['ylim'])

    # Set axes labels
    hist.set_xlabel(args['label'], fontsize=50)
    hist.set_ylabel("")
    hist.tick_params(labelsize=30)
    
    # Save figure to file
    fig.savefig(output_path, bbox_inches='tight')

def add_histogram_arguments(parser):
    parser.add_argument(
        "-o", "--output_dir",
        help="directory for output images",
        default="./"
    )
    parser.add_argument(
        "--uv_path",
        help="path for mesh with uv coordinates"
    )
    parser.add_argument(
        "--bin_min",
        help="minimum value for bin",
        type=float
    )
    parser.add_argument(
        "--bin_max",
        help="maximum value for bin",
        type=float
    )
    parser.add_argument(
        "--ylim",
        help="y limit for the histogam",
        type=float,
        default=100
    )
    parser.add_argument(
        "--label",
        help="label for histogram",
    )
    parser.add_argument(
        "--color",
        help="color for histogram",
        default='red'
    )
    parser.add_argument(
        "-H", "--height",
        help="image height",
        default=8.00
    )
    parser.add_argument(
        "-W", "--width",
        help="image width",
        default=12.80
    )

if __name__ == "__main__":
    # Parse arguments for the script
    parser = argparse.ArgumentParser("Generate histograms for mesh")
    add_histogram_arguments(parser)
    args = vars(parser.parse_args())

    Lp = 2
    solver = {"CG": 2, "Ch_LLT": 1}
    percents = [1, 3, 5]
    dif_err = [0.001, 0.01, 0.05]
    xs = defaultdict(list)  # Use defaultdict to auto-initialize lists

    for s in solver:
        for p in percents:
            for d in dif_err:
                folder = Path(f"output/test4/Lp_{solver[s]}_{p}_{d}/{s}")

                # Iterate through subdirectories (mesh folders)
                for mesh_dir in folder.iterdir():
                    if not mesh_dir.is_dir():
                        continue
                    
                    # Find JSON file in the mesh folder
                    json_files = list(mesh_dir.glob("*.json"))
                    if json_files:
                        json_file = json_files[0]
                        
                        try:
                            # Load JSON file
                            with open(json_file, 'r') as f:
                                json_data = json.load(f)
                            
                            # Read energy vector from JSON
                            X = np.array(json_data.get('energy', []))
                            
                            if len(X) == 0:
                                print(f"Warning: No energy data in {json_file.name}")
                                continue
                            
                            # Get mesh name from folder
                            base_mesh_name = json_file.stem.split("_")[0]
                            mesh_name = f"p={p}, err={d}"
                            
                            # Store with solver as key
                            xs[(base_mesh_name, s)].append([X, mesh_name, solver[s]])
                            print(f"Processed: {s} - {mesh_name} from {json_file.name}")
                            
                        except Exception as e:
                            print(f"Error processing {json_file.name}: {e}")

    print("\nProcessing complete. Generating histograms...")
    
    # Group by mesh and solver
    meshes_dict = defaultdict(lambda: defaultdict(list))
    
    for (mesh_base, solver_name), data_list in xs.items():
        meshes_dict[mesh_base][solver_name] = data_list
    
    # Generate separate histograms for CG and Ch_LLT for each mesh
    for mesh_base, solver_data in meshes_dict.items():
        for solver_name, data_list in solver_data.items():
            if data_list:
                plot_name = f"{mesh_base}_{solver_name}"
                histogram_energy_threshold_combined(data_list, plot_name, args)
                print(f"Finished histogram for {plot_name}")    # # Calculate weighted energy scores and find best pairs
    # print("\n" + "="*80)
    # print("WEIGHTED ENERGY SCORE ANALYSIS")
    # print("="*80)
    
    # best_cg, best_chllт, cg_scores, chllт_scores = find_best_solver_pair(xs, args)
    
    # print("\nCG Solver Scores:")
    # for (p, err), score in sorted(cg_scores.items()):
    #     print(f"  (n={p}, m={err}): {score:.2f}")
    
    # if best_cg:
    #     print(f"\nBest CG pair: (n={best_cg[0][0]}, m={best_cg[0][1]}) with score {best_cg[1]:.2f}")
    
    # print("\nCh_LLT Solver Scores:")
    # for (p, err), score in sorted(chllт_scores.items()):
    #     print(f"  (n={p}, m={err}): {score:.2f}")
    
    # if best_chllт:
    #     print(f"\nBest Ch_LLT pair: (n={best_chllт[0][0]}, m={best_chllт[0][1]}) with score {best_chllт[1]:.2f}")
    
    # print("="*80)