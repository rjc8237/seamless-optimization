from glob import glob
import json
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
import numpy as np
INPUT_DIR = Path("./output/test13_s")
OUTPUT_DIR = Path("./output/test13_s/graphs")
import igl

def plot_convergence_E_worst_all_meshes(output_base, solver, lp_value, p_value, err_value, ax=None, global_max_time=60.0):
    # Get all mesh folders
    base_path = output_base / (f"Lp_{lp_value}_{p_value}_{err_value}") / solver
    mesh_folders = sorted([f for f in base_path.iterdir() if f.is_dir()])
    if not mesh_folders:
        print(f"No mesh folders found in {base_path}")
        return ax, 0
    
    # Color for each mesh
    num_meshes = len(mesh_folders)
    colors_mesh = plt.cm.hsv(np.linspace(0, 0.95, num_meshes))
    mesh_colors = {folder.name: colors_mesh[i] for i, folder in enumerate(mesh_folders)}
    
    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    mesh_count = 0
    mesh_converged = 0
    max_time = 0.0
    for mesh_folder in mesh_folders:
        if not mesh_folder.name.endswith("_param"):
            continue

        mesh_name = mesh_folder.name
        json_files = list(mesh_folder.glob("*.json"))
        
        if not json_files:
            continue
        
        for json_file in json_files:
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                # Extract ONLY entries with BOTH elapsed_time AND E_worst
                opt_log = data.get("opt_log", [])
                
                time_list = []
                E_worst_list = []
                E_avg_list = []
                E_avg_step = []
                for entry in opt_log:
                    if "elapsed_time" in entry and "E_worst" in entry:
                        time_list.append(entry["elapsed_time"])
                        E_worst_list.append(entry["E_worst"])
                        E_avg_list.append(entry.get("E_avg"))                
                for i in range(1, len(E_avg_list)):
                    E_avg_step.append(abs(E_avg_list[i] - E_avg_list[i-1]) / E_avg_list[i-1])
                time = np.array(time_list)
                max_time = max(max_time, np.max(time))
                E_worst = np.array(E_worst_list)
                E_avg_step = np.array(E_avg_step)
                # Check if data is valid
                if len(E_worst) == 0 or len(time) == 0:
                    continue
                
                # Plot
                color = mesh_colors[mesh_name]
                style = '-'
                if data.get("converge_reason") != "Energy reached 1.0":
                    mesh_converged += 1
                    style = '--'
                ax.plot(time, E_worst, label=mesh_name, linewidth=2.5,
                       color=color, markersize=3, alpha=0.8, linestyle=style)
                mesh_count += 1
                
                break  # Only use first JSON file per mesh
                
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
    
    # Set labels and grid
    ax.set_xlabel("Time (s)", fontsize=9)
    ax.set_ylabel("E_worst", fontsize=9)
    ax.set_yscale("log")

    ax.set_xlim(0, global_max_time * 1.1)
    ax.set_title(f"p={p_value}, err={err_value}", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=8)

    # Add legend to the right of the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=8, frameon=True, ncol=2)

    return ax, mesh_count, mesh_converged

def create_convergence_plots(output_base, solver, lp_value, ps, errs, output_dir=None):
    """
    Create convergence plots (one per p and err combination) and save separately
    
    Args:
        output_base: Base output directory
        solver: Solver type ("CG" or "Ch_LLT")
        lp_value: Lp norm value
        ps: List of p values
        errs: List of error values
        output_dir: Directory to save plots
    """
    if output_dir is None:
        output_dir = output_base / ("Lp_" + str(lp_value))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot each combination
    for err in errs:
        for p in ps:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax, mesh_count, mesh_converged = plot_convergence_E_worst_all_meshes(
                output_base=output_base,
                solver=solver,
                lp_value=lp_value,
                p_value=p,
                err_value=err,
                ax=ax, global_max_time=120.0
            )
            percentage = (mesh_count - mesh_converged) / mesh_count * 100.0
            # Add text showing number of converged meshes
            ax.text(1.05, 1.00, f"E_worst reached 1.0: {mesh_count - mesh_converged}/{mesh_count} = {percentage:.2f}%",
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
            
            plt.tight_layout()
            
            # Save individual plot
            output_path = output_dir / f"{solver}_Lp{lp_value}_p{p}_err{err}_convergence.png"
            fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
            print(f"Saved plot to {output_path}")
            plt.close()

def get_mesh_folders(output_dir, Lp_value, solver, cgerr = 0.0, Lp_shift = 1.0):
    mesh_folders = {}

    if Lp_shift != 1.0:
        output_dir = output_dir / "Lp_shifted_" + str(Lp_shift)
    
    output_dir = output_dir / ("Lp_" + str(Lp_value))
    output_dir = output_dir / solver
    for folder in output_dir.iterdir():
        if not folder.is_dir():
            continue
    
        if not folder.name.endswith("_output"):
            continue

        # Check if folder contains .obj files
        obj_files = list(folder.glob("*.obj"))
        if obj_files:
            obj_path = obj_files[0]
            obj_name = obj_path.stem.split("_refined_with_uv_out_")[0]

            # Count faces in the OBJ file
            face_count = 0
            with open(obj_path, 'r') as f:
                for line in f:
                    if line.startswith('f '):
                        face_count += 1
        
            # Only include meshes with ~100K faces (allowing ±10% tolerance)
            if face_count >= 90000:
                mesh_folders[obj_path] = obj_name
                print(f"  Added {folder.name} with {face_count} faces")

    return mesh_folders


def compute_aspect_ratio(v3d, f):
    """Compute aspect ratio statistics for the mesh."""
    def triangle_aspect_ratio(v0, v1, v2):
        a = np.linalg.norm(v1 - v0)
        b = np.linalg.norm(v2 - v1)
        c = np.linalg.norm(v0 - v2)
        s = (a + b + c) / 2.0
        area = max(s * (s - a) * (s - b) * (s - c), 1e-10)**0.5
        inradius = area / s
        circumradius = (a * b * c) / (4.0 * area)
        return circumradius / inradius

    aspect_ratios = []
    for face in f:
        v0, v1, v2 = v3d[face]
        ar = triangle_aspect_ratio(v0, v1, v2)
        aspect_ratios.append(ar)

    aspect_ratios = np.array(aspect_ratios)
    return aspect_ratios

def histogram_of_ratios(mesh_folders):
    """Create a grid of aspect ratio histograms for all meshes"""
    
    num_meshes = len(mesh_folders)
    cols = 4
    rows = (num_meshes + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()
    
    for idx, (folder_name, mesh_name) in enumerate(mesh_folders.items()):
        ax = axes[idx]
        v3d, uv, _, f, fuv, _ = igl.readOBJ(folder_name)
        ar_stats = compute_aspect_ratio(v3d, f)
        print(f"for {mesh_name}, max ascpet ratio is {np.max(ar_stats)}")
        # Create histogram on subplot
        ax.hist(ar_stats, bins=50, color='blue', alpha=0.7, edgecolor='black', rwidth=0.85)
        ax.set_title(f"{mesh_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Aspect Ratio", fontsize=10)
        ax.set_ylabel("Frequency", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add statistics
        mean_ar = np.mean(ar_stats)
        max_ar = np.max(ar_stats)
        ax.text(0.98, 0.97, f"μ={mean_ar:.2f}\nmax={max_ar:.2f}",
                transform=ax.transAxes, fontsize=9, verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide unused subplots
    for idx in range(num_meshes, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle("Aspect Ratio Distribution for All Meshes", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_path = OUTPUT_DIR / "all_meshes_aspect_ratio_histograms.png"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved histogram grid to {output_path}")
    plt.close()

def main():
    folder = ["camel_output", "seahorse2_100K_output", "dancer2_output", "focal-octa_output", "raptor50K_output", "twirl_output", "shark_output", "bunnyBotsch_output", "pear_output", "femur_output"]
    parser = argparse.ArgumentParser(
        description="Plot a metric from N JSON solver logs in the same folder."
    )
    parser.add_argument("--folder", nargs="+",help="Folder containing JSON files")
    parser.add_argument("--files", nargs="+", help="List of JSON filenames")
    parser.add_argument("--name", default = "", help="Name for the output graph")
    parser.add_argument("--metric", default="residuals",
                        help="Which metric to plot")
    parser.add_argument("--newton")
    parser.add_argument("--Lp")
    args = parser.parse_args()

    # ps = [0.1, 0.5, 1, 3, 5, 7.5, 10]
    ps = [1, 3, 5, 8, 10]
    errs = [0.05]
    
    # Create separate plots for each solver
    lp_solver = {"CG": 2, "Ch_LLT": 1}
    for solver in ["CG", "Ch_LLT"]:
        create_convergence_plots(
            output_base=INPUT_DIR,
            solver=solver,
            lp_value=lp_solver[solver],
            ps=ps,
            errs=errs,
            output_dir=OUTPUT_DIR
        )

if __name__ == "__main__":
    main()
