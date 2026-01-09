from glob import glob
import json
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
import numpy as np
INPUT_DIR = Path("./output")
OUTPUT_DIR = Path("./output/graphs")
import igl

def plot_all_meshes_combined(output_base, lp_value, metrics=["E_worst", "E_avg"], solvers=["CG", "Ch_LLT"]):
    """Plot all meshes with multiple metrics and solvers on same plot"""
    
    # Get all mesh folders from CG directory
    base_path_cg = output_base / ("Lp_" + str(lp_value)) / "CG"
    mesh_folders_cg = sorted([f for f in base_path_cg.iterdir() if f.is_dir()])
    
    if not mesh_folders_cg:
        print(f"No mesh folders found in {base_path_cg}")
        return
    
    section_map = {
        "E_avg": "opt_log",
        "E_worst": "opt_log",
        "max_grad": "opt_log",
        "condition_numbers": "hessian_log",
        "time_solver": "hessian_log",
        "time_ls": "hessian_log",
        "iter_solver": "hessian_log",
        "ls_step_size": "hessian_log",
        "correction": "hessian_log",
        "newton_decr": "hessian_log",
        "residual": "hessian_log"
    }
    
    log_vals = ["max_grad", "E_avg", "E_worst", "newton_decr", "correction", "residual"]
    
    # Create grid: 1 column per mesh, rows = 1 (all on same plots)
    cols = 4
    rows = (len(mesh_folders_cg) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()
    
    metric_colors = {"E_worst": "blue", "E_avg": "red"}
    solver_linestyles = {"CG": "-", "Ch_LLT": "--", "max_grad": ":", "residual": "-."}
    
    # Iterate through each mesh
    for mesh_idx, mesh_folder_cg in enumerate(mesh_folders_cg):
        mesh_name = mesh_folder_cg.name
        ax = axes[mesh_idx]
        
        info_lines = []
        
        # For each solver
        for solver in solvers:
            if solver == "CG":
                mesh_folder = mesh_folder_cg
            else:  # Ch_LLT
                base_path_chllt = output_base / ("Lp_" + str(lp_value)) / "Ch_LLT"
                mesh_folder = base_path_chllt / mesh_name
            
            if not mesh_folder.exists():
                print(f"[warn] No data for {solver} in {mesh_name}")
                continue
            
            json_files = list(mesh_folder.glob("*.json"))
            
            if not json_files:
                print(f"[warn] No JSON found in {mesh_folder}")
                continue

            for json_file in json_files:
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    
                    solver_name = data.get("solver_type", json_file.stem)
                    
                    # Extract time once
                    time = np.array([entry.get("elapsed_time", 0) for entry in data.get("opt_log", [])])
                    
                    # Plot each metric
                    for metric in metrics:
                        section = section_map.get(metric)
                        entries = data.get(section, [])
                        series = np.array([entry.get(metric) for entry in entries if metric in entry])
                        
                        # Check if data is valid
                        if len(series) == 0 or len(time) == 0:
                            continue
                        
                        # Trim time to match series length
                        time_trimmed = time[:len(series)]
                        
                        # Apply log scale if needed
                        if metric in log_vals:
                            safe = np.maximum(series, 1e-10)
                            series = np.log10(safe)
                        
                        # Format label: solver - metric (cgerr)
                        label = f"{metric} - {solver}"
                        if solver == "CG" or solver == "CG_LLT" or solver == "CG_GS":
                            cgerr = data.get("args", {}).get("cg_rel_err", "?")
                            if cgerr != "?":
                                cgerr = f"{float(cgerr):.0e}"
                                label = f"{metric} - {solver} ({cgerr})"
                        
                        color = metric_colors.get(metric, "black")
                        linestyle = solver_linestyles.get(solver, "-")
                        ax.plot(time_trimmed, series, label=label, linewidth=2.5, color=color,
                               linestyle=linestyle, alpha=0.8)
                    
                    # Collect info for this solver
                    total_time = data.get("total_time", "?")
                    iters = data.get("iters", "?")
                    
                    # Get last values for each metric
                    metric_values = []
                    for metric in metrics:
                        section = section_map.get(metric)
                        entries = data.get(section, [])
                        series = np.array([entry.get(metric) for entry in entries if metric in entry])
                        if len(series) > 0:
                            last_val = series[-1]
                            if isinstance(last_val, (int, float)):
                                last_val = f"{last_val:.2e}"
                            metric_values.append(f"{metric}={last_val}")
                    
                    info = f"{solver}: time={total_time}s, iters={iters}, {', '.join(metric_values)}"
                    info_lines.append(info)
                    
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
                    continue
        
        # Set titles and labels
        ax.set_title(f"{mesh_name}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Energy (log scale)", fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=8)
        
        # Add info box below plot
        if info_lines:
            info_text = "\n".join(info_lines)
            ax.text(0.5, -0.20, info_text, transform=ax.transAxes,
                   fontsize=7, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide unused subplots
    for idx in range(len(mesh_folders_cg), len(axes)):
        axes[idx].set_visible(False)
    
    metrics_str = "_vs_".join(metrics)
    plt.suptitle(f"Metrics: {metrics_str} (L_{lp_value})", fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    base_path = output_base / ("Lp_" + str(lp_value))
    output_path = base_path / f"all_meshes_comparison_{metrics_str}.png"
    base_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved plot to {output_path}")
    plt.close()

def plot_graphs(series_list, time_list, labels, x_name, y_name, title, output_path, info_lines=None):
    plt.figure(figsize=(6, 6))
    colors = ["red", "green", "blue", "orange", "purple", "brown", "magenta", "cyan", "olive", "teal"]
    last_point = ["max_grad", "E_avg", "correction", "newton_decr", "residual", "iter_solver"]

    for i, series in enumerate(series_list):
        time = time_list[i]  # Use time_list instead of iteration count
        last_idx, last_val = len(series) - 1, series[-1]
        c = colors[i % len(series_list)]
        plt.plot(time, series, label=f"{labels[i]}", linewidth=2, color=c)
        plt.scatter(time[-1], series[-1], color=c, zorder=5)

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title if title else y_name)

    plt.legend()

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.subplots_adjust(bottom=0.25)

    if info_lines:
        sorted_info = [info for label, info in sorted(zip(labels, info_lines), key=lambda x: x[0], reverse=False)]
        plt.text(0.5, -0.12, "\n".join(sorted_info), ha="center", va="top",
                 transform=plt.gca().transAxes, fontsize=10,
                 bbox=dict(boxstyle="round", fc="white", ec="gray"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_n_jsons(folder, output_folder, files, name = "", metric="residuals", model = ""):
    # Resolve paths
    paths = [os.path.join(folder, f) for f in files]
    graphs_dir = output_folder
    os.makedirs(graphs_dir, exist_ok=True)  # auto-create once here, too
    if name != "":
        name = "_" + name

    for p in paths:
        print(f"Loading: {p}")

    # Load JSON data
    datas = []
    for p in paths:
        with open(p, "r") as fp:
            datas.append(json.load(fp))

    section_map = {
        "E_avg": "opt_log",
        "E_worst": "opt_log",
        "max_grad": "opt_log",
        "condition_numbers": "hessian_log",
        "time_solver": "hessian_log",
        "time_ls": "hessian_log",
        "iter_solver": "hessian_log",
        "ls_step_size": "hessian_log",
        "correction": "hessian_log",
        "newton_decr": "hessian_log",
        "residual": "hessian_log"
    }

    # Extract series and labels
    series_list = []
    time_list = []
    labels = []
    info_lines = []
    section = section_map.get(metric)
    yname = metric
    log_vals = ["max_grad", "E_avg", "newton_decr", "correction", "residual"]
    if metric in log_vals:
        yname = "log(" + metric + ")"
    for data, fname in zip(datas, files):
        entries = data.get("opt_log", [])
        time = [entry.get("elapsed_time", 0) for entry in entries]
        entries = data.get(section, [])
        series = [entry.get(metric) for entry in entries if metric in entry]
        if metric in log_vals:
            safe = np.maximum(np.array(series, dtype=float), 1e-10)
            series = np.log10(safe)

        series_list.append(series)
        time_list.append(time)
        solver = data.get("solver_type", os.path.splitext(fname)[0])
        if solver == "CG" or solver == "CG_LLT" or solver == "CG_GS":
            cgerr = data.get("args", {}).get("cg_rel_err", "?")
            cgerr = f"{float(cgerr):.0e}" if cgerr != "?" else "?"
            solver += f" ({cgerr})"
        labels.append(solver)

        args = data.get("args", {})
        info = f"{solver}: total_time={data.get('total_time','?')}, iters={data.get('iters','?')}, last_val={series[-1]:.2f}"
        if solver == "CG" or solver == "CG_LLT" or solver == "CG_GS":
            cgerr = args.get('cg_rel_err', '?')
            cgerr = f"{float(cgerr):.0e}" if cgerr != "?" else "?"
            info += f", rel_err={cgerr}"
        info_lines.append(info)

    print(len(series_list), "series to plot.")
    # Sort all lists by labels (solver names) in alphabetical order
    sorted_data = sorted(zip(labels, series_list, time_list, info_lines), key=lambda x: x[0])
    if sorted_data:
        labels, series_list, time_list, info_lines = zip(*sorted_data)
        labels = list(labels)
        series_list = list(series_list)
        time_list = list(time_list)
        info_lines = list(info_lines)

    output_path = os.path.join(graphs_dir, metric + "_" + model + ".png")
    plot_graphs(
        series_list, 
        time_list,
        labels,
        x_name="Total time (s)",
        y_name=yname,
        title=model,
        output_path=output_path,
        info_lines=info_lines
    )


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
            if 90000 <= face_count <= 110000:
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

    plot_all_meshes_combined(INPUT_DIR, args.Lp, metrics=["E_worst", "E_avg"], solvers=["CG", "Ch_LLT"])

if __name__ == "__main__":
    main()
