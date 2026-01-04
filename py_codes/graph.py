from glob import glob
import json
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path
import numpy as np
INPUT_DIR = Path("./output")
OUTPUT_DIR = Path("./output/graphs")

# plot all the meshes together
def plot_all_meshes_combined(output_base, lp_value, metric="residual", solvers=["CG"]):
    """Plot all meshes for a single solver in a grid layout"""
    
    base_path = output_base / ("Lp_" + str(lp_value))
    mesh_folders = sorted([f for f in base_path.iterdir() if f.is_dir()])
    
    if not mesh_folders:
        print(f"No mesh folders found in {base_path}")
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
    yname = metric

    if metric == "E_worst":
        yname = "||E_worst||_" + str(2 * lp_value)
    if metric in log_vals:
        yname = "log(" + yname + ")"

    section = section_map.get(metric)

    # Create grid: 4 columns, rows = number of meshes
    cols = 13
    rows = (len(mesh_folders) + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(78, 6 * rows))
    axes = axes.flatten()
    
    solver_colors = {"CG": "blue", "Ch_LLT": "red"}
    solver_linestyles = {"CG": "-", "Ch_LLT": "-"}
    
    for mesh_idx, mesh_folder in enumerate(mesh_folders):
        mesh_name = mesh_folder.name
        ax = axes[mesh_idx]
        
        info_lines = []
        
        # Plot both solvers on the same subplot
    for mesh_idx, mesh_folder in enumerate(mesh_folders):
        mesh_name = mesh_folder.name
        ax = axes[mesh_idx]
        
        info_lines = []
        
        for solver in solvers:
            solver_folder = mesh_folder / solver
            
            if not solver_folder.exists():
                print(f"[warn] Solver folder not found: {solver_folder}")
                continue
            
            json_files = list(solver_folder.glob("*.json"))

            for json_file in json_files:
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                    
                    solver_name = data.get("solver_type", json_file.stem)
                    
                    # Extract data
                    time = [entry.get("elapsed_time", 0) for entry in data.get("opt_log", [])]
                    entries = data.get(section, [])
                    series = [entry.get(metric) for entry in entries if metric in entry]
                    
                    # FIXED: Check length instead of truthiness of arrays
                    if len(series) == 0 or len(time) == 0:
                        continue
                    
                    # Apply log scale if needed
                    if metric in log_vals:
                        safe = np.maximum(np.array(series, dtype=float), 1e-10)
                        series = np.log10(safe)
                    
                    # Format label with solver info
                    label = solver_name
                    if solver == "CG" or solver == "CG_LLT" or solver == "CG_GS":
                        cgerr = data.get("args", {}).get("cg_rel_err", "?")
                        if cgerr != "?":
                            cgerr = f"{float(cgerr):.0e}"
                            label += f" ({cgerr})"
                    
                    color = solver_colors.get(solver, "black")
                    linestyle = solver_linestyles.get(solver, "-")
                    ax.plot(time, series, label=label, linewidth=2.5, color=color,
                           linestyle=linestyle, markersize=4, alpha=0.85)
                    
                    # Collect info for this solver
                    total_time = data.get("total_time", "?")
                    iters = data.get("iters", "?")
                    last_val = series[-1] if len(series) > 0 else "?"
                    if isinstance(last_val, (int, float)):
                        last_val = f"{last_val:.2e}"
                    
                    info = f"{solver_name}: total_time={total_time}s, iters={iters}, last_val={last_val}"
                    info_lines.append(info)
                    
                except Exception as e:
                    print(f"Error loading {json_file}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        # Set titles and labels
        ax.set_title(f"{mesh_name}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Time (s)", fontsize=9)
        ax.set_ylabel(yname, fontsize=9)
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.tick_params(labelsize=8)
        
        # Add info box below plot
        if info_lines:
            info_text = "\n".join(info_lines)
            ax.text(0.5, -0.1, info_text, transform=ax.transAxes,
                   fontsize=7, ha='center', va='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(len(mesh_folders), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(f"Metric: {yname} (Lp={lp_value})", fontsize=14, fontweight='bold')
    
    # Add space for the suptitle and adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave 2% space at top for suptitle
    
    output_path = base_path / f"all_meshes_{metric}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
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

    if args.folder:
        folder = args.folder
        print(folder)
    files = []
    if args.files:
        files = args.files

    solvers = ["CG", "Ch_LLT"]
    plot_all_meshes_combined(
        INPUT_DIR,  # Base output directory
        args.Lp,
        metric=args.metric,
        solvers=solvers
    )

    # for f in folder:
    #     folder_path = INPUT_DIR / ("Lp_" + str(args.Lp))
    #     folder_path /= f
    #     output_folder_path = OUTPUT_DIR / ("Lp_" + str(args.Lp))
    
    #     # if args.newton == "1":
    #     #     folder_path = folder_path / "projected_newton"
    #     if files:
    #         json_files = args.files
    #     else:  
    #         json_files = [os.path.basename(p) for p in folder_path.glob("*.json")]
    #     if not json_files:
    #         print(f"[warn] No JSON files found in {f}")
    #         continue
        
    #     if args.metric == "iter_solver":
    #         json_files = [fn for fn in json_files if not fn.endswith("_Ch_LLT.json") and not fn.endswith("_GS.json")]

    #     plot_n_jsons(str(folder_path), str(output_folder_path), json_files, args.name, args.metric, f)


if __name__ == "__main__":
    main()
