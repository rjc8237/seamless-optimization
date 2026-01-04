import json
import os
from pathlib import Path
import sys
import pandas as pd
import igl
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import symdir

OUTPUT_DIR = Path("./output/Lp_shifted")

def num_triangles_with_energy(uv_path, E_min):
    fname = os.path.basename(uv_path)
    dot_index = fname.rfind(".")
    m = fname[:dot_index]

    # Load uv information
    try:
        v3d, uv, _, f, fuv, _ = igl.readOBJ(uv_path)
    except:
        print(f"Error loading mesh: {uv_path}")
        return

    # Compute chosen vertex energy
    mesh_cutter = symdir.MeshCutter(v3d,uv, f, fuv)
    v_cut, _ = mesh_cutter.cut_mesh()
    X = symdir.symmetric_dirichlet_energy(v_cut, fuv, uv, 1) - 4.
    count = sum(1 for e in X if e >= E_min)
    e_max = max(X)
    return count, e_max

def create_mesh_table(lp_value=1, solver="CG"):
    """Create a table with mesh name, E_worst, and total_time"""
    
    base_path = OUTPUT_DIR / ("Lp_" + str(lp_value))
    
    # Get all mesh folders
    mesh_folders = sorted([f for f in base_path.iterdir() if f.is_dir()])
    
    if not mesh_folders:
        print(f"No mesh folders found in {base_path}")
        return
    
    print(f"Found {len(mesh_folders)} mesh folders")
    
    # Collect data
    data = {
        "Mesh Name": [],
        "# Faces": [],
        "E_worst": [],
        "Total Time": [],
        "Time when E_worst = 1.0": [],
        "# Triangles E >= 1": [],
        "E_max": []
    }
    
    for mesh_folder in mesh_folders:
        mesh_name = mesh_folder.name
        
        # Look for solver json file
        solver_folder = mesh_folder / solver
        if solver_folder.exists():
            json_files = list(solver_folder.glob("*.json"))
            if json_files:
                try:
                    with open(json_files[0], 'r') as f:
                        json_data = json.load(f)
                        e_worst = json_data.get("E_worst", "N/A")
                        faces = json_data["opt_log"][0].get("F_size", "N/A")
                        e_worst_1 = json_data.get("E_worst=1.0", "N/A")
                        total_time = json_data.get("total_time", "N/A")
                        data["Mesh Name"].append(mesh_name)
                        data["# Faces"].append(faces)
                        data["E_worst"].append(e_worst)
                        data["Total Time"].append(total_time)
                        data["Time when E_worst = 1.0"].append(e_worst_1)
                except Exception as e:
                    print(f"Error loading {solver} JSON for {mesh_name}: {e}")
            obj_files = list(solver_folder.glob("*.obj"))
            if obj_files:
                try:
                    with open(obj_files[0], 'r') as f:
                        num_triangles, E_max = num_triangles_with_energy(obj_files[0], 1.0)
                        data["# Triangles E >= 1"].append(num_triangles)
                        data["E_max"].append(E_max)
                except Exception as e:
                    print(f"Error loading {solver} JSON for {mesh_name}: {e}")

        else:
            print(f"[warn] {solver} folder not found for {mesh_name}")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Display table
    print("\n" + "="*80)
    print(f"Table for {solver} - Lp={lp_value}")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    # Save to CSV
    output_file = OUTPUT_DIR / "statistics" / "tables" / f"table_{solver}_Lp{lp_value}.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Table saved to {output_file}")
    
    return df


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create table of mesh metrics")
    parser.add_argument("--Lp", default="1", help="Lp value")
    parser.add_argument("--solver", default="CG", help="Solver type")
    
    args = parser.parse_args()
    
    df = create_mesh_table(lp_value=int(args.Lp), solver=args.solver)


if __name__ == "__main__":
    main()