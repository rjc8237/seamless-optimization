import igl
import os
import sys
import argparse
import numpy as np
import json
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import symdir
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
build_dir = repo_root / "build"
BIN = str(build_dir / "bin" / "symmetric_dirichlet")
DATA_ROOT = repo_root / "data" / "closed-Myles-dijkstra-01-11"
OUTPUT_DIR = repo_root / "output"
BASE_JSON = repo_root / "app" / "example.json"

def get_mesh_folders():
    mesh_names = []
    
    # Check if folder contains .obj files
    folder = DATA_ROOT
    obj_files = list(folder.glob("*_param.obj"))
    for obj in obj_files:
        # Count faces in the OBJ file
        face_count = 0
        with open(obj, 'r') as f:
            for line in f:
                if line.startswith('f '):
                    face_count += 1
    
        # Only include meshes with ~100K faces (allowing ±10% tolerance)
        if 100000 <= face_count:
            mesh_names.append(obj.stem)  # Extract just the filename without .obj
            print(f"  Added {obj.stem} with {face_count} faces")

    return mesh_names

def run_solver():
    solver = {"CG": 1}
    lps = [3]
    cgerrs = [1e-4]
    for s in solver:
        for lp in lps:
            for cgerr in cgerrs:
                folder_name = f"output/test2/{s}_{lp}"
                if s == "CG":
                    folder_name = folder_name + f"_{cgerr:.1e}"
                folder = Path(folder_name)
                
                # Iterate through subdirectories (mesh folders)
                for mesh_dir in folder.iterdir():
                    if not mesh_dir.is_dir():
                        continue
                    
                    # Find .obj file in the mesh folder
                    obj_files = list(mesh_dir.glob("*.obj"))
                    for obj_file in obj_files:
                        uv_path = str(obj_file)
                        
                        # Get mesh name
                        base_mesh_name = obj_file.stem.split("_refined")[0]
                        mesh_name = f"mesh={base_mesh_name}, L{solver[s]}"

                        try:
                            # Load mesh
                            v3d, uv, _, f, fuv, _ = igl.readOBJ(uv_path)
                            mesh_cutter = symdir.MeshCutter(v3d, uv, f, fuv)
                            v_cut, _ = mesh_cutter.cut_mesh()
                            
                            # Compute energy
                            X = symdir.symmetric_dirichlet_energy(v_cut, uv, fuv) - 4.0
                            # Save energy to txt file with same name as obj file
                            txt_path = obj_file.with_suffix('.txt')
                            np.savetxt(str(txt_path), X)
                        
                        except Exception as e:
                            print(f"Error processing {mesh_name}: {e}")
            
    print("Done!")

def compute_and_store_energy(a=1.0, b=10.0, s=10):
    """
    Compute number of energy elements in threshold ranges for every mesh in DATA_ROOT with >= 100K faces.
    Saves E>=a, E>=a+s, E>=a+2s, ... up to E>=b for each mesh.
    
    Parameters:
    - a: Start threshold
    - b: End threshold
    - s: Step size
    """
    mesh_names = get_mesh_folders()
    
    if not mesh_names:
        print("No meshes found with >= 100K faces")
        return
    
    # Create output file
    output_file = OUTPUT_DIR / "mesh_energies.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate thresholds
    thresholds = np.linspace(a, b, s)  # 10 points from 0.1 to 1.0
    
    with open(output_file, 'w') as f:
        f.write(f"Energy threshold counts (a={a}, b={b}, s={s})\n")
        f.write("=" * 80 + "\n\n")
    
    # Process each mesh
    for mesh_name in mesh_names:
        try:
            # Construct path to mesh OBJ file
            obj_path = DATA_ROOT / f"{mesh_name}.obj"
            
            if not obj_path.exists():
                print(f"Warning: OBJ file not found for {mesh_name}")
                continue
            
            # Load mesh
            v3d, uv, _, f, fuv, _ = igl.readOBJ(str(obj_path))
            mesh_cutter = symdir.MeshCutter(v3d, uv, f, fuv)
            v_cut, _ = mesh_cutter.cut_mesh()
            
            # Compute energy
            X = symdir.symmetric_dirichlet_energy(v_cut, uv, fuv) - 4.0
            
            # Compute counts for each threshold
            counts = []
            for threshold in thresholds:
                count = np.sum(X >= threshold)
                counts.append(count)
            
            # Append to output file
            with open(output_file, 'a') as f:
                # Remove _param suffix from mesh name
                clean_mesh_name = mesh_name.replace('_param', '')
                num_faces = len(fuv)
                f.write(f"{clean_mesh_name} {num_faces}\n")
                # Write counts for each threshold
                for threshold, count in zip(thresholds, counts):
                    f.write(f"{count}\n")
            
            print(f"Processed: {mesh_name}")
            print(f"  Faces: {len(fuv)}, Energy range: [{X.min():.6f}, {X.max():.6f}]")
        
        except Exception as e:
            print(f"Error processing {mesh_name}: {e}")
    
    print(f"\nEnergy threshold counts saved to {output_file}")

if __name__ == "__main__":
    run_solver()