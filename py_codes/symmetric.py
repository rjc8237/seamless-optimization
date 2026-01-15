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


if __name__ == "__main__":
    solver = {"CG": 2, "Ch_LLT": 1}
    percents = [1, 3, 5, 8, 10]
    dif_err = [0.05, 0.1]
    # percents = [5]
    # dif_err = [0.001]
    for s in solver:
        for p in percents:
            for d in dif_err:
                folder = Path(f"output/test13_s/Lp_{solver[s]}_{p}_{d}/{s}")
                
                # Iterate through subdirectories (mesh folders)
                for mesh_dir in folder.iterdir():
                    if not mesh_dir.is_dir():
                        continue
                    
                    # Find .obj file in the mesh folder
                    obj_files = list(mesh_dir.glob("*.obj"))
                    if obj_files:
                        obj_file = obj_files[0]
                        uv_path = str(obj_file)
                        
                        # Get mesh name
                        base_mesh_name = obj_file.stem.split("_refined")[0]
                        mesh_name = f"mesh={base_mesh_name}, L{solver[s]}, p={p}, err={d}"

                        try:
                            # Load mesh
                            v3d, uv, _, f, fuv, _ = igl.readOBJ(uv_path)
                            mesh_cutter = symdir.MeshCutter(v3d, uv, f, fuv)
                            v_cut, _ = mesh_cutter.cut_mesh()
                            
                            # Compute energy
                            X = symdir.symmetric_dirichlet_energy(v_cut, uv, fuv) - 4.0
                            
                            # Find corresponding JSON file
                            json_files = list(mesh_dir.glob("*.json"))
                            if json_files:
                                json_file = json_files[0]
                                
                                # Load JSON data
                                with open(json_file, 'r') as f:
                                    json_data = json.load(f)
                                
                                # Add energy vector to JSON
                                json_data['energy'] = X.tolist()  # Convert numpy array to list
                                
                                # Save updated JSON
                                with open(json_file, 'w') as f:
                                    json.dump(json_data, f, indent=2)
                                
                                print(f"Processed: {mesh_name}")
                                print(f"  Saved energy vector to {json_file.name}")
                            else:
                                print(f"Warning: No JSON file found in {mesh_dir}")
                        
                        except Exception as e:
                            print(f"Error processing {mesh_name}: {e}")
            
    print("Done!")