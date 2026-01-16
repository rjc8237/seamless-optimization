import igl
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import sys
import argparse
import numpy as np
import json
from pathlib import Path
from collections import defaultdict

OUTPUT_DIR = Path("./output/aspect_ratio_histograms")

def get_mesh_folders(output_dir, Lp_value = None, solver = None, cgerr = 0.0):
    mesh_folders = {}

    obj_files = list(output_dir.glob("*_param.obj"))
    print(len(obj_files))
    for obj in obj_files:
        obj_name = obj.stem.split("_param")[0]

        # Count faces in the OBJ file
        face_count = 0
        with open(obj, 'r') as f:
            for line in f:
                if line.startswith('f '):
                    face_count += 1
        
        # Only include meshes with ~100K faces (allowing ±10% tolerance)
        if face_count >= 90000:
            mesh_folders[obj] = obj_name
            print(f"  Added {obj.name} with {face_count} faces")

    return mesh_folders

def compute_aspect_ratio(vs, f):
    """Compute aspect ratio statistics for the mesh."""
    def triangle_aspect_ratio(v0, v1, v2):
        a = np.linalg.norm(v1 - v0)
        b = np.linalg.norm(v2 - v1)
        c = np.linalg.norm(v0 - v2)
        s = (a + b + c) / 2.0
        area = (s * (s - a) * (s - b) * (s - c))**0.5
        inradius = area / s
        circumradius = (a * b * c) / (4.0 * area)
        return circumradius / inradius

    aspect_ratios = []
    for face in f:
        v0, v1, v2 = vs[face]
        ar = triangle_aspect_ratio(v0, v1, v2)
        aspect_ratios.append(ar)

    aspect_ratios = np.array(aspect_ratios)
    return aspect_ratios

def table_worst_aspect_ratio(mesh_folders):
    """Create a table of worst (maximum) aspect ratio for 3D and UV triangles"""
    
    worst_ratios_3d = []
    worst_ratios_uv = []
    mesh_names = []
    
    for folder_name, mesh_name in mesh_folders.items():
        v3d, uv, _, f, fuv, _ = igl.readOBJ(folder_name)

        # Compute aspect ratio for 3D triangles
        ar_stats_3d = compute_aspect_ratio(v3d, f)
        worst_ratio_3d = np.max(ar_stats_3d)
        worst_ratios_3d.append(worst_ratio_3d)
        
        # Compute aspect ratio for UV triangles
        ar_stats_uv = compute_aspect_ratio(uv, fuv)
        worst_ratio_uv = np.max(ar_stats_uv)
        worst_ratios_uv.append(worst_ratio_uv)
        
        mesh_names.append(mesh_name)
        print(f"for {mesh_name}, max 3D aspect ratio is {worst_ratio_3d:.2f}, max UV aspect ratio is {worst_ratio_uv:.2f}")
    
    # Create CSV file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "worst_aspect_ratio_table.csv"
    
    with open(output_path, 'w') as f:
        f.write("Mesh,Worst 3D Aspect Ratio,Worst UV Aspect Ratio\n")
        for mesh_name, ratio_3d, ratio_uv in zip(mesh_names, worst_ratios_3d, worst_ratios_uv):
            f.write(f"{mesh_name},{ratio_3d:.2f},{ratio_uv:.2f}\n")
    
    print(f"Saved worst aspect ratio table to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Plot a metric from N JSON solver logs in the same folder."
    )
    parser.add_argument("--folder", nargs="+",help="Folder containing JSON files")
    args = parser.parse_args()
    
    output_dir = Path("./data/closed-Myles-dijkstra-01-11")

    mesh_folders = get_mesh_folders(output_dir)
    print(len(mesh_folders))
    table_worst_aspect_ratio(mesh_folders)
    
if __name__ == "__main__":
    main()
    