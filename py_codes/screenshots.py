import polyscope as ps
import numpy as np
import igl
from pathlib import Path
import os
import sys
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import symdir

def read_meshes_from_single_lp_folder(lp_folder_path):
    """
    Read meshes from a single Lp_* folder structure:
    Lp_{solver}_{p}_{err}/{solver}/name_mesh/
    
    Returns dict with mesh info
    """
    lp_folder = Path(lp_folder_path)
    meshes = []
    
    # Parse folder name: Lp_{solver}_{p}_{err}
    folder_name = lp_folder.name
    parts = folder_name.split("_")
    if len(parts) < 4:
        print(f"Invalid folder name format: {folder_name}")
        return meshes
    
    solver_id = parts[1]  # 1 or 2
    p_value = parts[2]
    err_value = parts[3]
    
    solver_name = "Ch_LLT" if solver_id == "1" else "CG"
    
    print(f"Processing: {folder_name}")
    print(f"Solver ID: {solver_id}, P: {p_value}, Err: {err_value}\n")
    
    # Iterate through solver subfolders
    for solver_folder in sorted(lp_folder.glob(solver_name)):
        if not solver_folder.is_dir():
            continue
        
        print(f"Found solver folder: {solver_folder}")
        
        # Iterate through mesh name folders
        for mesh_folder in sorted(solver_folder.glob("*")):
            if not mesh_folder.is_dir():
                continue
            
            mesh_name = mesh_folder.name
            
            # Look for OBJ files
            obj_files = list(mesh_folder.glob("*.obj"))
            if not obj_files:
                print(f"  No OBJ files in {mesh_folder}")
                continue
            
            obj_file = obj_files[0]
            
            try:
                # Read mesh with UV parameterization
                V, uv, _, F, Fuv, _ = igl.readOBJ(str(obj_file))
                mesh_cutter = symdir.MeshCutter(V, uv, F, Fuv)
                v_cut, f_cut = mesh_cutter.cut_mesh()

                meshes.append({
                    "name": mesh_name,
                    "solver": solver_name,
                    "solver_id": solver_id,
                    "p": p_value,
                    "err": err_value,
                    "lp": folder_name,
                    "path": str(obj_file),
                    "V": V,
                    "F": F,
                    "uv": uv,
                    "fuv": Fuv,
                    "v_cut": v_cut,
                    "f_cut": f_cut
                })
                print(f"  ✓ Loaded: {mesh_name} (V: {len(V)}, F: {len(F)}, UV: {len(uv)})")
            except Exception as e:
                print(f"  ✗ Error reading {obj_file}: {e}")
                import traceback
                traceback.print_exc()
    
    return meshes

def get_mesh_scale(V):
    """Calculate the diagonal length of the mesh bounding box."""
    if len(V) == 0: return 1.0
    min_coords = np.min(V, axis=0)
    max_coords = np.max(V, axis=0)
    diagonal = np.linalg.norm(max_coords - min_coords)
    return diagonal if diagonal > 0 else 1.0


def visualize_meshes(meshes, output_dir=None):
    """
    Visualize meshes in Polyscope and save screenshots.
    Uses add_parameterization_quantity for UV display.
    
    Args:
        meshes: List of mesh dicts
        output_dir: Base directory to save screenshots
    """
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
    
    ps.init()
    
    for i, mesh_data in enumerate(meshes):
        print(f"\n[{i+1}/{len(meshes)}] Visualizing: {mesh_data['name']}")
        
        ps.set_program_name(f"Mesh: {mesh_data['name']}")
        
        # Register 3D mesh with cut vertices
        mesh = ps.register_surface_mesh(
            "mesh",
            mesh_data["V"],
            mesh_data["F"]
        )
        mesh_scale = get_mesh_scale(mesh_data["V"])
        print(f"  -> Mesh Scale (BBox Diagonal): {mesh_scale:.4f}")

        # Add UV parameterization
        if mesh_data["uv"] is not None and len(mesh_data["uv"]) > 0:
            try:
                uv_corners = mesh_data["uv"][mesh_data["fuv"].flatten()]
                q_param = ps.get_surface_mesh("mesh").add_parameterization_quantity(
                    "UV Map",
                    uv_corners,
                    defined_on='corners',
                    enabled=True,
                    coords_type='world', viz_style='checker'
                )
            except Exception as e:
                print(f"  ✗ Error adding UV parameterization: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  ⚠ No UV data available")
        
        # Print mesh info
        print(f"  Vertices: {len(mesh_data['V'])}, Faces: {len(mesh_data['F'])}")
        print(f"  Solver: {mesh_data['solver']}, P: {mesh_data['p']}, Err: {mesh_data['err']}")
        
        if output_dir:
            lp_dir = output_path / mesh_data["lp"] / mesh_data["solver"]
            lp_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{mesh_data['name']}_p{mesh_data['p']}_err{mesh_data['err']}.png"
            screenshot_path = lp_dir / filename
            
            # --- CRITICAL FIX FOR CAMERA ---
            # This calculates the bounding box of the active mesh
            # and moves the camera to fit it perfectly in the view.
            ps.reset_camera_to_home_view() 
            
            # Optional: Add a slight delay or force redraw if needed, 
            # though polyscope usually handles this synchronously.
            
            ps.screenshot(str(screenshot_path), transparent_bg=False)
            print(f"  ✓ Saved: {screenshot_path}")
        
        # Clean up for next mesh
        ps.remove_all_structures()

if __name__ == "__main__":
    # Test with a single Lp_* folder
    base_path = Path("/Users/aa13586/Desktop/symmetric-dirichlet/output/test3")
    
    # Find first Lp_* folder
    lp_folders = sorted(base_path.glob("Lp_*"))
    
    if not lp_folders:
        print(f"No Lp_* folders found in {base_path}")
        exit(1)
    
    # Use first folder for quick testing
    print(lp_folders)
    output_dir = base_path / "mesh_screenshots"
    for lp_folder in lp_folders:    
        print(f"Base path: {base_path}")
        print(f"Testing with folder: {lp_folder.name}\n")
        
        meshes = read_meshes_from_single_lp_folder(lp_folder)
        
        if meshes:
            print(f"\n{'='*70}")
            print(f"Found {len(meshes)} meshes. Visualizing...")
            print(f"{'='*70}\n")
            
            visualize_meshes(meshes, output_dir)
            
            print(f"\n{'='*70}")
            print(f"Done! Screenshots saved to: {output_dir}")
            print(f"{'='*70}")
        else:
            print("No meshes loaded.")