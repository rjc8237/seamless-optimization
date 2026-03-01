from tkinter import Image
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
import polyscope.imgui as ps_imgui
import matplotlib.pyplot as plt
import io
from PIL import Image

def read_meshes_from_single_lp_folder(folder, num_meshes = 100):
    meshes = []
    
    # Iterate through solver subfolders
    for solver_folder in folder.iterdir():
        mesh_count = 0
        if not solver_folder.is_dir():
            continue
        
        print(f"Found solver folder: {solver_folder}")
        
        # Iterate through mesh name folders
        for mesh_folder in solver_folder.iterdir():
            if mesh_count >= num_meshes:
                break
            mesh_count += 1
            if not mesh_folder.is_dir():
                continue
            
            # Look for OBJ files
            obj_files = list(mesh_folder.glob("*.obj"))
            if not obj_files:
                print(f"  No OBJ files in {mesh_folder}")
                continue
            energy_min, energy_max = float('inf'), float('-inf')
            counter = 0
            for obj_file in obj_files:
                if "iter=" in obj_file.stem:
                    continue  # Skip iteration meshes
                
                # Read mesh with UV parameterization
                mesh_name = obj_file.stem
                V, uv, _, F, Fuv, _ = igl.readOBJ(str(obj_file))
                mesh_cutter = symdir.MeshCutter(V, uv, F, Fuv)
                v_cut, f_cut = mesh_cutter.cut_mesh()
                txt_files = list(mesh_folder.glob(f"{mesh_name}.txt"))
                energy_array = []
                if txt_files:
                    energy_array = np.loadtxt(str(txt_files[0]))
                    if energy_array.size > 0:
                        energy_min = min(energy_min, energy_array.min())
                        energy_max = max(energy_max, energy_array.max())
                if energy_array is None or len(energy_array) == 0:
                    energy_array = []
                if len(energy_array) != F.shape[0]:
                    energy_array = []
                meshes.append({
                    "mesh_name": mesh_name,
                    "mesh_folder": mesh_folder.name,
                    "solver": solver_folder.name,
                    "path": str(obj_file),
                    "V": V,
                    "F": F,
                    "uv": uv,
                    "fuv": Fuv,
                    "v_cut": v_cut,
                    "f_cut": f_cut,
                    "energy": energy_array,
                    "energy_minmax": ()
                })
                counter += 1
                print(f"  ✓ Loaded: {mesh_name} (V: {len(V)}, F: {len(F)}, UV: {len(uv)})")
            for i in range(len(meshes)-counter, len(meshes)):
                meshes[i]["energy_minmax"] = (energy_min, energy_max)
                print(f"  → Energy range for {meshes[i]['mesh_name']}: [{energy_min:.4e}, {energy_max:.4e}]")
    
    return meshes

def get_mesh_scale(V):
    """Calculate the diagonal length of the mesh bounding box."""
    if len(V) == 0: return 1.0
    min_coords = np.min(V, axis=0)
    max_coords = np.max(V, axis=0)
    diagonal = np.linalg.norm(max_coords - min_coords)
    return diagonal if diagonal > 0 else 1.0

def normalize_uv_scale_nx2(V, F, V_uv):
    """
    Scales UV coordinates (N, 2) so that the total UV area equals the 3D mesh area.
    
    Args:
        V (np.array): (N, 3) 3D vertices
        F (np.array): (M, 3) Faces indices
        V_uv (np.array): (N, 2) UV coordinates per vertex
    
    Returns:
        scaled_V_uv (np.array): The scaled (N, 2) UV coordinates
    """
    
    # --- 1. Compute 3D Surface Area ---
    # Get 3D coordinates for each triangle corner
    v0 = V[F[:, 0]]
    v1 = V[F[:, 1]]
    v2 = V[F[:, 2]]
    
    # Area = 0.5 * ||(v1 - v0) x (v2 - v0)||
    cross_prod = np.cross(v1 - v0, v2 - v0)
    area_3d = 0.5 * np.linalg.norm(cross_prod, axis=1).sum()

    # --- 2. Compute UV Area ---
    # We must "expand" the (N, 2) UVs to face-corners to calculate area
    # uv_corners shape becomes (M, 3, 2)
    uv_corners = V_uv[F]
    
    uv0 = uv_corners[:, 0, :]
    uv1 = uv_corners[:, 1, :]
    uv2 = uv_corners[:, 2, :]
    
    # 2D Cross product (determinant) for triangle area
    vec_a = uv1 - uv0
    vec_b = uv2 - uv0
    cross_2d = vec_a[:, 0] * vec_b[:, 1] - vec_a[:, 1] * vec_b[:, 0]
    area_uv = 0.5 * np.abs(cross_2d).sum()

    # --- 3. Apply Scale ---
    if area_uv < 1e-12:
        print("Warning: UV area is near zero. Skipping normalization.")
        return V_uv 
        
    scale_factor = np.sqrt(area_3d / area_uv)
    print(f"Scaling UVs by {scale_factor:.4f} to match physical scale.")
    
    return V_uv * scale_factor

def overlay_histogram(image_path, data, val_range=None):
    """
    Generates a matplotlib histogram of the data and overlays it 
    on the image at image_path (bottom-right corner).
    """
    try:
        # Create histogram
        plt.figure(figsize=(5, 3.5), dpi=100)
        # Use a nice color (viridis purple-like)
        plt.hist(data, bins=50, range=val_range, color='#440154', alpha=0.7, rwidth=0.9, density=False)
        plt.title('Log Energy Distribution', fontsize=10)
        plt.xlabel("Log(Energy)", fontsize=8)
        plt.ylabel("Face Count", fontsize=8)
        plt.grid(axis='y', alpha=0.3)
        if val_range:
            # Force x-axis to specific range
            plt.xlim(val_range)
            # Create ticks that span from min to max (e.g., 5 ticks include endpoints)
            ticks = np.linspace(val_range[0], val_range[1], 5)
            plt.xticks(ticks, fontsize=8)
        else:
            plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        plt.tight_layout()
        
        # Save plot to in-memory buffer - Solid white background (not transparent)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=False, facecolor='white')
        plt.close()
        buf.seek(0)
        
        # Overlay onto screenshot
        with Image.open(image_path) as main_img:
            main_img = main_img.convert("RGBA")
            with Image.open(buf) as hist_img:
                # Resize hist to ~30% of screenshot width
                target_w = int(main_img.width * 0.3)
                aspect = hist_img.height / hist_img.width
                target_h = int(target_w * aspect)
                hist_img = hist_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                
                # Position: Bottom Right with padding
                x = main_img.width - target_w - 20
                y = main_img.height - target_h - 20
                
                # Paste and save (simple paste for opaque overlay)
                main_img.paste(hist_img, (x, y))
                main_img.save(image_path)
                
    except Exception as e:
        print(f"  ⚠ Failed to add histogram overlay: {e}")

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
        if "iter=" in mesh_data["mesh_name"]:
            continue  # Skip iteration meshes
        print(f"\n[{i+1}/{len(meshes)}] Visualizing: {mesh_data['mesh_name']} (Solver: {mesh_data['solver']})")
        
        ps.set_program_name(f"Mesh: {mesh_data['mesh_name']}")
        
        # Register 3D mesh with cut vertices
        mesh = ps.register_surface_mesh(
            "mesh",
            mesh_data["V"],
            mesh_data["F"]
        )
        mesh_scale = get_mesh_scale(mesh_data["V"])
        print(f"  -> Mesh Scale (BBox Diagonal): {mesh_scale:.4f}")

        # Add energy coloring (log scale)
        if len(mesh_data["energy"]) > 0:
            # Take log of energy values (add small epsilon to avoid log(0))
            energy_log = []
            if mesh_data["energy"] is not None and len(mesh_data["energy"]) > 0:
                energy_log = np.log(mesh_data["energy"] + 1e-10)
            else:
                continue  # Skip if energy data is not valid
            energy_log_minmax = (np.log(mesh_data["energy_minmax"][0] + 1e-10), np.log(mesh_data["energy_minmax"][1] + 1e-10))
            # Add as face scalar quantity
            quantity = ps.get_surface_mesh("mesh").add_scalar_quantity(
                "Energy (log)",
                energy_log,
                defined_on='faces',
                enabled=True,
                cmap='viridis',
                vminmax = energy_log_minmax
            )
            print(f"  ✓ Energy coloring applied (log scale)")

        # Print mesh info
        print(f"  Vertices: {len(mesh_data['V'])}, Faces: {len(mesh_data['F'])}")
        print(f"  Solver: {mesh_data['solver']}, Mesh_name: {mesh_data['mesh_name']}")
        
        if output_dir:
            screen_dir = output_path / mesh_data["solver"] / mesh_data["mesh_folder"] / "energy_distr"
            screen_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{mesh_data['mesh_name']}.png"
            screenshot_path = screen_dir / filename
            
            # --- CRITICAL FIX FOR CAMERA ---
            # This calculates the bounding box of the active mesh
            # and moves the camera to fit it perfectly in the view.
            ps.reset_camera_to_home_view() 
            ps.screenshot(str(screenshot_path), transparent_bg=False)
            current_energy_log = []
            if mesh_data["energy"] is not None and len(mesh_data["energy"]) > 0:
                current_energy_log = np.log(mesh_data["energy"] + 1e-10)
            else:
                continue
            # Calculate global range for consistent histogram axis
            val_range = None
            if mesh_data.get("energy_minmax") and len(mesh_data["energy_minmax"]) == 2:
                val_range = (np.log(mesh_data["energy_minmax"][0] + 1e-10), np.log(mesh_data["energy_minmax"][1] + 1e-10))
            overlay_histogram(str(screenshot_path), current_energy_log, val_range=val_range)

            print(f"  ✓ Saved: {screenshot_path}")
        
        # Add UV parameterization
        if len(mesh_data["uv"]) > 0:
            try:
                uv = normalize_uv_scale_nx2(mesh_data["V"], mesh_data["F"], mesh_data["uv"])
                
                uv_corners = uv[mesh_data["fuv"].flatten()]
                q_param = ps.get_surface_mesh("mesh").add_parameterization_quantity(
                    "UV Map",
                    uv_corners,
                    defined_on='corners',
                    enabled=True,
                    coords_type='world', viz_style='checker', checker_size = 0.0005
                )
            except Exception as e:
                print(f"  ✗ Error adding UV parameterization: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"  ⚠ No UV data available")
        
        if output_dir:
            screen_dir = output_path / mesh_data["solver"] / mesh_data["mesh_folder"] / "checkerboard"
            screen_dir.mkdir(parents=True, exist_ok=True)
            filename = f"{mesh_data['mesh_name']}.png"
            screenshot_path = screen_dir / filename
            
            # --- CRITICAL FIX FOR CAMERA ---
            # This calculates the bounding box of the active mesh
            # and moves the camera to fit it perfectly in the view.
            ps.reset_camera_to_home_view() 
            ps.screenshot(str(screenshot_path), transparent_bg=False)
            print(f"  ✓ Saved: {screenshot_path}")
        # Clean up for next mesh
        ps.remove_all_structures()



if __name__ == "__main__":
    # Test with a single Lp_* folder
    base_path = Path("/Users/aa13586/Desktop/symmetric-dirichlet/output/test2")
    
    output_dir = base_path / "mesh_screenshots"
    meshes = read_meshes_from_single_lp_folder(base_path, 50)        
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
