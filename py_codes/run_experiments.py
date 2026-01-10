import itertools, subprocess, os, json, tempfile
from pathlib import Path
from multiprocessing import Pool
import random

repo_root = Path(__file__).resolve().parent.parent
build_dir = repo_root / "build"
BIN = str(build_dir / "bin" / "symmetric_dirichlet")
DATA_ROOT = repo_root / "data" / "closed_myles"
OUTPUT_DIR = repo_root / "output"
BASE_JSON = repo_root / "app" / "example.json"

#find all meshes in closed_myles
def get_mesh_folders():
    mesh_folders = {}
    
    for folder in DATA_ROOT.iterdir():
        if not folder.is_dir():
            continue
    
        if not folder.name.endswith("_output"):
            continue

        # Check if folder contains .obj files
        obj_files = list(folder.glob("*_refined_with_uv.obj"))
        if obj_files:
            obj_path = obj_files[0]
            obj_name = obj_path.name.replace(".obj", "")
        
            # Count faces in the OBJ file
            face_count = 0
            with open(obj_path, 'r') as f:
                for line in f:
                    if line.startswith('f '):
                        face_count += 1
        
            # Only include meshes with ~100K faces (allowing ±10% tolerance)
            if 90000 <= face_count <= 110000:
                mesh_folders[folder.name] = obj_name
                print(f"  Added {folder.name} with {face_count} faces")

    return mesh_folders



def run_solver(args):
    """Wrapper function for multiprocessing"""
    obj_name, folder_name, s, cfg, cgerr = args
    input_dir = DATA_ROOT / folder_name

    if not input_dir.is_dir():
        print(f"[warn] Skipping {obj_name}: missing folder {folder_name}")
        return

    output_dir = OUTPUT_DIR / ("Lp_" + str(cfg.get("Lp", 0))) / (s + "_4") / folder_name
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    
    cfg_copy = cfg.copy()
    if s == "CG" or s == "CG_GS" or s == "CG_LLT":
        tmp_json_path = os.path.join(tempfile.gettempdir(), f"cfg_{obj_name}_{s}_cg{cgerr:.0e}.json")
        cfg_copy["cg_rel_err"] = cgerr
    else:
        tmp_json_path = os.path.join(tempfile.gettempdir(), f"cfg_{obj_name}_{s}.json")

    with open(tmp_json_path, "w") as jf:
        json.dump(cfg_copy, jf)

    tag = f"{obj_name}_{s}" if (s != "CG" and s != "CG_GS" and s != "CG_LLT") else f"{obj_name}_{s}_cg-{cgerr:.0e}"
    log_path = output_dir / "logs" / f"{tag}.log"

    cmd = [
        BIN,
        "--input", str(input_dir),
        "--model", obj_name,
        "--json", str(tmp_json_path),
        "-o", str(output_dir),
        "-s", s,
        "-t", "4"
    ]
    
    print(f"Running model: {obj_name} with solver: {s}" + (f" and cg_rel_err: {cgerr}" if cgerr else ""))
    
    with open(log_path, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=lf, cwd=build_dir)
        if result.returncode != 0:
            print(f"[error] {tag} exited code {result.returncode}")
        else:
            print(f"[success] {tag} completed")

if __name__ == '__main__':
    with open(BASE_JSON) as f:
        base_cfg = json.load(f)

    # Automatically discover mesh folders
    mesh_folders = get_mesh_folders()
    print(f"Found {len(mesh_folders)} mesh folders:")
    for m in sorted(mesh_folders.keys()):
        print(f"  - {m}")

    # Select random 10% of meshes
    # Set seed for reproducibility
    # random.seed(42)
    # num_to_select = max(1, len(mesh_folders) // 5)  # At least 1 mesh
    
    # # Convert dict keys to list, sample them, then reconstruct dict
    # all_folder_names = list(mesh_folders.keys())
    # selected_folder_names = random.sample(all_folder_names, num_to_select)
    # mesh_folders = {name: mesh_folders[name] for name in selected_folder_names}
    
    # print(f"\nSelected {len(mesh_folders)} meshes (20%):")
    # for m in sorted(mesh_folders.keys()):
    #     print(f"  - {m}")
        
    # Solvers
    solvers = ["CG"]
    cg_rel_errs = [1e-4]


    tasks = []
    for s in solvers:
        if s == "CG" or s == "CG_GS" or s == "CG_LLT":
            for cgerr in cg_rel_errs:
                for folder_name, obj_name in mesh_folders.items():
                    tasks.append((obj_name, folder_name, s, base_cfg, cgerr))
                    # run_solver(m, s, base_cfg, cgerr)
        else:
            for folder_name, obj_name in mesh_folders.items():
                tasks.append((obj_name, folder_name, s, base_cfg, None))
                # run_solver(m, s, base_cfg)

    print(f"\nTotal tasks: {len(tasks)}")

    # Run in parallel
    num_workers = 4
    with Pool(num_workers) as pool:
        pool.map(run_solver, tasks)

    print("All done.")