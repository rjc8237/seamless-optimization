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

    output_dir = OUTPUT_DIR / ("Lp_" + str(cfg.get("Lp", 0)) + "_" + str(cfg.get("percent", 0)) + "_" + str(cfg.get("diff_err", 0))) / s / folder_name
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    
    if s in ["CG", "CG_GS", "CG_LLT"] and cgerr is not None:
        tmp_json_path = os.path.join(tempfile.gettempdir(), f"cfg_{obj_name}_{s}_cg{cgerr:.0e}.json")
    else:
        tmp_json_path = os.path.join(tempfile.gettempdir(), f"cfg_{obj_name}_{s}.json")

    with open(tmp_json_path, "w") as jf:
        json.dump(cfg, jf)

    tag = f"{obj_name}_{s}" if (s != "CG" and s != "CG_GS" and s != "CG_LLT") else f"{obj_name}_{s}_cg-{cgerr:.0e}"
    log_path = output_dir / "logs" / f"{tag}.log"

    cmd = [
        BIN,
        "--input", str(input_dir),
        "--model", obj_name,
        "--json", str(tmp_json_path),
        "-o", str(output_dir),
        "-s", s,
        "-t", "1"
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
    solvers = ["Ch_LLT", "CG"]
    cg_rel_errs = [1e-3]
    # ns = [1, 3, 5]
    # ms = [0.001, 0.01, 0.05]
    ns = [1, 3]
    ms = [0.05]
    tasks = []
    for s in solvers:
        for n in ns:
            for m in ms:
                if s == "CG" or s == "CG_GS" or s == "CG_LLT":
                    for cgerr in cg_rel_errs:
                        cfg_copy = base_cfg.copy()
                        cfg_copy["percent"] = n
                        cfg_copy["diff_err"] = m
                        cfg_copy["Lp"] = 2
                        cfg_copy["cg_rel_err"] = cgerr
                        for folder_name, obj_name in mesh_folders.items():
                            tasks.append((obj_name, folder_name, s, cfg_copy, cgerr))
                else:
                    cfg_copy = base_cfg.copy()
                    cfg_copy["percent"] = n
                    cfg_copy["diff_err"] = m
                    cfg_copy["Lp"] = 1
                    for folder_name, obj_name in mesh_folders.items():
                        tasks.append((obj_name, folder_name, s, cfg_copy, None))
                        # run_solver(m, s, base_cfg)

    print(f"\nTotal tasks: {len(tasks)}")

    # Run in parallel
    num_workers = 8
    with Pool(num_workers) as pool:
        pool.map(run_solver, tasks)

    print("All done.")