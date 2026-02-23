import itertools, subprocess, os, json, tempfile, uuid
from pathlib import Path
from multiprocessing import Pool
import random

repo_root = Path(__file__).resolve().parent.parent
build_dir = repo_root / "build"
BIN = str(build_dir / "bin" / "symmetric_dirichlet")
DATA_ROOT = repo_root / "data" / "closed-Myles-01-20"
OUTPUT_DIR = repo_root / "output"
BASE_JSON = repo_root / "app" / "example.json"

#find all meshes in closed_myles
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

def run_solver(args):
    """Wrapper function for multiprocessing"""
    obj_name, s, cfg, cgerr, folder_test = args
    input_dir = DATA_ROOT

    if not input_dir.is_dir():
        print(f"[warn] Skipping {obj_name}: missing DATA_ROOT")
        return

    output_dir = OUTPUT_DIR / folder_test / (s + f"_{cgerr:.1e}") / obj_name
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    
    unique_id = str(uuid.uuid4())[:8]
    if s in ["CG", "CG_GS", "CG_LLT"] and cgerr is not None:
        tmp_json_path = os.path.join(tempfile.gettempdir(), f"cfg_{obj_name}_{s}_cg{cgerr:.0e}_{unique_id}.json")
    else:
        tmp_json_path = os.path.join(tempfile.gettempdir(), f"cfg_{obj_name}_{s}_{unique_id}.json")

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
        "-t", "8"
    ]

    env_vars = os.environ.copy()
    env_vars["OMP_NUM_THREADS"] = "8"
    env_vars["MKL_NUM_THREADS"] = "8"
    env_vars["OPENBLAS_NUM_THREADS"] = "8"
    env_vars["VECLIB_MAXIMUM_THREADS"] = "8"
    env_vars["NUMEXPR_NUM_THREADS"] = "8"
    
    print(f"Running model: {obj_name} with solver: {s}" + (f" and cg_rel_err: {cgerr}" if cgerr else ""))
    
    with open(log_path, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=lf, cwd=build_dir, env=env_vars)
        if result.returncode != 0:
            print(f"[error] {tag} exited code {result.returncode}")
        else:
            print(f"[success] {tag} completed")

if __name__ == '__main__':
    with open(BASE_JSON) as f:
        base_cfg = json.load(f)

    # Automatically discover mesh folders
    mesh_names = get_mesh_folders()
    print(f"Found {len(mesh_names)} mesh files:")
    for m in sorted(mesh_names):
        print(f"  - {m}")

    # # Select random subset if needed
    # num_to_select = max(1, len(mesh_names) // 5)  # At least 1 mesh
    # mesh_names = random.sample(mesh_names, num_to_select)
    
    # print(f"\nSelected {len(mesh_names)} meshes (20%):")
    # for m in sorted(mesh_names):
    #     print(f"  - {m}")
        
    # Solvers
    solvers = ["CG"]
    cg_rel_errs = [1e-4, 1e-5, 1e-6]

    E_worst_2_target_converge = False
    folder_test = "test16"

    tasks = []
    for s in solvers:
        if s == "CG" or s == "CG_GS" or s == "CG_LLT":
            for cgerr in cg_rel_errs:
                for obj_name in mesh_names:
                    cfg_copy = base_cfg.copy()
                    cfg_copy["Lp"] = 2
                    cfg_copy["cg_rel_err"] = cgerr
                    tasks.append((obj_name, s, cfg_copy, cgerr, folder_test))
        else:
            for obj_name in mesh_names:
                cfg_copy = base_cfg.copy()
                cfg_copy["Lp"] = 1
                tasks.append((obj_name, s, cfg_copy, None, folder_test))

    print(f"\nTotal tasks: {len(tasks)}")

    # Run in parallel
    num_workers = 6
    with Pool(num_workers) as pool:
        pool.map(run_solver, tasks)

    print("All done.")