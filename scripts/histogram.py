# Script to generate histograms for per-vertex colormaps used for rendering meshes

<<<<<<< HEAD
=======
import glob
>>>>>>> alish
import igl
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
import sys
import argparse
import numpy as np
script_dir = os.path.dirname(__file__)
module_dir = os.path.join(script_dir, '..', 'py')
sys.path.append(module_dir)
import symdir


def add_histogram_arguments(parser):
    parser.add_argument(
        "-o", "--output_dir",
        help="directory for output images",
        default="./"
    )
    parser.add_argument(
        "--uv_path",
        help="path for mesh with uv coordinates"
    )
    parser.add_argument(
<<<<<<< HEAD
=======
        "--mesh_dir",
        help="directory containing multiple .obj files to process"
    ) # argument to process all files in a directory
    parser.add_argument(
>>>>>>> alish
        "--bin_min",
        help="minimum value for bin",
        type=float
    )
    parser.add_argument(
        "--bin_max",
        help="maximum value for bin",
        type=float
    )
    parser.add_argument(
        "--ylim",
        help="y limit for the histogam",
        type=float,
        default=100
    )
    parser.add_argument(
        "--label",
        help="label for histogram",
    )
    parser.add_argument(
        "--color",
        help="color for histogram",
        default='red'
    )
    parser.add_argument(
        "-H", "--height",
        help="image height",
        default=8.00
    )
    parser.add_argument(
        "-W", "--width",
        help="image width",
        default=12.80
    )

<<<<<<< HEAD
if __name__ == "__main__":
    # Parse arguments for the script
    parser = argparse.ArgumentParser("Generate histograms for mesh")
    add_histogram_arguments(parser)
    args = vars(parser.parse_args())

    # Get mesh and test name
    fname = os.path.basename(args['uv_path'])
=======
def process_single_mesh(uv_path, output_dir, bin_min, bin_max, ylim, label, color, width, height):
        # Get mesh and test name
    fname = os.path.basename(uv_path)
>>>>>>> alish
    dot_index = fname.rfind(".")
    m = fname[:dot_index]

    # Load uv information
    try:
<<<<<<< HEAD
        uv_path = args['uv_path']
        v3d, uv, _, f, fuv, _ = igl.read_obj(args['uv_path'])
    except:
        exit()
=======
        v3d, uv, _, f, fuv, _ = igl.readOBJ(uv_path)
    except:
        print(f"Error loading mesh: {uv_path}")
        return
>>>>>>> alish

    # Get histogram color
    color_dict = {
        'red': "#b90f29",
        'blue': "#3c4ac8"
    }
    color = args['color']
    if color in color:
        color = color_dict[color]
    colors = [color,]
    sns.set_palette(colors)

    # Compute chosen vertex energy
    mesh_cutter = symdir.MeshCutter(v3d,uv, f, fuv)
    v_cut, _ = mesh_cutter.cut_mesh()
    X = symdir.symmetric_dirichlet_energy(v_cut, fuv, uv, 1) - 4.

<<<<<<< HEAD
=======
    # X_sort = np.sort(X)
    # X_worst = X_sort[int(0.95*len(X_sort)):]
    # X_worst_log = np.log(np.maximum(X_worst, 1e-10))
    # if X_worst_log[-1] < 3:
    #     return

>>>>>>> alish
    # Get bin range (or None if no range values provided)
    if (args['bin_min'] and args['bin_max']):
        binrange = (args['bin_min'], args['bin_max'])
    elif args['bin_min']:
        binrange = (args['bin_min'], np.max(X))
    elif args['bin_max']:
        binrange = (np.min(X), args['bin_max'])
    else:
        binrange = None

    # Generate histogram and save to file
<<<<<<< HEAD
    os.makedirs(args['output_dir'], exist_ok=True)
    output_path = os.path.join(
        args['output_dir'],
        m+"_sym_dir.png"
    )

    matplotlib.rcParams['figure.figsize'] = (args['width'], args['height'])

    # Set percentage or absolute scale for y axis
    fig, ax = plt.subplots(1)
    bins=21
    hist = sns.histplot(X, bins = bins, stat='percent', binrange=binrange, ax=ax)
=======
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir,
        m+"_sym_dir.png"
    )

    matplotlib.rcParams['figure.figsize'] = (width, height)

    # Set percentage or absolute scale for y axis
    fig, ax = plt.subplots(1)
    bins=25
    hist = sns.histplot(X, bins = bins, stat='percent', binrange=binrange, ax=ax)
    # hist = sns.histplot(X_worst_log, bins = bins, stat='percent', binrange=binrange, ax=ax)
>>>>>>> alish
    hist.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylim(0, args['ylim'])

    # Set axes labels
    hist.set_xlabel(args['label'], fontsize=50)
    hist.set_ylabel("")
    hist.tick_params(labelsize=30)
    
    # Save figure to file
<<<<<<< HEAD
    fig.savefig(output_path, bbox_inches='tight')
=======
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    # Parse arguments for the script
    parser = argparse.ArgumentParser("Generate histograms for mesh")
    add_histogram_arguments(parser)
    args = vars(parser.parse_args())
    
    if args['mesh_dir']:
        # Process all .obj files in directory
        obj_files = glob.glob(os.path.join(args['mesh_dir'], "*.obj"))
        if not obj_files:
            print(f"[error] No .obj files found in {args['mesh_dir']}")
            exit(1)
        print(f"[info] Found {len(obj_files)} .obj file(s) in {args['mesh_dir']}")
        for obj_path in obj_files:
            process_single_mesh(
                obj_path,
                args['output_dir'],
                args['bin_min'],
                args['bin_max'],
                args['ylim'],
                args['label'],
                args['color'],
                args['width'],
                args['height']
            )
    elif args['uv_path']:
        # Process single file
        process_single_mesh(
            args['uv_path'],
            args['output_dir'],
            args['bin_min'],
            args['bin_max'],
            args['ylim'],
            args['label'],
            args['color'],
            args['width'],
            args['height']
        )
    else:
        print("[error] Must provide --uv_path or --mesh_dir")
        exit(1)
>>>>>>> alish
