# Script to generate histograms for per-vertex colormaps used for rendering meshes

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

if __name__ == "__main__":
    # Parse arguments for the script
    parser = argparse.ArgumentParser("Generate histograms for mesh")
    add_histogram_arguments(parser)
    args = vars(parser.parse_args())

    # Get mesh and test name
    fname = os.path.basename(args['uv_path'])
    dot_index = fname.rfind(".")
    m = fname[:dot_index]

    # Load uv information
    try:
        uv_path = args['uv_path']
        v3d, uv, _, f, fuv, _ = igl.readOBJ(args['uv_path'])
    except:
        print("FUCK")
        exit()

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
    X = symdir.symmetric_dirichlet_energy(v_cut, uv, fuv) - 4.0
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
    hist.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.set_ylim(0, args['ylim'])

    # Set axes labels
    hist.set_xlabel(args['label'], fontsize=50)
    hist.set_ylabel("")
    hist.tick_params(labelsize=30)
    
    # Save figure to file
    fig.savefig(output_path, bbox_inches='tight')
