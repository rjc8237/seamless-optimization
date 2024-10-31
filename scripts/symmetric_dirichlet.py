# Script to project a marked metric to holonomy constraints with feature alignment

import os, sys
base_dir = os.path.dirname(__file__)
module_dir = os.path.join(base_dir, '..', 'py')
sys.path.append(module_dir)
script_dir = os.path.join(base_dir, '..', 'ext', 'penner-optimization', 'scripts')
sys.path.append(script_dir)
import numpy as np
import penner
import symdir_py as symdir
import igl
import optimization_scripts.script_util as script_util

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def run_one(args, fname):
    # Get mesh and test name
    dot_index = fname.rfind(".")
    m = fname[:dot_index]
    name = m
    fmesh = name + '_refined_with_uv.obj'

    # Create output directory for the mesh
    output_dir = script_util.get_mesh_output_directory(args['output_dir'], m)
    os.makedirs(output_dir, exist_ok=True)

    # Get logger
    log_path = os.path.join(output_dir, name+'_optimize_aligned_angles.log')
    logger = script_util.get_logger(log_path)
    logger.info("Running symmetric dirichlet on {}".format(fmesh))

    try:
        V, uv, _, F, FT, _ = igl.read_obj(os.path.join(args['input_dir'], name + '_output', fmesh))
    except:
        logger.info("Could not open mesh data")
        return

    # cut the mesh
    meshcutter = symdir.MeshCutter(V, uv, F, FT)
    V_cut, EE = meshcutter.cut_mesh()
    FE_init = meshcutter.load_feature_edges(os.path.join(args['input_dir'], name + '_output', fmesh))
    FE = meshcutter.reindex_feature_edges(FE_init)

    alg_params = symdir.Parameters()
    alg_params.model_name = name
    alg_params.save_meshes = args['save_meshes']
    alg_params.do_feature_alignment = args['do_feature_alignment']
    alg_params.max_iters = args['max_iters']
    alg_params.smooth_only_iters = args['smooth_only_iters']
    alg_params.do_newton = args['do_newton']
    alg_params.local_smooth = args['local_smooth']
    alg_params.global_smooth = args['global_smooth']
    alg_params.ls_iters = args['ls_iters']
    alg_params.E_target = args['E_target']
    alg_params.elen_alpha = args['elen_alpha']
    alg_params.with_cons = args['with_cons']
    alg_params.do_projection = args['do_projection']
    alg_params.use_max_energy = args['use_max_energy']
    alg_params.Lp = args['Lp']

    # Additional print statements to verify correct assignments

    opt_log = {}
    opt_log['model_name'] = name
    #opt_log['config'] = alg_params

    extremeopt = symdir.ExtremeOpt(V_cut, FT)
    extremeopt.create_mesh(V_cut, FT, uv)
    extremeopt.m_params = alg_params

    extremeopt.init_constraints(EE, FE)

    if extremeopt.check_constraints(1e-7):
        print("initial constraints satisfied")
    else:
        print("initial constraints are not satisfied")
    
    extremeopt.do_optimization(opt_log)

    logger.info("check constraints inside wmtk")
    if extremeopt.check_constraints(1e-7):
        print("constraints satisfied")
    else:
        print("constraints are not satisfied")
    
    extremeopt.export_mesh(V, FT, uv)

    extremeopt.export_EE(EE)

    uv_mesh_path = os.path.join(output_dir, name + '_out.obj')
    logger.info("Saving optimized mesh at {}".format(uv_mesh_path))
    penner.write_obj_with_uv(uv_mesh_path, V, F, uv, FT)

    # save form to output file
    output_path = os.path.join(output_dir, name + '_EE.txt')
    np.savetxt(output_path, EE)
    output_path = os.path.join(output_dir, name + '_FE.txt')
    np.savetxt(output_path, FE)

def run_many(args):
    script_util.run_many(run_one, args)

def add_arguments(parser):
    alg_params = symdir.Parameters()
    parser.add_argument("-f", "--fname",         help="filenames of the obj file", 
                                                     nargs='+')
    parser.add_argument("-i", "--input_dir",     help="input folder that stores obj files and Th_hat")
    parser.add_argument("--do_feature_alignment", help="do feature alignment",
                                                     type=bool, default=alg_params.do_feature_alignment)
    parser.add_argument("--max_iters",   help="maximum number of iterations for the symmetric dirichlet optimization",
                                                     type=int, default=alg_params.max_iters)
    parser.add_argument("--smooth_only_iters", help="maximum number of smooth only iterations",
                                                     type=int, default=alg_params.smooth_only_iters)
    parser.add_argument("--do_newton",      help="do newton iteration",
                                                     type=bool, default=alg_params.do_newton)
    parser.add_argument("--local_smooth",      help="do local smoothing",
                                                     type=bool, default=alg_params.local_smooth)
    parser.add_argument("--global_smooth",      help="do global smoothing",
                                                     type=bool, default=alg_params.global_smooth)
    parser.add_argument("--ls_iters",           help="number of line search iterations", 
                                                    type=int, default=alg_params.ls_iters)
    parser.add_argument("--E_target",           help="target energy value", 
                                                    type=float, default=alg_params.E_target)
    parser.add_argument("--elen_alpha",         help="element alpha value", 
                                                    type=float, default=alg_params.elen_alpha)
    parser.add_argument("--do_projection",      help="do projection", 
                                                    type=bool, default=alg_params.do_projection)
    parser.add_argument("--with_cons",          help="apply constraints", 
                                                    type=bool, default=alg_params.with_cons)
    parser.add_argument("--save_meshes",        help="save meshes", 
                                                    type=bool, default=alg_params.save_meshes)
    parser.add_argument("--Lp",                 help="Lp value", 
                                                    type=int, default=alg_params.Lp)
    parser.add_argument("--use_max_energy",                 help="Use max energy", 
                                                    type=bool, default=alg_params.use_max_energy)
    parser.add_argument("-o",  "--output_dir",
                        help="directory for output lambdas and logs")
    
if __name__ == "__main__":
    # Parse arguments for the script
    parser = script_util.generate_parser("Optimize angles with relaxed alignment")
    add_arguments(parser)
    args = vars(parser.parse_args())

    # Run parallel method
    run_many(args)
