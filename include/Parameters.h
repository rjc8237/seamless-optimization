#pragma once
#include <string>

namespace SymDir {
struct Parameters
{
    std::string model_name;
    bool save_meshes = false;

    bool do_feature_alignment = true;
    bool fix_misaligned = false;

    int max_iters = 500;
    int max_time = 180;
    int smooth_only_iters = 100;
    bool do_newton = false;
    bool local_smooth = false;
    bool global_smooth = true;
    int ls_iters = 200;
    double E_target = 10.0;
    double elen_alpha = 2.0;
    bool with_cons = true;
    bool do_projection = true;
    
    bool use_max_energy = false;
    int Lp = 4;
    double symdir_weight = 0.001;
    double alignment_weight = 1.0;
    bool use_rref = true;

    // for worst n% triangles energies
    double percent = 5.0;
    std::string solver_type = "LDLT"; // solver type
    double E_rel_err = 1e-6; //tolerance for E_worst convergence
    double E_abs_err = 1e-6; // absolute tolerance for E_worst convergence
    double grad_rel_err = 1e-6; //tolerance for gradient convergence
    double grad_abs_err = 1e-6; // absolute tolerance for gradient convergence
    double cg_rel_err = 5e-4; // relative error for cg solver
    double diff_err = 1e-2; // relative change for energy convergence

    bool precompute_seamless = false; // use explicit seamless subspace construction
 
    bool projected_newton = false; // use projected newton method

    // use soft max for energy computation
    bool soft_max = false; 
    double t = 1.0; // temperature for soft max
    
    double E_min = 1.0;
    /* data */
};

} // namespace SymDir
