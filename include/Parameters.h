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
    double norm_p = 1.0;
    bool use_rref = true;

    // for worst n% triangles energies
    double percent = 5.0;
    int p_energy = 5; // for p-norm (\sum_{T}(E_T)^p)^(1/p))
    std::string solver_type = "LDLT"; // solver type
    double E_rel_tol = 1e-3; //tolerance for E_worst convergence
    double cg_rel_err = 5e-4; // relative error for cg solver

    /* data */
};

} // namespace SymDir
