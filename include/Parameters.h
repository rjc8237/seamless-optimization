#pragma once
#include <string>

namespace SymDir {
struct Parameters
{
    std::string model_name;
    std::string output_dir;
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
    double degenerate_weight = 1.0;
    bool use_rref = true;

    // for worst n% triangles energies
    std::vector<double> percentages = {5.0};
    double percentage_target = 5.0;
    double percentage_target_value = 1.0;
    bool save_percentages_meshes = false;
    
    std::string solver_type = "LDLT"; // solver type
    double E_rel_err = 1e-6; //tolerance for E_worst convergence
    double E_abs_err = 1e-6; // absolute tolerance for E_worst convergence
    double grad_rel_err = 1e-6; //tolerance for gradient convergence
    double grad_abs_err = 1e-6; // absolute tolerance for gradient convergence
    double cg_rel_err = 5e-4; // relative error for cg solver
    double cg_iters = 10000; // iterations for cg solver
    double diff_err = 1e-2; // relative change for energy convergence

    bool precompute_seamless = false; // use explicit seamless subspace construction
 
    bool projected_newton = false; // use projected newton method

    // use soft max for energy computation
    bool soft_max = false; 
    double t = 1.0; // temperature for soft max
    
    double E_min = 1.0;

    bool last_screenshot_after_optimization = false;
    int screenshot_interval = 5;
    std::string output_dir_for_screenshots = "./output_screenshots/";
    double uv_scale_for_screenshots = 1.0; // Adjust the grid size for better visibility
    double angle_to_rotate_model_for_screenshots = 0.0; // Rotate the model for a better view
    bool screenshot_during_optimization = false;

    bool percentage_target_converge = true;
    bool max_grad_abs_converge = true;
    bool max_grad_rel_converge = true;
    bool energy_diff_converge = true;
    bool use_worst_n_energy_in_ls = true;
    bool E_abs_converge = true;
    bool E_rel_converge = true;

    bool fix_boundary = false; // fix all boundary edges instead of using seamless constraints

    bool degenerate_vertices_preconditioner = false;
    int precond_dim = 2; // dim = 2 - for degenerate uv triangles, dim = 3 - for degenerate 3d triangles
    double triangle_threshold = 25.0; // threshold for marking degenerate triangles for preconditioner and line search
    /* data */
};

} // namespace SymDir
