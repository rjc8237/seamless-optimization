#pragma once
#include <string>

namespace SymDir {
struct Parameters
{
    std::string model_name;
    bool save_meshes = false;

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
    /* data */
    /* solver type*/
    std::string solver_type = "LDLT"; // "CG" or "LDLT"
};

} // namespace SymDir
