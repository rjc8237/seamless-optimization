
//#include <Eigen/src/Core/util/Constants.h>
#include <igl/Timer.h>

#include <Eigen/Sparse>
#include <array>

#include <igl/upsample.h>
#include <igl/writeOBJ.h>
#include <limits>
#include <optional>
#include "energy.h"
#include "ExtremeOpt.h"
#include "spdlog/spdlog.h"


namespace SymDir{

void ExtremeOpt::do_optimization(json& opt_log)
{
    igl::Timer timer;
    igl::Timer total_timer;
    double time;
    double total_time = 0;
    opt_log["total_time"] = total_time;


    // get edge length thresholds for collapsing operation
    Eigen::MatrixXd V, uv;
    Eigen::MatrixXi F;
    export_mesh(V, F, uv);
    elen_threshold = sqrt(
        pow((uv.col(0).maxCoeff() - uv.col(0).minCoeff()), 2) +
        pow((uv.col(1).maxCoeff() - uv.col(1).minCoeff()), 2));
    elen_threshold_3d = sqrt(
        pow((V.col(0).maxCoeff() - V.col(0).minCoeff()), 2) +
        pow((V.col(1).maxCoeff() - V.col(1).minCoeff()), 2) +
        pow((V.col(2).maxCoeff() - V.col(2).minCoeff()), 2));
    elen_threshold *= m_params.elen_alpha;
    elen_threshold_3d *= m_params.elen_alpha;

    double E = get_quality();
    spdlog::info("Start Energy E = {}", E);

    double E_max = get_quality_max();
    spdlog::info("Start E_max = {}", E_max);

    opt_log["opt_log"].push_back(
        {{"F_size", get_faces().size()},
         {"V_size", get_vertices().size()},
         {"E_max", E_max},
         {"E", E}});
    double E_old = E;
    int V_size, F_size;

    double max_grad = 0;
    total_timer.start();
    for (int i = 1; i <= m_params.max_iters; i++) {
        double E_max;

        if (this->m_params.global_smooth) {
            timer.start();
            max_grad = smooth_global(1);
            time = timer.getElapsedTime();
            spdlog::info("GLOBAL smoothing operation time serial: {}s", time);

            // E = get_quality();
            E = get_quality_avg_for_smooth_only();
            E_max = get_quality_max();

            spdlog::info("After GLOBAL smoothing {}, E = {}", i, E);
            spdlog::info("E_max = {}", E_max);
            spdlog::info("max gradient = {}", max_grad);
        }

        opt_log["opt_log"].push_back(
            {{"F_size", F_size}, {"V_size", V_size}, {"E_max", E_max}, {"E_avg", E}, {"max_grad", max_grad}});
    
        double grad_thres = 1e-5;
        if (max_grad < grad_thres) {
            spdlog::info(
                "Reach target gradient({}), optimization succeed!",
                grad_thres);
            break;
        }

        // TODO: terminate criteria
        if (E < m_params.E_target) {
            spdlog::info(
                "Reach target energy({}), optimization succeed!",
                m_params.E_target);
            break;
        }

        E_old = E;
        std::cout << std::endl;
    }
    total_time = total_timer.getElapsedTime();
    spdlog::info("Total optimization time: {}s", total_time);
    opt_log["total_time"] = total_time;
}
} // namespace SymDir
