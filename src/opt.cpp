
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
#include "rref.h"
#include <math.h>       

#include <igl/facet_components.h>

namespace SymDir{

void buildAeq(
    const Eigen::MatrixXi& EE,
    const Eigen::MatrixXi& FE,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F,
    Eigen::SparseMatrix<double>& Aeq)
{
    int N = uv.rows();
    int c = 0;
    int m = EE.rows() / 2;
    int fes = FE.rows();

    // get u aligned edges for each component
    auto [min_v_diffs, min_v_diff_ids, min_v_diff_next_ids] = find_u_aligned_edges(uv, F);
    int num_components = min_v_diffs.size();

    int n_fix_dof;
    if (fes > 0) 
    {
        n_fix_dof = 2 * num_components;
    }
    else
    {
        n_fix_dof = 3 * num_components;
    }

    std::set<std::pair<int, int>> added_e;
    typedef Eigen::Triplet<double> Trip;
    std::vector<Trip> trips;
    trips.reserve(12 * EE.rows());

    Aeq.resize(2 * m + n_fix_dof + fes, uv.rows() * 2);
    int A2, B2, C2, D2;
    for (int i = 0; i < EE.rows(); i++) {
        int A2 = EE(i, 0);
        int B2 = EE(i, 1);
        int C2 = EE(i, 2);
        int D2 = EE(i, 3);
        auto e0 = std::make_pair(A2, B2);
        auto e1 = std::make_pair(C2, D2);
        if (added_e.find(e0) != added_e.end() || added_e.find(e1) != added_e.end()) continue;
        added_e.insert(e0);
        added_e.insert(e1);

        Eigen::Matrix<double, 2, 1> e_ab = uv.row(B2) - uv.row(A2);
        Eigen::Matrix<double, 2, 1> e_dc = uv.row(C2) - uv.row(D2);

        Eigen::Matrix<double, 2, 1> e_ab_perp;
        e_ab_perp(0) = -e_ab(1);
        e_ab_perp(1) = e_ab(0);
        double angle = atan2(-e_ab_perp.dot(e_dc), e_ab.dot(e_dc));
        int r = (int)(round(2 * angle / double(igl::PI)) + 2) % 4;

        std::vector<Eigen::Matrix<double, 2, 2>> r_mat(4);
        r_mat[0] << -1, 0, 0, -1;
        r_mat[1] << 0, 1, -1, 0;
        r_mat[2] << 1, 0, 0, 1;
        r_mat[3] << 0, -1, 1, 0;

        trips.push_back(Trip(c, A2, 1));
        trips.push_back(Trip(c, B2, -1));
        trips.push_back(Trip(c + 1, A2 + N, 1));
        trips.push_back(Trip(c + 1, B2 + N, -1));

        trips.push_back(Trip(c, C2, r_mat[r](0, 0)));
        trips.push_back(Trip(c, D2, -r_mat[r](0, 0)));
        trips.push_back(Trip(c, C2 + N, r_mat[r](0, 1)));
        trips.push_back(Trip(c, D2 + N, -r_mat[r](0, 1)));
        trips.push_back(Trip(c + 1, C2, r_mat[r](1, 0)));
        trips.push_back(Trip(c + 1, D2, -r_mat[r](1, 0)));
        trips.push_back(Trip(c + 1, C2 + N, r_mat[r](1, 1)));
        trips.push_back(Trip(c + 1, D2 + N, -r_mat[r](1, 1)));
        c = c + 2;
    }

    for (int ci = 0; ci < num_components; ++ci)
    {
        int min_v_diff_id = min_v_diff_ids[ci];
        if (min_v_diff_id == -1)
        {
            spdlog::warn("for component {}, skipping edge fix", ci);
            n_fix_dof -= 2;
            continue;
        }
        spdlog::debug("for component {}, fixing {}", ci, min_v_diff_id);
        trips.push_back(Trip(c, min_v_diff_id, 1));
        trips.push_back(Trip(c + 1, min_v_diff_id + N, 1));
        c = c + 2;
    }
    // fix rotation
    if (fes == 0)
    {
        for (int ci = 0; ci < num_components; ++ci)
        {
            int min_v_diff_id = min_v_diff_ids[ci];
            if (min_v_diff_id == -1)
            {
                n_fix_dof -= 1;
                continue;
            }
            spdlog::info("for component {}, fixing rotation {}", ci, min_v_diff_next_ids[ci]);
            trips.push_back(Trip(c, min_v_diff_next_ids[ci], 1));
            c = c + 1;
        }
    }
    else {
        //std::set<std::pair<int, int>> added_fe;

        // feature edge constraints
        for (int i = 0; i < fes; ++i) {
            int v1 = FE(i, 0);
            int v2 = FE(i, 1);
            auto e0 = std::make_pair(v1, v2);
            //if (added_fe.find(e0) != added_fe.end())
            //{
            //    spdlog::warn("Edge added twice");
            //    continue;
            //}
            //added_fe.insert(e0);
            
            Eigen::Vector2d e_ab = uv.row(v2) - uv.row(v1);

            // constrain u or v depending on initial position
            if (FE(i, 2) == 0) {
                trips.push_back(Trip(c, v1, -1));
                trips.push_back(Trip(c, v2, 1));
                c += 1;
            }
            else if (FE(i, 2) == 1) {
                trips.push_back(Trip(c, v1 + N, -1));
                trips.push_back(Trip(c, v2 + N, 1));
                c += 1;
            }
            else
            {
                spdlog::error("Feature edge does not have a tag");
            }
        }
    }
    Aeq.resize(2 * m + n_fix_dof + fes, uv.rows() * 2);
    Aeq.setFromTriplets(trips.begin(), trips.end());
}

void buildBeq(
    const Eigen::MatrixXi& ME,
    const Eigen::MatrixXd& uv,
    Eigen::SparseMatrix<double>& Beq)
{
    int N = uv.rows();
    int num_misaligned = ME.rows();
    spdlog::info("Adding {} misalignment constraints", num_misaligned);
    typedef Eigen::Triplet<double> Trip;
    std::vector<Trip> trips;
    trips.reserve(2 * num_misaligned);

    for (int i = 0; i < num_misaligned; ++i)
    {
        int v1 = ME(i, 0);
        int v2 = ME(i, 1);
		Eigen::Vector2d e_ab = uv.row(v2) - uv.row(v1);

		// constrain u or v depending on initial position
		if (std::abs(e_ab[0]) < std::abs(e_ab[1])) {
            spdlog::info("Adding ({}, {}) u constraint with error {}", v1, v2, e_ab[0]);
            trips.push_back(Trip(i, v1, -1));
            trips.push_back(Trip(i, v2, 1));
		} else {
            spdlog::info("Adding ({}, {}) v constraint with error {}", v1, v2, e_ab[1]);
            trips.push_back(Trip(i, v1 + N, -1));
            trips.push_back(Trip(i, v2 + N, 1));
		}
	}
    Beq.resize(num_misaligned, uv.rows() * 2);
    Beq.setFromTriplets(trips.begin(), trips.end());
}

void ExtremeOpt::do_optimization(json& opt_log)
{
    igl::Timer timer;
    double time;

    std::cout << "Number of threads: " << Eigen::nbThreads() << std::endl;

    igl::Timer total_timer;
    double total_time = 0;
    int iters = 0;
    opt_log["total_time"] = total_time;
    opt_log["iters"] = iters;

    // get edge length thresholds for collapsing operation
    Eigen::MatrixXd uv;
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

    int V_size = get_vertices().size();
    int F_size = get_faces().size();
    
    // get input operators
    spdlog::info("Building constraints");
    igl::doublearea(V, F, area);
    get_grad_op(V, F, G);
    buildAeq(EE, FE, uv, F, Aeq);
    buildBeq(ME, uv, Beq);
    AeqT = Aeq.transpose();

    // build reduced system
    if (m_params.use_rref)
    {
        spdlog::info("Eliminating constraints");
        elim_constr(Aeq, Q2);
        Q2.makeCompressed();
        Q2T = Q2.transpose();
    }

    total_timer.start();
    
    std::vector<HessianStats> hessian_log;
    bool failed = false;

    double val_e_min = 1.0;
    m_params.E_min = val_e_min;
    double max_grad_0 = smooth_global(failed, hessian_log);
    m_params.E_min = 1.0;
    double max_grad = max_grad_0;

    double E = get_quality_avg_for_smooth_only();
    double E_worst = get_quality_avg_worst_for_smooth_only();
    spdlog::info("Initial E = {}, E_worst = {}, E_min = {}", E, E_worst, m_params.E_min);

    max_grad_0 = max_grad_0 * std::pow(E, (1.0 - 2 * m_params.Lp) / (2 * m_params.Lp));
    E = std::pow(E, 1.0 / (2 * m_params.Lp));
    E_worst = std::pow(E_worst, 1.0 / (2 * m_params.Lp));
    double E_worst_2 = get_quality_avg_worst_for_smooth_only(1.0);
    double E_2 = get_quality_avg_for_smooth_only(1.0);
    opt_log["opt_log"].push_back(
        {{"F_size", F_size}, {"V_size", V_size}, {"E_avg", E_2}, {"E_worst", E_worst_2}, {"max_grad", max_grad}, {"elapsed_time", total_timer.getElapsedTime()}});

    std::vector<double> residuals;
    double E_0 = E_worst;
    bool reached_one = false;
    for (int i = 1; i <= m_params.max_iters; i++) {
        double E_old = E_worst;
    
        if (E_worst_2 < 1.0 && reached_one == false) {
            total_time = total_timer.getElapsedTime();
            reached_one = true;
        }

        // if times exceeds 3 minutes, stop optimization
        if (total_timer.getElapsedTime() > 180.0) {
            spdlog::info("Time limit exceeded (>{}s). Stopping optimization early.", 180);
            iters = i; // last completed iteration
            break;
        }


        double E_max;
        failed = false;
        if (this->m_params.global_smooth) {
            timer.start();
            m_params.E_min = val_e_min;
            max_grad = smooth_global(failed, hessian_log);
            m_params.E_min = 1.0;
            time = timer.getElapsedTime();
            spdlog::info("GLOBAL smoothing operation time serial: {}s", time);

            // E = get_quality();
            E = get_quality_avg_for_smooth_only();
            E_worst = get_quality_avg_worst_for_smooth_only();

            E_2 = get_quality_avg_for_smooth_only(1.0);
            E_worst_2 = get_quality_avg_worst_for_smooth_only(1.0);
            //E_max = get_quality_max();

            // spdlog::info("After GLOBAL smoothing {}, E = {}", i, E);
            // spdlog::info("E_max = {}", E_max);
            spdlog::info("max gradient = {}", max_grad);
        }
        max_grad = max_grad * std::pow(E, (1.0 - 2 * m_params.Lp) / (2 * m_params.Lp));
        E = std::pow(E, 1.0 / (2 * m_params.Lp));
        E_worst = std::pow(E_worst, 1.0 / (2 * m_params.Lp));

        // opt_log["opt_log"].push_back(
        //     {{"F_size", F_size}, {"V_size", V_size}, {"E_max", E_max}, {"E_avg", E}, {"E_worst", E_worst}, {"max_grad", max_grad}});
        opt_log["opt_log"].push_back(
            {{"F_size", F_size}, {"V_size", V_size}, {"E_avg", E_2}, {"E_worst", E_worst_2}, {"max_grad", max_grad}, {"elapsed_time", total_timer.getElapsedTime()}});
        
        iters = i;

        // TODO: terminate criteria
        if (max_grad < m_params.grad_abs_err || max_grad < max_grad_0 * m_params.grad_rel_err) {
            spdlog::info(
                "Reach target gradient({}), optimization succeed!",
                m_params.grad_abs_err);
            break;
        }
        if (fabs(E_worst - E_old) < 1e-4) {
            spdlog::info("Energy change too small ({}), optimization converge!", fabs(E_worst - E_old));
            break;
        }
        if (fabs(E_worst) < m_params.E_rel_err * E_0 || fabs(E_worst) < m_params.E_abs_err) {
            spdlog::info("Energy converged!", fabs(E_worst), m_params.E_rel_err * E_0, m_params.E_abs_err);
            break;
        }
        
        if (failed) {
            spdlog::info(
                "Line search step failed. stopping optimization early."
            );
            break;
        }

        std::cout << std::endl;
    }
    if (reached_one == true) {
        opt_log["E_worst=1.0"] = total_time;
    }
    total_time = total_timer.getElapsedTime();
    spdlog::info("Total optimization time: {}s", total_time);
    double total_time_rounded = std::round(total_time);
    opt_log["total_time"] = total_time_rounded;
    opt_log["iters"] = iters;
    opt_log["hessian_log"] = hessian_log;
    E_worst = get_quality_avg_worst_for_smooth_only(1.0);
    opt_log["E_worst"] = E_worst;

}
} // namespace SymDir
