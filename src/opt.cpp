
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

#include <igl/boundary_loop.h>
#include <igl/facet_components.h>

namespace SymDir{

std::vector<int> propagate_component_labels(const Eigen::MatrixXi& F, const Eigen::VectorXi& C, int N)
{
    std::vector<int> component_vertices(N, -1);
    for (int f = 0; f < F.rows(); ++f)
    {
        for (int i = 0; i < 3; ++i)
        {
            int vi = F(f, i);
            component_vertices[vi] = C[f];
        }
    }
    return component_vertices;
}

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

    std::vector<std::vector<int>> bds;
    igl::boundary_loop(F, bds);

    // get face components
    Eigen::VectorXi C;
    int num_components = igl::facet_components(F, C);
    std::vector<int> component_faces(num_components);
    std::vector<int> component_vertices{ propagate_component_labels(F, C, N) };
    for (int f = 0; f < F.rows(); ++f)
    {
        component_faces[C[f]] = f;
    }

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

    std::vector<double> min_u_diffs(num_components, 1e10);
    std::vector<int> min_u_diff_ids(num_components, -1);
    std::vector<int> min_u_diff_next_ids(num_components, -1);

    for (const auto& l : bds)
    {
        for (int i = 0; i < l.size(); i++) {
            double u_diff = abs(uv(l[i], 0) - uv(l[(i + 1) % l.size()], 0));
            int component = component_vertices[l[i]];
            if (u_diff < min_u_diffs[component]) {
                min_u_diffs[component] = u_diff;
                min_u_diff_ids[component] = l[i];
                min_u_diff_next_ids[component] = l[(i + 1) % l.size()];
            }
        }
    }

    for (int ci = 0; ci < num_components; ++ci)
    {
        int min_u_diff_id = min_u_diff_ids[ci];
        if (min_u_diff_id == -1)
        {
            spdlog::warn("for component {}, skipping edge fix", ci);
            n_fix_dof -= 2;
            continue;
        }
        spdlog::debug("for component {}, fixing {}", ci, min_u_diff_id);
        trips.push_back(Trip(c, min_u_diff_id, 1));
        trips.push_back(Trip(c + 1, min_u_diff_id + N, 1));
        c = c + 2;
    }
    // fix rotation
    if (fes == 0)
    {
        for (int ci = 0; ci < num_components; ++ci)
        {
            int min_u_diff_id = min_u_diff_ids[ci];
            if (min_u_diff_id == -1)
            {
                n_fix_dof -= 1;
                continue;
            }
            spdlog::info("for component {}, fixing rotation {}", ci, min_u_diff_next_ids[ci]);
            trips.push_back(Trip(c, min_u_diff_next_ids[ci], 1));
            c = c + 1;
        }
    }
    else {
        std::set<std::pair<int, int>> added_fe;

        // feature edge constraints
        for (int i = 0; i < fes; ++i) {
            int v1 = FE(i, 0);
            int v2 = FE(i, 1);
            auto e0 = std::make_pair(v1, v2);
            if (added_fe.find(e0) != added_fe.end())
            {
                spdlog::warn("Edge added twice");
                continue;
            }
            added_fe.insert(e0);
            
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

void ExtremeOpt::do_optimization(json& opt_log)
{
    igl::Timer timer;
    double time;


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

    // get input operators
    igl::doublearea(V, F, area);
    get_grad_op(V, F, G);
    buildAeq(EE, FE, uv, F, Aeq);
    AeqT = Aeq.transpose();

    // build reduced system
    elim_constr(Aeq, Q2);
    Q2.makeCompressed();
    Q2T = Q2.transpose();

    double max_grad = 0;
    for (int i = 1; i <= m_params.max_iters; i++) {
        double E_max;

        if (this->m_params.global_smooth) {
            timer.start();
            max_grad = smooth_global();
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

    
        double grad_thres = 1e-4;
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
}
} // namespace SymDir
