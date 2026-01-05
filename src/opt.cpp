
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

#include <igl/facet_components.h>

namespace SymDir{

/*
void buildAeq_explicit(
    const Eigen::MatrixXi& EE,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F,
    Eigen::SparseMatrix<double>& Aeq)
{
    int N = uv.rows();
    int c = 0;
    int m = EE.rows() / 2;

    // get u aligned edges for each component
    auto [min_v_diffs, min_v_diff_ids, min_v_diff_next_ids] = find_u_aligned_edges(uv, F);
    int num_components = min_v_diffs.size();

    int n_fix_dof = 3 * num_components;

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
    Aeq.resize(2 * m + n_fix_dof + fes, uv.rows() * 2);
    Aeq.setFromTriplets(trips.begin(), trips.end());
}
*/

struct CoordinateVector
{
    void resize(int n)
    {
        // nothing to do
    }


    Eigen::SparseVector<double> data;
};


Eigen::SparseMatrix<double> build_seamless_subspace(
    const Eigen::MatrixXi& EE,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F)
{
    int num_vertices = uv.rows();
    int num_seam_edges = EE.rows();

    // mark all seam vertices
    std::vector<bool> is_seam_vertex(num_vertices, false);
    std::vector<bool> is_independent_vertex(num_vertices, false);
    std::vector<bool> is_dependent_vertex(num_vertices, false);
    std::vector<int> out(num_vertices, -1);
    std::vector<int> in(num_vertices, -1);
    for (int eij = 0; eij < num_seam_edges; ++eij)
    {
        for (int i = 0; i < 4; ++i)
        {
            is_seam_vertex[EE(eij, i)] = true;
        }
        int A2 = EE(eij, 0);
        int B2 = EE(eij, 1);
        int C2 = EE(eij, 2);
        int D2 = EE(eij, 3);
        out[A2] = eij;
        in[B2] = eij;
        out[C2] = eij;
        in[D2] = eij;
    }

    // get reduced seam subspace
    std::vector<int> seam_index(num_vertices, -1);
    std::vector<int> seam_vertices = {};
    seam_vertices.reserve(num_vertices);
    for (int vi = 0; vi < num_vertices; ++vi)
    {
        if (is_seam_vertex[vi])
        {
            seam_index[vi] = seam_vertices.size();
            seam_vertices.push_back(vi);
        }
    }
    int num_seam_vertices = seam_vertices.size();
    spdlog::info("{} seam vertices", num_seam_vertices);

    // set all leaves as independent
    std::vector<int> edges_to_process = {};
    std::vector<bool> is_edge_seen(num_seam_edges, false);
    bool use_independent_leaves = true;
    if (use_independent_leaves)
    {
        for (int e = 0; e < EE.rows(); e++)
        {
            int A2 = EE(e, 0);
            int B2 = EE(e, 1);
            int C2 = EE(e, 2);
            int D2 = EE(e, 3);
            int vi = -1;
            if (A2 == D2) vi = A2;
            else if (B2 == C2) vi = B2;
            else continue;

            is_independent_vertex[vi] = true;
            is_edge_seen[e] = true;
            edges_to_process.push_back(e);
        }
    }

    bool use_tree_traversal = false;
    if (use_tree_traversal)
    {
        int e_start = 0;
        while ((EE(e_start, 0) != EE(e_start, 3)) && (EE(e_start, 1) != EE(e_start, 2)))
        {
            e_start++;
        }

        int v_start = (EE(e_start, 0) == EE(e_start, 3)) ? EE(e_start, 0) : EE(e_start, 2);
        int v_curr = v_start;
        do
        {
            int e = out[v_curr];

            // get next vertex
            if (EE(e, 0) == v_curr) v_curr = EE(e, 1);
            else v_curr = EE(e, 3);

            // determine if vertex is dependent or independent (if not already set)
            if (!is_independent_vertex[v_curr])
            {
                if (!is_edge_seen[e]) is_independent_vertex[v_curr] = true;
                else is_dependent_vertex[v_curr] = true;
            }
            is_edge_seen[e] = true;
        } while (v_curr != v_start);
        edges_to_process.resize(num_seam_edges);
        std::iota(edges_to_process.begin(), edges_to_process.end(), 0);
    }
    //typedef Eigen::VectorXd CoordVector;
    typedef Eigen::SparseVector<double> CoordVector;
    std::vector<CoordVector> coords(2 * num_seam_vertices); 
    auto make_ind_coord = [&](int i) {
        coords[2 * i].coeffRef(2 * i) = 1.; 
        coords[(2 * i) + 1].coeffRef((2 * i) + 1) = 1.; 
    };

    int count = 0;
    for (int i = 0; i < num_seam_vertices; ++i)
    {
        int vi = seam_vertices[i];
        coords[2 * i].resize(2 * num_seam_vertices); 
        coords[2 * i + 1].resize(2 * num_seam_vertices); 
        if (is_independent_vertex[vi])
        {
            make_ind_coord(i);
            ++count;
        }
    }

    // build rotation matrix for different matchings
    std::vector<Eigen::Matrix<double, 2, 2>> r_mat(4);
    r_mat[0] << 1, 0, 0, 1;
    r_mat[1] << 0, -1, 1, 0;
    r_mat[2] << -1, 0, 0, -1;
    r_mat[3] << 0, 1, -1, 0;

    std::vector<Eigen::Matrix<double, 2, 2>> r_rev(4);
    r_rev[0] << 1, 0, 0, 1;
    r_rev[1] << 0, 1, -1, 0;
    r_rev[2] << -1, 0, 0, -1;
    r_rev[3] << 0, -1, 1, 0;

    std::vector<bool> is_set_vertex = is_independent_vertex;
    std::vector<bool> is_set_edge(num_seam_edges, false);
    //int safety_count = 0;
    while (count < num_seam_vertices)
    {
        if (edges_to_process.empty()) break;
        int e = edges_to_process.back();
        edges_to_process.pop_back();

        // only process edges with one unset vertex
        int num_set = 0;
        int unset_index = -1;
        int num_v_in_edge = 4;
        for (int i = 0; i < 4; ++i)
        {
            if (is_set_vertex[EE(e, i)]) ++num_set;
            else unset_index = i;
        }
        if ((EE(e, 0) == EE(e, 3)) && (!is_set_vertex[EE(e, 0)])) num_v_in_edge--;
        if ((EE(e, 1) == EE(e, 2)) && (!is_set_vertex[EE(e, 1)])) num_v_in_edge--;
        //if (num_set == 4) continue;
        //if (num_set == 2) spdlog::info("{} has 2 set vertices", e);
        spdlog::info("{} set in {}", num_set, e);
        if (num_set == (num_v_in_edge - 2))
        {
            spdlog::info("setting independent vertex in {}", e);
            int i = 0;
            while (is_set_vertex[EE(e, i)]) ++i;
            int v = EE(e, i);
            is_independent_vertex[v] = true;
            is_dependent_vertex[v] = false;
            int index = seam_index[v];
            make_ind_coord(index);
            is_set_vertex[v] = true;
            ++count;
            edges_to_process.push_back(in[v]);
            edges_to_process.push_back(out[v]);
            num_set++;
        }
        if (num_set != (num_v_in_edge - 1)) continue;
        if (num_v_in_edge < 4) spdlog::warn("{} vertices in edge", num_v_in_edge);
        spdlog::info("Setting dependent vertex in {}", e);

        int A2 = EE(e, 0);
        int B2 = EE(e, 1);
        int C2 = EE(e, 2);
        int D2 = EE(e, 3);

        // get oriented eddge positions
        Eigen::Matrix<double, 2, 1> e_ab = uv.row(B2) - uv.row(A2);
        Eigen::Matrix<double, 2, 1> e_dc = uv.row(C2) - uv.row(D2);

        // get ccw rotation of edge ab
        Eigen::Matrix<double, 2, 1> e_ab_perp;
        e_ab_perp(0) = -e_ab(1);
        e_ab_perp(1) = e_ab(0);
        double angle = atan2(-e_ab_perp.dot(e_dc), e_ab.dot(e_dc));
        int r = (int)(round(2 * angle / double(igl::PI)) + 2) % 4;

        // get current coordinate vectors
        CoordVector& Au = coords[2 * seam_index[A2]];
        CoordVector& Bu = coords[2 * seam_index[B2]];
        CoordVector& Cu = coords[2 * seam_index[C2]];
        CoordVector& Du = coords[2 * seam_index[D2]];
        CoordVector& Av = coords[2 * seam_index[A2] + 1];
        CoordVector& Bv = coords[2 * seam_index[B2] + 1];
        CoordVector& Cv = coords[2 * seam_index[C2] + 1];
        CoordVector& Dv = coords[2 * seam_index[D2] + 1];

        if (!is_set_vertex[A2])
        {
            if (A2 == D2)
            {
                spdlog::info("skipping A tail vertex {} / {}", count, num_seam_vertices);
                //Au - r_mat[r](0, 0) * Au - r_mat[r](0, 1) * Av = Bu - r_mat[r](0, 0) * Du - r_mat[r](0, 1) * Dv
                //Av - r_mat[r](1, 0) * Au - r_mat[r](1, 1) * Av = Bv - r_mat[r](1, 0) * Du - r_mat[r](1, 1) * Dv
                //Av = Bv - r_mat[r](1, 0) * (Du - Cu) - r_mat[r](1, 1) * (Dv - Cv);
                is_independent_vertex[A2] = true;
                is_dependent_vertex[A2] = false;
                int i = seam_index[A2];
                make_ind_coord(i);
            }
            else
            {
                // build constrained values
                Au = Bu - r_mat[r](0, 0) * (Du - Cu) - r_mat[r](0, 1) * (Dv - Cv);
                Av = Bv - r_mat[r](1, 0) * (Du - Cu) - r_mat[r](1, 1) * (Dv - Cv);
                is_independent_vertex[A2] = false;
                is_dependent_vertex[A2] = true;
            }
            edges_to_process.push_back(in[A2]);
            is_set_vertex[A2] = true;
            is_set_edge[e] = true;
            ++count;
        }
        else if (!is_set_vertex[B2])
        {
            if (B2 == C2)
            {
                spdlog::info("skipping B tail vertex {} / {}", count, num_seam_vertices);
                // Bu + r_mat[r](0, 0) * Bu + r_mat[r](0, 1) * Bv = Au + r_mat[r](0, 0) * Du + r_mat[r](0, 1) * Dv
                // Bv + r_mat[r](1, 0) * Bu + r_mat[r](1, 1) * Bv = Au + r_mat[r](1, 0) * Du + r_mat[r](1, 1) * Dv
                is_independent_vertex[B2] = true;
                is_dependent_vertex[B2] = false;
                int i = seam_index[B2];
                make_ind_coord(i);
            }
            else
            {
                // build constrained values
                Bu = Au + r_mat[r](0, 0) * (Du - Cu) + r_mat[r](0, 1) * (Dv - Cv);
                Bv = Av + r_mat[r](1, 0) * (Du - Cu) + r_mat[r](1, 1) * (Dv - Cv);
                is_independent_vertex[B2] = false;
                is_dependent_vertex[B2] = true;
            }
            edges_to_process.push_back(out[B2]);
            is_set_vertex[B2] = true;
            is_set_edge[e] = true;
            ++count;
        }
        else if (!is_set_vertex[C2])
        {
            if (C2 == B2)
            {
                spdlog::info("skipping C tail vertex {} / {}", count, num_seam_vertices);
                is_independent_vertex[C2] = true;
                is_dependent_vertex[C2] = false;
                int i = seam_index[C2];
                make_ind_coord(i);
            }
            else
            {
                // build constrained values
                Cu = Du - r_rev[r](0, 0) * (Bu - Au) - r_rev[r](0, 1) * (Bv - Av);
                Cv = Dv - r_rev[r](1, 0) * (Bu - Au) - r_rev[r](1, 1) * (Bv - Av);
                is_independent_vertex[C2] = false;
                is_dependent_vertex[C2] = true;
            }
            edges_to_process.push_back(in[C2]);
            is_set_vertex[C2] = true;
            is_set_edge[e] = true;
            ++count;
        }
        else if (!is_set_vertex[D2])
        {
            if (D2 == A2)
            {
                spdlog::info("skipping D tail vertex {} / {}", count, num_seam_vertices);
                is_independent_vertex[D2] = true;
                is_dependent_vertex[D2] = false;
                int i = seam_index[D2];
                make_ind_coord(i);
            }
            else
            {
                // build constrained values
                Du = Cu + r_rev[r](0, 0) * (Bu - Au) + r_rev[r](0, 1) * (Bv - Av);
                Dv = Cv + r_rev[r](1, 0) * (Bu - Au) + r_rev[r](1, 1) * (Bv - Av);
                is_independent_vertex[D2] = false;
                is_dependent_vertex[D2] = true;
            }
            edges_to_process.push_back(out[D2]);
            is_set_vertex[D2] = true;
            is_set_edge[e] = true;
            ++count;
        }
        else
        {
            spdlog::error("unset variable that should not exist");
        }
        //spdlog::info("{}/{} variables set", count, num_seam_vertices);
    }

    std::vector<int> dep_vertices = {};
    std::vector<int> indep_vertices = {};
    std::vector<int> reduced_index(num_vertices, -1);
    dep_vertices.reserve(num_vertices);
    indep_vertices.reserve(num_vertices);
    int num_reduced = 0;
    for (int vi = 0; vi < num_vertices; ++vi)
    {
        if (is_independent_vertex[vi])
        {
            indep_vertices.push_back(vi);
        }
        if (is_dependent_vertex[vi])
        {
            dep_vertices.push_back(vi);
        }
        else
        {
            reduced_index[vi] = num_reduced;
            ++num_reduced;
        }
    }
    int num_indep_vertices = indep_vertices.size();
    spdlog::info("{} independent vertices", indep_vertices.size());
    spdlog::info("{} dependent vertices", dep_vertices.size());

    typedef Eigen::Triplet<double> Trip;
    std::vector<Trip> trips;
    trips.reserve(12 * EE.rows());
    int N = num_vertices;
    for (int vi = 0; vi < num_vertices; ++vi)
    {
        if (is_dependent_vertex[vi])
        {
            CoordVector& Au = coords[2 * seam_index[vi]];
            CoordVector& Av = coords[2 * seam_index[vi] + 1];
            //Au.prune([&](int /*i*/, double val) { return std::abs(val) > 1e-12; });
            //Av.prune([&](int /*i*/, double val) { return std::abs(val) > 1e-12; });
            Au.prune(1e-12);
            Av.prune(1e-12);
            spdlog::info("{} u coord nonzeros for dependent {}", Au.nonZeros(), vi);
            spdlog::info("{} v coord nonzeros for dependent {}", Av.nonZeros(), vi);
            for (CoordVector::InnerIterator it(Au); it; ++it)
            {
                int i = it.index() >> 1;
                int vj = seam_vertices[i];
                int index = reduced_index[vj];
                double coord = it.value();
                if ((it.index() % 2) == 0) trips.push_back(Trip(vi, index, coord));
                else trips.push_back(Trip(vi, index + num_reduced, coord));
            }

            for (CoordVector::InnerIterator it(Av); it; ++it)
            {
                int i = it.index() >> 1;
                int vj = seam_vertices[i];
                int index = reduced_index[vj];
                double coord = it.value();
                if ((it.index() % 2) == 0) trips.push_back(Trip(vi + N, index, coord));
                else trips.push_back(Trip(vi + N, index + num_reduced, coord));
            }

            /*
            for (int i = 0; i < num_indep_vertices; ++i)
            {
                int vj = indep_vertices[i];
                int index = reduced_index[vj];
                if (Au[2 * i] != 0.)
                {
                    trips.push_back(Trip(vi, index, Au[2 * i]));
                }
                if (Au[2 * i + 1] != 0.)
                {
                    trips.push_back(Trip(vi, index + num_reduced, Au[2 * i + 1]));
                }
                if (Av[2 * i] != 0.)
                {
                    trips.push_back(Trip(vi + N, index, Av[2 * i]));
                }
                if (Av[2 * i + 1] != 0.)
                {
                    trips.push_back(Trip(vi + N, index + num_reduced, Av[2 * i + 1]));
                }
            }
            */
        }
        else
        {
            trips.push_back(Trip(vi, reduced_index[vi], 1));
            trips.push_back(Trip(vi + N, reduced_index[vi] + num_reduced, 1));
        }
    }


    Eigen::SparseMatrix<double> basis(2 * N, 2 * num_reduced);
    basis.setFromTriplets(trips.begin(), trips.end());

    return basis;
}

Eigen::SparseMatrix<double> build_seamless_subspace_(
    const Eigen::MatrixXi& EE,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F)
{
    int num_vertices = uv.rows();
    int num_seam_edges = EE.rows();

    // mark all seam vertices
    std::vector<bool> is_seam_vertex(num_vertices, false);
    for (int eij = 0; eij < num_seam_edges; ++eij)
    {
        for (int i = 0; i < 4; ++i)
        {
            is_seam_vertex[EE(eij, i)] = true;
        }
        //is_dependent_vertex[EE(eij, 1)] = true;
        //is_independent_vertex[EE(eij, 3)] = true;
    }

    // get reduced seam subspace
    std::vector<int> seam_vertices = {};
    //std::vector<int> dep_vertices = {};
    std::vector<int> seam_index(num_vertices, -1);
    //dep_vertices.reserve(num_vertices);
    int num_reduced = 0;
    for (int vi = 0; vi < num_vertices; ++vi)
    {
        if (is_seam_vertex[vi])
        {
            seam_index[vi] = seam_vertices.size();
            seam_vertices.push_back(vi);
        }
    }

    int num_seam_vertices = seam_vertices.size();
    spdlog::info("{} seam vertices", num_seam_vertices);

    std::vector<Eigen::VectorXd> coords(2 * num_seam_vertices); 
    for (int i = 0; i < num_seam_vertices; ++i)
    {
        coords[2 * i].setZero(2 * num_seam_vertices);
        coords[(2 * i) + 1].setZero(2 * num_seam_vertices);
    }

    std::vector<int> reduced_index(num_vertices, -1);
    seam_vertices.reserve(num_vertices);

    std::vector<bool> is_set(num_vertices, false);
    std::vector<int> indep_vertices = {};
    indep_vertices.reserve(num_vertices);
    int count = 0;
    for (int e = 0; e < EE.rows(); e++)
    {
        int A2 = EE(e, 0);
        int B2 = EE(e, 1);
        int C2 = EE(e, 2);
        int D2 = EE(e, 3);
        int i = -1;
        if (A2 == D2) i = seam_index[A2];
        else if (B2 == C2) i = seam_index[B2];
        else continue;

        coords[2 * i][2 * i] = 1.; 
        coords[(2 * i) + 1][(2 * i) + 1] = 1.; 
        is_set[seam_vertices[i]] = true;
        indep_vertices.push_back(i);
        count++;
    }
    spdlog::info("{} vertices determined", count);

    // build rotation matrix for different matchings
    std::vector<Eigen::Matrix<double, 2, 2>> r_mat(4);
    r_mat[0] << 1, 0, 0, 1;
    r_mat[1] << 0, -1, 1, 0;
    r_mat[2] << -1, 0, 0, -1;
    r_mat[3] << 0, 1, -1, 0;

    std::vector<Eigen::Matrix<double, 2, 2>> r_rev(4);
    r_rev[0] << 1, 0, 0, 1;
    r_rev[1] << 0, 1, -1, 0;
    r_rev[2] << -1, 0, 0, -1;
    r_rev[3] << 0, -1, 1, 0;


    while (count < num_seam_vertices)
    {
        for (int e = 0; e < EE.rows(); e++)
        {
            int A2 = EE(e, 0);
            int B2 = EE(e, 1);
            int C2 = EE(e, 2);
            int D2 = EE(e, 3);

            // check if all set
            if ((is_set[A2] && is_set[D2]) && (is_set[B2] && is_set[C2])) continue;

            // check if at least two set
            if ((is_set[A2] && is_set[D2]) || (is_set[B2] && is_set[C2]))
            {
                // get oriented eddge positions
                Eigen::Matrix<double, 2, 1> e_ab = uv.row(B2) - uv.row(A2);
                Eigen::Matrix<double, 2, 1> e_dc = uv.row(C2) - uv.row(D2);

                // get ccw rotation of edge ab
                Eigen::Matrix<double, 2, 1> e_ab_perp;
                e_ab_perp(0) = -e_ab(1);
                e_ab_perp(1) = e_ab(0);
                double angle = atan2(-e_ab_perp.dot(e_dc), e_ab.dot(e_dc));
                int r = (int)(round(2 * angle / double(igl::PI)) + 2) % 4;

                // get current coordinate vectors
                Eigen::VectorXd& Au = coords[2 * seam_index[A2]];
                Eigen::VectorXd& Bu = coords[2 * seam_index[B2]];
                Eigen::VectorXd& Cu = coords[2 * seam_index[C2]];
                Eigen::VectorXd& Du = coords[2 * seam_index[D2]];
                Eigen::VectorXd& Av = coords[2 * seam_index[A2] + 1];
                Eigen::VectorXd& Bv = coords[2 * seam_index[B2] + 1];
                Eigen::VectorXd& Cv = coords[2 * seam_index[C2] + 1];
                Eigen::VectorXd& Dv = coords[2 * seam_index[D2] + 1];

                if (is_set[A2] && is_set[D2])
                {
                    if ((!is_set[C2]) && (!is_set[B2]))
                    {
                        int i = seam_index[C2];
                        coords[2 * i][2 * i] = 1.; 
                        coords[(2 * i) + 1][(2 * i) + 1] = 1.; 
                        is_set[C2] = true;
                        indep_vertices.push_back(i);
                        ++count;

                        // build constrained values
                        Bu = Au + r_mat[r](0, 0) * (Du - Cu) + r_mat[r](0, 1) * (Dv - Cv);
                        Bv = Av + r_mat[r](1, 0) * (Du - Cu) + r_mat[r](1, 1) * (Dv - Cv);
                        is_set[B2] = true;
                        ++count;
                    }
                    else if ((is_set[C2]) && (!is_set[B2]))
                    {
                        // build constrained values
                        Bu = Au + r_mat[r](0, 0) * (Du - Cu) + r_mat[r](0, 1) * (Dv - Cv);
                        Bv = Av + r_mat[r](1, 0) * (Du - Cu) + r_mat[r](1, 1) * (Dv - Cv);
                        is_set[B2] = true;
                        ++count;
                    }
                    else if ((!is_set[C2]) && (is_set[B2]))
                    {
                        // build constrained values
                        is_set[C2] = true;
                        ++count;
                    }
                }
                else if (is_set[B2] && is_set[C2])
                {
                    if ((!is_set[A2]) && (!is_set[D2]))
                    {
                        int i = seam_index[A2];
                        coords[2 * i][2 * i] = 1.; 
                        coords[(2 * i) + 1][(2 * i) + 1] = 1.; 
                        is_set[A2] = true;
                        indep_vertices.push_back(i);
                        ++count;

                        // build constrained values
                        Du = Cu + r_rev[r](0, 0) * (Bu - Au) + r_rev[r](0, 1) * (Bv - Av);
                        Dv = Cv + r_rev[r](1, 0) * (Bu - Au) + r_rev[r](1, 1) * (Bv - Av);
                        is_set[D2] = true;
                        ++count;
                    }
                    else if ((is_set[A2]) && (!is_set[D2]))
                    {
                        // build constrained values
                        Du = Cu + r_rev[r](0, 0) * (Bu - Au) + r_rev[r](0, 1) * (Bv - Av);
                        Dv = Cv + r_rev[r](1, 0) * (Bu - Au) + r_rev[r](1, 1) * (Bv - Av);
                        is_set[D2] = true;
                        ++count;
                    }
                    else if ((!is_set[A2]) && (is_set[D2]))
                    {
                        is_set[A2] = true;
                        ++count;
                    }
                }
                spdlog::info("{}/{} variables set", count, num_seam_vertices);
            }
        }
    }
    spdlog::info("{} independent vertices", indep_vertices.size());

    typedef Eigen::Triplet<double> Trip;
    std::vector<Trip> trips;
    trips.reserve(12 * EE.rows());
    int N = num_vertices;
    /*
    for (int vi = 0; vi < num_vertices; ++vi)
    {
        if (is_dependent_vertex[vi])
        {
            Eigen::VectorXd& Au = coords[2 * seam_index[vi]];
            Eigen::VectorXd& Av = coords[2 * seam_index[vi] + 1];
            for (int i = 0; i < num_indep_vertices; ++i)
            {
                int vj = indep_vertices[i];
                int index = reduced_index[vj];
                if (Au[2 * i] != 0.)
                {
                    trips.push_back(Trip(vi, index, Au[2 * i]));
                }
                if (Au[2 * i + 1] != 0.)
                {
                    trips.push_back(Trip(vi, index + num_reduced, Au[2 * i + 1]));
                }
                if (Av[2 * i] != 0.)
                {
                    trips.push_back(Trip(vi + N, index, Av[2 * i]));
                }
                if (Av[2 * i + 1] != 0.)
                {
                    trips.push_back(Trip(vi + N, index + num_reduced, Av[2 * i + 1]));
                }
            }
        }
        else
        {
            trips.push_back(Trip(vi, reduced_index[vi], 1));
            trips.push_back(Trip(vi + N, reduced_index[vi] + num_reduced, 1));
        }
    }
    */


    Eigen::SparseMatrix<double> basis(2 * N, 2 * num_reduced);
    basis.setFromTriplets(trips.begin(), trips.end());

    return basis;
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
    //int m = EE.rows() / 2;
    int m = EE.rows();
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

    //std::set<std::pair<int, int>> added_e;
    typedef Eigen::Triplet<double> Trip;
    std::vector<Trip> trips;
    trips.reserve(12 * EE.rows());

    int A2, B2, C2, D2;
    for (int i = 0; i < EE.rows(); i++) {
        int A2 = EE(i, 0);
        int B2 = EE(i, 1);
        int C2 = EE(i, 2);
        int D2 = EE(i, 3);
        //auto e0 = std::make_pair(A2, B2);
        //auto e1 = std::make_pair(C2, D2);
        //if (added_e.find(e0) != added_e.end() || added_e.find(e1) != added_e.end()) continue;
        //added_e.insert(e0);
        //added_e.insert(e1);

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
    spdlog::info("Building constraints");
    igl::doublearea(V, F, area);
    get_grad_op(V, F, G);
    buildAeq(EE, FE, uv, F, Aeq);
    buildBeq(ME, uv, Beq);
    AeqT = Aeq.transpose();

    // build reduced system
    if (m_params.use_rref)
    {
        bool precompute_explicit = true;
        if (precompute_explicit)
        {
            Eigen::SparseMatrix<double> Q1 = build_seamless_subspace(EE, uv, F);
            spdlog::info("partially reduced system matrix has {} nonzeros", Q1.nonZeros());
            Eigen::SparseMatrix<double> Ared = Aeq * Q1;

            spdlog::info("Eliminating constraints");
            Eigen::SparseMatrix<double> Q3;
            elim_constr(Ared, Q3);
            Q2 = Q1 * Q3;
            Q2.makeCompressed();
            Q2T = Q2.transpose();
            spdlog::info("reduced system matrix has {} nonzeros", Q2.nonZeros());
        }
        else
        {
            spdlog::info("Eliminating constraints");
            elim_constr(Aeq, Q2);
            Q2.makeCompressed();
            Q2T = Q2.transpose();
            spdlog::info("reduced system matrix has {} nonzeros", Q2.nonZeros());
        }
    }

    if (m_params.use_rref)
    {
    }

    double max_grad = 0;
    for (int i = 1; i <= m_params.max_iters; i++) {
        double E_max;
        bool failed = false;
        if (this->m_params.global_smooth) {
            timer.start();
            max_grad = smooth_global(failed);
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
        if (failed) {
            spdlog::info(
                "Line search step failed. stopping optimization early."
            );
            break;
        }

        E_old = E;
        std::cout << std::endl;
    }
}
} // namespace SymDir
