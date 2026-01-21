
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

#ifdef ENABLE_VISUALIZATION
#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#endif

namespace SymDir{

struct CoordinateVector
{
    void resize(int n)
    {
        // nothing to do
    }


    Eigen::SparseVector<double> data;
};


class SeamlessSubspaceGenerator {
public:
SeamlessSubspaceGenerator() {}

void view(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const std::vector<int>& v_map
)
{
    int num_vertices = seam_vertices.size();
    Eigen::MatrixXd P(num_vertices, 3);
    std::vector<int> u_nonzeros(V.rows(), 0); 
    std::vector<int> v_nonzeros(V.rows(), 0); 
    for (int i = 0; i < num_vertices; ++i)
    {
        int vi = seam_vertices[i];
        P.row(i) = V.row(vi);
        u_nonzeros[vi] = coords[2 * i].nonZeros();
        v_nonzeros[vi] = coords[2 * i + 1].nonZeros();
        //if (is_independent_vertex[vi]) independent[vi] = 1;
    }

    std::vector<int> feature_degrees = compute_feature_degrees(v_map);
    std::vector<int> cut_degrees(uv.rows());
    for (int vi = 0; vi < uv.rows(); ++vi)
    {
        cut_degrees[vi] = feature_degrees[v_map[vi]];
    }

#ifdef ENABLE_VISUALIZATION
    polyscope::init();
    polyscope::registerSurfaceMesh("seam mesh", V, F);
    polyscope::getSurfaceMesh("seam mesh")->addVertexParameterizationQuantity("uv", uv);
    polyscope::getSurfaceMesh("seam mesh")->addVertexScalarQuantity("is independent", is_independent_vertex);
    polyscope::getSurfaceMesh("seam mesh")->addVertexScalarQuantity("u nonzeros", u_nonzeros);
    polyscope::getSurfaceMesh("seam mesh")->addVertexScalarQuantity("v nonzeros", v_nonzeros);
    polyscope::getSurfaceMesh("seam mesh")->addVertexScalarQuantity("order", seam_order);
    polyscope::getSurfaceMesh("seam mesh")->addVertexScalarQuantity("degree", cut_degrees);
    polyscope::registerPointCloud("seam vertices", P);
    //polyscope::getPointCloud("seam vertices")->addScalarQuantity("is independent", independent);
    polyscope::show();
#endif
}

typedef Eigen::SparseVector<double> CoordVector;
int count;
std::vector<int> seam_vertices;
std::vector<int> seam_index;
std::vector<bool> is_seam_vertex;
std::vector<bool> is_independent_vertex;
std::vector<bool> is_dependent_vertex;
std::vector<int> seam_order;
std::vector<CoordVector> coords;

std::deque<int> edges_to_process;
std::vector<bool> is_edge_seen;
std::vector<int> out;
std::vector<int> in;

void set_independent_leaves(const Eigen::MatrixXi& EE)
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

        make_ind_coord(seam_index[vi]);
        seam_order[vi] = count;
        ++count;
        is_edge_seen[e] = true;
        edges_to_process.push_back(e);
    }
}

int get_leaf_vertex(const Eigen::MatrixXi& EE)
{
    int e_start = 0;
    while ((EE(e_start, 0) != EE(e_start, 3)) && (EE(e_start, 1) != EE(e_start, 2)))
    {
        e_start++;
    }

    return (EE(e_start, 0) == EE(e_start, 3)) ? EE(e_start, 0) : EE(e_start, 2);
}

void traverse_tree(const Eigen::MatrixXi& EE)
{
    int num_seam_edges = EE.rows();
    int v_start = get_leaf_vertex(EE);
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
            if (!is_edge_seen[e])
            {
                make_ind_coord(seam_index[v_curr]);
                seam_order[v_curr] = count;
                ++count;
            }
        }
        is_edge_seen[e] = true;
    } while (v_curr != v_start);
    edges_to_process.resize(num_seam_edges);
    std::iota(edges_to_process.begin(), edges_to_process.end(), 0);
}

void build_seam_connectivity(const Eigen::MatrixXi& EE)
{
    int num_seam_edges = EE.rows();
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
}

void make_ind_coord(int index) {
    int i = index;
    coords[2 * i].coeffRef(2 * i) = 1.; 
    coords[(2 * i) + 1].coeffRef((2 * i) + 1) = 1.; 
    is_independent_vertex[seam_vertices[i]] = true;
}

int get_edge_rotation(const Eigen::MatrixXd& uv, const Eigen::MatrixXi& EE, int e)
{
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
    return r;
}

int count_set(
    const std::vector<bool>& is_set_vertex,
    const Eigen::MatrixXi& EE,
    int e)
{
    int num_set = 0;
    for (int i = 0; i < 4; ++i)
    {
        if (is_set_vertex[EE(e, i)]) ++num_set;
    }
    return num_set;
}

std::vector<int> compute_feature_degrees(
    const std::vector<int>& v_map)
{
    int num_vertices = (*std::max_element(v_map.begin(), v_map.end())) + 1;
    spdlog::info("computing degree for mesh with {} vertices", num_vertices);
    std::vector<int> feature_degrees(num_vertices, 0);
    for (int vi : seam_vertices)
    {
        feature_degrees[v_map[vi]] += 1;
    }

    return feature_degrees;
}

Eigen::SparseMatrix<double> run(
    const Eigen::MatrixXi& EE,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F,
    const std::vector<int>& v_map)
{
    int num_vertices = uv.rows();
    int num_seam_edges = EE.rows();

    // mark all seam vertices
    is_seam_vertex = std::vector<bool>(num_vertices, false);
    is_independent_vertex = std::vector<bool>(num_vertices, false);
    is_dependent_vertex = std::vector<bool>(num_vertices, false);
    seam_order = std::vector<int>(num_vertices, -1);
    out = std::vector<int>(num_vertices, -1);
    in = std::vector<int>(num_vertices, -1);

    // get reduced seam subspace
    seam_vertices = {};
    seam_vertices.reserve(num_vertices);
    seam_index = std::vector<int>(num_vertices, -1);
    build_seam_connectivity(EE);
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

    // compute junctions
    std::vector<int> feature_degrees = compute_feature_degrees(v_map);

    // set coordinate vector
    coords.resize(2 * num_seam_vertices); 
    for (int i = 0; i < num_seam_vertices; ++i)
    {
        int vi = seam_vertices[i];
        coords[2 * i].resize(2 * num_seam_vertices); 
        coords[2 * i + 1].resize(2 * num_seam_vertices); 
    }

    // set all leaves as independent
    count = 0;
    edges_to_process.clear();
    is_edge_seen = std::vector<bool>(num_seam_edges, false);

    bool use_independent_leaves = false;
    if (use_independent_leaves) set_independent_leaves(EE);

    bool use_tree_traversal = false;
    if (use_tree_traversal) traverse_tree(EE);

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

    int v_start = get_leaf_vertex(EE);
    int v_curr = v_start;
    do
    {
        int e = out[v_curr];

        // get next vertex
        if (EE(e, 0) == v_curr) v_curr = EE(e, 1);
        else v_curr = EE(e, 3);

        int degree = feature_degrees[v_map[v_curr]];

        if ((count_set(is_set_vertex, EE, e) != 3) || (degree != 2))
        {
            int index = seam_index[v_curr];
            make_ind_coord(index);
        }
        else
        {

            int A2 = EE(e, 0);
            int B2 = EE(e, 1);
            int C2 = EE(e, 2);
            int D2 = EE(e, 3);
            int r = get_edge_rotation(uv, EE, e);

            // get current coordinate vectors
            CoordVector& Au = coords[2 * seam_index[A2]];
            CoordVector& Bu = coords[2 * seam_index[B2]];
            CoordVector& Cu = coords[2 * seam_index[C2]];
            CoordVector& Du = coords[2 * seam_index[D2]];
            CoordVector& Av = coords[2 * seam_index[A2] + 1];
            CoordVector& Bv = coords[2 * seam_index[B2] + 1];
            CoordVector& Cv = coords[2 * seam_index[C2] + 1];
            CoordVector& Dv = coords[2 * seam_index[D2] + 1];

            // build constrained values
            if (v_curr == EE(e, 1))
            {
                Bu = Au + r_mat[r](0, 0) * (Du - Cu) + r_mat[r](0, 1) * (Dv - Cv);
                Bv = Av + r_mat[r](1, 0) * (Du - Cu) + r_mat[r](1, 1) * (Dv - Cv);
            }
            else
            {
                Du = Cu + r_rev[r](0, 0) * (Bu - Au) + r_rev[r](0, 1) * (Bv - Av);
                Dv = Cv + r_rev[r](1, 0) * (Bu - Au) + r_rev[r](1, 1) * (Bv - Av);
            }

            //if (coords[2 * seam_index[v_curr]].nonZeros() > 100)
            if (false)
            {
                coords[2 * seam_index[v_curr]].setZero();
                coords[2 * seam_index[v_curr] + 1].setZero();
                make_ind_coord(seam_index[v_curr]);
            }
            else
            {
                is_dependent_vertex[v_curr] = true;
            }
        }

        is_set_vertex[v_curr] = true;
        seam_order[v_curr] = count;
        ++count;
    } while (v_curr != v_start);

    while (count < num_seam_vertices)
    {
        break;
        if (edges_to_process.empty())
        {
            bool edge_found = false;
            for (int e = 0; e < num_seam_edges; ++e)
            {
                for (int i = 0; i < 4; ++i)
                {
                    if (!is_set_vertex[EE(e, i)])
                    {
                        int v = EE(e, i);
                        is_independent_vertex[v] = true;
                        is_dependent_vertex[v] = false;
                        int index = seam_index[v];
                        make_ind_coord(index);
                        is_set_vertex[v] = true;
                        seam_order[v] = count;
                        ++count;
                        edges_to_process.push_back(in[v]);
                        edges_to_process.push_back(out[v]);
                        edge_found = true;
                    }
                    if (edge_found) break;
                }
                if (edge_found) break;
            }
        }

        int e = edges_to_process.front();
        edges_to_process.pop_front();

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
        spdlog::debug("{} set in {}", num_set, e);
        if ((!use_tree_traversal) && (num_set == (num_v_in_edge - 2)))
        {
            spdlog::debug("setting independent vertex in {}", e);
            int i = 0;
            while (is_set_vertex[EE(e, i)]) ++i;
            int v = EE(e, i);
            is_independent_vertex[v] = true;
            is_dependent_vertex[v] = false;
            int index = seam_index[v];
            make_ind_coord(index);
            is_set_vertex[v] = true;
            seam_order[v] = count;
            ++count;
            edges_to_process.push_back(in[v]);
            edges_to_process.push_back(out[v]);
            num_set++;
        }
        if (num_set != (num_v_in_edge - 1)) continue;
        if (num_v_in_edge < 4) spdlog::warn("{} vertices in edge", num_v_in_edge);
        spdlog::debug("Setting dependent vertex in {}", e);

        int A2 = EE(e, 0);
        int B2 = EE(e, 1);
        int C2 = EE(e, 2);
        int D2 = EE(e, 3);
        int r = get_edge_rotation(uv, EE, e);

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
            seam_order[A2] = count;
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
            seam_order[B2] = count;
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
            seam_order[C2] = count;
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
            seam_order[D2] = count;
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
            Au.prune(1e-12);
            Av.prune(1e-12);
            spdlog::debug("{} u coord nonzeros for dependent {}", Au.nonZeros(), vi);
            spdlog::debug("{} v coord nonzeros for dependent {}", Av.nonZeros(), vi);
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
};

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
                spdlog::debug("{}/{} variables set", count, num_seam_vertices);
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

void ExtremeOpt::make_screenshot(int iter) {
#ifdef ENABLE_VISUALIZATION
    // screenshot every N iterations (adjust interval as needed)
    polyscope::init();
    // Rotate mesh for visualization
    Eigen::Matrix3d rot_matrix;
    rot_matrix = Eigen::AngleAxisd(igl::PI * m_params.angle_to_rotate_model_for_screenshots / 180.0, Eigen::Vector3d::UnitY());
    V = (rot_matrix * V.transpose()).transpose();

    auto mesh = polyscope::registerSurfaceMesh("optimization_mesh", V, F);
    
    // Add parameterization UV
    Eigen::MatrixXd uv_current;
    export_mesh(V, F, uv_current);
    uv_current = uv_current * m_params.uv_scale_for_screenshots;
    auto param_qty = mesh->addVertexParameterizationQuantity("UV parameterization", uv_current);
    param_qty->setEnabled(true);
    
    // Save screenshot
    std::string screenshot_path = fmt::format("{}/iter_{:05d}.png", m_params.output_dir_for_screenshots, iter);
    polyscope::screenshot(screenshot_path);
    spdlog::info("Saved screenshot to {}", screenshot_path);
    
    polyscope::removeAllStructures();
#endif
}

void ExtremeOpt::do_optimization(json& opt_log)
{
    igl::Timer timer;
    double time;

    std::cout << "Number of threads: " << Eigen::nbThreads() << std::endl;

    igl::Timer total_timer;
    total_timer.start();
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
    timer.start();
    if (m_params.use_rref)
    {
        if (m_params.precompute_seamless)
        {
            SeamlessSubspaceGenerator seamless_subspace_generator;
            Eigen::SparseMatrix<double> Q1 = seamless_subspace_generator.run(EE, uv, F, v_map);
            //seamless_subspace_generator.view(V, F, uv, v_map);
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
    time = timer.getElapsedTime();
    opt_log["time_log"]["constraint_elimination_time"] = time;
    spdlog::info("constraint elimination time serial: {}s", time);

    std::vector<HessianStats> hessian_log;
    bool failed = false;

    double max_grad_0 = smooth_global(failed, hessian_log);
    double max_grad = max_grad_0;

    double E = get_quality_avg_for_smooth_only();
    double E_worst_2 = get_quality_avg_worst_for_smooth_only();
    double E_2 = get_quality_avg_for_smooth_only(1.0);

    spdlog::info("Initial E = {}, E_worst_2 = {}", E, E_worst_2, m_params.E_min);

    opt_log["opt_log"].push_back(
        {{"F_size", F_size}, {"V_size", V_size}, {"E_avg", E}, {"E_worst", E_worst_2}, {"max_grad", max_grad}, {"elapsed_time", total_timer.getElapsedTime()}});

    std::vector<double> residuals;
    double E_0 = E;
    bool reached_one = false;
    int count = 0;
    double avr_step = 0;
    for (int i = 1; i <= m_params.max_iters; i++) {
        double E_old = E;
        double E_old_2 = E_2;
        // if times exceeds 3 minutes, stop optimization
        if (total_timer.getElapsedTime() > m_params.max_time) {
            spdlog::info("Time limit exceeded (>{}s). Stopping optimization early.", m_params.max_time);
            iters = i; // last completed iteration
            break;
        }


        double E_max;
        failed = false;
        
        if (this->m_params.global_smooth) {
            timer.start();
            max_grad = smooth_global(failed, hessian_log);
            spdlog::info("GLOBAL smoothing operation time serial: {}s", timer.getElapsedTime());
            // E = get_quality();

            E = get_quality_avg_for_smooth_only();
            E_worst_2 = get_quality_avg_worst_for_smooth_only();
            E_2 = get_quality_avg_for_smooth_only(1.0);
            //E_max = get_quality_max();

            // spdlog::info("After GLOBAL smoothing {}, E = {}", i, E);
            // spdlog::info("E_max = {}", E_max);
            spdlog::info("max gradient = {}", max_grad);

        }

        // opt_log["opt_log"].push_back(
        //     {{"F_size", F_size}, {"V_size", V_size}, {"E_max", E_max}, {"E_avg", E}, {"E_worst", E_worst}, {"max_grad", max_grad}});
        opt_log["opt_log"].push_back(
            {{"F_size", F_size}, {"V_size", V_size}, {"E_avg", E}, {"E_worst", E_worst_2}, {"max_grad", max_grad}, {"elapsed_time", total_timer.getElapsedTime()}});
        
        iters = i;

        // TODO: terminate criteria
        // 1. gradient stopping condition
        if (max_grad < m_params.grad_abs_err && m_params.max_grad_abs_converge) {
            std::string reason = fmt::format("Reach target gradient({}) with abs err {}, optimization succeed!", m_params.grad_abs_err, m_params.grad_abs_err);
            spdlog::info(reason);
            opt_log["converge_reason"] = reason;
            break;
        }
        if (max_grad < max_grad_0 * m_params.grad_rel_err && m_params.max_grad_rel_converge) {
            std::string reason = fmt::format("Reach target gradient({}) with rel err {}, optimization succeed!", max_grad_0 * m_params.grad_rel_err, m_params.grad_rel_err);
            spdlog::info(reason);
            opt_log["converge_reason"] = reason;
            break;
        }
        // 2. energy stopping condition
        if (E_worst_2 < m_params.E_worst_2_target + 1e-8 && m_params.E_worst_2_target_converge) {
            std::string reason = fmt::format("Energy reached {}", m_params.E_worst_2_target);
            spdlog::info("Energy reached {}, optimization converge!", m_params.E_worst_2_target);
            opt_log["converge_reason"] = reason;
            break;        

        }
        avr_step += fabs(E - E_old) / E_old;
        count += 1;
        if (count % 3) {
            if (avr_step / 3.0 < m_params.diff_err && m_params.energy_diff_converge) {
                std::string reason = fmt::format("Energy change too small ({}) in {} steps", avr_step / 3.0, count);
                spdlog::info("Energy change too small ({}) in {} steps, optimization succeed!", avr_step / 3.0, count);
                opt_log["converge_reason"] = reason;
                break;        
            }
            avr_step = 0;
        }

        if (fabs(E) < m_params.E_abs_err && m_params.E_abs_converge) {
            std::string reason = fmt::format("Energy converged to {} with abs err {}, optimization succeed!", E, m_params.E_abs_err);
            spdlog::info(reason);
            opt_log["converge_reason"] = reason;
            break;
        }
        if (fabs(E) < m_params.E_rel_err * E_0 && m_params.E_rel_converge) {
            std::string reason = fmt::format("Energy converged to {} with rel err {}, optimization succeed!", E, m_params.E_rel_err);
            spdlog::info(reason);
            opt_log["converge_reason"] = reason;
            break;
        }
        // VISUALIZATION: take screenshot at each iteration
        if (m_params.screenshot_during_optimization) {
            if (i % m_params.screenshot_interval == 0 || i == 1) {  
                make_screenshot(i);
            }
        }

        if (failed) {
            std::string reason = "Line search step failed.";
            spdlog::info(reason);
            opt_log["converge_reason"] = reason;
            break;
        }
    }
    if (m_params.last_screenshot_after_optimization) {
        make_screenshot(iters);
    }

    std::string reason = "Reached max iteration.";
    spdlog::info(reason);
    if (!opt_log.contains("converge_reason"))
        opt_log["converge_reason"] = reason;

    total_time = total_timer.getElapsedTime();
    spdlog::info("Total optimization time: {}s", total_time);
    opt_log["iters"] = iters;
    opt_log["hessian_log"] = hessian_log;

    double time_ls = 0.0;
    double time_solver = 0.0;
    double time_grad_hessian = 0.0;
    for (size_t i = 0; i < hessian_log.size(); ++i)
    {
        time_ls += hessian_log[i].time_ls;
        time_solver += hessian_log[i].time_solver;
        time_grad_hessian += hessian_log[i].time_grad_hessian;
    }
    opt_log["time_log"]["line_search_time"] = time_ls;
    opt_log["time_log"]["solver_time"] = time_solver;
    opt_log["time_log"]["grad_hessian_time"] = time_grad_hessian;
    opt_log["E_worst"] = E_worst_2;
}
} // namespace SymDir
