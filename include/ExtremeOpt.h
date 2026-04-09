#pragma once

#include <igl/PI.h>
#include <igl/Timer.h>
#include <igl/AABB.h>
#include <Eigen/Sparse>
#include "Parameters.h"
#include "json.hpp"
#include "spdlog/spdlog.h"
#ifdef ENABLE_VISUALIZATION
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/curve_network.h"
#endif
#include "energy.h"

using json = nlohmann::json;


namespace SymDir {

/**
 * @brief Find edges per connected component as aligned with the positive u-axis as possible
 * 
 * @param uv: parameterization vertices
 * @param F: parameterization faces
 * @return list of changes in the v coordinate per edge
 * @return list of edge start vertices
 * @return list of edge end vertices
 */
std::tuple<std::vector<double>, std::vector<int>, std::vector<int>>
find_u_aligned_edges(
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F);

void get_grad_op(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    Eigen::SparseMatrix<double>& grad_op);

Eigen::VectorXd symmetric_dirichlet_energy(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    double norm_p);

std::vector<bool> mark_degenerate_faces(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& uv, const std::vector<int>& v_map, int dim=2, double threshold=1e5);
std::vector<bool> mark_degenerate_vertices(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& uv, const std::vector<int>& v_map, int dim=2, double threshold=1e5);

class VertexAttributes
{
public:
    Eigen::Vector2d pos = Eigen::Vector2d::Zero();
    Eigen::Vector3d pos_3d = Eigen::Vector3d::Zero();

    //size_t partition_id = 0; // TODO this should not be here

    // Vertices marked as fixed cannot be modified by any local operation
    //bool fixed = false;
};

class Mesh;

// equivalent to a halfedge
class Tuple
{
public:
    Tuple()
    : hij(-1)
    , is_reversed(false)
    {}

    Tuple(int _h, bool _is_reversed=false)
    : hij(_h)
    , is_reversed(_is_reversed)
    {}

    int vid(const Mesh& m) const;
    int fid(const Mesh& m) const;
    int eid(const Mesh& m) const;

    Tuple switch_vertex(const Mesh& m) const;
    Tuple switch_face(const Mesh& m) const;
    Tuple switch_edge(const Mesh& m) const;

    // only relevant when boundary implemented
    bool has_value() const {return (hij >= 0);}
    Tuple value() const {return *this;}

    int hij;
    bool is_reversed;
// TODO
//    Tuple(int fijk, int i)
//    : face_index(fijk)
//    , local_corner_index(i)
//    {
//
//    }
//    int face_index;
//    int local_corner_index;
};

class FaceAttributes
{
public:
    double area_3d;
};

class EdgeAttributes
{
public:
    Tuple pair;
};

std::vector<int> compose(const std::vector<int>& right_f, const std::vector<int>& left_f);
bool is_one_sided_inverse(const std::vector<int>& right_f, const std::vector<int>& left_f);
bool is_invariant(const std::vector<int>& label, const std::vector<int>& permutation);
bool are_vectors_equal(const std::vector<int>& f, const std::vector<int>& g);

class Mesh {
public:
    Mesh() {}

    Mesh(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F
     )
    : input_V(V)
    , input_F(F)
    {
        // resize arrays
        int num_vertices = V.rows();
        int num_faces = F.rows();
        int num_halfedges = num_faces * 3;
        next.resize(num_halfedges);
        prev.resize(num_halfedges);
        opposite.resize(num_halfedges);
        he2f.resize(num_halfedges);
        to.resize(num_halfedges);
        from.resize(num_halfedges);
        he2e.resize(num_halfedges);
        f2he.resize(num_faces);
        out.resize(num_vertices);
        vv2he.resize(num_vertices, num_vertices);

        // iterate over faces to build halfedge connectivity
        typedef Eigen::Triplet<int> Trip;
        std::vector<Trip> vv2he_triplets;
        for (int fijk = 0; fijk < num_faces; ++fijk)
        {
            for (int i = 0; i < 3; ++i)
            {
                // build local and global indices
                int j = (i + 1) % 3;
                int k = (i + 2) % 3;
                int vj = F(fijk, j);
                int vk = F(fijk, k);
                int hjk = (3 * fijk) + i;
                int hki = (3 * fijk) + j;

                // halfedge data
                next[hjk] = hki;
                prev[hki] = hjk;

                // face data
                he2f[hjk] = fijk;

                // vertex data
                from[hjk] = vj;
                to[hjk] = vk;
                out[vk] = hki;
                vv2he_triplets.push_back(Trip(vj, vk, hjk + 1)); // offset to 1 indexing
            }

            // set face halfedge to point to F(fijk, 0)
            f2he[fijk] = (3 * fijk) + 1;
        }

        // Create vertex  maps
        vv2he.setFromTriplets(vv2he_triplets.begin(), vv2he_triplets.end());

        // build opposite mapping
        for (int hij = 0; hij < num_halfedges; ++hij)
        {
            int vi = from[hij];
            int vj = to[hij];
            int hji = vv2he.coeff(vj, vi) - 1; // return to 0 indexing
            opposite[hij] = hji;
        }

        // build edge identification
        int edge_count = 0;
        e2he.reserve(num_halfedges);
        for (int hij = 0; hij < num_halfedges; ++hij)
        {
            int hji = opposite[hij];
            if (hji < 0)
            {
                e2he.push_back(hij);
                he2e[hij] = edge_count;
                ++edge_count;
            }
            else
            {
                if (hij < hji) continue; // only process once
                e2he.push_back(hij);
                he2e[hij] = edge_count;
                he2e[hji] = edge_count;
                ++edge_count;
            }
        }

        if (!is_valid_mesh())
        {
            spdlog::error("mesh not built correctly");
        }
    }

    Eigen::MatrixXd input_V;
    Eigen::MatrixXi input_F;

    std::vector<Tuple> get_vertices() const;
    std::vector<Tuple> get_faces() const;
    std::vector<Tuple> get_edges() const;
    std::vector<Tuple> get_one_ring_tris_for_vertex(const Tuple& t) const;
    std::array<Tuple, 3> oriented_tri_vertices(const Tuple& t) const;
    bool is_boundary_edge(const Tuple& t) const { return (opposite[t.hij] < 0); }
    int num_vertices() const { return out.size(); }
    int num_faces() const { return f2he.size(); }
    int num_edges() const { return e2he.size(); }

    bool are_vertices_valid() const
    {
        if (input_V.rows() != out.size()) return false;
        if (!is_one_sided_inverse(from, out))
        {
            spdlog::error("from-out not identity");
            return false;
        }
        if (!are_vectors_equal(to, compose(from, next)))
        {
            spdlog::error("from-next not to");
            return false;
        }
        // FIXME handle opposite logic
        //if (!is_invariant(to, compose(opposite, next)))
        //{
        //    spdlog::error("to not invariant under vertex tip circulation");
        //    return false;
        //}
        std::vector<Tuple> vertices = get_vertices();
        for (int vi = 0; vi < num_vertices(); ++vi)
        {
            if (vertices[vi].vid(*this) != vi)
            {
                spdlog::error("vertex tuple list not complete");
                return false;
            }
        }

        return true;
    }

    bool are_faces_valid() const
    {
        if (input_F.rows() != f2he.size()) return false;
        if (!is_one_sided_inverse(he2f, f2he)) return false;
        if (!is_invariant(he2f, next)) return false;

        return true;
    }

    bool are_halfedges_valid() const
    {
        if (!is_one_sided_inverse(next, prev))
        {
            spdlog::error("next-prev not identity");
            return false;
        }
        if (!is_one_sided_inverse(prev, next))
        {
            spdlog::error("prev-next not identity");
            return false;
        }

        // TODO opposite check (with -1)

        return true;
    }

    bool is_valid_mesh() const
    {
        if (!are_vertices_valid())
        {
            spdlog::error("mesh vertices invalid");
            return false;
        }
        if (!are_faces_valid())
        {
            spdlog::error("mesh faces invalid");
            return false;
        }
        if (!are_halfedges_valid())
        {
            spdlog::error("mesh halfedges invalid");
            return false;
        }

        return true;
    }

    Tuple tuple_from_vertex(int vi) const {
        int hij = out[vi];
        return Tuple(hij, true);
    }

    Tuple tuple_from_face(int fijk) const
    {
        int hij = f2he[fijk];
        return Tuple(hij);
    }

    Tuple tuple_from_edge(int fijk, int k) const
    {
        int j = (k + 2) % 3;
        int vj = input_F(fijk, j);
        int hij = f2he[fijk];
        while (to[hij] != vj)
        {
            hij = next[hij];
        }
        return Tuple(hij);
    }

    std::vector<int> next;
    std::vector<int> prev;
    std::vector<int> opposite;
    std::vector<int> he2f;
    std::vector<int> f2he;
    std::vector<int> out;
    std::vector<int> to;
    std::vector<int> from;
    std::vector<int> he2e;
    std::vector<int> e2he;
    Eigen::SparseMatrix<int> vv2he;
    
};

class ExtremeOpt : public Mesh
{
public:
    ExtremeOpt() {}
    ExtremeOpt(
        const Eigen::MatrixXd& V,
        const Eigen::MatrixXi& F
     )
     : Mesh(V, F)
     {
        vertex_attrs.resize(num_vertices());
        face_attrs.resize(num_faces());
        edge_attrs.resize(num_edges());
     }

    double elen_threshold;
    double elen_threshold_3d;
    Parameters m_params;
    igl::AABB<Eigen::MatrixXd, 3> tree; // for closest point queries

    std::vector<VertexAttributes> vertex_attrs;
    std::vector<FaceAttributes> face_attrs;
    std::vector<EdgeAttributes> edge_attrs;
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXi EE;
    Eigen::MatrixXi FE;
    Eigen::MatrixXi ME;
    Eigen::SparseMatrix<double> G; // modified grad for use with symdir
    Eigen::SparseMatrix<double> Grad; // original grad operator from igl::grad
    Eigen::SparseMatrix<double> Aeq, AeqT, Beq;
    Eigen::SparseMatrix<double> Q2, Q2T;
    Eigen::VectorXd area;
    Eigen::VectorXi matchings;
    Eigen::MatrixXd PD1;
    Eigen::MatrixXd PD2;
    std::vector<double> min_v_diffs;
    std::vector<int> min_v_diff_ids;
    std::vector<int> min_v_diff_next_ids;
    std::vector<int> v_map;
    Eigen::VectorXi C;
    int num_components;
    double correction, residual;
    int iter_solver;

    // previous direction for line search
    Eigen::VectorXd prev_dir;

    std::vector<std::vector<VertexAttributes>> e_worst_v_attrs; // vertex_attrs for E_worst computations
    std::vector<int> e_worst_v_attrs_ind; // index into iter_v_attrs for each E_worst computation
    std::vector<int> e_worst_iters;

    std::vector<std::vector<VertexAttributes>> iter_v_attrs; // vertex_attrs for each iteration
    int last_iter = 0;

    // Optimization
    int tri_capacity() const { return face_attrs.size(); }
    int vert_capacity() const { return vertex_attrs.size(); }
    void make_screenshot(int iter, double percentage=0);
    void do_optimization(json& opt_log);
    double compute_energy(const Eigen::MatrixXd& aaa, double Lp = 0);
    std::vector<double> compute_worst_n_energy(const Eigen::MatrixXd& aaa);
    double compute_threshold_energy(const Eigen::MatrixXd& aaa);

    void export_uv(Eigen::MatrixXd& uv);
    void export_EE(Eigen::MatrixXi& EE);
    void export_FE(Eigen::MatrixXi& FE);
    void export_mesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& uv);
    void export_mesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& uv, std::vector<VertexAttributes> v_attrs_input);

    double get_quality();
    double get_quality_max();
    double get_quality_avg_for_smooth_only(double Lp = 0);
    std::vector<double> get_quality_avg_worst_for_smooth_only();
    double get_threshold_energy();

    
    //statistics for solver
    struct HessianStats {
        int iteration;
        double condition_number;
        double residual;
        double correction;
        double time_solver;
        double time_ls;
        double time_grad_hessian;
        int iter_solver;
        double ls_step_size;
        double newton_decr;
        double grad_norm;

        // Custom JSON serialization
        friend void to_json(json& j, const HessianStats& s) {
            j = json{
                {"iteration", s.iteration},
                {"condition_number", s.condition_number},
                {"residual", s.residual},
                {"correction", s.correction},
                {"time_solver", s.time_solver},
                {"time_ls", s.time_ls},
                {"time_grad_hessian", s.time_grad_hessian},
                {"iter_solver", s.iter_solver},
                {"ls_step_size", s.ls_step_size},
                {"newton_decr", s.newton_decr},
                {"grad_norm", s.grad_norm}
            };
        }

    };

    Eigen::VectorXd misaligned_newton_direction(
        Eigen::MatrixXd& uv,
        double& energy,
        Eigen::VectorXd& grad,
        Eigen::SparseMatrix<double>& hessian
    );
    Eigen::VectorXd kkt_newton_direction(
        Eigen::MatrixXd& uv,
        double& energy,
        Eigen::VectorXd& grad,
        Eigen::SparseMatrix<double>& hessian
    );
    Eigen::VectorXd gs_newton_direction(
        Eigen::MatrixXd& uv,
        double& energy,
        Eigen::VectorXd& grad,
        Eigen::SparseMatrix<double>& hessian
    );
    Eigen::VectorXd reduced_newton_direction(
        Eigen::MatrixXd& uv,
        double& energy,
        Eigen::VectorXd& grad,
        Eigen::SparseMatrix<double>& hessian
    );

    double get_hessian(Eigen::SparseMatrix<double>& hessian);
    double smooth_global(bool& failed, std::vector<HessianStats>& hessian_log);

    void create_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& uv);
    void set_v_map(const Eigen::MatrixXi& F, const Eigen::MatrixXi& FT);

    void init_constraints(const std::vector<std::vector<int>>& EE_e);
    void consolidate_mesh_cons();
    bool check_constraints(double eps = 1e-7);

    // Writes a triangle mesh in OBJ format
    void write_obj(const std::string& path);

    void view() {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        Eigen::MatrixXd uv;
        export_mesh(V, F, uv);

#ifdef ENABLE_VISUALIZATION
        polyscope::init();
        polyscope::registerPointCloud("vertices", V);
        polyscope::registerSurfaceMesh("mesh", V, F);
        polyscope::registerCurveNetwork("features", V, FE.leftCols(2))
        ->addEdgeScalarQuantity("alignment", FE.col(2));
        polyscope::show();
#endif
    }

    std::vector<int> propagate_component_labels(const Eigen::MatrixXi& F, const Eigen::VectorXi& C, int N);

    std::tuple<
    Eigen::MatrixXd, 
    Eigen::VectorXd, 
    Eigen::MatrixXi> load_reference_field(const std::string& ffield_file);

    Eigen::VectorXd rotate_vector(
                    const Eigen::VectorXd& V,
                    double angle,
                    const Eigen::VectorXd& B1,
                    const Eigen::VectorXd& B2);
    
    std::tuple<std::deque<int>, Eigen::VectorXi> initialize_matchings(
    const Eigen::MatrixXd frame_field, 
    const Eigen::MatrixXd B1,
    const Eigen::MatrixXd B2);
    
    void comb_matchings(const std::string& ffield_file);
    
    void check_cross_field_alignment();

    double get_energy_grad_and_hessian(const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXd& Guv,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hessian,
    bool get_hessian);
    Eigen::SparseMatrix<double> compute_area_weight_matrix();
    
    /*
    // Energy Assigned to undefined energy
    // TODO: why not the max double?
    const double MAX_ENERGY = 1e50;


    bool has_degenerate_tris(const std::vector<Tuple>& tris) const;

    ExtremeOpt(){};

    virtual ~ExtremeOpt(){};

    // Store the per-vertex and per-face attributes



    struct PositionInfoCache
    {
        int vid1;
        int vid2;
        Eigen::Vector3d V1;
        Eigen::Vector3d V2;
        Eigen::Vector2d uv1;
        Eigen::Vector2d uv2;
        bool is_v1_bd;
        bool is_v2_bd;
        Tuple bd_e1;
        Tuple bd_e2;
        double E_before;

        bool debug_switch;
    };
    tbb::enumerable_thread_specific<PositionInfoCache> position_cache;

    struct SwapInfoCache
    {
        Tuple t1;
        Tuple t2;
        double E_old;
    };
    tbb::enumerable_thread_specific<SwapInfoCache> swap_cache;
    // Initializes the mesh
    void update_constraints_EE_v(const Eigen::MatrixXi& EE);
    void export_mesh_vtu(const std::string& dir, const std::string& filename);

    // Export constraints EE


    // Computes the quality of the mesh
    Eigen::VectorXd get_quality_all();

    // compute the max_E of a one ring
    int get_mesh_onering(
        const Tuple& t,
        Eigen::MatrixXd& V_local,
        Eigen::MatrixXd& uv_local,
        Eigen::MatrixXi& F_local);
    double get_e_max_onering(const Tuple& t);
    double get_e_onering_edge(const Tuple& t);
    void get_mesh_onering_edge(
        const Tuple& t,
        Eigen::MatrixXd& V_local,
        Eigen::MatrixXd& uv_local,
        Eigen::MatrixXi& F_local);
    // Check if a triangle is inverted
    bool is_inverted(const Tuple& loc) const;
    bool is_3d_degenerated(const Tuple& loc) const;


    double get_vertex_angle(const Tuple& t);

    // Vertex Smoothing
    bool smooth_before(const Tuple& t);
    bool smooth_after(const Tuple& t);
    void smooth_all_vertices();
    */
};

} // namespace SymDir
