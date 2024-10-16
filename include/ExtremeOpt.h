#pragma once

#include <igl/PI.h>
#include <igl/Timer.h>
#include <igl/AABB.h>
#include <Eigen/Sparse>
#include "Parameters.h"
#include "json.hpp"
#include "spdlog/spdlog.h"
#include "polyscope/point_cloud.h"
#include "polyscope/surface_mesh.h"

using json = nlohmann::json;


namespace SymDir {

void get_grad_op(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    Eigen::SparseMatrix<double>& grad_op);

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
                f2he[fijk] = hjk;

                // vertex data
                from[hjk] = vj;
                to[hjk] = vk;
                out[vk] = hki;
                vv2he_triplets.push_back(Trip(vj, vk, hjk + 1)); // offset to 1 indexing
            }
        }

        // Create vertex  maps
        Eigen::SparseMatrix<int> vv2he(num_vertices, num_vertices);
        vv2he.setFromTriplets(vv2he_triplets.begin(), vv2he_triplets.end());

        // build opposite mapping
        for (int hij = 0; hij < num_halfedges; ++hij)
        {
            int vi = from[hij];
            int vj = to[hij];
            int hji = vv2he.coeffRef(vj, vi) - 1; // return to 0 indexing
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
    Eigen::MatrixXi EE;
    Eigen::MatrixXi FE;
    std::vector<int> FE_alignments;

    // Optimization
    int tri_capacity() const { return face_attrs.size(); }
    int vert_capacity() const { return vertex_attrs.size(); }
    void do_optimization(json& opt_log);

    void export_EE(Eigen::MatrixXi& EE);
    void export_FE(Eigen::MatrixXi& FE);
    void export_mesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& uv);

    double get_quality();
    double get_quality_max();
    double get_quality_avg_for_smooth_only();

    double smooth_global(int steps);

    void create_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, const Eigen::MatrixXd& uv);

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

        polyscope::init();
        polyscope::registerPointCloud("vertices", V);
        polyscope::registerSurfaceMesh("mesh", V, F);
        polyscope::show();
    }
    
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
