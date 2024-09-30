#pragma once

#include <igl/PI.h>
#include <igl/Timer.h>
#include <igl/AABB.h>
#include <Eigen/Sparse>
#include "Parameters.h"
#include "json.hpp"

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

// equivalent to a halfedge
class Tuple
{
// TODO
    int face_index;
    int local_corner_index;
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


class ExtremeOpt
{
public:
    Eigen::MatrixXd input_V;
    Eigen::MatrixXi input_F;

    double elen_threshold;
    double elen_threshold_3d;
    Parameters m_params;
    igl::AABB<Eigen::MatrixXd, 3> tree; // for closest point queries



    std::vector<Tuple> get_vertices() const;
    std::vector<Tuple> get_faces() const;

    std::vector<VertexAttributes> vertex_attrs;
    std::vector<FaceAttributes> face_attrs;
    std::vector<EdgeAttributes> edge_attrs;

    // Optimization
    int tri_capacity() const { return face_attrs.size(); }
    int vert_capacity() const { return vertex_attrs.size(); }
    void do_optimization(json& opt_log);

    void export_EE(Eigen::MatrixXi& EE);
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
