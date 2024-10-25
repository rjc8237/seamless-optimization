#include <spdlog/common.h>
#include "spdlog/spdlog.h"
#include "ExtremeOpt.h"
#include "polyscope/curve_network.h"
#include "MeshCutter.h"

#include <igl/read_triangle_mesh.h>
#include <CLI/CLI.hpp>
using json = nlohmann::json;

using namespace SymDir;

void view(ExtremeOpt &extremeopt, const Eigen::MatrixXi &EE, const Eigen::MatrixXi &FE);
void transform_EE(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& EE_v,
    std::vector<std::vector<int>>& EE_e);
bool find_edge_in_F(const Eigen::MatrixXi& F, int v0, int v1, int& fid, int& eid);

int main(int argc, char** argv)
{
    //ZoneScopedN("extreme_opt_main");

    CLI::App app{argv[0]};
    std::string input_dir = "../data";
    std::string model = "";
    std::string suffix = "";
    app.add_option("-i,--input", input_dir, "Input mesh dir.");
    app.add_option("-m,--model", model, "Input model name.");
    app.add_option("-s,--suffix", suffix, "Input model suffix.");

    CLI11_PARSE(app, argc, argv);


    std::string input_file = input_dir + "/" + model + "_" + suffix + ".obj";
    // Loading the input mesh
    Eigen::MatrixXd V, uv, N;
    Eigen::MatrixXi F, FT, FN;
    igl::readOBJ(input_file, V, uv, N, F, FT, FN);
    spdlog::info("Input mesh F size {}, V size {}, uv size {}", F.rows(), V.rows(), uv.rows());

    // Loading the seamless boundary constraints
    //Eigen::MatrixXi EE;
    //int EE_rows;
    //std::ifstream EE_in(input_dir + "/EE/" + model + "_EE.txt");
    //EE_in >> EE_rows;
    //EE.resize(EE_rows, 4);
    //for (int i = 0; i < EE.rows(); i++) {
    //    EE_in >> EE(i, 0) >> EE(i, 1) >> EE(i, 2) >> EE(i, 3);
    //}
    //spdlog::info("Input EE size {}", EE.rows());

    Eigen::MatrixXi FE;
    int FE_rows;
    //std::ifstream FE_in(input_dir + "/FE/" + model + "_FE.txt");
    //FE_in >> FE_rows;
    //FE.resize(FE_rows, 3);
    //for (int i = 0; i < FE.rows(); i++) {
    //    FE_in >> FE(i, 0) >> FE(i, 1) >> FE(i, 2);
    //}
    spdlog::info("Input FE size {}", FE.rows());

    MeshCutter meshcutter(V, uv, F, FT);
    auto [V_cut, EE] = meshcutter.cut_mesh();

    ExtremeOpt extremeopt(V_cut, FT);
    extremeopt.create_mesh(V_cut, FT, uv);
    //extremeopt.view();
    view(extremeopt, EE, FE);

}


void view(ExtremeOpt &extremeopt, const Eigen::MatrixXi &EE, const Eigen::MatrixXi &FE)
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd uv;
    extremeopt.export_mesh(V, F, uv);
    
    Eigen::VectorXd f_scalar = Eigen::VectorXd::Zero(F.rows());    
    std::unordered_set<int> unique_seamless_vertices;
    std::unordered_set<int> unique_feature_vertices;

    
    for (int i = 0; i < EE.rows(); ++i)
    {   
        unique_seamless_vertices.insert(EE(i, 0));
        unique_seamless_vertices.insert(EE(i, 1));
    }
    
    std::vector<int> seamless_vertex_indices(unique_seamless_vertices.begin(), unique_seamless_vertices.end());
    
    std::unordered_map<int, int> old_to_new_index;
    for (size_t i = 0; i < seamless_vertex_indices.size(); ++i)
    {
        old_to_new_index[seamless_vertex_indices[i]] = i;
    }
    
    Eigen::MatrixXd V_seamless(seamless_vertex_indices.size(), 3);
    for (size_t i = 0; i < seamless_vertex_indices.size(); ++i)
    {
        V_seamless.row(i) = V.row(seamless_vertex_indices[i]);
    }
    
    std::vector<std::array<size_t, 2>> edges_seamless;
    for (int i = 0; i < EE.rows(); ++i)
    {
        edges_seamless.push_back({
            static_cast<size_t>(old_to_new_index[EE(i, 0)]), 
            static_cast<size_t>(old_to_new_index[EE(i, 1)])
        });
    }

    for (int i = 0; i < FE.rows(); ++i)
    {   
        unique_feature_vertices.insert(FE(i, 0));
        unique_feature_vertices.insert(FE(i, 1));
    }

    std::vector<int> feature_vertex_indices(unique_feature_vertices.begin(), unique_feature_vertices.end());

    old_to_new_index.clear();
    for (size_t i = 0; i < feature_vertex_indices.size(); ++i)
    {
        old_to_new_index[feature_vertex_indices[i]] = i;
    }

    Eigen::MatrixXd V_feature(feature_vertex_indices.size(), 3);
    for (size_t i = 0; i < feature_vertex_indices.size(); ++i)
    {
        V_feature.row(i) = V.row(feature_vertex_indices[i]);
    }
    
    std::vector<std::array<size_t, 2>> edges_feature;
    for (int i = 0; i < FE.rows(); ++i)
    {
        edges_feature.push_back({
            static_cast<size_t>(old_to_new_index[FE(i, 0)]), 
            static_cast<size_t>(old_to_new_index[FE(i, 1)])
        });
    }

    /*
    std::vector<std::vector<int>> EE_e;
    transform_EE(F, EE, EE_e);
    for (int i = 0; i < EE_e.size(); i++) {
        auto t1 = extremeopt.tuple_from_edge(EE_e[i][0], EE_e[i][1]);
        auto t2 = extremeopt.tuple_from_edge(EE_e[i][2], EE_e[i][3]);
        int eid1 = t1.eid(extremeopt);
        int eid2 = t2.eid(extremeopt);
        e_scalar[eid1] = 1;
        e_scalar[eid2] = 1;
    }*/
    
    polyscope::init();
    polyscope::registerPointCloud("vertices", V);
    auto mesh = polyscope::registerSurfaceMesh("mesh", V, F);
    polyscope::registerCurveNetwork("seamless edges", V_seamless, edges_seamless);
    polyscope::registerCurveNetwork("feature edges", V_feature, edges_feature);
    mesh->addVertexParameterizationQuantity("Seamless parameterization", uv);
    //mesh->addFaceScalarQuantity("seamless", f_scalar);
    //mesh->addEdgeScalarQuantity("seamless", e_scalar);
    polyscope::show();
}

bool find_edge_in_F(const Eigen::MatrixXi& F, int v0, int v1, int& fid, int& eid)
{
    fid = -1;
    eid = -1;
    for (int i = 0; i < F.rows(); i++) {
        for (int j = 0; j < 3; j++) {
            if (F(i, j) == v0 && F(i, (j + 1) % 3) == v1) {
                fid = i;
                eid = 3 - j - ((j + 1) % 3);
                return true;
            }
        }
    }
    return false;
}

void transform_EE(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& EE_v,
    std::vector<std::vector<int>>& EE_e)
{
    EE_e.resize(EE_v.rows());
    for (int i = 0; i < EE_v.rows(); i++) {
        std::vector<int> one_row;
        int v0 = EE_v(i, 0), v1 = EE_v(i, 1);
        int fid, eid;
        if (find_edge_in_F(F, v0, v1, fid, eid)) {
            one_row.push_back(fid);
            one_row.push_back(eid);
            // one_row.push_back(3 * fid + eid);
        } else {
            std::cout << "Something Wrong in transform_EE: edge not found in F" << std::endl;
        }

        v0 = EE_v(i, 2);
        v1 = EE_v(i, 3);
        if (find_edge_in_F(F, v0, v1, fid, eid)) {
            one_row.push_back(fid);
            one_row.push_back(eid);
            // one_row.push_back(3 * fid + eid);
        } else {
            std::cout << "Something Wrong in transform_EE: edge not found in F" << std::endl;
        }
        EE_e[i] = one_row;
    }
}