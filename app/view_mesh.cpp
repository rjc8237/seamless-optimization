#include <spdlog/common.h>
#include "spdlog/spdlog.h"
#include "ExtremeOpt.h"
#include "polyscope/curve_network.h"
#include "MeshCutter.h"
#include <igl/grad.h>

#include <igl/read_triangle_mesh.h>
#include <CLI/CLI.hpp>
using json = nlohmann::json;

using namespace SymDir;

glm::vec3 BEIGE(0.867, 0.765, 0.647);
glm::vec3 BLACK_BROWN(0.125, 0.118, 0.125);
glm::vec3 TAN(0.878, 0.663, 0.427);
glm::vec3 MUSTARD(0.890, 0.706, 0.282);
glm::vec3 FOREST_GREEN(0.227, 0.420, 0.208);
glm::vec3 TEAL(0., 0.375, 0.5);
glm::vec3 DARK_TEAL(0., 0.5*0.375, 0.5*0.5);
glm::vec3 BLUE(0., 0., 1.);
glm::vec3 RED(1., 0., 0.);

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
    std::string ffield = "";
    std::string suffix = "";
    app.add_option("-i,--input", input_dir, "Input mesh dir.");
    app.add_option("-f,--field", ffield, "Input frame field");
    app.add_option("-m,--model", model, "Input model name.");
    app.add_option("-s,--suffix", suffix, "Input model suffix.");

    CLI11_PARSE(app, argc, argv);

    ffield = input_dir + "/" + ffield;
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

    //std::ifstream FE_in(input_dir + "/FE/" + model + "_FE.txt");
    //if (!FE_in.is_open()) {
    //    return -1;
    //}
    //std::vector<Eigen::Vector3i> rows;
    //int a, b, c;

    // Read data line by line
    //while (FE_in >> a >> b >> c) {
    //    rows.emplace_back(a, b, c);
    //}
    //FE_in.close();
    //int FE_rows = rows.size();

    //Eigen::MatrixXi FE(FE_rows, 3);

    //for (int i = 0; i < FE_rows; ++i) {
    //    FE.row(i) = rows[i];
    //}
    //spdlog::info("Input FE size {}", FE.rows());

    MeshCutter meshcutter(V, uv, F, FT);
    auto [V_cut, EE] = meshcutter.cut_mesh();
    Eigen::MatrixXi FE_i = meshcutter.load_feature_edges(input_file);
    Eigen::MatrixXi FE = meshcutter.reindex_feature_edges(FE_i);

    ExtremeOpt extremeopt(V_cut, FT);
    extremeopt.create_mesh(V_cut, FT, uv);
    extremeopt.EE = EE;
    extremeopt.FE = FE;
    extremeopt.m_params.do_feature_alignment = true;
    extremeopt.comb_matchings(ffield);
    //extremeopt.view();
    view(extremeopt, EE, FE);

}


void view(ExtremeOpt &extremeopt, const Eigen::MatrixXi &EE, const Eigen::MatrixXi &FE)
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd uv;
    extremeopt.export_mesh(V, F, uv);

    Eigen::SparseMatrix<double> G;
    igl::grad(V, F, G);

    Eigen::MatrixXd Guv = G * uv;

    Eigen::MatrixXd G_u = Eigen::Map<const Eigen::MatrixXd>(Guv.col(0).data(), F.rows(), 3);
    Eigen::MatrixXd G_v = Eigen::Map<const Eigen::MatrixXd>(Guv.col(1).data(), F.rows(), 3);
    
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
    mesh->addFaceVectorQuantity("PD1", extremeopt.PD1)
        ->setVectorColor(FOREST_GREEN)
        ->setVectorRadius(0.0005)
        ->setVectorLengthScale(0.005)
        ->setEnabled(true);
    mesh->addFaceVectorQuantity("PD2", extremeopt.PD2)
        ->setVectorColor(BLACK_BROWN)
        ->setVectorRadius(0.0005)
        ->setVectorLengthScale(0.005)
        ->setEnabled(true);
    mesh->addFaceVectorQuantity("grad_u", G_u)
        ->setVectorColor(BLUE)
        ->setVectorRadius(0.0005)
        ->setVectorLengthScale(0.005)
        ->setEnabled(true);
    mesh->addFaceVectorQuantity("grad_v", G_v)
        ->setVectorColor(RED)
        ->setVectorRadius(0.0005)
        ->setVectorLengthScale(0.005)
        ->setEnabled(true);
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