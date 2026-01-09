#include <spdlog/common.h>
#include "spdlog/spdlog.h"
#include "ExtremeOpt.h"
#include "polyscope/curve_network.h"
#include "MeshCutter.h"
#include <igl/grad.h>

#include <igl/read_triangle_mesh.h>
#include <CLI/CLI.hpp>
#include "energy.h"
#include <numeric>

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

void view(std::vector<ExtremeOpt>& extremeopts, const Eigen::MatrixXi &EE, const Eigen::MatrixXi &FE);
void transform_EE(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& EE_v,
    std::vector<std::vector<int>>& EE_e);
bool find_edge_in_F(const Eigen::MatrixXi& F, int v0, int v1, int& fid, int& eid);
Eigen::VectorXd get_worst_faces(const Eigen::VectorXd& face_energies, double percentile = 0.05);

int main(int argc, char** argv)
{
    //ZoneScopedN("extreme_opt_main");

    CLI::App app{argv[0]};
    std::vector<std::string> models{};
    std::string ffield = "";
    std::string suffix = "";
    app.add_option("-f,--field", ffield, "Input frame field");
    app.add_option("-m,--models", models, "Input model names.")->expected(-1);
    app.add_option("-s,--suffix", suffix, "Input model suffix.");

    CLI11_PARSE(app, argc, argv);
    // Loading the input mesh
    Eigen::MatrixXd V, N;
    Eigen::MatrixXi F, FT, FN;
    std::vector<Eigen::MatrixXd> uvs;
    std::vector<json> json_configs;

    for (const auto& m : models)
    {
        // std::string input_file = m + "_" + suffix + ".obj";
        std::string input_file = m + ".obj";
        Eigen::MatrixXd uv;
        igl::readOBJ(input_file, V, uv, N, F, FT, FN);
        uvs.push_back(uv);
        // std::string input_json = m + ".json";
        // std::ifstream js_in(input_json);
        // json config = json::parse(js_in);
        // json_configs.push_back(config);
    }
    spdlog::info("Input mesh F size {}, V size {}, uv size {}", F.rows(), V.rows(), uvs[0].rows());

    

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

    MeshCutter meshcutter(V, uvs[0], F, FT);
    auto [V_cut, EE] = meshcutter.cut_mesh();
    // Eigen::MatrixXi FE_i = meshcutter.load_feature_edges(models[0] + "_" + suffix + ".obj");
    Eigen::MatrixXi FE_i = meshcutter.load_feature_edges(models[0] + ".obj");
    Eigen::MatrixXi FE = meshcutter.reindex_feature_edges(FE_i);

    std::vector<ExtremeOpt> extremeopts;
    for (int i = 0; i < uvs.size(); ++i)
    {
        ExtremeOpt extremeopt(V_cut, FT);
        extremeopt.create_mesh(V_cut, FT, uvs[i]);

        extremeopt.EE = EE;
        extremeopt.FE = FE;
        extremeopt.m_params.do_feature_alignment = true;
        if (ffield != "") extremeopt.comb_matchings(ffield);
        // extremeopt.m_params.Lp = json_configs[i]["args"]["Lp"];
        extremeopts.push_back(extremeopt);
        spdlog::warn("PD1 = {}, PD2 = {} expected {}", extremeopt.PD1.size(), extremeopt.PD2.size(), 3 * F.rows());

    }
    
    
    //extremeopt.view();
    view(extremeopts, EE, FE);

}

Eigen::VectorXd get_worst_faces(const Eigen::VectorXd& face_energies, double percent)
{
    // Sort indices based on energy values
    std::vector<int> indices(face_energies.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
            [&](int i1, int i2) { return face_energies[i1] > face_energies[i2]; });

    // Calculate how many triangles to include
    int num_tris = static_cast<int>(std::ceil(percent / 100.0 * indices.size()));

    // Create vector with energy for worst faces, zero otherwise
    Eigen::VectorXd worst_faces = Eigen::VectorXd::Zero(face_energies.size());
    double threshold_energy = face_energies[indices[num_tris - 1]];
    double log_threshold = std::log(std::max(threshold_energy, 1e-10)); // avoid log(0)

    for (int i = 0; i < num_tris; ++i) {
        int face_idx = indices[i];
        worst_faces[face_idx] = std::log(std::max(face_energies[face_idx], 1e-10));
    }

    for (int i = num_tris; i < indices.size(); ++i) {
        int face_idx = indices[i];
        worst_faces[face_idx] = log_threshold;
    }
    spdlog::info("Top {:.1f}% worst faces: {} faces, max energy {}", 
                 percent * 100.0, num_tris, face_energies[indices[0]]);
    
    return worst_faces;
}

void view(std::vector<ExtremeOpt>& extremeopts, const Eigen::MatrixXi &EE, const Eigen::MatrixXi &FE)
{
    if (extremeopts.empty()) {
        spdlog::error("No ExtremeOpt instances; aborting view.");
        return;
    }

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    std::vector<Eigen::MatrixXd> uvs;

    std::vector<double> alignment_energies;
    std::vector<double> symdir_energies;
    std::vector<Eigen::VectorXd> alignment_face_energies;
    std::vector<Eigen::VectorXd> symdir_face_energies;
    std::vector<Eigen::VectorXd> symdir_face_worst;

    std::vector<Eigen::MatrixXd> G_us;
    std::vector<Eigen::MatrixXd> G_vs;

    for (auto& extremeopt : extremeopts)
    {
        Eigen::MatrixXd uv;
        extremeopt.export_mesh(V, F, uv);
        uvs.push_back(uv);
        SymDir::get_grad_op(V, F, extremeopt.G);
        igl::doublearea(V, F, extremeopt.area);

        Eigen::MatrixXd Guv = extremeopt.Grad * uv;

        Eigen::MatrixXd G_u = Eigen::Map<const Eigen::MatrixXd>(Guv.col(0).data(), F.rows(), 3);
        Eigen::MatrixXd G_v = Eigen::Map<const Eigen::MatrixXd>(Guv.col(1).data(), F.rows(), 3);
        G_us.push_back(G_u);
        G_vs.push_back(G_v);
        Eigen::MatrixXd uT_vT(3*F.rows(), 2);
        uT_vT.col(0) = Eigen::Map<const Eigen::VectorXd>(extremeopt.PD1.data(), 3 * F.rows());
        uT_vT.col(1) = Eigen::Map<const Eigen::VectorXd>(extremeopt.PD2.data(), 3 * F.rows());

        Eigen::MatrixXd R = Guv - uT_vT;
        Eigen::MatrixXd Rx = R.topRows(F.rows());
        Eigen::MatrixXd Ry = R.middleRows(F.rows(), F.rows());
        Eigen::MatrixXd Rz = R.bottomRows(F.rows());

        alignment_energies.push_back(R.array().square().sum());
        extremeopt.m_params.alignment_weight = 0.0;
        extremeopt.m_params.symdir_weight = 1.0;
        symdir_energies.push_back(extremeopt.compute_energy(uv));

        Eigen::MatrixXd Ji;
        SymDir::jacobian_from_uv(extremeopt.G, uv, Ji);
        Eigen::VectorXd symdir_e(F.rows());
        
        for (int i = 0; i < F.rows(); ++i)
        {
            Eigen::MatrixXd J = Ji.row(i);
            symdir_e[i] = SymDir::symmetric_dirichlet_energy_t(J(0), J(1), J(2), J(3), 1.0);
        }
        symdir_face_energies.push_back(symdir_e);
        // symdir_face_worst.push_back(get_worst_faces(symdir_e, 0.05));

        Eigen::VectorXd residuals = (Rx.array().square() 
                            + Ry.array().square() 
                            + Rz.array().square()).rowwise().sum();
        alignment_face_energies.push_back(residuals);
    }

    
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
    //polyscope::registerPointCloud("vertices", V);
    auto mesh = polyscope::registerSurfaceMesh("mesh", V, F);
    for (int i = 0; i < uvs.size(); ++i)
    {
        std::cout << i << " Total Alignment energy: " << alignment_energies[i] << '\n';
        std::cout << i << " Total Symdir energy: " << symdir_energies[i] << '\n';
        mesh->addVertexParameterizationQuantity("Seamless parameterization " + std::to_string(i), uvs[i]);
        mesh->addFaceScalarQuantity("alignment_error " + std::to_string(i), alignment_face_energies[i]);
        mesh->addFaceScalarQuantity("symdir_energy " + std::to_string(i), symdir_face_energies[i]);
        mesh->addFaceVectorQuantity("grad_u" + std::to_string(i), G_us[i])
            ->setVectorColor(BLUE)
            ->setVectorRadius(0.0005)
            ->setVectorLengthScale(0.005);
        mesh->addFaceVectorQuantity("grad_v" + std::to_string(i), G_vs[i])
            ->setVectorColor(RED)
            ->setVectorRadius(0.0005)
            ->setVectorLengthScale(0.005);
    }

    polyscope::registerCurveNetwork("seamless edges", V_seamless, edges_seamless);
    polyscope::registerCurveNetwork("feature edges", V_feature, edges_feature);
    mesh->addFaceVectorQuantity("PD1", extremeopts[0].PD1)
        ->setVectorColor(FOREST_GREEN)
        ->setVectorRadius(0.0005)
        ->setVectorLengthScale(0.005)
        ->setEnabled(true);
    mesh->addFaceVectorQuantity("PD2", extremeopts[0].PD2)
        ->setVectorColor(BLACK_BROWN)
        ->setVectorRadius(0.0005)
        ->setVectorLengthScale(0.005)
        ->setEnabled(true);
    mesh->addFaceScalarQuantity("matching", extremeopts[0].matchings)->setEnabled(true);
    
    
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