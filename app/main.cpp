#include <spdlog/common.h>
#include "spdlog/spdlog.h"
#include "ExtremeOpt.h"
#include "MeshCutter.h"

#include <igl/PI.h>
#include <igl/boundary_loop.h>
#include <igl/read_triangle_mesh.h>
#include <igl/upsample.h>
#include <igl/writeOBJ.h>
#include <CLI/CLI.hpp>
#include "main_helper.h"

//#include "json.hpp"
using json = nlohmann::json;

using namespace SymDir;

int main(int argc, char** argv)
{
    //ZoneScopedN("extreme_opt_main");

    CLI::App app{argv[0]};
    std::string input_dir = "../data";
    std::string output_dir = "./";
    std::string input_json = "../app/example.json";
    std::string model = "";
    std::string ffield = "";
    std::string feature_edges_filename = "";
    Parameters param;
    app.add_option("-i,--input", input_dir, "Input mesh dir.");
    app.add_option("-m,--model", model, "Input model name.");
    app.add_option("-f,--field", ffield, "Input frame field");
    app.add_option("-j,--json", input_json, "Input arguments.");
    app.add_option("-o,--output", output_dir, "Output dir.");

    app.add_option("-s,--solver", param.solver_type, "Solver type");
    CLI11_PARSE(app, argc, argv);


    std::string input_file = input_dir + "/" + model + ".obj";
    std::string ffield_path = input_dir + "/" + ffield;
    // Loading the input mesh
	Eigen::MatrixXd V_init, uv, N;
	Eigen::MatrixXi F_init, F, FN;
	igl::readOBJ(input_file, V_init, uv, N, F_init, F, FN);
    spdlog::info("Input mesh F size {}, V size {}, uv size {}", F.rows(), V_init.rows(), uv.rows());

    std::ifstream js_in(input_json);
    json config = json::parse(js_in);
    
    param.max_iters = config["max_iters"]; // iterations
    param.smooth_only_iters = config["smooth_only_iters"];
    param.E_target = config["E_target"]; // Energy target
    param.ls_iters = config["ls_iters"]; // param for linesearch in smoothing operation
    param.do_newton = config["do_newton"]; // do newton/gd steps for smoothing operation
    // do global/local smooth (local smooth does not optimize boundary vertices)
    param.local_smooth = config["local_smooth"];
    param.global_smooth = config["global_smooth"];
    param.elen_alpha = config["elen_alpha"];
    param.do_projection = config["do_projection"];
    param.with_cons = config["with_cons"];
    param.Lp = config["Lp"];
    param.save_meshes = config["save_meshes"];
    param.model_name = model;
    param.do_feature_alignment = config["do_feature_alignment"]; // align feature edges
    param.symdir_weight = config["symdir_weight"];
    param.alignment_weight = config["alignment_weight"];
    param.fix_misaligned = config["fix_misaligned"];
    param.use_rref = config["use_rref"];
    // param.solver_type = config["solver_type"];
    param.percent = config["percent"];
    param.p_energy = config["p_energy"];
    param.E_rel_tol = config["E_rel_tol"];
    param.cg_rel_err = config["cg_rel_err"];
    
    if (ffield == "")
    {
        spdlog::info("no field provided: disabling alignment");
        param.alignment_weight = config["alignment_weight"];
    }

	MeshCutter meshcutter(V_init, uv, F_init, F);

	auto [V, EE] = meshcutter.cut_mesh();

    Eigen::MatrixXi FE_init;
    Eigen::MatrixXi FE(0, 0);
    Eigen::MatrixXi ME(0, 0);
    if (param.do_feature_alignment)
    {
        // Loading the feature edge constraints
        FE_init = meshcutter.load_feature_edges(input_file);
        FE = meshcutter.reindex_feature_edges(FE_init);
        if (param.fix_misaligned)
        {
            std::string misaligned_file = input_dir + "/" + model + "_misaligned_edges";
            ME = meshcutter.load_misaligned_edges(misaligned_file);
        }
        //FE = meshcutter.remove_cycles_and_duplicates(FE_init, FE_full);
    }
    
    double cons_residual = check_constraints(EE, FE, uv, F);
    spdlog::info("Initial constraints error {}", cons_residual);

    json opt_log;
    opt_log["model_name"] = model;
    opt_log["solver_type"] = param.solver_type;
    opt_log["args"] = config;
    std::ofstream js_out(output_dir + "/" + model + "_" + param.solver_type + ".json");

    //std::vector<std::vector<int>> bds;
    //igl::boundary_loop(F, bds);
    //std::cout << "#bd loops:" << bds.size() << std::endl;
    //int Nv_bds = 0;
    //for (auto bd : bds) Nv_bds += bd.size();
    //spdlog::info("Boundary size: {}", Nv_bds);

    Eigen::MatrixXi new_F;
    Eigen::MatrixXd new_V, new_uv;
    ExtremeOpt extremeopt(V, F);
    extremeopt.m_params = param;
    
    extremeopt.create_mesh(V, F, uv);

    if (extremeopt.m_params.with_cons)
    {
        std::vector<std::vector<int>> EE_e = transform_EE(F, EE);
        std::vector<std::vector<int>> FE_e;
        if (extremeopt.m_params.do_feature_alignment) {
            FE_e = transform_FE(F, FE);
        }
        extremeopt.init_constraints(EE_e);
        extremeopt.EE = EE;
        extremeopt.FE = FE;
        extremeopt.ME = ME;
        // assert(extremeopt.check_mesh_connectivity_validity());
        std::cout << "check constraints inside wmtk" << std::endl;
        if (extremeopt.check_constraints()) {
            std::cout << "initial constraints satisfied" << std::endl;
        } else {
            std::cout << "fails" << std::endl;
        }
        //extremeopt.EE = EE;
        //extremeopt.FE = FE;
    }
    //extremeopt.view();
    if (ffield == "") extremeopt.comb_matchings(ffield_path);
    extremeopt.do_optimization(opt_log);

    if (extremeopt.m_params.with_cons)
    {
        std::cout << "check constraints inside wmtk" << std::endl;
        if (extremeopt.check_constraints()) {
            std::cout << "constraints satisfied" << std::endl;
        } else {
            std::cout << "fails" << std::endl;
        }
    }

    extremeopt.export_mesh(V, F, uv);

    cons_residual = check_constraints(EE, FE, uv, F);
    spdlog::info("Final constraints error {}", cons_residual);

    if (extremeopt.m_params.with_cons) extremeopt.export_EE(EE);

    igl::writeOBJ(output_dir + "/" + model + "_out_" + param.solver_type + ".obj", V_init, F_init, N, FN, uv, F);

    // open output file
    std::string output_filename = output_dir + "/" + model + "_out.obj";
    std::ofstream output_file(output_filename, std::ios::out | std::ios::app);

    // write all feature edge vertices
    for (int eij = 0; eij < FE_init.rows(); ++eij)
    {
        int vi = FE_init(eij, 0);
        int vj = FE_init(eij, 1);
        output_file << "l " << vi + 1 << " " << vj + 1 << std::endl;
    }

    // close output file
    output_file.close();
    
    if (extremeopt.m_params.with_cons)
    {
        std::ofstream EE_out(output_dir + "/EE/" + model + "_EE.txt");
        EE_out << EE.rows() << std::endl;
        EE_out << EE << std::endl;
        EE_out.close();

        if (extremeopt.m_params.do_feature_alignment)
        {
        std::ofstream FE_out(output_dir + "/FE/" + model + "_FE.txt");
        FE_out << FE.rows() << std::endl;
        FE_out << FE << std::endl;
        FE_out.close();
        }
    }

    js_out << std::setw(4) << opt_log << std::endl;
    return 0;
}

