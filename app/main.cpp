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
#include <filesystem>
#include <algorithm>

#include <stdlib.h>  // for setenv

//#include "json.hpp"
using json = nlohmann::json;

using namespace SymDir;

static std::string sci_short(double x)
{
    if (x == 0.0) return "0";
    int exp = static_cast<int>(std::floor(std::log10(std::fabs(x))));
    double mant = x / std::pow(10.0, exp);

    // Trim trailing zeros in mantissa
    std::ostringstream ms;
    ms << std::fixed << std::setprecision(6) << mant; // cap decimals
    std::string m = ms.str();
    // remove trailing zeros and possible trailing dot
    while (!m.empty() && m.back() == '0') m.pop_back();
    if (!m.empty() && m.back() == '.') m.pop_back();
    if (m.empty()) m = "1"; // fallback

    std::ostringstream out;
    out << m << "e" << exp; // exp already has sign if negative
    return out.str();
}

void export_obj_mesh(
    ExtremeOpt extremeopt,
    Parameters param,
    Eigen::MatrixXd V,
    Eigen::MatrixXi F,
    Eigen::MatrixXd uv,
    std::vector<VertexAttributes> v_attrs_input,
    Eigen::MatrixXd V_init,
    Eigen::MatrixXi F_init,
    Eigen::MatrixXi FE_init,
    Eigen::MatrixXd N,
    Eigen::MatrixXi FN,
    Eigen::MatrixXi EE,
    Eigen::MatrixXi FE,
    std::string output_dir,
    std::string model,
    std::string obj_name)
{
    extremeopt.export_mesh(V, F, uv, v_attrs_input);

    double cons_residual = check_constraints(EE, FE, uv, F);
    spdlog::info("Final constraints error {}", cons_residual);
    if (extremeopt.m_params.with_cons) extremeopt.export_EE(EE);

    igl::writeOBJ(obj_name, V_init, F_init, N, FN, uv, F);

    std::ofstream output_file(obj_name, std::ios::out | std::ios::app);
    for (int eij = 0; eij < FE_init.rows(); ++eij)
    {
        int vi = FE_init(eij, 0);
        int vj = FE_init(eij, 1);
        output_file << "l " << vi + 1 << " " << vj + 1 << std::endl;
    }
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

}

int main(int argc, char** argv)
{
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
    int num_threads = 1;
    app.add_option("-t, --threads", num_threads, "Number of threads");

    CLI11_PARSE(app, argc, argv);
    num_threads = std::max(1, num_threads);
    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);
    Eigen::setNbThreads(num_threads);
    std::cout << "========================================" << std::endl;
    std::cout << "Is OpenMP recognized? ";
    #ifdef _OPENMP
        std::cout << "YES (Version: " << _OPENMP << ")" << std::endl;
    #else
        std::cout << "NO (Macro not defined)" << std::endl;
    #endif

    std::cout << "Max Threads available: " << omp_get_max_threads() << " Set number of threads: " << num_threads << std::endl;
    std::cout << "========================================" << std::endl;

    // Ensure base output directory exists
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);

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
    param.max_time = config["max_time"]; // time in seconds
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
    param.output_dir = output_dir;
    param.do_feature_alignment = config["do_feature_alignment"]; // align feature edges
    param.symdir_weight = config["symdir_weight"];
    param.alignment_weight = config["alignment_weight"];
    param.degenerate_weight = config["degenerate_weight"];
    param.fix_misaligned = config["fix_misaligned"];
    param.use_rref = config["use_rref"];
    // param.solver_type = config["solver_type"];
    param.cg_rel_err = config["cg_rel_err"];
    param.percentages = config["percentages"].get<std::vector<double>>();
    param.percentage_target = config["percentage_target"];

    param.E_abs_err = config["E_abs_err"];
    param.E_rel_err = config["E_rel_err"];
    param.diff_err = config["diff_err"];
    param.grad_abs_err = config["grad_abs_err"];
    param.grad_rel_err = config["grad_rel_err"];
    
    param.precompute_seamless = config["precompute_seamless"];
    param.projected_newton = config["projected_newton"];
    param.soft_max = config["soft_max"];
    param.t = config["t"];
    param.precompute_seamless = config["precompute_seamless"];
    
    param.percentage_target_converge = config["percentage_target_converge"];
    param.max_grad_abs_converge = config["max_grad_abs_converge"];
    param.max_grad_rel_converge = config["max_grad_rel_converge"];
    param.energy_diff_converge = config["energy_diff_converge"];
    param.use_worst_n_energy_in_ls = config["use_worst_n_energy_in_ls"];
    param.E_abs_converge = config["E_abs_converge"];
    param.E_rel_converge = config["E_rel_converge"];

    param.last_screenshot_after_optimization = config["last_screenshot_after_optimization"];
    param.screenshot_interval = config["screenshot_interval"];
    param.output_dir_for_screenshots = config["output_dir_for_screenshots"];
    param.uv_scale_for_screenshots = config["uv_scale_for_screenshots"];
    param.angle_to_rotate_model_for_screenshots = config["angle_to_rotate_model_for_screenshots"];
    param.screenshot_during_optimization = config["screenshot_during_optimization"];
    param.output_dir_for_screenshots = param.output_dir_for_screenshots + "/" + model;
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(param.output_dir_for_screenshots);

    json opt_log;
    opt_log["model_name"] = model;
    opt_log["solver_type"] = param.solver_type;
    opt_log["num_threads"] = num_threads;
    opt_log["args"] = config;

    if (ffield == "")
    {
        spdlog::info("no field provided: disabling alignment");
        param.alignment_weight = 0.;
    }

    igl::Timer timer;
    double time = 0;
    timer.start();
	MeshCutter meshcutter(V_init, uv, F_init, F);
	auto [V, EE] = meshcutter.cut_mesh();
    opt_log["time_log"]["cutting_time"] = timer.getElapsedTime();
    Eigen::MatrixXi FE_init;
    Eigen::MatrixXi FE(0, 0);
    Eigen::MatrixXi ME(0, 0);
    time = timer.getElapsedTime();
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
    opt_log["time_log"]["loading_feature_edges_time"] = timer.getElapsedTime() - time;
    double cons_residual = check_constraints(EE, FE, uv, F);
    spdlog::info("Initial constraints error {}", cons_residual);


    // Choose output JSON filename
    std::string json_name = output_dir + "/" + model + "_" + param.solver_type;

    if (param.solver_type == "CG" || param.solver_type == "CG_LLT" || param.solver_type == "CG_GS" || param.solver_type == "Parallel_CG") {
        // append cg_rel_err for CG runs
        json_name += "_" + sci_short(param.cg_rel_err);
    }
    json_name += "_" + std::to_string(num_threads);
    json_name += ".json";

    std::ofstream js_out(json_name);

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
    extremeopt.set_v_map(F_init, F);

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
    if (ffield != "") extremeopt.comb_matchings(ffield_path);
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

    // export intermediate meshes for worst n energy stopping condition
    std::string obj_name;
    for (int i = 0; i < extremeopt.e_worst_v_attrs_ind.size(); i++)
    {
        if (extremeopt.e_worst_v_attrs_ind[i] == -1)
        {
            continue;
        }
        obj_name = output_dir + "/" + model + "_out_" + param.solver_type;
        if (param.solver_type == "CG" || param.solver_type == "CG_LLT" || param.solver_type == "CG_GS" || param.solver_type == "Parallel_CG") {
            // append cg_rel_err for CG runs
            obj_name += "_" + sci_short(param.cg_rel_err);
        }
        obj_name += "_Lp=" + std::to_string(param.Lp);
        obj_name += "_perc=" + fmt::format("{:.2f}", param.percentages[i]);
        obj_name += "_Lp-worst=" + std::to_string(1.0);
        obj_name += "_" + std::to_string(num_threads);
        obj_name += ".obj";
        export_obj_mesh(extremeopt, param, V, F, uv, extremeopt.e_worst_v_attrs[extremeopt.e_worst_v_attrs_ind[i]], V_init, F_init, FE_init, N, FN, EE, FE, output_dir, model, obj_name);
    }
    for (int i = 0; i < extremeopt.iter_v_attrs.size(); i++)
    {
        obj_name = output_dir + "/" + model + "_out_" + param.solver_type;
        if (param.solver_type == "CG" || param.solver_type == "CG_LLT" || param.solver_type == "CG_GS" || param.solver_type == "Parallel_CG") {
            // append cg_rel_err for CG runs
            obj_name += "_" + sci_short(param.cg_rel_err);
        }
        obj_name += "_Lp=" + std::to_string(param.Lp);
        obj_name += "_iter=" + fmt::format("{}", std::min(i * 10, extremeopt.last_iter));
        obj_name += "_" + std::to_string(num_threads);
        obj_name += ".obj";
        export_obj_mesh(extremeopt, param, V, F, uv, extremeopt.iter_v_attrs[i], V_init, F_init, FE_init, N, FN, EE, FE, output_dir, model, obj_name);
    }
    
    // export final mesh
    obj_name = output_dir + "/" + model + "_out_" + param.solver_type;
    if (param.solver_type == "CG" || param.solver_type == "CG_LLT" || param.solver_type == "CG_GS" || param.solver_type == "Parallel_CG") {
        // append cg_rel_err for CG runs
        obj_name += "_" + sci_short(param.cg_rel_err);
    }
    obj_name += "_Lp=" + std::to_string(param.Lp);
    obj_name += "_" + std::to_string(num_threads);
    obj_name += ".obj";
    export_obj_mesh(extremeopt, param, V, F, uv, extremeopt.vertex_attrs, V_init, F_init, FE_init, N, FN, EE, FE, output_dir, model, obj_name);

    js_out << std::setw(4) << opt_log << std::endl;
    return 0;
}

