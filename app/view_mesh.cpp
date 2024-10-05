#include <spdlog/common.h>
#include "spdlog/spdlog.h"
#include "ExtremeOpt.h"

#include <igl/read_triangle_mesh.h>
#include <CLI/CLI.hpp>
using json = nlohmann::json;

using namespace SymDir;

int main(int argc, char** argv)
{
    //ZoneScopedN("extreme_opt_main");

    CLI::App app{argv[0]};
    std::string input_dir = "../data";
    std::string model = "";
    app.add_option("-i,--input", input_dir, "Input mesh dir.");
    app.add_option("-m,--model", model, "Input model name.");

    CLI11_PARSE(app, argc, argv);


    std::string input_file = input_dir + "/" + model + "_init.obj";
    // Loading the input mesh
    Eigen::MatrixXd V, uv;
    Eigen::MatrixXi F;
    igl::readOBJ(input_file, V, uv, uv, F, F, F);
    spdlog::info("Input mesh F size {}, V size {}, uv size {}", F.rows(), V.rows(), uv.rows());

    // Loading the seamless boundary constraints
    Eigen::MatrixXi EE;
    int EE_rows;
    std::ifstream EE_in(input_dir + "/EE/" + model + "_EE.txt");
    EE_in >> EE_rows;
    EE.resize(EE_rows, 4);
    for (int i = 0; i < EE.rows(); i++) {
        EE_in >> EE(i, 0) >> EE(i, 1) >> EE(i, 2) >> EE(i, 3);
    }
    spdlog::info("Input EE size {}", EE.rows());

    ExtremeOpt extremeopt(V, F);
    extremeopt.create_mesh(V, F, uv);
    extremeopt.view();

}
