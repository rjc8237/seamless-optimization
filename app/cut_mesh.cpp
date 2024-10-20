#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/boundary_facets.h>
#include <igl/triangle_triangle_adjacency.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <fstream>
#include <sstream>

#include <CLI/CLI.hpp>
#include "spdlog/spdlog.h"
#include "MeshCutter.h"

Eigen::MatrixXi get_feature_edges(std::ifstream& inf);
/// Join two filepaths.
///
/// @param[in] first_path: first path to join
/// @param[in] second_path: second path to join
/// @return combined path
std::filesystem::path join_path(
	const std::filesystem::path& first_path,
	const std::filesystem::path& second_path) {
	return first_path / second_path;
}

void write_obj_with_uv(
	const std::string& filename,
	const Eigen::MatrixXd& V,
	const Eigen::MatrixXi& F,
	const Eigen::MatrixXd& uv,
	const Eigen::MatrixXi& F_uv) {
	Eigen::MatrixXd N;
	Eigen::MatrixXi FN;
	igl::writeOBJ(filename, V, F, N, FN, uv, F_uv);
}

int main(int argc, char* argv[]) {
	// Get command line arguments
	CLI::App app{ "Generate a conformal similarity parameterization." };
	std::string mesh_name = "";
	std::string mesh_filename = "";
	std::string feature_edges_filename = "";
	std::string output_dir = "./";
	app.add_option("--mesh_name", mesh_name, "Name of the mesh")->required();
	app.add_option("--mesh", mesh_filename, "Mesh filepath")->check(CLI::ExistingFile)->required();
	app.add_option("--feature_edges", feature_edges_filename, "Feature edges filepath")->check(CLI::ExistingFile);
	app.add_option("-o,--output", output_dir, "Output directory")->check(CLI::ExistingDirectory);
	CLI11_PARSE(app, argc, argv);

	// Get input mesh
	Eigen::MatrixXd V, uv, N;
	Eigen::MatrixXi F, FT, FN;
	spdlog::info("Using mesh at {}", mesh_filename);
	igl::readOBJ(mesh_filename, V, uv, N, F, FT, FN);

	MeshCutter meshcutter(V, uv, F, FT);

	auto [V_cut, EE] = meshcutter.cut_mesh();

	Eigen::MatrixXi FE_reindex(0, 0);
	if (feature_edges_filename != "") {
		Eigen::MatrixXi FE = meshcutter.load_feature_edges(feature_edges_filename);
		FE_reindex = meshcutter.reindex_feature_edges(FE);
	}


	std::string output_filename = join_path(join_path(output_dir, "EE"), mesh_name + "_EE.txt");
	std::ofstream output_file(output_filename, std::ios::out | std::ios::trunc);
	output_file << EE.rows() << std::endl;
	for (int i = 0; i < EE.rows(); ++i) {
		output_file << EE(i, 0) << " " << EE(i, 1) << " "
			<< EE(i, 2) << " " << EE(i, 3) << "\n";
	}

	output_file.close();
	std::string feature_edge_filename = join_path(join_path(output_dir, "FE"), mesh_name + "_FE.txt");
	std::ofstream feature_file(feature_edge_filename, std::ios::out | std::ios::trunc);
	feature_file << FE_reindex.rows() << std::endl;
	for (int i = 0; i < FE_reindex.rows(); ++i) {
		feature_file << FE_reindex(i, 0) << " " << FE_reindex(i, 1) << " "
			<< FE_reindex(i, 2) << "\n";
	}
	feature_file.close();
	// Write the refined output
	output_filename = join_path(output_dir, mesh_name + "_init.obj");
	write_obj_with_uv(output_filename, V_cut, FT, uv, FT);
}