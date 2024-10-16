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

Eigen::MatrixXi get_feature_edges(std::ifstream& inf);
/// Join two filepaths.
///
/// @param[in] first_path: first path to join
/// @param[in] second_path: second path to join
/// @return combined path
std::filesystem::path join_path(
    const std::filesystem::path& first_path,
    const std::filesystem::path& second_path)
{
    return first_path / second_path;
}


void write_obj_with_uv(
    const std::string& filename,
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F_uv)
{
    Eigen::MatrixXd N;
    Eigen::MatrixXi FN;
    igl::writeOBJ(filename, V, F, N, FN, uv, F_uv);
}

int main(int argc, char* argv[])
{
  // Get command line arguments
  CLI::App app{"Generate a conformal similarity parameterization."};
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

  int num_uv_vertices = uv.rows();
  int num_uv_faces = FT.rows();

	// Get boundary of the uv map
	Eigen::MatrixXi TT, TTi;
	igl::triangle_triangle_adjacency(F, TT, TTi);
	Eigen::MatrixXd bd;
	igl::boundary_facets(FT, bd);
	spdlog::info("{} boundary faces", bd.rows());

  // Check input validity
  if (F.rows() != num_uv_faces)
  {
    spdlog::error("F and FT have a different number of faces");
    return 1;
  }
  Eigen::MatrixXi FE;
  if (feature_edges_filename != "")
  {
    std::ifstream inf{ feature_edges_filename };
    if (!inf)
    {
      spdlog::error("Failed to load feature file\n");
      return 1;
    }
    FE = get_feature_edges(inf);
  }
  std::unordered_map<int, int> fe_to_f;
  for (int i = 0; i < FE.rows(); ++i)
  {
    bool found = false;
    for (int f = 0; f < F.rows(); ++f)
    {
      for (int p = 0; p < 3; ++p)
      {
        int q = (p + 1) % 3;
        
        if ((FE(i, 0) == F(f, p)) && (FE(i, 1) == F(f, q)))
        {
          fe_to_f[i] = f;
          found = true;
          break;
        }
        if ((FE(i, 1) == F(f, p)) && (FE(i, 0) == F(f, q)))
        {
          fe_to_f[i] = f;
          found = true;
          break;
        }
      }
      if (found) break;
    }
    if (!found)
    { 
      std::cerr << "Did not find feature edge" << FE(i, 0) << " " << FE(i, 1) << "\n";
      return 1;
    }
  }


  // Copy by face index correspondences
  Eigen::MatrixXd V_cut(num_uv_vertices, 3);
	std::vector<std::array<int, 4>> edge_pairings;
	edge_pairings.reserve(num_uv_faces);
  std::unordered_map<int, std::set<int>> V_map;
  //std::set<std::pair<int, int>> cut_edge_set;
  std::vector<std::array<int, 2>> feature_edges;

  for (int f = 0; f < num_uv_faces; ++f)
  {
		// Cut triangles according to uv seams
    for (int i = 0; i < 3; ++i)
    {
      int vi = F(f, i);
      int uvi = FT(f, i);
      V_cut.row(uvi) = V.row(vi);
      V_map[vi].insert(uvi);
    }

		// Check if an edge is a cut
    for (int i = 0; i < 3; ++i)
    {
			int j = (i + 1) % 3;
			int f_opp = TT(f, i);
			//if (f > f_opp) continue;
			assert(F(f, j) == F(f_opp, TTi(f, i)));
			assert(F(f, i) == F(f_opp, (TTi(f, i) + 1)%3));
			if (F(f, j) != F(f_opp, TTi(f, i))) spdlog::error("Incorrect 1");
			if (F(f, i) != F(f_opp, (TTi(f, i) + 1)%3))spdlog::error("Incorrect 2");
			int uvi = FT(f, i);
			int uvj = FT(f, j);
			int uvi_opp = FT(f_opp, (TTi(f, i) + 1)%3);
			int uvj_opp = FT(f_opp, TTi(f, i));
			if ((uvi != uvi_opp) || (uvj != uvj_opp))
			{
				//edge_pairings.push_back({uvi + 1, uvj + 1, uvi_opp + 1, uvj_opp + 1});
				edge_pairings.push_back({uvi, uvj, uvj_opp, uvi_opp});
        //cut_edge_set.emplace(uvi, uvj);
        //cut_edge_set.emplace(uvj_opp, uvi_opp);
			}
		}
  }

  // TODO Handle vertex reindexing
  for (int i = 0; i < FE.rows(); ++i)
  {
    int f = fe_to_f[i];
    bool found{false};
    for (int p = 0; p < 3; ++p)
    {
      int q = (p + 1) % 3;
      int f_opp = TT(f, p);
			//if (f > f_opp) continue;
			assert(F(f, q) == F(f_opp, TTi(f, p)));
			assert(F(f, p) == F(f_opp, (TTi(f, p) + 1)%3));
			if (F(f, q) != F(f_opp, TTi(f, p))) spdlog::error("Incorrect 1");
			if (F(f, p) != F(f_opp, (TTi(f, p) + 1)%3))spdlog::error("Incorrect 2");
      int uvi{};
      int uvj{};
      uvi = FT(f, p);
      uvj = FT(f, q);
      int uvi_opp = FT(f_opp, (TTi(f, p) + 1)%3);
			int uvj_opp = FT(f_opp, TTi(f, p));
      if ((FE(i, 0) == F(f, p)) && (FE(i, 1) == F(f, q)))
      {
        feature_edges.push_back({uvi, uvj});
        /*
        if (cut_edge_set.find(std::make_pair(uvi, uvj)) == cut_edge_set.end() && 
            cut_edge_set.find(std::make_pair(uvj, uvi)) == cut_edge_set.end())
        {
          std::cout << "adding feature edge\n";
          edge_pairings.push_back({uvi, uvj, uvj, uvi});
        }
        */
      }
      
      else if ((FE(i, 1) == F(f, p)) && (FE(i, 0) == F(f, q)))
      {
        feature_edges.push_back({uvj_opp, uvi_opp});
        /*
        if (cut_edge_set.find(std::make_pair(uvi, uvj)) == cut_edge_set.end() && 
            cut_edge_set.find(std::make_pair(uvj, uvi)) == cut_edge_set.end())
        {
          std::cout << "adding feature edge\n";
          edge_pairings.push_back({uvi, uvj, uvj, uvi});
        }
        */
      }
    }
  }

	std::string output_filename = join_path(join_path(output_dir, "EE"), mesh_name + "_EE.txt");
	std::ofstream output_file(output_filename, std::ios::out | std::ios::trunc);
	output_file << edge_pairings.size() << std::endl; 
  for (const auto& ep : edge_pairings)
  {
    output_file << ep[0] << " " << ep[1] << " " << ep[2] << " " << ep[3] << std::endl; 
	}
  output_file.close();
  std::string feature_edge_filename = join_path(join_path(output_dir, "FE"), mesh_name + "_FE.txt");
  std::ofstream feature_file(feature_edge_filename, std::ios::out | std::ios::trunc);
  feature_file << feature_edges.size() << std::endl; 
  for (const auto& ep : feature_edges)
  {
    feature_file << ep[0] << " " << ep[1] << std::endl; 
	}
  feature_file.close();
	// Write the refined output
	output_filename = join_path(output_dir, mesh_name + "_init.obj");
	write_obj_with_uv(output_filename, V_cut, FT, uv, FT);

}

Eigen::MatrixXi get_feature_edges(std::ifstream& inf)
{
  int num_edges{};
  inf >> num_edges;
  Eigen::MatrixXi FE(num_edges, 2);
  std::string line{};
  std::getline(inf, line);
  int l{0};
  while (std::getline(inf, line))
  {
    char label;
    int v1;
    int v2;
    std::istringstream iss(line);
    iss >> label >> v1 >> v2;
    FE(l, 0) = v1;
    FE(l, 1) = v2;
    ++l;
  }

  return FE.array() - 1;
}