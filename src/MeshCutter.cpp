#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <igl/boundary_facets.h>
#include <igl/triangle_triangle_adjacency.h>
#include "MeshCutter.h"
#include "spdlog/spdlog.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <fstream>
#include <sstream>
#include <Eigen/Dense>

MeshCutter::MeshCutter(const Eigen::MatrixXd& V,
	const Eigen::MatrixXd& uv,
	const Eigen::MatrixXi& F,
	const Eigen::MatrixXi& FT) :
	V{ V }, uv{ uv }, F{ F }, FT{ FT } {
	igl::triangle_triangle_adjacency(F, TT, TTi);
	// Get boundary of the uv map
	Eigen::MatrixXd bd;
	igl::boundary_facets(FT, bd);
	spdlog::info("{} boundary faces", bd.rows());

	// Check input validity
	if (F.rows() != FT.rows()) {
		spdlog::error("F and FT have a different number of faces");
		exit(EXIT_FAILURE);
	}
}

std::pair<Eigen::MatrixXd, Eigen::MatrixXi> MeshCutter::cut_mesh() {
	int num_uv_vertices = uv.rows();
	int num_uv_faces = FT.rows();


	// Copy by face index correspondences
	Eigen::MatrixXd V_cut(num_uv_vertices, 3);
	std::vector<std::array<int, 4>> edge_pairings;
	edge_pairings.reserve(num_uv_faces);

	for (int f = 0; f < num_uv_faces; ++f) {
		// Cut triangles according to uv seams
		for (int i = 0; i < 3; ++i) {
			int vi = F(f, i);
			int uvi = FT(f, i);
			V_cut.row(uvi) = V.row(vi);
		}

		// Check if an edge is a cut
		for (int i = 0; i < 3; ++i) {
			int j = (i + 1) % 3;
			int f_opp = TT(f, i);
			// if (f > f_opp) continue;
			assert(F(f, j) == F(f_opp, TTi(f, i)));
			assert(F(f, i) == F(f_opp, (TTi(f, i) + 1) % 3));
			if (F(f, j) != F(f_opp, TTi(f, i)))
				spdlog::error("Incorrect 1");
			if (F(f, i) != F(f_opp, (TTi(f, i) + 1) % 3))
				spdlog::error("Incorrect 2");
			int uvi = FT(f, i);
			int uvj = FT(f, j);
			int uvi_opp = FT(f_opp, (TTi(f, i) + 1) % 3);
			int uvj_opp = FT(f_opp, TTi(f, i));
			if ((uvi != uvi_opp) || (uvj != uvj_opp)) {
				edge_pairings.push_back({ uvi, uvj, uvj_opp, uvi_opp });
			}
		}
	}
	Eigen::MatrixXi EE(edge_pairings.size(), 4);
	for (int i = 0; i < edge_pairings.size(); ++i) {
		EE.row(i) << edge_pairings[i][0], edge_pairings[i][1], edge_pairings[i][2], edge_pairings[i][3];
	}

	return { V_cut, EE };
}

Eigen::MatrixXi MeshCutter::load_feature_edges(std::string_view fe_filename) {
	std::ifstream inf{ fe_filename };
	if (!inf) {
		spdlog::error("Failed to load feature edges file\n");
		exit(EXIT_FAILURE);
	}


	Eigen::MatrixXi FE(0, 2);
	std::string line{};
	std::getline(inf, line);
	int l{ 0 };
	while (std::getline(inf, line)) {
		FE.conservativeResize(l + 1, 2);
		char label;
		std::istringstream iss(line);
		iss >> label;
		if (label != 'l') 
			continue;
		int v1;
		int v2;
		iss >> v1 >> v2;
		FE(l, 0) = v1;
		FE(l, 1) = v2;
		++l;
	}

	return FE.array() - 1;
}

Eigen::MatrixXi MeshCutter::reindex_feature_edges(const Eigen::MatrixXi& FE) {
	std::unordered_map<int, int> fe_to_f;
	std::vector<std::array<int, 2>> feature_edges;
	for (int i = 0; i < FE.rows(); ++i) {
		bool found = false;
		for (int f = 0; f < F.rows(); ++f) {
			for (int p = 0; p < 3; ++p) {
				int q = (p + 1) % 3;

				if ((FE(i, 0) == F(f, p)) && (FE(i, 1) == F(f, q))) {
					fe_to_f[i] = f;
					found = true;
					break;
				}
				if ((FE(i, 1) == F(f, p)) && (FE(i, 0) == F(f, q))) {
					fe_to_f[i] = f;
					found = true;
					break;
				}
			}
			if (found) break;
		}
		if (!found) {
			spdlog::error("Did not find feature edge {} {}\n", FE(i, 0), FE(i, 1));
			exit(EXIT_FAILURE);
		}
	}

	for (int i = 0; i < FE.rows(); ++i) {
		int f = fe_to_f[i];
		bool found{ false };
		for (int p = 0; p < 3; ++p) {
			int q = (p + 1) % 3;
			int f_opp = TT(f, p);
			//if (f > f_opp) continue;
			assert(F(f, q) == F(f_opp, TTi(f, p)));
			assert(F(f, p) == F(f_opp, (TTi(f, p) + 1) % 3));
			if (F(f, q) != F(f_opp, TTi(f, p))) spdlog::error("Incorrect 1");
			if (F(f, p) != F(f_opp, (TTi(f, p) + 1) % 3))spdlog::error("Incorrect 2");
			int uvi{};
			int uvj{};
			uvi = FT(f, p);
			uvj = FT(f, q);
			int uvi_opp = FT(f_opp, (TTi(f, p) + 1) % 3);
			int uvj_opp = FT(f_opp, TTi(f, p));
			if ((FE(i, 0) == F(f, p)) && (FE(i, 1) == F(f, q))) {
				feature_edges.push_back({ uvi, uvj });
			}

			else if ((FE(i, 1) == F(f, p)) && (FE(i, 0) == F(f, q))) {
				feature_edges.push_back({ uvj_opp, uvi_opp });
			}
		}
	}
	Eigen::MatrixXi FE_reindex(feature_edges.size(), 3);
	int r = 0;
	for (auto [v1, v2] : feature_edges) {
		Eigen::Vector2d e_ab = uv.row(v2) - uv.row(v1);
		// constrain u or v depending on initial position
		//if (-1e-12 < e_ab[0] && e_ab[0] < 1e-12) {
		//	FE_reindex.row(r) << v1, v2, 0;
		//}
		//else if (-1e-12 < e_ab[1] && e_ab[1] < 1e-12) {
		//	FE_reindex.row(r) << v1, v2, 1;
		//}

		// constrain u or v depending on initial position
		if (std::abs(e_ab[0]) < std::abs(e_ab[1])) {
			FE_reindex.row(r) << v1, v2, 0;
		}
		else {
			FE_reindex.row(r) << v1, v2, 1;
		}
		++r;
	}
	return FE_reindex;
}