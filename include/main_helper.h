#pragma once
#include <spdlog/common.h>
#include "spdlog/spdlog.h"

#include <igl/PI.h>
#include <igl/boundary_loop.h>
#include <igl/read_triangle_mesh.h>
#include <igl/upsample.h>
#include <igl/writeOBJ.h>

// TODO: This is very redundant-replace
double check_constraints(
    const Eigen::MatrixXi& EE,
    const Eigen::MatrixXi& FE,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F);

bool find_edge_in_F(const Eigen::MatrixXi& F, int v0, int v1, int& fid, int& eid);
bool find_edge_in_F(const Eigen::MatrixXi& F, const Eigen::SparseMatrix<int>& vv2f, int v0, int v1, int& fid, int& eid);
Eigen::SparseMatrix<int> generate_VV_to_face_map(const Eigen::MatrixXi& F);

std::vector<std::vector<int>> transform_EE(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& EE_v);

std::vector<std::vector<int>> transform_FE(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& FE_v);