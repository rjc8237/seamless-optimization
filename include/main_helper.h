// Copyright (C) 2026 Ryan Capouellez
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

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
