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
#include <tuple>

class MeshCutter {
public:
    MeshCutter(const Eigen::MatrixXd& V,
        const Eigen::MatrixXd& uv,
        const Eigen::MatrixXi& F,
        const Eigen::MatrixXi& FT);

    std::pair<Eigen::MatrixXd, Eigen::MatrixXi> cut_mesh();

    Eigen::MatrixXi load_feature_edges(const std::string& fe_filename);

    Eigen::MatrixXi load_misaligned_edges(const std::string& me_filename);

    Eigen::MatrixXi reindex_feature_edges(const Eigen::MatrixXi& FE);

    Eigen::MatrixXi remove_cycles_and_duplicates(const Eigen::MatrixXi& FE, const Eigen::MatrixXi& FE_reindex);

private:
    Eigen::MatrixXd V;
    Eigen::MatrixXd uv;
    Eigen::MatrixXi F;
    Eigen::MatrixXi FT;
    Eigen::MatrixXi TT;
    Eigen::MatrixXi TTi;
};
