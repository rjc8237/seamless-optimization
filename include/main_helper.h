#pragma once
#include <spdlog/common.h>
#include "spdlog/spdlog.h"

#include <igl/PI.h>
#include <igl/boundary_loop.h>
#include <igl/read_triangle_mesh.h>
#include <igl/upsample.h>
#include <igl/writeOBJ.h>

double check_constraints(
    const Eigen::MatrixXi& EE,
    const Eigen::MatrixXi& FE,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F)
{
    Eigen::SparseMatrix<double> Aeq;
    int N = uv.rows();
    int c = 0;
    int m = EE.rows() / 2;
    int fes = FE.rows();

    //std::vector<std::vector<int>> bds;
    //igl::boundary_loop(F, bds);

    // if there are no constraints then the constraint error should be 0?
    if (m == 0) {
        return 0.0;
    }
    double ret = 0;
    std::set<std::pair<int, int>> added_e;
    typedef Eigen::Triplet<double> Trip;
    std::vector<Trip> trips;
    trips.reserve(12 * EE.rows());
    for (int i = 0; i < EE.rows(); i++) {
        int A2 = EE(i, 0);
        int B2 = EE(i, 1);
        int C2 = EE(i, 2);
        int D2 = EE(i, 3);
        auto e0 = std::make_pair(A2, B2);
        auto e1 = std::make_pair(C2, D2);
        if (added_e.find(e0) != added_e.end() || added_e.find(e1) != added_e.end()) continue;
        added_e.insert(e0);
        added_e.insert(e1);

        Eigen::Vector2d e_ab = uv.row(B2) - uv.row(A2);
        Eigen::Vector2d e_dc = uv.row(C2) - uv.row(D2);

        Eigen::Vector2d e_ab_perp;
        e_ab_perp(0) = -e_ab(1);
        e_ab_perp(1) = e_ab(0);
        double angle = atan2(-e_ab_perp.dot(e_dc), e_ab.dot(e_dc));
        int r = (int)(round(2 * angle / igl::PI) + 2) % 4;

        std::vector<Eigen::Matrix<double, 2, 2>> r_mat(4);
        r_mat[0] << -1, 0, 0, -1;
        r_mat[1] << 0, 1, -1, 0;
        r_mat[2] << 1, 0, 0, 1;
        r_mat[3] << 0, -1, 1, 0;

        trips.push_back(Trip(c, A2, 1));
        trips.push_back(Trip(c, B2, -1));
        trips.push_back(Trip(c + 1, A2 + N, 1));
        trips.push_back(Trip(c + 1, B2 + N, -1));

        trips.push_back(Trip(c, C2, r_mat[r](0, 0)));
        trips.push_back(Trip(c, D2, -r_mat[r](0, 0)));
        trips.push_back(Trip(c, C2 + N, r_mat[r](0, 1)));
        trips.push_back(Trip(c, D2 + N, -r_mat[r](0, 1)));
        trips.push_back(Trip(c + 1, C2, r_mat[r](1, 0)));
        trips.push_back(Trip(c + 1, D2, -r_mat[r](1, 0)));
        trips.push_back(Trip(c + 1, C2 + N, r_mat[r](1, 1)));
        trips.push_back(Trip(c + 1, D2 + N, -r_mat[r](1, 1)));
        c = c + 2;
    }
    
    // feature edge constraints
    if (fes > 0)
    {
        for (int i = 0; i < FE.rows(); ++i) {
            int v1 = FE(i, 0);
            int v2 = FE(i, 1);
            auto e0 = std::make_pair(v1, v2);

            bool constrained = false;
            Eigen::Vector2d e_ab = uv.row(v2) - uv.row(v1);
            // constrain u or v depending on initial position
            if (FE(i, 2) == 0 && (std::abs(e_ab[0]) < std::abs(e_ab[1]))) {
                trips.push_back(Trip(c, v1, -1));
                trips.push_back(Trip(c, v2, 1));
                c += 1;
                constrained = true;
            }
            else if (FE(i, 2) == 1 && (std::abs(e_ab[0]) >= std::abs(e_ab[1]))) {
                trips.push_back(Trip(c, v1 + N, -1));
                trips.push_back(Trip(c, v2 + N, 1));
                c += 1;
                constrained = true;
            }
            if (!constrained)
            {
                std::cout << "Feature edge not aligned on u or v\n";
            }
        }
    }

    Aeq.resize(2 * m + fes, uv.rows() * 2);
    Aeq.setFromTriplets(trips.begin(), trips.end());
    Eigen::VectorXd flat_uv = Eigen::Map<const Eigen::VectorXd>(uv.data(), uv.size());
    auto res = Aeq * flat_uv;
    ret = res.cwiseAbs().maxCoeff();

    return ret;
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
std::vector<std::vector<int>> transform_EE(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& EE_v)
{
    std::vector<std::vector<int>> EE_e{};
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
    return EE_e;
}

std::vector<std::vector<int>> transform_FE(
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXi& FE_v)
{
    std::vector<std::vector<int>> FE_e{};
    FE_e.resize(FE_v.rows());
    for (int i = 0; i < FE_v.rows(); i++) {
        std::vector<int> one_row;
        int v0 = FE_v(i, 0), v1 = FE_v(i, 1);
        int fid, eid;
        if (find_edge_in_F(F, v0, v1, fid, eid)) {
            one_row.push_back(fid);
            one_row.push_back(eid);
        } else {
            std::cout << "Something Wrong in transform_FE: edge not found in F" << std::endl;
        }
        FE_e[i] = one_row;
    }

    return FE_e;
}