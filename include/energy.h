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

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Sparse>
#include <iostream>

#include "autodiff_jakob.h"

// #define SOFT_MAX
// #define TEST_AMIPS_OLD
namespace SymDir {
template <typename T>
T symmetric_dirichlet_energy_t(T a, T b, T c, T d, double norm_p, bool soft_max = false, double t = 1.0, double E_min = 1.0)
{
    auto det = a * d - b * c;
    auto frob2 = a * a + b * b + c * c + d * d;
    auto energy = frob2 * (1.0 + 1.0 / (det * det)) - 4.0;
    #ifndef TEST_AMIPS_OLD
    if (soft_max) {
        return exp(energy / t / E_min);
    }
    
    return pow(energy / E_min, norm_p);
#else
    return pow((frob2 / det), norm_p);
#endif
    // return frob2 / det; // amips
}

template <typename Derived>
inline Eigen::VectorXd symmetric_dirichlet_energy(
    const Eigen::MatrixBase<Derived>& a,
    const Eigen::MatrixBase<Derived>& b,
    const Eigen::MatrixBase<Derived>& c,
    const Eigen::MatrixBase<Derived>& d,
    double norm_p, bool soft_max = false, double t = 1.0, double E_min = 1.0)
{
    auto det = a.array() * d.array() - b.array() * c.array();
    auto frob2 = a.array().abs2() + b.array().abs2() + c.array().abs2() + d.array().abs2();
    auto energy = frob2 * (1.0 + det.abs2().inverse()) - 4.0;
#ifndef TEST_AMIPS_OLD
    if (soft_max) {
        auto energy_exp = (energy / t / E_min).array().exp();
        return energy_exp.matrix();
    }
    return (energy / E_min).pow(norm_p).matrix();
#else
    auto result = (frob2 * det.inverse()).pow(norm_p);
    return result.matrix();
#endif

}

template <typename Scalar>
Scalar compute_energy_from_jacobian(
    const Eigen::Matrix<Scalar, -1, -1>& J,
    const Eigen::Matrix<Scalar, -1, 1>& areas,
    double norm_p, bool soft_max = false, double t = 1.0, double E_min = 1.0,
    bool uniform = false);
template <typename Scalar> 
std::vector<Scalar> compute_worst_n_energy_from_jacobian(
    const Eigen::Matrix<Scalar, -1, -1>& J,
    const Eigen::Matrix<Scalar, -1, 1>& area,
    int norm_p,
    std::vector<double> percentages, bool soft_max = false, double t = 1.0, double E_min = 1.0);
template <typename Scalar>
Scalar compute_threshold_energy_from_jacobian(
    const Eigen::Matrix<Scalar, -1, -1>& J,
    const Eigen::Matrix<Scalar, -1, 1>& area,
    int norm_p,
    double percent, bool soft_max = false, double t = 1.0);
template <typename Scalar>
    Eigen::ArrayXd get_sym_dirich_per_triangle_from_jacobian(const Eigen::Matrix<Scalar, -1, -1> &J);

template <typename Scalar>
Scalar get_grad_and_hessian(
    const Eigen::SparseMatrix<Scalar>& G,
    const Eigen::Matrix<Scalar, -1, 1>& area,
    const Eigen::Matrix<Scalar, -1, -1>& uv,
    Eigen::Matrix<Scalar, -1, 1>& grad,
    Eigen::SparseMatrix<Scalar>& hessian,
    bool get_hessian,
    double norm_p, bool projected_newton, bool soft_max = false, double t = 1.0, double E_min = 1.0);

template <typename Scalar>
void jacobian_from_uv(
    const Eigen::SparseMatrix<Scalar>& G,
    const Eigen::Matrix<Scalar, -1, -1>& uv,
    Eigen::Matrix<Scalar, -1, -1>& Ji);
} // namespace SymDir
namespace jakob
{
    template <typename Scalar>
    Scalar gradient_and_hessian_from_J(const Eigen::Matrix<Scalar, 1, 4> &J,
                                       Eigen::Matrix<Scalar, 1, 4> &local_grad,
                                       Eigen::Matrix<Scalar, 4, 4> &local_hessian,
                                       double norm_p, bool soft_max = false, double t = 1.0, double E_min = 1.0);
}
