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

#ifndef RREF_H
#define RREF_H
#include <Eigen/Sparse>
#include <iostream>

void swap_two_rows(
    Eigen::SparseMatrix<double> &R,
    int i,
    int k
);
template<typename mat>
void rref(
    const mat &A_in,
    mat &R,
    std::vector<int> &jb,
    double tol = 2e-12);

void elim_constr(
    const Eigen::SparseMatrix<double> &C,
    Eigen::SparseMatrix<double> &T
);

void elim_constr(
    const Eigen::SparseMatrix<double> &C,
    const Eigen::VectorXd &d,
    Eigen::SparseMatrix<double> &T_out,
    Eigen::VectorXd &b
);
#endif
