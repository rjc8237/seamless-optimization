#include "smooth_utils.h"
#include "spdlog/spdlog.h"
#include "ExtremeOpt.h"
#include <Eigen/CholmodSupport>
#include <Eigen/Core>
//#include <Spectra/SymEigsSolver.h>
//#include <Spectra/SymEigsShiftSolver.h>
//#include <Spectra/MatOp/SparseSymMatProd.h>
//#include <Spectra/MatOp/SparseSymShiftSolve.h>


namespace SymDir {
    Solver::Solver(const Eigen::SparseMatrix<double>& A, const std::string& solver_name, const double cg_rel_err): solver_name_(solver_name){
        if (solver_name_ == "CG") {
            cg_.setTolerance(cg_rel_err);
            cg_.setMaxIterations(10000);
        }
        if (solver_name_ == "CG_GS") {
            cg_gs.setTolerance(cg_rel_err);
            cg_gs.setMaxIterations(10000);
        }
        if (solver_name_ == "BiCGSTAB") {
            bicgstab.setTolerance(cg_rel_err);
        }
        if (solver_name_ == "CG_LLT") {
            cg_llt.setTolerance(cg_rel_err);
            cg_llt.setMaxIterations(4000);
        }
    }

    void Solver::compute(const Eigen::SparseMatrix<double>& A) {
        if (solver_name_ == "CG") {
            Eigen::SparseMatrix<double, Eigen::RowMajor> A_rowmajor = A;
            cg_.compute(A_rowmajor);
        } else if (solver_name_ == "Ch_LLT") {
            cholmod_super_llt_.compute(A);
        } else if (solver_name_ == "LDLT") {
            ldlt_.compute(A);
        } else if (solver_name_ == "LLT") {
            llt_.compute(A);
        } else if (solver_name_ == "BiCGSTAB") {
            bicgstab.compute(A);
        } else if (solver_name_ == "CG_GS") {
            cg_gs.compute(A);
        } else if (solver_name_ == "CG_LLT") {
            cg_llt.compute(A);
        } else {
            throw std::invalid_argument("Unknown solver name: " + solver_name_);
        }
    }

    Eigen::VectorXd Solver::solve(const Eigen::VectorXd& b) {
        if (solver_name_ == "CG") return cg_.solve(b);
        else if (solver_name_ == "Ch_LLT") return cholmod_super_llt_.solve(b);
        else if (solver_name_ == "LDLT") return ldlt_.solve(b);
        else if (solver_name_ == "LLT") return llt_.solve(b);
        else if (solver_name_ == "BiCGSTAB") return bicgstab.solve(b);
        else if (solver_name_ == "CG_GS") return cg_gs.solve(b);
        else if (solver_name_ == "CG_LLT") return cg_llt.solve(b);
        throw std::invalid_argument("Unknown solver: " + solver_name_);
    }

    Eigen::ComputationInfo Solver::info() const {
        if (solver_name_ == "CG") return cg_.info();
        else if (solver_name_ == "Ch_LLT") return cholmod_super_llt_.info();
        else if (solver_name_ == "LDLT") return ldlt_.info();
        else if (solver_name_ == "LLT") return llt_.info();
        else if (solver_name_ == "BiCGSTAB") return bicgstab.info();
        else if (solver_name_ == "CG_GS") return cg_gs.info();
        else if (solver_name_ == "CG_LLT") return cg_llt.info();
        return Eigen::InvalidInput;
    }

    int Solver::iterations() const {
        if (solver_name_ == "CG") return cg_.iterations();
        else if (solver_name_ == "BiCGSTAB") return bicgstab.iterations();
        else if (solver_name_ == "CG_GS") return cg_gs.iterations();
        else if (solver_name_ == "CG_LLT") return cg_llt.iterations();
        throw std::invalid_argument("iterations is only available for iterative solvers");
    }

    double get_cond_num_from_hessian(const Eigen::SparseMatrix<double>& hessian){
        spdlog::error("not implemented");
        return 0.;

//        Spectra::SparseSymMatProd<double> op(hessian);
//        Spectra::SymEigsSolver<Spectra::SparseSymMatProd<double>> eigs_larg(op, 2, 5);
//        eigs_larg.init();
//        eigs_larg.compute();
//        if (eigs_larg.info() != Spectra::CompInfo::Successful) {
//            spdlog::error("Failed to compute eigenvalues");
//            return -1;
//        }
//        double largest_eigen = eigs_larg.eigenvalues().maxCoeff();
//
//
//        Spectra::SparseSymShiftSolve<double> op_small(hessian);
//        Spectra::SymEigsShiftSolver<Spectra::SparseSymShiftSolve<double>> eigs_small(op_small, 2, 5, 0.0);
//        eigs_small.init();
//        eigs_small.compute();
//        if (eigs_small.info() != Spectra::CompInfo::Successful) {
//            spdlog::error("Failed to compute eigenvalues");
//            return -1;
//        }
//        double smallest_eigen = eigs_small.eigenvalues().minCoeff();
//        spdlog::info("Conditions number computation done: {}, {}", largest_eigen, smallest_eigen);
//        if (smallest_eigen == 0) {
//            spdlog::warn("Smallest eigenvalue is zero, condition number is infinite");
//            return std::numeric_limits<double>::infinity();
//        }
//        double cond_num = largest_eigen / smallest_eigen;
//        return cond_num;
    }

    template <typename Scalar>
    void projected_local_hessian(Eigen::Matrix<Scalar, 4, 4>& local_hessian) {
        // Eigen decomposition
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 4, 4>> es(local_hessian);
        Eigen::Matrix<Scalar, 4, 1> evals = es.eigenvalues();
        Eigen::Matrix<Scalar, 4, 4> evecs = es.eigenvectors();

        // Find the minimum eigenvalue and shift to make all eigenvalues positive
        const Scalar rel_eps = Scalar(1e-12);
        const Scalar eps = std::max(rel_eps, std::abs(local_hessian.trace()) * rel_eps);

        // Add shift to all eigenvalues and reconstruct
        for (int i = 0; i < 4; i++) {
            if (evals[i] < eps) {
                evals[i] = eps;
            }
        }

        // Reconstruct the projected Hessian
        local_hessian = evecs * evals.asDiagonal() * evecs.transpose();   
    }

    bool solveGaussSeidel(const Eigen::SparseMatrix<double>& A, 
                      const Eigen::VectorXd& b, 
                      Eigen::VectorXd& x, 
                      int max_iters, 
                      double tolerance) {
        
        int n = A.rows();
        Eigen::VectorXd invDiag(n);
        // Pre-compute Diagonal Inverse
        for (int i = 0; i < n; ++i) {
            double d = A.coeff(i, i);
            if (std::abs(d) < 1e-12) {
                std::cerr << "Error: Zero diagonal at index " << i << std::endl;
                return false;
            }
            invDiag(i) = 1.0 / d;
        }

        // Iteration Loop
        for (int iter = 0; iter < max_iters; ++iter) {
            double max_error = 0.0;

            // Loop over every row (vertex)
            for (int i = 0; i < n; ++i) {
                double sigma = 0.0;

                // Efficient Sparse Iteration over the row
                // We sum A_ij * x_j for all j != i
                for (Eigen::SparseMatrix<double>::InnerIterator it(A, i); it; ++it) {
                    if (it.col() != i) {
                        sigma += it.value() * x(it.col());
                    }
                }

                // 4. Compute new x_i
                // x_new = (b_i - sigma) / A_ii
                double x_new = (b(i) - sigma) * invDiag(i);

                // Track how much x changed (for convergence check)
                // Note: This is a faster proxy for residual checking than calculating (b-Ax).norm()
                double diff = std::abs(x_new - x(i));
                if (diff > max_error) max_error = diff;

                // 5. Immediate Update
                // This is what distinguishes Gauss-Seidel from Jacobi.
                // We update x(i) instantly so the next row sees the new value.
                x(i) = x_new;
            }

            // 6. Check Convergence
            if (max_error < tolerance) {
                std::cout << "Gauss-Seidel converged at iteration " << iter + 1 << std::endl;
                return true;
            }
        }
        std::cout << "Gauss-Seidel reached max iterations without full convergence." << std::endl;
        return false;
    }


    using RowMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
    using Vec = Eigen::VectorXd;

    void spmv(const RowMat& a, const Vec& x, Vec& y) {
        y.resize(a.rows());
        #pragma omp parallel for schedule(static)
        for (Eigen::Index i = 0; i < a.rows(); ++i) {
            double sum = 0.0;
            for (RowMat::InnerIterator it(a, i); it; ++it) {
            sum += it.value() * x[it.col()];
            }
            y[i] = sum;
        }
    }

    double dot(const Vec& a, const Vec& b) {
        double sum = 0.0;
        #pragma omp parallel for reduction(+ : sum) schedule(static)
        for (Eigen::Index i = 0; i < a.size(); ++i) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    void axpy(double alpha, const Vec& x, Vec& y) {
        #pragma omp parallel for schedule(static)
        for (Eigen::Index i = 0; i < y.size(); ++i) {
            y[i] += alpha * x[i];
        }
    }

    CgResult conjugate_gradient(const RowMat& a, const Vec& b, Vec& x, int max_iter, double tol) {
        spdlog::info("CG solver starting with {} OpenMP threads", omp_get_max_threads());
        const Eigen::Index n = b.size();

        x.setZero(n);
        Vec r = b;
        Vec p = r;
        Vec ap(n);

        const double b_norm = std::max(1e-30, std::sqrt(dot(b, b)));
        double rsold = dot(r, r);
        double rel_residual = std::sqrt(rsold) / b_norm;

        int iter = 0;
        bool converged = false;
        for (; iter < max_iter && rel_residual > tol; ++iter) {
            spmv(a, p, ap);
            const double p_ap = dot(p, ap);
            if (p_ap == 0.0) {
                break;
            }

            const double alpha = rsold / p_ap;
            axpy(alpha, p, x);
            axpy(-alpha, ap, r);

            const double rsnew = dot(r, r);
            rel_residual = std::sqrt(rsnew) / b_norm;
            if (rel_residual <= tol) {
                converged = true;  // Mark as converged
                rsold = rsnew;
                break;
            }

            const double beta = rsnew / rsold;
            #pragma omp parallel for schedule(static)
            for (Eigen::Index i = 0; i < n; ++i) {
                p[i] = r[i] + beta * p[i];
            }
            rsold = rsnew;
        }

        return {iter, rel_residual, converged};

    }

    template void SymDir::projected_local_hessian<double>(Eigen::Matrix<double, 4, 4>& local_hessian);
}

