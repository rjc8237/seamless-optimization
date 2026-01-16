#pragma once

#include <ExtremeOpt.h>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/CholmodSupport>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseLU> // if needed
#include <string>

namespace SymDir {
// A custom Preconditioner for Eigen that implements 
// Symmetric Gauss-Seidel (Forward sweep + Backward sweep).

template <typename MatrixType>
class SymmetricGaussSeidelPreconditioner {
typedef typename MatrixType::Scalar Scalar;
    typedef typename MatrixType::Index Index;

    // We store a local Row-Major copy of the matrix for fast row iteration
    Eigen::SparseMatrix<Scalar, Eigen::RowMajor> m_rowMajorA;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> m_invDiag;

public:
    typedef typename MatrixType::Scalar ScalarType;

    SymmetricGaussSeidelPreconditioner() {}

    // 1. Compute: Converts input matrix to RowMajor and pre-calcs diagonals
    void compute(const MatrixType& A) {
        m_rowMajorA = A; 

        Index n = m_rowMajorA.rows();
        m_invDiag.resize(n);

        for (Index i = 0; i < n; ++i) {
            Scalar d = m_rowMajorA.coeff(i, i);
            if (std::abs(d) < 1e-12) d = 1.0; // Regularization for zero diagonals
            m_invDiag(i) = Scalar(1.0) / d;
        }
    }

    // 2. Solve: Runs Symmetric Gauss-Seidel (Forward + Backward)
    template <typename Rhs>
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> solve(const Rhs& b) const {
        Index n = m_rowMajorA.rows();
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> x = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(n);

        // --- Forward Sweep (0 -> n-1) ---
        for (Index i = 0; i < n; ++i) {
            Scalar sigma = 0;
            // Iterate over the Row-Major matrix efficiently
            for (typename Eigen::SparseMatrix<Scalar, Eigen::RowMajor>::InnerIterator it(m_rowMajorA, i); it; ++it) {
                if (it.col() != i) {
                    sigma += it.value() * x(it.col());
                }
            }
            x(i) = (b(i) - sigma) * m_invDiag(i);
        }

        // --- Backward Sweep (n-1 -> 0) ---
        for (Index i = n - 1; i >= 0; --i) {
            Scalar sigma = 0;
            for (typename Eigen::SparseMatrix<Scalar, Eigen::RowMajor>::InnerIterator it(m_rowMajorA, i); it; ++it) {
                if (it.col() != i) {
                    sigma += it.value() * x(it.col());
                }
            }
            x(i) = (b(i) - sigma) * m_invDiag(i);
        }

        return x;
    }

    Eigen::ComputationInfo info() { return Eigen::Success; }
};

class Solver {
public:
    Solver(const Eigen::SparseMatrix<double>& A, const std::string& solver_name, const double cg_rel_err=0);

    void compute(const Eigen::SparseMatrix<double>& A);
    Eigen::VectorXd solve(const Eigen::VectorXd& b);
    Eigen::ComputationInfo info() const;
    int iterations() const;

private:
    std::string solver_name_;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> cg_;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper, Eigen::IncompleteCholesky<double>> cg_llt;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper, 
    SymmetricGaussSeidelPreconditioner<Eigen::SparseMatrix<double>>> cg_gs;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper> ldlt_;
    Eigen::LDLT<Eigen::MatrixXd, Eigen::RowMajor> ldlt_matrix_;
    Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> llt_;
    Eigen::CholmodSupernodalLLT<Eigen::SparseMatrix<double>> cholmod_super_llt_;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double>> bicgstab;
};
template<typename Scalar>
void projected_local_hessian(Eigen::Matrix<Scalar, 4, 4> &local_hessian);

bool solveGaussSeidel(const Eigen::SparseMatrix<double>& A, 
                      const Eigen::VectorXd& b, 
                      Eigen::VectorXd& x, 
                      int max_iters = 2000, 
                      double tolerance = 1e-10);
// double get_cond_num_from_hessian(const Eigen::SparseMatrix<double>& hessian);

struct CgResult {
    int iterations = 0;
    double rel_residual = 0.0;
    bool converged = false;
};

CgResult conjugate_gradient(const Eigen::SparseMatrix<double, Eigen::RowMajor>& a,
    const Eigen::VectorXd& b,
    Eigen::VectorXd& x,
    int max_iter,
    double tol);

} // namespace SymDir
