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

template <typename Scalar>
class DegenerateVerticesPreconditioner {
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::SparseMatrix<Scalar> SparseMat;

    // Components of the preconditioner
    Vector inv_diag_A;                                // B^-1 (Jacobi)
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> ss_solver;       // A_ss^-1
    int g_size;                                       // Offset for ss block
    SparseMat A_copy;                           // Copy of full A
    int iter = 0;

public:
    typedef int StorageIndex; // <--- This tells Eigen how to index your matrices
    enum {
            ColsAtCompileTime = Eigen::Dynamic,
            MaxColsAtCompileTime = Eigen::Dynamic
    };
    
    DegenerateVerticesPreconditioner() : A_copy(SparseMat(0,0)), g_size(0) {}

    Eigen::Index rows() const { return A_copy.rows(); }
    Eigen::Index cols() const { return A_copy.cols(); }

    // Initialize the solvers and diagonal
    DegenerateVerticesPreconditioner& compute(const SparseMat& A) {
        A_copy = A;

        // 1. Build B^-1 (Jacobi)
        inv_diag_A = A.diagonal().cwiseInverse();

        // 2. Extract and Factorize A_ss
        // Assumes you've already permuted A so that ss is the bottom-right block
        int total_size = A.rows();
        int s_size = total_size - g_size; 
        SparseMat Ass = A_copy.bottomRightCorner(s_size, s_size);
        ss_solver.compute(Ass);

        return *this;
    }

    // Set the partition boundary (must be set before compute)
    void set_g_size(int size) { g_size = size; }

    // This implements: z_i = M^-1 * r
    template<typename Rhs, typename Dest>
    void _solve_impl(const Rhs& r, Dest& z) const {
        int n = r.size();
        int s_size = n - g_size;

        Vector r_vec = r;

        // Stage 1: z_i1 = B^-1 * r
        Vector z_i1 = inv_diag_A.cwiseProduct(r);

        // Stage 2: z_i2 = z_i1 + [0 ; Ass^-1] * (r - A*z_i1)
        Vector res1 = r_vec - A_copy * z_i1;
        Vector z_i2 = z_i1;
        // The correction only applies to the 's' indices (bottom part)
        z_i2.tail(s_size) += ss_solver.solve(res1.tail(s_size));

        // Stage 3: z_i = z_i2 + B^-1 * (r - A*z_i2)
        Vector res2 = r - A_copy * z_i2;
        z = z_i2 + inv_diag_A.cwiseProduct(res2);
    }

    template<typename Rhs>
    inline const Eigen::Solve<DegenerateVerticesPreconditioner, Rhs>
    solve(const Eigen::MatrixBase<Rhs>& b) const {
        // Return the lazy evaluation blueprint, passing b's exact type
        return Eigen::Solve<DegenerateVerticesPreconditioner, Rhs>(*this, b.derived());
    }

    Eigen::ComputationInfo info() const { return ss_solver.info(); }
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
