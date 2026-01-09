#include "rref.h"
#include <igl/slice.h>
#include <igl/slice_into.h>
#include <Eigen/Core>
#include <igl/list_to_matrix.h>
#include <igl/Timer.h>
#include "spdlog/spdlog.h"
#include "spdlog/fmt/ostr.h"
#include <Eigen/CholmodSupport>
#include <Eigen/SPQRSupport>
#include <numeric>

//#include <SuiteSparseQR.hpp>
//#include <cholmod.h>

Eigen::VectorXi get_seq(int start, int end)
{
    Eigen::VectorXi seq(end - start + 1);
    for (int i = start; i <= end; i++)
    {
        seq(i - start) = i;
    }
    return seq;
}

template <typename mat>
void slice_mat(
    mat &A_in,
    int i_start, int i_end,
    int j_start, int j_end,
    mat &A_out)
{
    Eigen::VectorXi II = get_seq(i_start, i_end);
    Eigen::VectorXi JJ = get_seq(j_start, j_end);
    igl::slice(A_in, II, JJ, A_out);
}

void slice_into_sparse(
    Eigen::SparseMatrix<double> &A,
    const Eigen::VectorXi &II,
    const Eigen::VectorXi &JJ,
    Eigen::SparseMatrix<double> &B)
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> dyn_A(A);
    // Iterate over rows
    for (int i = 0; i < B.rows(); i++)
    {
        // Iterate over cols
        for (int j = 0; j < B.cols(); j++)
        { 
            dyn_A.coeffRef(II(i),JJ(j)) = B.coeff(i, j);
        }
    }
    A = Eigen::SparseMatrix<double>(dyn_A);
}

void slice_into_mat(
    Eigen::SparseMatrix<double> &A,
    int i_start, int i_end,
    int j_start, int j_end,
    Eigen::SparseMatrix<double> &B)
{
    Eigen::VectorXi II = get_seq(i_start, i_end);
    Eigen::VectorXi JJ = get_seq(j_start, j_end);

    slice_into_sparse(A, II, JJ, B);
}

void slice_into_mat(
    Eigen::MatrixXd &A,
    int i_start, int i_end,
    int j_start, int j_end,
    Eigen::MatrixXd &B)
{
    Eigen::VectorXi II = get_seq(i_start, i_end);
    Eigen::VectorXi JJ = get_seq(j_start, j_end);
    igl::slice_into(B, II, JJ, A);
}


template <typename mat>
int find_pivot(
    mat &A,
    int i,
    int j,
    double &p)
{
    int m = A.rows() - 1;

    int k = i;
    p = A.coeff(i, j) * A.coeff(i, j);
    for (int l = i + 1; l <= m; l++)
    {
        double tmp = A.coeff(l, j) * A.coeff(l, j);
        if (tmp > p)
        {
            p = tmp;
            k = l;
        }
    }

    return k;
}

template <typename mat>
void rref(
    const mat &A_in,
    mat &R,
    std::vector<int> &jb,
    double tol)
{
    jb.clear();
    R = A_in;
    int m = R.rows() - 1, n = R.cols() - 1;

igl::Timer timer;
timer.start();
    // loop over the entire matrix
    int i = 0, j = 0;
    // Eigen::SparseMatrix<double, Eigen::RowMajor> R_rm(R);
    Eigen::SparseMatrix<double, Eigen::RowMajor> R_rm(R);
    while (i <= m && j <= n)
    {
        double p = 0;
        // find value(p) and index(k) of largest element in the remainder of column j
auto tm_pv0 = timer.getElapsedTime();        
        int k = find_pivot(R_rm, i, j, p);
        // int k = find_pivot(R_rm, i, j, p);
auto tm_pv1 = timer.getElapsedTime();

        // std::cout << "k = " << k+1 << " p = " << p << std::endl;
// std::cout << "find_pivot time: " << tm_pv1 - tm_pv0;
        if (p <= tol)
        {
            // the column is negligible, zero it out
            // R.prune([i, j](int ii, int jj, double) { return !(ii >= i && jj == j); });
            j++;
        }
        else
        {
            // remember column index
            jb.push_back(j);

auto tm_swap0 = timer.getElapsedTime();        
            // swap i-th and k-th rows
            // swap_two_rows(R, i, k);
            Eigen::SparseMatrix<double, Eigen::RowMajor> tmp_i = R_rm.row(i);
            Eigen::SparseMatrix<double, Eigen::RowMajor> tmp_k = R_rm.row(k);
    
            R_rm.row(i) = tmp_k;
            R_rm.row(k) = tmp_i;
auto tm_swap1 = timer.getElapsedTime();        
// std::cout << "\tswap time: " << tm_swap1 - tm_swap0;
auto tm_subtract0 = timer.getElapsedTime();        
            // divide the pivot row by the pivot element
            Eigen::SparseMatrix<double, Eigen::RowMajor> Ai = R_rm.row(i) / R_rm.coeff(i, j);
            Eigen::SparseMatrix<double, Eigen::ColMajor> colj = R_rm.col(j);
auto tm_subtract1 = timer.getElapsedTime();        

            //Eigen::SparseMatrix<double, Eigen::RowMajor> tmp = colj * Ai;
            //Eigen::SparseMatrix<double, Eigen::RowMajor> tmp = R_rm.col(j) * Ai;

auto tm_subtract2 = timer.getElapsedTime();        

            //R_rm = R_rm - tmp;
            //R_rm -= tmp;
            //for (int ri = 0; ri < tmp.outerSize(); ++ri)
            //{
            //    for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(tmp, ri); it; ++it)
            //    {
            //        R_rm.coeffRef(it.row(), it.col()) -= it.value();
            //    }
            //}
            for (Eigen::SparseMatrix<double, Eigen::ColMajor>::InnerIterator it(colj, 0); it; ++it)
            {
                R_rm.row(it.row()) = (R_rm.row(it.row()) - it.value() * Ai).pruned();
            }
            R_rm.row(i) = Ai;
            //R_rm = R_rm.pruned();

auto tm_subtract3 = timer.getElapsedTime();        
// std::cout << "\tsubtract time: " << tm_subtract1 - tm_subtract0 << ", " << tm_subtract2 - tm_subtract1 << ", " << tm_subtract3 - tm_subtract2;
            i++;
            j++;
        }
auto tm_end = timer.getElapsedTime();        
// std::cout << "\ttotal time:" << tm_end - tm_pv0 << std::endl;    
    }
    R = mat(R_rm);
}

int qr(
    const Eigen::SparseMatrix<double>& A_in,
    Eigen::SparseMatrix<double>& R_out,
    Eigen::SparseMatrix<double>& P_out)
{
    typedef int32_t Int;
    typedef Eigen::SparseMatrix<double, Eigen::ColMajor, Int> CholmodMatrix;

    cholmod_common c;
    cholmod_start(&c); // Initialize CHOLMOD

    // Define your sparse matrix A in CHOLMOD format
    int ordering = 7;
    double pivotThreshold = -2;
    CholmodMatrix A_temp(A_in);
    A_temp.makeCompressed();
    cholmod_sparse A = Eigen::viewAsCholmod<double, Eigen::ColMajor, Int>(A_temp);
    //cholmod_dense* B = cholmod_ones(A.nrow, 1, A.xtype, &c);
    //SuiteSparseQR_min2norm<double, Int>(ordering, pivotThreshold, &A, B, &c);
    //SuiteSparseQR<double, Int>(&A, B, &c);
    spdlog::info("{} nonzeros in A", A_temp.nonZeros());


    // Perform QR factorization with column permutation
    //cholmod_sparse* Q = nullptr;
    cholmod_sparse* R = nullptr;
    //cholmod_dense* beta = nullptr;
    //cholmod_sparse* P = nullptr; // Column permutation matrix
    Int* P = nullptr; // Column permutation matrix

    // Use SuiteSparseQR_factorize to compute QR factorization with permutation
    //Int econ = A_in.rows();
    Int econ = 0;
        //SPQR_DEFAULT_TOL,      // Default tolerance
    Int rank = SuiteSparseQR<double, Int>(
        SPQR_ORDERING_DEFAULT, // Column permutation ordering (e.g., COLAMD)
        2e-12,      // Default tolerance
        econ,
        &A,                     // Input sparse matrix
        &R, &P,            // Output Q, R, and P
        &c                     // CHOLMOD workspace
    );
    spdlog::info("Rank is {}", rank);

    // Use the results (Q, R, and P) as neededS
    //Q_out = Eigen::viewAsEigen<double, Eigen::RowMajor, int64_t>(*Q);
    R_out = Eigen::viewAsEigen<double, Eigen::ColMajor, Int>(*R);
    int n = A_in.cols();
    P_out.resize(n, n);
    typedef Eigen::Triplet<double> Trip;
    std::vector<Trip> trips;
    trips.reserve(n);
    for (int i = 0; i < n; ++i)
    {
        trips.push_back(Trip(P[i], i, 1.));
    }
    P_out.setFromTriplets(trips.begin(), trips.end());
    spdlog::info("{} nonzeros in R", R_out.nonZeros());

    // Clean up
    //cholmod_free_sparse(&Q, &c);
    cholmod_free_sparse(&R, &c);
    //cholmod_free_sparse(&P, &c);
    cholmod_finish(&c);

    return rank;
}

void _backslash(
    const Eigen::SparseMatrix<double>& A_in,
    const Eigen::SparseMatrix<double>& B_in,
    Eigen::SparseMatrix<double>& C_out)
{
    typedef int32_t Int;
    typedef Eigen::SparseMatrix<double, Eigen::ColMajor, Int> CholmodMatrix;

    cholmod_common c;
    cholmod_start(&c); // Initialize CHOLMOD

    // Define your sparse matrix A, B in CHOLMOD format
    CholmodMatrix A_temp(A_in);
    CholmodMatrix B_temp(B_in);
    A_temp.makeCompressed();
    B_temp.makeCompressed();
    cholmod_sparse A = Eigen::viewAsCholmod<double, Eigen::ColMajor, Int>(A_temp);
    cholmod_sparse B = Eigen::viewAsCholmod<double, Eigen::ColMajor, Int>(B_temp);


    // Perform QR factorization with column permutation
    cholmod_sparse* C = SuiteSparseQR<double, Int>(
        SPQR_ORDERING_DEFAULT, 
        2e-12,
        &A,
        &B,
        &c);

    //cholmod_sparse* C = cholmod_solve(CHOLMOD_A, &A, &B, &c);

    // Use the results (Q, R, and P) as neededS
    //Q_out = Eigen::viewAsEigen<double, Eigen::RowMajor, int64_t>(*Q);
    C_out = Eigen::viewAsEigen<double, Eigen::ColMajor, Int>(*C);
    spdlog::info("{} nonzeros in C", C_out.nonZeros());

    // Clean up
    cholmod_finish(&c);
}

void backslash(
    const Eigen::SparseMatrix<double>& R,
    const Eigen::SparseMatrix<double>& A,
    Eigen::SparseMatrix<double>& B)
{
    int n = A.rows();
    int m = A.cols();
    Eigen::SparseMatrix<double, Eigen::RowMajor> R_rm(R);
    Eigen::SparseMatrix<double, Eigen::RowMajor> A_rm(A);
    Eigen::SparseMatrix<double, Eigen::RowMajor> B_rm(A);
    for (int i = n - 1; i >= 0; --i)
    {
        double value = 0.;
        for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(R_rm, i); it; ++it)
        {
            if (it.col() == i)
            {
                value = it.value();
                //B_rm.row(i) += (A_rm.row(i) / it.value());
            }
            else
            {
                B_rm.row(i) -= (it.value() * B_rm.row(it.col()));
            }
        }
        B_rm.row(i) = (B_rm.row(i) / value).pruned();
    }
    spdlog::info("{} nonzeros in B", B_rm.nonZeros());

    B = Eigen::SparseMatrix<double>(B_rm);
    spdlog::info("{} nonzeros in B", B.nonZeros());
}

template <typename mat>
void rref_qr(
    const mat &A_in,
    mat &A,
    mat &P,
    std::vector<int> &jb,
    double tol)
{
    jb.clear();

    Eigen::SparseMatrix<double> R;
    spdlog::info("Factorizing {}x{} matrix", A_in.rows(), A_in.cols());
    int rank = qr(A_in, R, P);
    //Eigen::SparseQR<mat, Eigen::COLAMDOrdering<int>> spqr;
    //spdlog::info("Factorizing {}x{} matrix", A_in.rows(), A_in.cols());
    //spqr.analyzePattern(A_in);
    //spqr.factorize(A_in);
    spdlog::info("Factorizaiton complete");

    //R = spqr.matrixR();
    //auto rank = spqr.rank();
    spdlog::info("Matrix rank: {}", rank);

    // jb = 1:r
    jb.resize(rank);
    std::iota(jb.begin(), jb.end(), 0);
    Eigen::VectorXi jb_vec = Eigen::Map<Eigen::VectorXi>(jb.data(), jb.size());

    // size(R, 2)-r:size(R, 2)
    spdlog::info("Removing redundant constraints");
    //std::vector<int> col_indices(rank);
    //std::iota(col_indices.begin(), col_indices.end(), R.cols() - rank);
    //Eigen::VectorXi cols = Eigen::Map<Eigen::VectorXi>(col_indices.data(), col_indices.size());
    std::vector<int> col_indices(R.cols() - rank);
    std::iota(col_indices.begin(), col_indices.end(), rank);
    Eigen::VectorXi cols = Eigen::Map<Eigen::VectorXi>(col_indices.data(), col_indices.size());

    spdlog::info("Splitting upper triangular matrices");
    Eigen::SparseMatrix<double> R1;
    Eigen::SparseMatrix<double> R2;
    // R1 = R(1:r,1:r)
    igl::slice(R, jb_vec, jb_vec, R1);
    // R2 = R(1:r,size(R,2)-r:size(R,2) )
    igl::slice(R, jb_vec, cols, R2);

    // R1 \ R2
    spdlog::info("Computing dependent variables");
    //Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
    //solver.compute(R1);
    //if (solver.info() != Eigen::Success)
    //{
    //    std::cout << "Decomposition of R1 failed\n";
    //    exit(EXIT_FAILURE);
    //}
    //Eigen::SparseMatrix<double> R1_inv_R2 = solver.solve(R2);
    Eigen::SparseMatrix<double> R1_inv_R2;
    backslash(R1, R2, R1_inv_R2);

    // A = [ eye(r) R1 \ R2]
    spdlog::info("Building matrix for independent to full variables");
    Eigen::SparseMatrix<double> I(rank, rank);
    I.setIdentity();
    A.resize(rank, rank + R1_inv_R2.cols());
    A.reserve(I.nonZeros() + R1_inv_R2.nonZeros());

    for (int k = 0; k < I.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(I, k); it; ++it) {
            A.insert(it.row(), it.col()) = it.value();
        }
    }

    for (int k = 0; k < R1_inv_R2.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(R1_inv_R2, k); it; ++it) {
            A.insert(it.row(), it.col() + rank) = it.value();
        }
    }
    A.makeCompressed();

    spdlog::info("Row reduction complete");
}


void elim_constr(
    const Eigen::SparseMatrix<double> &C,
    Eigen::SparseMatrix<double> &T_out)
{
    spdlog::debug("eliminating constraints for {}x{} matrix", C.rows(), C.cols());
    int nvars = C.cols();
    std::vector<int> dep_list, indep_list;
    Eigen::SparseMatrix<double> R, P;
    //rref(C, R, dep_list);
    rref_qr(C, R, P, dep_list, 2e-12);
    spdlog::info("reduced matrix is {}x{}", R.rows(), R.cols());
    spdlog::info("Final row has norm {}", R.row(R.rows() - 1).norm());

    // matlab code: indep = setdiff(1:nvars, dep);
    int k = 0;
    for (int i = 0; i < nvars; i++)
    {
        if (k == dep_list.size() || i != dep_list[k])
        {
            indep_list.push_back(i);
        }
        else
        {
            k++;
        }
    }
    spdlog::info("{} independent and {} dependent constraints", indep_list.size(), dep_list.size());
    // std::cout << nvars << " dep: " << dep_list.size() << " indep: " << indep_list.size() << std::endl;
    Eigen::VectorXi dep, indep;
    igl::list_to_matrix(dep_list, dep);
    igl::list_to_matrix(indep_list, indep);
    Eigen::VectorXi all_rows = get_seq(0, R.rows() - 1);
    int num_dep = dep_list.size();
    // std::cout << "dep list:" << dep << std::endl;
    // std::cout << "indep list:" << indep << std::endl;

    // matlab code: T = [-R(:,indep); speye(size(indep,2),size(indep,2))];
    Eigen::SparseMatrix<double> R_ind;
    igl::slice(R, all_rows, indep, R_ind);
    Eigen::SparseMatrix<double> T(num_dep + indep_list.size(), indep_list.size());
    T.reserve(R_ind.nonZeros() + indep.size());
    for (Eigen::Index c = 0; c < T.cols(); ++c)
    {
        T.startVec(c);

        for (typename Eigen::SparseMatrix<double>::InnerIterator itR_ind(R_ind, c); itR_ind; ++itR_ind)
        {
            if (itR_ind.row() >= num_dep) continue;
            T.insertBack(itR_ind.row(), c) = -itR_ind.value();
        }
        T.insertBack(c + num_dep, c) = 1;
    }
    T.finalize();

    // std::cout << "T after adding spyeye:\n" << T << std::endl;

    // matlab code: T([dep indep],:) = T;
    Eigen::VectorXi dep_indep(dep.size() + indep.size());
    Eigen::VectorXi all_cols_T;
    all_cols_T = get_seq(0, T.cols()-1);
    dep_indep << dep, indep;
    Eigen::VectorXi dep_indep_rev(dep_indep.size());
    for (int i = 0; i < dep_indep.size(); i++)
    {
        dep_indep_rev[dep_indep[i]] = i;
    }
    // std::cout << "start slicing:" << std::endl;
    T_out.resize(T.rows(), T.cols());
    igl::slice(T, dep_indep_rev, 1, T_out);
    // slice_into_sparse(T_out, dep_indep, all_cols_T, T);
    // std::cout << "T output:\n" << T_out <<  std::endl;

    spdlog::info("test q2 on R: {}", (R * T_out * Eigen::VectorXd::Random(T_out.cols())).norm());
    T_out = P * T_out;
    spdlog::info("{} nonzeros", T_out.nonZeros());
}


void swap_two_rows(
    Eigen::SparseMatrix<double> &R,
    int i,
    int k
)
{
    // Eigen::SparseMatrix<double> tmp_i = R.row(i);
    // Eigen::SparseMatrix<double> tmp_k = R.row(k);
    // slice_into_mat(R, i, i, 0, R.cols()-1, tmp_k);
    // slice_into_mat(R, k, k, 0, R.cols()-1, tmp_i);

    Eigen::SparseMatrix<double, Eigen::RowMajor> R_rm(R);
    Eigen::SparseMatrix<double, Eigen::RowMajor> tmp_i = R_rm.row(i);
    Eigen::SparseMatrix<double, Eigen::RowMajor> tmp_k = R_rm.row(k);
    
    R_rm.row(i) = tmp_k;
    R_rm.row(k) = tmp_i;

    R = Eigen::SparseMatrix<double>(R_rm);
}

void elim_constr(
    const Eigen::SparseMatrix<double> &C,
    const Eigen::VectorXd &d,
    Eigen::SparseMatrix<double> &T_out,
    Eigen::VectorXd &b
)
{
    int nvars = C.cols();
    std::vector<int> dep_list, indep_list;
    Eigen::VectorXd rb;
    Eigen::SparseMatrix<double> R;
    Eigen::SparseMatrix<double> C_app = C;

    C_app.conservativeResize(C_app.rows(), C_app.cols() + 1);
    C_app.col(C_app.cols() - 1) = d.sparseView();
    rref(C_app, R, dep_list);

    rb = R.col(R.cols() - 1);
    R.conservativeResize(R.rows(), R.cols() - 1);
    // matlab code: indep = setdiff(1:nvars, dep);
    int k = 0;
    for (int i = 0; i < nvars; i++)
    {
        if (k == dep_list.size() || i != dep_list[k])
        {
            indep_list.push_back(i);
        }
        else
        {
            k++;
        }
    }
    // std::cout << nvars << " dep: " << dep_list.size() << " indep: " << indep_list.size() << std::endl;
    Eigen::VectorXi dep, indep;
    igl::list_to_matrix(dep_list, dep);
    igl::list_to_matrix(indep_list, indep);
    Eigen::VectorXi all_rows = get_seq(0, R.rows() - 1);

    // matlab code: b = [rb; zeros(size(indep,2),1)]; 
    rb.conservativeResize(nvars);
    for (int i = dep_list.size(); i < nvars; i++)
    {
        rb(i) = 0;
    }
    // matlab code: T = [-R(:,indep); speye(size(indep,2),size(indep,2))];
    Eigen::SparseMatrix<double> R_ind;
    igl::slice(R, all_rows, indep, R_ind);
    Eigen::SparseMatrix<double> T(R_ind.rows() + indep_list.size(), indep_list.size());
    T.reserve(R_ind.nonZeros() + indep.size());
    for (Eigen::Index c = 0; c < T.cols(); ++c)
    {
        T.startVec(c);

        for (typename Eigen::SparseMatrix<double>::InnerIterator itR_ind(R_ind, c); itR_ind; ++itR_ind)
        {
            T.insertBack(itR_ind.row(), c) = -itR_ind.value();
        }
        T.insertBack(c + R_ind.rows(), c) = 1;
    }
    T.finalize();

    // matlab code: T([dep indep],:) = T;
    Eigen::VectorXi dep_indep(dep.size() + indep.size());
    Eigen::VectorXi all_cols_T;
    all_cols_T = get_seq(0, T.cols()-1);
    dep_indep << dep, indep;
    Eigen::VectorXi dep_indep_rev(dep_indep.size());
    for (int i = 0; i < dep_indep.size(); i++)
    {
        dep_indep_rev[dep_indep[i]] = i;
    }
    // std::cout << "start slicing:" << std::endl;
    T_out.resize(T.rows(), T.cols());
    igl::slice(T, dep_indep_rev, 1, T_out);
    igl::slice(rb, dep_indep_rev, 1, b);
}

