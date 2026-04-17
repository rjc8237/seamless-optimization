#include <igl/predicates/predicates.h>
#include <igl/writeOBJ.h>
#include <algorithm>
#include <igl/grad.h>
#include "ExtremeOpt.h"
#include "energy.h"
#include "spdlog/spdlog.h"
#include <Eigen/CholmodSupport>
#include <Eigen/Core>
#include <umfpack.h>
#include "smooth_utils.h"
#include <unsupported/Eigen/KroneckerProduct>

namespace SymDir {

/*
class ExtremeOptSmoothVertexOperation : public SymDir::TriMeshOperationShim<
                                            ExtremeOpt,
                                            ExtremeOptSmoothVertexOperation,
                                            SymDir::TriMeshSmoothVertexOperation>
{
public:
    ExecuteReturnData execute(ExtremeOpt& m, const Tuple& t)
    {
        return SymDir::TriMeshSmoothVertexOperation::execute(m, t);
    }
    bool before(ExtremeOpt& m, const Tuple& t)
    {
        if (SymDir::TriMeshSmoothVertexOperation::before(m, t)) {
            return m.smooth_before(t);
        }
        return false;
    }
    bool after(ExtremeOpt& m, ExecuteReturnData& ret_data)
    {
        ret_data.success &= SymDir::TriMeshSmoothVertexOperation::after(m, ret_data);
        if (ret_data.success) {
            ret_data.success &= m.smooth_after(ret_data.tuple);
        }
        return ret_data;
    }
    bool invariants(ExtremeOpt& m, ExecuteReturnData& ret_data)
    {
        ret_data.success &= SymDir::TriMeshSmoothVertexOperation::invariants(m, ret_data);
        if (ret_data.success) {
            ret_data.success &= m.invariants(ret_data.new_tris);
        }
        return ret_data;
    }
};

template <typename Executor>
void addCustomOps(Executor& e)
{
    e.add_operation(std::make_shared<ExtremeOptSmoothVertexOperation>());
}

*/

Eigen::VectorXd UMFPACK_solve(const Eigen::SparseMatrix<double>& A_eigen,
                                 const Eigen::VectorXd& b_eigen, int& status)
{
    int n = static_cast<int>(A_eigen.rows());
    const int* Ap = A_eigen.outerIndexPtr();
    const int* Ai = A_eigen.innerIndexPtr();
    const double* Ax = A_eigen.valuePtr();

    void* Symbolic = nullptr;
    void* Numeric  = nullptr;

    status = umfpack_di_symbolic(n, n, Ap, Ai, Ax, &Symbolic, nullptr, nullptr);
    if (status != UMFPACK_OK)
    {
        spdlog::error("UMFPACK symbolic factorization failed");
        return Eigen::VectorXd::Zero(n);
    }

    status = umfpack_di_numeric(Ap, Ai, Ax, Symbolic, &Numeric, nullptr, nullptr);
    umfpack_di_free_symbolic(&Symbolic);
    if (status != UMFPACK_OK)
    {
        umfpack_di_free_numeric(&Numeric);
        spdlog::error("UMFPACK numeric factorization failed");
        return Eigen::VectorXd::Zero(n);
    }

    Eigen::VectorXd x_eigen(n);
    status = umfpack_di_solve(UMFPACK_A, Ap, Ai, Ax, 
                                    x_eigen.data(),
                                    b_eigen.data(), 
                                    Numeric,
                                    nullptr, 
                                    nullptr);

    umfpack_di_free_numeric(&Numeric);
    if (status != UMFPACK_OK)
    {
        spdlog::error("UMFPACK solve failed");
        return Eigen::VectorXd::Zero(n);
    }
    
    return x_eigen;
}


void buildkkt(
    Eigen::SparseMatrix<double>& hessian,
    Eigen::SparseMatrix<double>& Aeq,
    Eigen::SparseMatrix<double>& AeqT,
    Eigen::SparseMatrix<double>& kkt)
{
    kkt.reserve(hessian.nonZeros() + Aeq.nonZeros() + AeqT.nonZeros());
    for (Eigen::Index c = 0; c < kkt.cols(); ++c) {
        kkt.startVec(c);
        if (c < hessian.cols()) {
            for (typename Eigen::SparseMatrix<double>::InnerIterator ithessian(hessian, c);
                 ithessian;
                 ++ithessian)
                kkt.insertBack(ithessian.row(), c) = ithessian.value();
            for (typename Eigen::SparseMatrix<double>::InnerIterator itAeq(Aeq, c); itAeq; ++itAeq)
                kkt.insertBack(itAeq.row() + hessian.rows(), c) = itAeq.value();
        } else {
            for (typename Eigen::SparseMatrix<double>::InnerIterator itAeqT(
                     AeqT,
                     c - hessian.cols());
                 itAeqT;
                 ++itAeqT)
                kkt.insertBack(itAeqT.row(), c) = itAeqT.value();
        }
    }
    kkt.finalize();
}

int check_flip(const Eigen::MatrixXd& uv, const Eigen::MatrixXi& Fn)
{
    int fl = 0;
    for (int i = 0; i < Fn.rows(); i++) {
        Eigen::Matrix<double, 1, 2> a_db(uv(Fn(i, 0), 0), uv(Fn(i, 0), 1));
        Eigen::Matrix<double, 1, 2> b_db(uv(Fn(i, 1), 0), uv(Fn(i, 1), 1));
        Eigen::Matrix<double, 1, 2> c_db(uv(Fn(i, 2), 0), uv(Fn(i, 2), 1));
        if (igl::predicates::orient2d(a_db, b_db, c_db) != igl::predicates::Orientation::POSITIVE) {
            fl++;
        }
    }
    return fl;
}


Eigen::SparseMatrix<double> ExtremeOpt::compute_area_weight_matrix()
{
    int num_faces = area.size();
    Eigen::SparseMatrix<double> weights(3 * num_faces, 3 * num_faces);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(3 * num_faces);
    double total_area = area.sum();

    for (int f = 0; f < num_faces; ++f)
    {
        double area_f = area[f] / total_area;
        for (int i = 0; i < 3; ++i)
        {
            int j = f + (i * num_faces);
            triplets.emplace_back(j, j, area_f);
        }
    }

    weights.setFromTriplets(triplets.begin(), triplets.end());
    return weights;
}

double compute_degenerate_quartic_energy(
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXd& uv0,
    const Eigen::MatrixXi& F,
    const std::vector<bool>& is_degenerate_face
) {
    int num_faces = F.rows();
    double energy = 0.;
    for (int f = 0; f < num_faces; ++f)
    {
        if (!is_degenerate_face[f]) continue;
        for (int i = 0; i < 3; ++i)
        {
            int j = (i + 1) % 3;
            int vi = F(f, i);
            int vj = F(f, j);
            double dl = (uv.row(vi) - uv.row(vj)).squaredNorm();
            double dl0 = (uv0.row(vi) - uv0.row(vj)).squaredNorm();
            energy += 0.5 * ((dl - dl0) * (dl - dl0)) / (dl0);
        }
    }

    return energy;
}

double ExtremeOpt::compute_energy(const Eigen::MatrixXd& aaa, double Lp) {
    Eigen::MatrixXd Ji;
    SymDir::jacobian_from_uv(G, aaa, Ji);
    
    double energy = 0.;
    if (m_params.alignment_weight > 0.)
    {
        Eigen::SparseMatrix<double> weights = compute_area_weight_matrix();
        Eigen::MatrixXd Guv = Grad * aaa;

        Eigen::MatrixXd uT_vT(3*input_F.rows(), 2);

        uT_vT.col(0) = Eigen::Map<const Eigen::VectorXd>(PD1.data(), 3 * input_F.rows());
        uT_vT.col(1) = Eigen::Map<const Eigen::VectorXd>(PD2.data(), 3 * input_F.rows());

        Eigen::MatrixXd R = Guv - uT_vT;
        energy = (R.transpose() * (weights * R)).trace();
    }

    if (Lp == 0) {
        Lp = m_params.Lp;
    }
    return m_params.alignment_weight*energy + m_params.symdir_weight*SymDir::compute_energy_from_jacobian(Ji, area, Lp, m_params.soft_max, m_params.t, m_params.E_min);
    // return compute_worst_n_energy(Ji, area, m_params.Lp, m_params.percent, m_params.p_energy);
}

std::vector<double> ExtremeOpt::compute_worst_n_energy(const Eigen::MatrixXd& aaa) {
    Eigen::MatrixXd Ji;
    SymDir::jacobian_from_uv(G, aaa, Ji);
    return compute_worst_n_energy_from_jacobian(Ji, area, 1.0, m_params.percentages, m_params.soft_max, m_params.t, m_params.E_min);
}

double ExtremeOpt::compute_threshold_energy(const Eigen::MatrixXd& aaa) {
    Eigen::MatrixXd Ji;
    SymDir::jacobian_from_uv(G, aaa, Ji);
    return SymDir::compute_threshold_energy_from_jacobian(Ji, area, m_params.Lp, 5.0, m_params.soft_max, m_params.t);
}

double ExtremeOpt::get_energy_grad_and_hessian(const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXd& Guv,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hessian,
    bool get_hessian)
{
    Eigen::SparseMatrix<double> weights = compute_area_weight_matrix();
    double energy = m_params.symdir_weight * SymDir::get_grad_and_hessian(G, area, uv, grad, hessian, get_hessian, m_params.Lp, m_params.projected_newton, m_params.soft_max, m_params.t, m_params.E_min);
    grad *= m_params.symdir_weight;
    hessian *= m_params.symdir_weight;

    if (m_params.alignment_weight > 0.)
    {
        Eigen::MatrixXd uT_vT(3*F.rows(), 2);

        uT_vT.col(0) = Eigen::Map<const Eigen::VectorXd>(PD1.data(), 3 * F.rows());
        uT_vT.col(1) = Eigen::Map<const Eigen::VectorXd>(PD2.data(), 3 * F.rows());

        Eigen::MatrixXd R = Guv - uT_vT;
        double alignment_energy = m_params.alignment_weight * (R.transpose() * (weights * R)).trace();
        spdlog::info("sym Dirichlet energy: {}", energy);
        spdlog::info("alignment energy: {}", alignment_energy);
        energy += alignment_energy;
    
        Eigen::MatrixXd grad_E = m_params.alignment_weight * 2 * Grad.transpose() * (weights * R);
        Eigen::VectorXd grad_E_vec = Eigen::Map<const Eigen::VectorXd>(grad_E.data(), 2*uv.rows());
        grad += grad_E_vec;

        Eigen::SparseMatrix<double> gradSquared = 2 * Grad.transpose() * (weights * Grad);
        int n = gradSquared.rows();

        Eigen::SparseMatrix<double> hessian_E(2*n, 2*n);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(2 * gradSquared.nonZeros());

        for (int k = 0; k < gradSquared.outerSize(); ++k) {
            for (Eigen::SparseMatrix<double>::InnerIterator it(gradSquared, k); it; ++it) {
                // Top-left block
                triplets.emplace_back(it.row(), it.col(), it.value());
                // Bottom-right block
                triplets.emplace_back(it.row() + n, it.col() + n, it.value());
            }
        }

        hessian_E.setFromTriplets(triplets.begin(), triplets.end());
    
        hessian += m_params.alignment_weight*hessian_E;
    }

    /*
    std::cout << Grad.rows() << ", " << Grad.cols() << '\n';
    std::cout << grad.rows() << ", " << grad.cols() << '\n';
    std::cout << uv.rows() << ", " << uv.cols() << '\n';
    std::cout << Guv.rows() << ", " << Guv.cols() << '\n';
    std::cout << R.rows() << ", " << R.cols() << '\n';
    std::cout << grad_E.rows() << ", " << grad_E.cols() << '\n';
    std::cout << grad_E_vec.rows() << ", " << grad_E_vec.cols() << '\n';
    std::cout << hessian.rows() << ", " << hessian.cols() << '\n';
    */
    return energy;
}

double ExtremeOpt::get_hessian(Eigen::SparseMatrix<double>& hessian) {
    Eigen::MatrixXd uv;
    export_uv(uv);
    Eigen::MatrixXd Guv = Grad * uv;

    Eigen::VectorXd newton, initial_guess;
    // get grad and hessian
    Eigen::VectorXd grad;

    double energy_0 = get_energy_grad_and_hessian(input_V, input_F, uv, Guv, grad, hessian, true);
    return energy_0;
}

Eigen::VectorXd ExtremeOpt::misaligned_newton_direction(
    Eigen::MatrixXd& uv,
    double& energy_0,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hessian
)
{
    double misalignment_weight = 1.;

    // add augmentation term to lagrangian
    Eigen::VectorXd uv_flat = Eigen::Map<Eigen::VectorXd>(uv.data(), 2*V.rows());
    Eigen::SparseMatrix<double> aug_hessian = (Beq.transpose() * Beq);
    hessian = hessian + misalignment_weight * aug_hessian;
    grad = grad + misalignment_weight * aug_hessian * uv_flat;
    energy_0 = energy_0 + 0.5 * misalignment_weight * uv_flat.dot(aug_hessian * uv_flat);

    // Compute corrected descent direction
    double a = 0;
    Eigen::SparseMatrix<double> mat;
    Eigen::VectorXd rhs, newton;
    Eigen::SparseMatrix<double> cons;
    while (true)
    {
        if (a == 0)
        {
            //mat = hessian; // Use newton step
            mat = Q2T * hessian * Q2;
            cons = Beq * Q2;
        }
        else 
        {     
            // Create identity
            Eigen::SparseMatrix<double> id(hessian.rows(), hessian.rows());
            id.setIdentity();
            
            // Create matrix with correction
            mat = Q2T * ((hessian + a*id) * Q2);
            cons = (1 + a) * Beq * Q2;
            //mat = hessian + a*id; // use newton step
        }

        spdlog::trace("Building kkt with {}x{} constraints", Beq.rows(), Beq.cols());
        //Eigen::SparseMatrix<double> cons = Beq;
        spdlog::trace("Reduced constraint: {}x{}", cons.rows(), cons.cols());
        Eigen::SparseMatrix<double> consT = cons.transpose();
        Eigen::SparseMatrix<double> kkt(mat.rows() + cons.rows(), mat.cols() + cons.rows());
        buildkkt(mat, cons, consT, kkt);
        rhs.setZero(kkt.rows());
        rhs.topRows(Q2T.rows()) << Q2T* grad;
        //rhs.topRows(Q2T.rows()) << grad;

        spdlog::trace("Solving {}x{} kkt with {} rhs", kkt.rows(), kkt.cols(), rhs.size());
        //Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        //solver.analyzePattern(kkt);
        //solver.factorize(kkt);
        int status{};
        newton = -UMFPACK_solve(kkt, rhs, status);
        if (status != UMFPACK_OK) {
            std::cout << "cannot solve newton system" << std::endl;
            mat.setIdentity();
            buildkkt(mat, cons, consT, kkt);
            //solver.analyzePattern(kkt);
            //solver.factorize(kkt);
            newton = -UMFPACK_solve(kkt, rhs, status);
        }
        //newton = -solver.solve(rhs);

        spdlog::trace("Extracting unreduced newton direction");
        double newton_decr = newton.dot(rhs);
        std::cout << "gradient norm is " << grad.dot(grad) << std::endl;
        std::cout << "newton norm is " << newton.dot(newton) << std::endl;
        std::cout << "projected gradient is " << newton_decr << std::endl;
        if (status == UMFPACK_OK && newton_decr < 0)
        {
            break;
        }
        else if (a == 0)
        {
            a = 1; // We did not try the correction yet, start from arbitrary value 1
            spdlog::info(" Starting correction.");
        }
        else
        {
            a *= 2; // Correction was not enough, increase weight of id
        }
    }

    newton = Q2 * (newton.topRows(mat.rows()));

    return newton;
}

Eigen::VectorXd ExtremeOpt::kkt_newton_direction(
    Eigen::MatrixXd& uv,
    double& energy_0,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hessian
)
{
    double a = 0;
    Eigen::VectorXd newton;
    Eigen::SparseMatrix<double> mat;
    while (true)
    {
        if (a == 0)
        {
            mat = hessian;
        }
        else 
        {     
            // Create identity
            Eigen::SparseMatrix<double> id(hessian.rows(), hessian.rows());
            id.setIdentity();
            
            // Create matrix with correction
            mat = hessian + a*id;
        }

        // build kkt system
        Eigen::SparseMatrix<double> kkt(hessian.rows() + Aeq.rows(), hessian.cols() + Aeq.rows());
        buildkkt(mat, Aeq, AeqT, kkt);
        Eigen::VectorXd rhs(kkt.rows());
        rhs.setZero();
        rhs.topRows(grad.rows()) << -grad;

        // solve the system
        int status{};
        newton = UMFPACK_solve(kkt, rhs, status);

        double newton_decr = newton.topRows(grad.rows()).dot(grad);
        std::cout << "gradient norm is " << grad.dot(grad) << std::endl;
        std::cout << "newton norm is " << newton.dot(newton) << std::endl;
        std::cout << "projected gradient is " << newton_decr << std::endl;
        if (status == UMFPACK_OK && newton_decr < 0)
        {
            break;
        }
        else if (a == 0)
        {
            a = 1; // We did not try the correction yet, start from arbitrary value 1
            spdlog::info(" Starting correction.");
        }
        else
        {
            a *= 2; // Correction was not enough, increase weight of id
        }
    }

    return newton;
}

Eigen::VectorXd ExtremeOpt::gs_newton_direction(
    Eigen::MatrixXd& uv,
    double& energy_0,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hessian
)
{
    Eigen::VectorXd rhs = Q2T * grad;
    Eigen::VectorXd newton;

    // Compute corrected descent direction
    double a = 0;
    newton.setZero(uv.rows() * 2);  // initialize as zero vector with correct size
    while (true){
        Eigen::SparseMatrix<double> mat;
        if (a == 0)
        {
            // mat = hessian; // Use newton step
            mat = Q2T * hessian * Q2;
        }
        else 
        {     
            // Create identity
            Eigen::SparseMatrix<double> id(hessian.rows(), hessian.rows());
            id.setIdentity();
            
            // Create matrix with correction
            mat = Q2T * ((hessian + a*id) * Q2);
            // mat = (hessian + a*id);
        }
        mat.makeCompressed();

        bool result = solveGaussSeidel(mat, rhs, newton);
        newton = -newton;
        if (result){
            spdlog::info("Gauss-Seidel iteration converged");
            residual = (mat * newton + rhs).norm();
        }

        newton = Q2 * newton;

        double newton_decr = newton.dot(grad);
        std::cout << "gradient norm is " << grad.dot(grad) << std::endl;
        std::cout << "newton norm is " << newton.dot(newton) << std::endl;
        std::cout << "projected gradient is " << newton_decr << std::endl;
        if (result && newton_decr < 0)
        {
            break;
        }
        else if (a == 0)
        {
            a = 1; // We did not try the correction yet, start from arbitrary value 1
            spdlog::info("Starting correction.");
        }
        else
        {
            a *= 2; // Correction was not enough, increase weight of id
            spdlog::info("Correction {}.", a);
        }
    }
    correction = a;

    return newton;
}

    Eigen::VectorXd conjugate_gradient(
        const Eigen::SparseMatrix<double>& A,
        const Eigen::VectorXd& b,
        const Eigen::VectorXd& x0,
        int max_iter,
        double rel_thres,
        double abs_thres)
    {
        // initialize with guess
        Eigen::VectorXd x = x0;
        spdlog::info("initial guess norm is {}", x.norm());

        // build CG solver
        int batch_size = 1000;
        Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower|Eigen::Upper> cg;
        cg.setMaxIterations(batch_size);          // advance a few iterations at a time
        cg.setTolerance(rel_thres);
        cg.compute(A);
        int num_batches = max_iter / batch_size;

		for (int i = 0; i < num_batches; ++i)
		{
			x = cg.solveWithGuess(b, x); // one CG iteration from current x

            // check for convergence
			double abs_res = (A * x - b).norm();
            spdlog::info("residual at batch {} is {}", i, abs_res);
			if ((abs_res < abs_thres) || (abs_res < rel_thres * b.norm()))
			{
                break;
			}
		}

        return x;
    }

Eigen::VectorXd ExtremeOpt::reduced_newton_direction(
    Eigen::MatrixXd& uv,
    double& energy_0,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hessian
)
{
    Eigen::VectorXd rhs = Q2T * grad;
    // Eigen::VectorXd rhs = grad;
    Eigen::VectorXd newton;
    // Compute corrected descent direction
    double a = 0;
    spdlog::debug("{}x{} constraint matrix", Aeq.rows(), Aeq.cols());
    spdlog::debug("{}x{} reduced matrix", Q2.rows(), Q2.cols());
    while (true){
        Eigen::SparseMatrix<double> mat;
        newton.setZero(rhs.size());  // reduced space size

        if (a == 0)
        {
            // mat = hessian; // Use newton step
            mat = Q2T * hessian * Q2;
        }
        else 
        {     
            // Create identity
            Eigen::SparseMatrix<double> id(hessian.rows(), hessian.rows());
            id.setIdentity();
            
            // Create matrix with correction
            mat = Q2T * ((hessian + a*id) * Q2);
            // mat = mat + a * id;
            // mat = (hessian + a*id);
        }
        mat.makeCompressed();

        Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> perm;
        if (m_params.degenerate_vertices_preconditioner) {
            std::vector<bool> is_degenerate_uv = mark_degenerate_vertices(V, F, uv, v_map, m_params.precond_dim, m_params.triangle_threshold);
            int N = is_degenerate_uv.size();
            // mark variables corresponding to degenerate uv as degenerate
            std::vector<bool> is_degenerate_var(rhs.size(), false);
            is_degenerate_var = is_degenerate_uv;
            // for (int k = 0; k < Q2.outerSize(); ++k) {
            //     for (Eigen::SparseMatrix<double>::InnerIterator it(Q2, k); it; ++it) {
            //         if (is_degenerate_uv[it.row() % N]) {
            //             is_degenerate_var[it.col()] = true;
            //         }
            //     }
            // }

            int gg_size = std::count(is_degenerate_var.begin(), is_degenerate_var.end(), false);
            spdlog::info("Preconditioning with {:.2f} degenerate variables", (double)(is_degenerate_var.size() - gg_size) / (double)is_degenerate_var.size() * 100);
            Eigen::VectorXi perm_indices(is_degenerate_var.size() * 2);
            int curr = 0;
            // Fill G indices first
            for(int v_i = 0; v_i < is_degenerate_var.size(); v_i++) {
                if (!is_degenerate_var[v_i]) {
                    // perm_indices[curr++] = v_i;
                    perm_indices[curr++] = 2 * v_i;
                    perm_indices[curr++] = 2 * v_i + 1;
                }
            }
            // Fill S indices second
            for(int v_i = 0; v_i < is_degenerate_var.size(); v_i++) {
                if (is_degenerate_var[v_i]) {
                    // perm_indices[curr++] = v_i;
                    perm_indices[curr++] = 2 * v_i;
                    perm_indices[curr++] = 2 * v_i + 1;

                }
            }

            perm.resize(perm_indices.size());
            perm.indices() = perm_indices;

            mat = perm.transpose() * mat * perm;
            rhs = perm.transpose() * rhs;
        }
        Solver solver(mat, m_params.solver_type, m_params.cg_rel_err, m_params.cg_iters);
        CgResult result;

        if (m_params.solver_type == "Parallel_CG") {
            result = conjugate_gradient(mat, rhs, newton, m_params.cg_iters, m_params.cg_rel_err);
            newton = -newton;
        } else if (m_params.solver_type == "Guess_CG") {
            if (prev_dir.size() != 0)
            {
                newton = prev_dir;
            }
            newton = conjugate_gradient(mat, rhs, newton, m_params.cg_iters, m_params.cg_rel_err, 0.1 * m_params.cg_rel_err);
            newton = -newton;
            prev_dir = newton;
        } else {
            solver.compute(mat);
            newton = -solver.solve(rhs);
            spdlog::info("rhs norm is {}", rhs.norm());
            spdlog::info("residual of reduced system is {}", (mat * newton + rhs).norm());
            if (m_params.degenerate_vertices_preconditioner) {
                // unpermute the solution
                Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> perm_inv = perm.inverse();
                newton = perm_inv.transpose() * newton;
            }
        }
        residual = (mat * newton + rhs).norm();                    

        bool freeze_degenerate = false;
        if (freeze_degenerate)
        {
            std::vector<bool> is_degenerate_uv = mark_degenerate_vertices(V, F, uv, v_map, m_params.precond_dim, m_params.triangle_threshold);

            // mark 
            int N = is_degenerate_uv.size();
            std::vector<bool> is_degenerate_var(newton.size(), false);
            for (int k = 0; k < Q2.outerSize(); ++k) {
                for (Eigen::SparseMatrix<double>::InnerIterator it(Q2, k); it; ++it) {
                    if (is_degenerate_uv[it.row() % N]) is_degenerate_var[it.col()] = true;
                }
            }

            int num_degen = 0;
            for (int vi = 0; vi < is_degenerate_var.size(); ++vi)
            {
                if (is_degenerate_var[vi])
                {
                    newton[vi] = 0.;
                    ++num_degen;
                }
            }
            spdlog::info("{}/{} variables fixed", num_degen, is_degenerate_var.size());
        }

        // int status{};
        // newton = -UMFPACK_solve(mat, rhs, status);
        newton = Q2 * newton;

        double newton_decr = newton.dot(grad);
        
        std::cout << "gradient norm is " << grad.dot(grad) << std::endl;
        std::cout << "newton norm is " << newton.dot(newton) << std::endl;
        std::cout << "projected gradient is " << newton_decr << std::endl;
        spdlog::info("residual of full system is {}", (hessian * newton + grad).norm());
        // if (status == UMFPACK_OK && newton_decr < 0)
        // {
        //     break;
        // }

        if (m_params.solver_type == "Parallel_CG") {
            if (result.converged && newton_decr < 0)
            {
                iter_solver = result.iterations;
                std::cout << "CG converged in " << result.iterations << " iterations with rel res " << result.rel_residual << std::endl;
                break;
            }
        }
        if (m_params.solver_type == "Guess_CG" && newton_decr < 0) {
            break;
        }

        // for CG, solver failure may just be iteration count
        if (m_params.solver_type == "CG" && newton_decr < 0) {
            iter_solver = solver.iterations();
            spdlog::info("{} iterations of CG", iter_solver);
            break;
        }

        if (solver.info() == Eigen::Success && newton_decr < 0)
        {
            // cond_num = get_cond_num_from_hessian(hessian);
            if (m_params.solver_type == "CG" || m_params.solver_type == "CG_LLT" || m_params.solver_type == "CG_GS") {
                iter_solver = solver.iterations();
            }
            break;
        }
        else if (a == 0)
        {
            a = grad.norm(); // We did not try the correction yet, start from gradient norm
            spdlog::info("Starting correction.");
        }
        else
        {
            a *= 2; // Correction was not enough, increase weight of id
        }

    }                
    correction = a;

    return newton;
}

Eigen::VectorXd compute_lbfgs_direction(
    const std::deque<Eigen::VectorXd>& variables,
    const std::deque<Eigen::VectorXd>& gradients,
    const Eigen::VectorXd& gradient)
{
    int m = variables.size() - 1;
    std::deque<Eigen::VectorXd> delta_variables;
    std::deque<Eigen::VectorXd> delta_gradients;
    for (int i = 0; i < m; ++i) {
        delta_variables.push_back(variables[i + 1] - variables[i]);
        delta_gradients.push_back(gradients[i + 1] - gradients[i]);
    }

    // Initialize descent direction with a forward pass
    Eigen::VectorXd q = gradient;
    std::vector<double> rho(m);
    std::vector<double> alpha(m);
    for (int i = 0; i < m; ++i) {
        Eigen::VectorXd dx = delta_variables[i];
        Eigen::VectorXd dg = delta_gradients[i];
        rho[i] = 1.0 / dx.dot(dg);
        alpha[i] = rho[i] * dx.dot(q);
        q -= alpha[i] * dg;
    }

    // Scale descent direction
    double gamma = delta_variables[0].dot(delta_gradients[0]) / delta_gradients[0].squaredNorm();
    Eigen::VectorXd z = gamma * q;

    // Finish computation of descent direction with a back pass
    std::vector<double> beta(m);
    for (int i = m - 1; i >= 0; --i) {
        beta[i] = rho[i] * delta_gradients[i].dot(z);
        z += (alpha[i] - beta[i]) * delta_variables[i];
    }

    return -z;
}

double ExtremeOpt::smooth_global(bool& failed, std::vector<HessianStats>& hessian_log)
{
    Eigen::MatrixXd uv;
    export_uv(uv);
    Eigen::MatrixXd Guv = Grad * uv;

    Eigen::VectorXd newton, initial_guess;
    // get grad and hessian
    Eigen::VectorXd grad;
    Eigen::SparseMatrix<double> hessian;

    igl::Timer timer;
    double time_grad_hessian = 0.0;
    timer.start();
    double energy_0 = get_energy_grad_and_hessian(input_V, input_F, uv, Guv, grad, hessian, m_params.do_newton);
    time_grad_hessian = timer.getElapsedTime();

    bool use_rref = m_params.use_rref;
    double misalignment_weight = 1.;

    double cond_num = 0;
    double residual = 0;

    double time_solver = timer.getElapsedTime();
    double newton_decr = 0;
    double grad_max = 0.;
    double grad_norm = 0.;

    if (m_params.solver_type == "LBFGS")
    {
        Eigen::VectorXd uv_flat = Eigen::Map<Eigen::VectorXd>(uv.data(), 2*V.rows());
        Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
        solver.compute(Q2T * Q2);
        Eigen::VectorXd x = solver.solve(Q2T * uv_flat);
        variables.push_front(x);
        gradients.push_front(Q2T * grad);
        if (variables.size() > 20) variables.pop_back();
        if (gradients.size() > 20) gradients.pop_back();
    }

    std::vector<bool> is_degenerate_face = mark_degenerate_faces(input_V, input_F, uv, v_map, m_params.precond_dim, m_params.triangle_threshold);
    if (m_params.degenerate_weight > 0.)
    {
        std::vector<Eigen::Triplet<double>> length_trips;

        int num_faces = F.rows();
        int N = uv.rows();
        int count = 0;
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(num_faces * 6);
        for (int f = 0; f < num_faces; ++f)
        {
            if (!is_degenerate_face[f]) continue;
            for (int i = 0; i < 3; ++i)
            {
                int j = (i + 1) % 3;
                int vi = F(f, i);
                int vj = F(f, j);

                length_trips.emplace_back(2 * count, vi, 1.);
                length_trips.emplace_back(2 * count, vj, -1.);
                length_trips.emplace_back(2 * count + 1, N + vi, 1.);
                length_trips.emplace_back(2 * count + 1, N + vj, -1.);
                rhs[2 * count] = uv(vi, 0) - uv(vj, 0);
                rhs[2 * count + 1] = uv(vi, 1) - uv(vj, 1);
                ++count;
            }
        }
        spdlog::info("{}/{} edges fixed", count, 3 * num_faces);
        Eigen::SparseMatrix<double> edge_lengths(6 * num_faces, 2 * N);
        edge_lengths.setFromTriplets(length_trips.begin(), length_trips.end());
        spdlog::info("degenerate weight: {}", m_params.degenerate_weight);

        bool use_quartic = true;
        if (use_quartic)
        {
            std::vector<Eigen::Triplet<double>> hess_trips;
            Eigen::VectorXd uv_flat = Eigen::Map<Eigen::VectorXd>(uv.data(), 2*V.rows());
            Eigen::VectorXd lengths = edge_lengths * uv_flat;
            for (int i = 0; i < 3 * num_faces; ++i)
            {
                double l1 = lengths[2 * i];
                double l2 = lengths[2 * i + 1];
                hess_trips.emplace_back(2 * i, 2 * i, 8 * l1 * l1);
                hess_trips.emplace_back(2 * i, 2 * i + 1, 8 * l1 * l2);
                hess_trips.emplace_back(2 * i + 1, 2 * i, 8 * l1 * l2);
                hess_trips.emplace_back(2 * i + 1, 2 * i + 1, 8 * l2 * l2);
            }
            Eigen::SparseMatrix<double> length_hess(6 * num_faces, 6 * num_faces);
            length_hess.setFromTriplets(hess_trips.begin(), hess_trips.end());
            Eigen::SparseMatrix<double> degenerate_hessian = edge_lengths.transpose() * (length_hess * edge_lengths);
            hessian = hessian + m_params.degenerate_weight * degenerate_hessian;
        }
        else
        {
            hessian = hessian + m_params.degenerate_weight * (edge_lengths.transpose() * edge_lengths);
            grad = grad + m_params.degenerate_weight * (edge_lengths.transpose() * rhs);
        }
    }
    

    // find newton direction
    if (ME.rows() > 0) {
        spdlog::info("Fixing misalignment");
        newton = misaligned_newton_direction(uv, energy_0, grad, hessian);
    } else if (!use_rref) {
        newton = kkt_newton_direction(uv, energy_0, grad, hessian);
    } else if (m_params.solver_type == "GS") {
        newton = gs_newton_direction(uv, energy_0, grad, hessian);
    } else if (m_params.solver_type == "LBFGS") {
        // Check if the previous gradient and descent direction are trivial
        if (variables.size() < 2) {
            spdlog::info("using gradient for initial direction");
            newton = -Q2 * (Q2T * grad);
        } else {
            //newton = Q2 * (Q2T * (compute_lbfgs_direction(variables, gradients, grad)));
            newton = Q2 * compute_lbfgs_direction(variables, gradients, Q2T * grad);
        }
        if  (newton.dot(grad) >= 0)
        {
            newton = -Q2 * (Q2T * grad);
            gradients.clear();
            variables.clear();
        }
        grad_norm = (Q2T * grad).norm();
        grad_max = (Q2T * grad).cwiseAbs().maxCoeff();
    } else if (m_params.solver_type == "gradient") {
        newton = -Q2 * (Q2T * grad);
        grad_norm = (Q2T * grad).norm();
        grad_max = (Q2T * grad).cwiseAbs().maxCoeff();
    } else {
        newton = reduced_newton_direction(uv, energy_0, grad, hessian);
        residual = (Q2T * (hessian * newton + grad)).norm();                    
        grad_norm = (Q2T * grad).norm();
        grad_max = (Q2T * grad).cwiseAbs().maxCoeff();
    }
    //int iter = static_cast<int>(hessian_log.size());
    //write_sparse_matrix(hessian, "hessian_" + std::to_string(iter) + ".mat", "matlab");

    time_solver = timer.getElapsedTimeInSec() - time_solver;
    double time_ls = 0;
    timer.start();
    // do lineserach
    newton_decr = newton.dot(grad);
    Eigen::MatrixXd search_dir = Eigen::Map<Eigen::MatrixXd>(newton.data(), V.rows(), 2);

        constexpr double armijo_c1 = 1e-4;
    constexpr double wolfe_c2 = 0.9;
    constexpr double ls_backtrack = 0.8;

    auto evaluate_linesearch_objective =
        [&](const Eigen::MatrixXd& sample_uv,
            double& sample_energy,
            Eigen::VectorXd& sample_grad) {
            Eigen::MatrixXd sample_Guv = Grad * sample_uv;
            Eigen::SparseMatrix<double> sample_hessian;
            sample_energy = get_energy_grad_and_hessian(
                input_V,
                input_F,
                sample_uv,
                sample_Guv,
                sample_grad,
                sample_hessian,
                false);
        };

    auto new_x = uv;
    double ls_step_size = 1.0;
    bool ls_good = false;
    std::vector<double> E_worst_0;
    if (m_params.use_worst_n_energy_in_ls) {
        E_worst_0 = compute_worst_n_energy(uv);
    }
    int count_e = 0, count_f = 0;
    for (int i = 0; i < m_params.ls_iters; i++) {
        new_x = uv + ls_step_size * search_dir;
        double new_E = compute_energy(new_x);
        if (ME.rows() > 0) 
        {
            Eigen::VectorXd uv_flat = Eigen::Map<Eigen::VectorXd>(uv.data(), 2*V.rows());
            double misalignment_energy = 0.5 * (Beq * uv_flat).squaredNorm();
            new_E += misalignment_weight * misalignment_energy;
        }

        // add degenerate weight term
        if (m_params.degenerate_weight > 0)
        {
            double degenerate_energy = compute_degenerate_quartic_energy(new_x, uv, F, is_degenerate_face);
            new_E += m_params.degenerate_weight * degenerate_energy;
            spdlog::info("degenerate energy is {}", degenerate_energy);
        }

        // check armijo conditions
        double armijo_bound = energy_0 + armijo_c1 * ls_step_size * newton_decr;
        bool armijo_ok = std::isfinite(new_E) && (new_E <= armijo_bound);

        // check flip conditions
        bool flip_ok = (check_flip(new_x, F) == 0);

        // check wolfe conditions
        //Eigen::VectorXd new_grad;
        //evaluate_linesearch_objective(new_x, new_E, new_grad);
        //double wolfe_decr = new_grad.dot(newton);
        //bool wolfe_ok = new_grad.allFinite() && (wolfe_decr >= wolfe_c2 * newton_decr);
        bool wolfe_ok = true;

        //if (new_E < energy_0 && check_flip(new_x, F) == 0) {
        if (armijo_ok && wolfe_ok && flip_ok) {
            std::cout << "energy from " << energy_0 << " to " << new_E << std::endl;
            if (m_params.use_worst_n_energy_in_ls) {
                std::vector<double> new_E_worst = compute_worst_n_energy(new_x);
                if (new_E_worst[new_E_worst.size() - 1] < E_worst_0[E_worst_0.size() - 1]) {
                    std::cout << "E_worst_2 from " << E_worst_0[E_worst_0.size() - 1] << " to " << new_E_worst[new_E_worst.size() - 1] << std::endl;
                    ls_good = true;
                    break;
                }
                std::cout << "E_worst_2 did not improve: " << new_E_worst[new_E_worst.size() - 1] << std::endl;
            } else {
                ls_good = true;
                break;
            }
        } else {
            if (new_E >= energy_0) {
                count_e++;
            } 
            if (check_flip(new_x, F) > 0) {
                count_f++;
            }
        }
        ls_step_size *= 0.8;
    }
    spdlog::info("Linesearch failures: energy {}, flip {}", count_e, count_f);
    if (ME.rows() > 0) 
    {
        Eigen::VectorXd uv_flat = Eigen::Map<Eigen::VectorXd>(uv.data(), 2*V.rows());
        double misalignment_energy = 0.5 * (Beq * uv_flat).squaredNorm();
        spdlog::info("Misalignment error is {}", misalignment_energy);
    }

    if (ls_good) {
        // update vertex_attrs
        std::cout << "ls_step_size = " << ls_step_size << std::endl;
        for (int i = 0; i < new_x.rows(); i++) {
            vertex_attrs[i].pos = new_x.row(i);
        }
    } else {
        std::cout << "smooth failed" << std::endl;
        failed = true;
    }
    time_ls = timer.getElapsedTimeInSec();
    // Store in log

    hessian_log.push_back({
        static_cast<int>(hessian_log.size()),
        cond_num,
        residual,
        correction,
        time_solver,
        time_ls,
        time_grad_hessian,
        iter_solver,
        ls_step_size,
        -newton_decr,
        grad_norm,
    });

    return grad_max;
}

} // namespace SymDir
