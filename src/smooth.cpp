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

double ExtremeOpt::compute_worst_n_energy(const Eigen::MatrixXd& aaa, double Lp) {
    Eigen::MatrixXd Ji;
    SymDir::jacobian_from_uv(G, aaa, Ji);
    if (Lp == 0) {
        Lp = m_params.Lp;
    }
    return compute_worst_n_energy_from_jacobian(Ji, area, Lp, m_params.percent, m_params.soft_max, m_params.t, m_params.E_min);
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


double ExtremeOpt::smooth_global(bool& failed, std::vector<HessianStats>& hessian_log)
{
    Eigen::MatrixXd uv;
    export_uv(uv);
    Eigen::MatrixXd Guv = Grad * uv;

    Eigen::VectorXd newton, initial_guess;
    // get grad and hessian
    Eigen::SparseMatrix<double> hessian;
    Eigen::VectorXd grad;

    igl::Timer timer;
    double time_grad_hessian = 0.0;
    timer.start();
    double energy_0 = get_energy_grad_and_hessian(input_V, input_F, uv, Guv, grad, hessian, m_params.do_newton);
    time_grad_hessian = timer.getElapsedTime();

    double misalignment_weight = 1.;

    bool use_rref = m_params.use_rref;

    double cond_num = 0;
    double residual = 0;

    double time_solver = timer.getElapsedTime();
    int iter_solver = 0;
    double correction = 0;
    double newton_decr = 0;

    if (ME.rows() > 0) {
        spdlog::info("Fixing misalignment");

        // add augmentation term to lagrangian
        Eigen::VectorXd uv_flat = Eigen::Map<Eigen::VectorXd>(uv.data(), 2*V.rows());
        Eigen::SparseMatrix<double> aug_hessian = (Beq.transpose() * Beq);
        hessian = hessian + misalignment_weight * aug_hessian;
        grad = grad + misalignment_weight * aug_hessian * uv_flat;
        energy_0 = energy_0 + 0.5 * misalignment_weight * uv_flat.dot(aug_hessian * uv_flat);

        // Compute corrected descent direction
        double a = 0;
        Eigen::SparseMatrix<double> mat;
        Eigen::VectorXd rhs;
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
            newton_decr = newton.dot(rhs);
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
        grad = rhs.topRows(mat.rows());
    } else {
        if (!use_rref) {
            double a = 0;
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
                // Solver solver(mat, m_params.solver_type, m_params.cg_rel_err);
                // solver.analyzePattern(kkt);
                // solver.factorize(kkt);
                // newton = solver.solve(rhs);
                int status{};
                newton = UMFPACK_solve(kkt, rhs, status);
                // if (solver.info() != Eigen::Success) {
                //    std::cout << "cannot solve newton system" << std::endl;
                //    hessian.setIdentity();
                //    buildkkt(hessian, Aeq, AeqT, kkt);
                //    solver.analyzePattern(kkt);
                //    solver.factorize(kkt);
                //    newton = solver.solve(rhs);
                // }

                newton_decr = newton.topRows(grad.rows()).dot(grad);
                std::cout << "gradient norm is " << grad.dot(grad) << std::endl;
                std::cout << "newton norm is " << newton.dot(newton) << std::endl;
                std::cout << "projected gradient is " << newton_decr << std::endl;
                if (status == UMFPACK_OK && newton_decr < 0)
                {
                    break;
                }
                // if (solver.info() == Eigen::Success && newton_decr < 0)
                // {
                //     break;
                // }
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
        } else {
            spdlog::debug("{}x{} constraint matrix", Aeq.rows(), Aeq.cols());
            spdlog::debug("{}x{} reduced matrix", Q2.rows(), Q2.cols());
            // std::cout << "test q2:" << (Aeq * Q2 * Eigen::VectorXd::Random(Q2.cols())).norm()
            //         << std::endl;
            // hessian = Q2T * hessian * Q2;
            Eigen::VectorXd rhs = Q2T * grad;
            initial_guess.setZero(rhs.size());  // initialize as zero vector with correct size

            // Compute corrected descent direction
            double a = 0;
            // if (hessian_log.size() > 0) {
            //     a = hessian_log.back().correction;
            // }
            if (m_params.solver_type == "GS") {
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

                    newton_decr = newton.dot(grad);
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
            } else {
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
                        // mat = mat + a * id;
                        // mat = (hessian + a*id);
                    }
                    mat.makeCompressed();
                    // Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;

                    Solver solver(mat, m_params.solver_type, m_params.cg_rel_err);
                    CgResult result;
                    if (m_params.solver_type == "Parallel_CG") {
                        result = conjugate_gradient(mat, rhs, newton, 10000, m_params.cg_rel_err);
                        newton = -newton;
                    } else {
                        solver.compute(mat);
                        newton = -solver.solve(rhs);
                    }

                    residual = (mat * newton + rhs).norm();                    

                    // int status{};
                    // newton = -UMFPACK_solve(mat, rhs, status);
                    newton = Q2 * newton;

                    newton_decr = newton.dot(grad);
                    
                    std::cout << "gradient norm is " << grad.dot(grad) << std::endl;
                    std::cout << "newton norm is " << newton.dot(newton) << std::endl;
                    std::cout << "projected gradient is " << newton_decr << std::endl;
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
                        a = 1; // We did not try the correction yet, start from arbitrary value 1
                        spdlog::info("Starting correction.");
                    }
                    else
                    {
                        a *= 2; // Correction was not enough, increase weight of id
                    }
                }                
            }
            correction = a;
            grad = rhs;
        }
    }
    time_solver = timer.getElapsedTimeInSec() - time_solver;
    double time_ls = 0;
    timer.start();
    // do lineserach
    Eigen::MatrixXd search_dir = Eigen::Map<Eigen::MatrixXd>(newton.data(), V.rows(), 2);
    auto new_x = uv;
    double ls_step_size = 1.0;
    bool ls_good = false;
    double E_worst_0 = compute_worst_n_energy(uv);
    for (int i = 0; i < m_params.ls_iters; i++) {
        new_x = uv + ls_step_size * search_dir;
        double new_E = compute_energy(new_x);
        if (ME.rows() > 0) 
        {
            Eigen::VectorXd uv_flat = Eigen::Map<Eigen::VectorXd>(uv.data(), 2*V.rows());
            double misalignment_energy = 0.5 * (Beq * uv_flat).squaredNorm();
            new_E += misalignment_weight * misalignment_energy;
        }
        if (new_E < energy_0 && check_flip(new_x, F) == 0) {
            std::cout << "energy from " << energy_0 << " to " << new_E << std::endl;
            // ls_good = true;
            // break;
            double new_E_worst = compute_worst_n_energy(new_x);
            if (new_E_worst < E_worst_0) {
                std::cout << "E_worst_2 from " << E_worst_0 << " to " << new_E_worst << std::endl;
                ls_good = true;
                break;
            }
            std::cout << "E_worst_2 did not improve: " << new_E_worst << std::endl;
        }
        ls_step_size *= 0.8;
    }

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
        fabs(newton_decr),
    });

    double grad_norm;
    grad_norm = grad.cwiseAbs().maxCoeff();
    return grad_norm;
}

/*
void ExtremeOpt::smooth_all_vertices()
{
    auto collect_all_ops = std::vector<std::pair<std::string, Tuple>>();
    for (auto& loc : get_vertices()) {
        collect_all_ops.emplace_back("vertex_smooth", loc);
    }

    auto executor = SymDir::ExecutePass<ExtremeOpt, SymDir::ExecutionPolicy::kSeq>();
    addCustomOps(executor);
    executor(*this, collect_all_ops);
}

for (int k =0; k < Beq.outerSize(); ++k)
{
    for (Eigen::SparseMatrix<double>::InnerIterator it (Beq, k); it; ++it)
    {
        spdlog::info("{}, {}: {}", it.row(), it.col(), it.value());
    }
}
*/

} // namespace SymDir
