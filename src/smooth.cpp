#include <igl/predicates/predicates.h>
#include <igl/writeOBJ.h>
#include <algorithm>
#include <igl/grad.h>
#include "ExtremeOpt.h"
#include "energy.h"
#include "spdlog/spdlog.h"

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

double ExtremeOpt::compute_energy(const Eigen::MatrixXd& aaa, double lambda) {
    Eigen::MatrixXd Ji;
    SymDir::jacobian_from_uv(G, aaa, Ji);

    Eigen::MatrixXd Guv = Grad * aaa;

    Eigen::MatrixXd uT_vT(3*F.rows(), 2);

    uT_vT.col(0) = Eigen::Map<const Eigen::VectorXd>(PD1.data(), 3 * F.rows());
    uT_vT.col(1) = Eigen::Map<const Eigen::VectorXd>(PD2.data(), 3 * F.rows());

    Eigen::MatrixXd R = Guv - uT_vT;
    double energy = R.array().square().sum();

    return energy + lambda*SymDir::compute_energy_from_jacobian(Ji, area);
}

double ExtremeOpt::get_energy_grad_and_hessian(const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXd& Guv,
    Eigen::VectorXd& grad,
    Eigen::SparseMatrix<double>& hessian,
    double lambda,
    bool get_hessian)
{
    double energy = lambda * SymDir::get_grad_and_hessian(G, area, uv, grad, hessian, get_hessian);

    Eigen::MatrixXd uT_vT(3*F.rows(), 2);

    uT_vT.col(0) = Eigen::Map<const Eigen::VectorXd>(PD1.data(), 3 * F.rows());
    uT_vT.col(1) = Eigen::Map<const Eigen::VectorXd>(PD2.data(), 3 * F.rows());

    Eigen::MatrixXd R = Guv - uT_vT;
    energy += R.array().square().sum();
    
    Eigen::MatrixXd grad_E = 2 * Grad.transpose() * R;
    Eigen::VectorXd grad_E_vec = Eigen::Map<const Eigen::VectorXd>(grad_E.data(), 2*uv.rows());
    grad = lambda*grad + grad_E_vec;

    Eigen::SparseMatrix<double> gradSquared = 2 * Grad.transpose() * Grad;
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
    
    hessian = lambda*hessian + hessian_E;

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

double ExtremeOpt::smooth_global()
{
    Eigen::MatrixXd uv;
    export_uv(uv);
    Eigen::MatrixXd Guv = Grad * uv;

    Eigen::VectorXd newton;
    // get grad and hessian
    Eigen::SparseMatrix<double> hessian;
    Eigen::VectorXd grad;
    double lambda = 0.001;
    double energy_0 = get_energy_grad_and_hessian(input_V, input_F, uv, Guv, grad, hessian, lambda, m_params.do_newton);

    bool use_rref = true;
    if (!use_rref) {
        // build kkt system
        Eigen::SparseMatrix<double> kkt(hessian.rows() + Aeq.rows(), hessian.cols() + Aeq.rows());
        buildkkt(hessian, Aeq, AeqT, kkt);
        Eigen::VectorXd rhs(kkt.rows());
        rhs.setZero();
        rhs.topRows(grad.rows()) << -grad;
        // solve the system
        Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
        solver.analyzePattern(kkt);
        solver.factorize(kkt);
        newton = solver.solve(rhs);
        if (solver.info() != Eigen::Success) {
            std::cout << "cannot solve newton system" << std::endl;
            hessian.setIdentity();
            buildkkt(hessian, Aeq, AeqT, kkt);
            solver.analyzePattern(kkt);
            solver.factorize(kkt);
            newton = solver.solve(rhs);
        }
    } else {
        spdlog::debug("{}x{} constraint matrix", Aeq.rows(), Aeq.cols());
        spdlog::debug("{}x{} reduced matrix", Q2.rows(), Q2.cols());
        std::cout << "test q2:" << (Aeq * Q2 * Eigen::VectorXd::Random(Q2.cols())).norm()
                  << std::endl;
        //hessian = Q2T * hessian * Q2;
        Eigen::VectorXd rhs = Q2T* grad;

		// Compute corrected descent direction
		double a = 0;
		while (true)
		{
		  Eigen::SparseMatrix<double> mat;
		  if (a == 0)
		  {
			//mat = hessian; // Use newton step
            mat = Q2T * hessian * Q2;
		  }
		  else 
		  {     
			// Create identity
			Eigen::SparseMatrix<double> id(hessian.rows(), hessian.rows());
			id.setIdentity();
			
			// Create matrix with correction
			mat = Q2T * ((hessian + a*id) * Q2);
		  }

		  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
		  solver.compute(mat);
		  newton = -solver.solve(rhs);
	 	  newton = Q2 * newton;
		  double newton_decr = newton.dot(grad);
		  std::cout << "gradient norm is " << grad.dot(grad) << std::endl;
		  std::cout << "newton norm is " << newton.dot(newton) << std::endl;
		  std::cout << "projected gradient is " << newton_decr << std::endl;
		  if (solver.info() == Eigen::Success && newton_decr < 0)
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
        grad = rhs;
    }

    // do lineserach
    Eigen::MatrixXd search_dir = Eigen::Map<Eigen::MatrixXd>(newton.data(), V.rows(), 2);
    auto new_x = uv;
    double ls_step_size = 1.0;
    bool ls_good = false;
    for (int i = 0; i < m_params.ls_iters; i++) {
        new_x = uv + ls_step_size * search_dir;
        double new_E = compute_energy(new_x, lambda);
        if (new_E < energy_0 && check_flip(new_x, F) == 0) {
            std::cout << "energy from " << energy_0 << " to " << new_E << std::endl;
            ls_good = true;
            break;
        }
        ls_step_size *= 0.8;
    }

    if (ls_good) {
        // update vertex_attrs
        std::cout << "ls_step_size = " << ls_step_size << std::endl;
        for (int i = 0; i < new_x.rows(); i++) {
            vertex_attrs[i].pos = new_x.row(i);
        }
    } else {
        std::cout << "smooth failed" << std::endl;
    }

    return grad.cwiseAbs().maxCoeff();
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
*/

} // namespace SymDir
