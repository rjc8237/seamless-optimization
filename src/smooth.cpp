#include <igl/boundary_loop.h>
#include <igl/facet_components.h>
#include <igl/predicates/predicates.h>
#include <igl/writeOBJ.h>
#include "ExtremeOpt.h"
#include "energy.h"
#include "rref.h"
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

void buildAeq(
    const Eigen::MatrixXi& EE,
    const Eigen::MatrixXi& FE,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F,
    Eigen::SparseMatrix<double>& Aeq)
{
    int N = uv.rows();
    int c = 0;
    int m = EE.rows() / 2;
    int fes = FE.rows();

    std::vector<std::vector<int>> bds;
    igl::boundary_loop(F, bds);

    // get face components
    Eigen::VectorXi C;
    int num_components = igl::facet_components(F, C);
    std::vector<int> component_faces(num_components);
    for (int f = 0; f < F.rows(); ++f)
    {
        component_faces[C[f]] = f;
    }

    int n_fix_dof;
    if (fes > 0) {
        n_fix_dof = 2 * num_components;
    }
    else
    {
        n_fix_dof = 3;
    }

    std::set<std::pair<int, int>> added_e;
    typedef Eigen::Triplet<double> Trip;
    std::vector<Trip> trips;
    trips.reserve(12 * EE.rows());

    Aeq.resize(2 * m + n_fix_dof + fes, uv.rows() * 2);
    int A2, B2, C2, D2;
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

        Eigen::Matrix<double, 2, 1> e_ab = uv.row(B2) - uv.row(A2);
        Eigen::Matrix<double, 2, 1> e_dc = uv.row(C2) - uv.row(D2);

        Eigen::Matrix<double, 2, 1> e_ab_perp;
        e_ab_perp(0) = -e_ab(1);
        e_ab_perp(1) = e_ab(0);
        double angle = atan2(-e_ab_perp.dot(e_dc), e_ab.dot(e_dc));
        int r = (int)(round(2 * angle / double(igl::PI)) + 2) % 4;

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
    

    if (fes == 0) {
        double min_u_diff = 1e10;
        int min_u_diff_id = 0;
        auto l = bds[0];
        for (int i = 0; i < l.size(); i++) {
            double u_diff = abs(uv(l[i], 0) - uv(l[(i + 1) % l.size()], 0));
            if (u_diff < min_u_diff) {
                min_u_diff = u_diff;
                min_u_diff_id = i;
            }
        }
        std::cout << "fix " << l[min_u_diff_id] << std::endl;
        trips.push_back(Trip(c, l[min_u_diff_id], 1));
        trips.push_back(Trip(c + 1, l[min_u_diff_id] + N, 1));

        std::cout << "fix " << l[(min_u_diff_id + 1) % l.size()] << std::endl;
        trips.push_back(Trip(c, l[(min_u_diff_id + 1) % l.size()], 1));
        c = c + 1;
    }
    else {
        for (int ci = 0; ci < num_components; ++ci)
        {
            int f = component_faces[ci];
            int vi = F(f, 0);
            trips.push_back(Trip(c, vi, 1));
            trips.push_back(Trip(c + 1, vi + N, 1));
            c = c + 2;
        }
    
        std::set<std::pair<int, int>> added_fe;
        // feature edge constraints
        for (int i = 0; i < fes; ++i) {
            int v1 = FE(i, 0);
            int v2 = FE(i, 1);
            auto e0 = std::make_pair(v1, v2);
            if (added_fe.find(e0) != added_fe.end())
            {
                spdlog::warn("Edge added twice");
                continue;
            }
            added_fe.insert(e0);
            
            Eigen::Vector2d e_ab = uv.row(v2) - uv.row(v1);

            // constrain u or v depending on initial position
            if (FE(i, 2) == 0) {
                trips.push_back(Trip(c, v1, -1));
                trips.push_back(Trip(c, v2, 1));
                c += 1;
            }
            else if (FE(i, 2) == 1) {
                trips.push_back(Trip(c, v1 + N, -1));
                trips.push_back(Trip(c, v2 + N, 1));
                c += 1;
            }
            else
            {
                spdlog::error("Feature edge does not have a tag");
            }
        }
    }
    Aeq.resize(2 * m + n_fix_dof + fes, uv.rows() * 2);
    Aeq.setFromTriplets(trips.begin(), trips.end());
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

double ExtremeOpt::smooth_global(int steps)
{
    Eigen::MatrixXi F;
    Eigen::MatrixXd V, uv;
    export_mesh(V, F, uv);
    
    // build edge mesh
    Eigen::MatrixXi EE;
    export_EE(EE);

    // build feature matrix if doing feature alignment
    Eigen::MatrixXi FE(0, 0);
    if (m_params.do_feature_alignment)
    {
        export_FE(FE);
    }

    Eigen::VectorXd area;
    Eigen::SparseMatrix<double> G;
    igl::doublearea(V, F, area);
    get_grad_op(V, F, G);
    Eigen::SparseMatrix<double> Aeq;
    buildAeq(EE, FE, uv, F, Aeq);
    Eigen::SparseMatrix<double> AeqT = Aeq.transpose();

    auto compute_energy = [G, area](Eigen::MatrixXd aaa) {
        Eigen::MatrixXd Ji;
        SymDir::jacobian_from_uv(G, aaa, Ji);
        return SymDir::compute_energy_from_jacobian(Ji, area);
    };
    Eigen::VectorXd newton;
    // get grad and hessian
    Eigen::SparseMatrix<double> hessian;
    Eigen::VectorXd grad;
    double energy_0 = SymDir::get_grad_and_hessian(G, area, uv, grad, hessian, m_params.do_newton);

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
        Eigen::SparseMatrix<double> Q2(Aeq.cols(), Aeq.cols() - Aeq.rows()), Q2T;
        elim_constr(Aeq, Q2);
        Q2.makeCompressed();
        Q2T = Q2.transpose();
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

		  //Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
		  Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
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
        double new_E = compute_energy(new_x);
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
