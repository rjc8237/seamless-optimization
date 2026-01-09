#include "energy.h"
#include "spdlog/spdlog.h"
#include <numeric>
#include <smooth_utils.h>

// From AutoDiff
namespace jakob
{
#include "autodiff_jakob.h"
  DECLARE_DIFFSCALAR_BASE();

  template <typename Scalar>
  Scalar gradient_and_hessian_from_J(const Eigen::Matrix<Scalar, 1, 4> &J,
                                     Eigen::Matrix<Scalar, 1, 4> &local_grad,
                                     Eigen::Matrix<Scalar, 4, 4> &local_hessian,
                                     double norm_p, bool soft_max = false, double t = 1.0, double E_min = 1.0)
  {
      typedef DScalar2<Scalar, Eigen::Matrix<Scalar, 4, 1>, Eigen::Matrix<Scalar, 4, 4>> DScalar;
      DiffScalarBase::setVariableCount(4);

      DScalar a(0, J(0));
      DScalar b(1, J(1));
      DScalar c(2, J(2));
      DScalar d(3, J(3));
      auto sd = SymDir::symmetric_dirichlet_energy_t(a, b, c, d, norm_p, soft_max, t, E_min);

      local_grad = sd.getGradient();
      local_hessian = sd.getHessian();
      DiffScalarBase::setVariableCount(0);
      return sd.getValue();
  }
} // namespace jakob

namespace SymDir{
    
    template <typename Scalar>
    void jacobian_from_uv(const Eigen::SparseMatrix<Scalar> &G, const Eigen::Matrix<Scalar, -1, -1> &uv, Eigen::Matrix<Scalar, -1, -1> &Ji)
    {
        Eigen::Matrix<Scalar, -1, 1> altJ = G * Eigen::Map<const Eigen::Matrix<Scalar, -1, 1>>(uv.data(), uv.size());
        Ji = Eigen::Map<Eigen::Matrix<Scalar, -1, -1>>(altJ.data(), G.rows() / 4, 4);
    }
    
    template <typename Scalar>
    Scalar compute_energy_from_jacobian(const Eigen::Matrix<Scalar, -1, -1> &J, const Eigen::Matrix<Scalar, -1, 1> &area, double norm_p, bool soft_max, double t, double E_min, bool uniform)
    {
        Scalar energy = symmetric_dirichlet_energy(J.col(0), J.col(1), J.col(2), J.col(3), norm_p, soft_max, t, E_min).dot(area) / area.sum();
        if (!uniform)
        {
            if (soft_max) {
                energy = std::log(std::max(energy, 1e-10)) * t;
            }
            return energy;
        }
        else
        {
            return symmetric_dirichlet_energy(J.col(0), J.col(1), J.col(2), J.col(3), norm_p, soft_max, t).sum() / Scalar(J.rows());
        }
    }
    
    template <typename Scalar>
    Scalar compute_worst_n_energy_from_jacobian(const Eigen::Matrix<Scalar, -1, -1>& J,  const Eigen::Matrix<Scalar, -1, 1>& area, int norm_p, double percent, bool soft_max, double t, double E_min)
    {
        // Compute per-triangle energy vector
        Eigen::VectorXd energy_per_tri = symmetric_dirichlet_energy(J.col(0), J.col(1), J.col(2), J.col(3), norm_p, soft_max, t, E_min).array();
        // Sort indices based on energy values
        std::vector<int> indices(energy_per_tri.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(),
                [&](int i1, int i2) { return energy_per_tri[i1] > energy_per_tri[i2]; });

        // Calculate how many triangles to include
        int num_tris = static_cast<int>(std::ceil(percent / 100.0 * indices.size()));
        // Compute p-norm of the N% largest energies
        Scalar p_norm_sum = 0;
        Scalar area_sum = 0;
        for (int i = 0; i < indices.size(); i++) {
            area_sum += area[indices[i]];
        }
        for (int i = 0; i < num_tris; i++) {
            p_norm_sum += energy_per_tri[indices[i]] * area[indices[i]];
        }
        // Return average of worst N% triangles
        return p_norm_sum / area_sum;
    }
    template <typename Scalar>
    Scalar compute_threshold_energy_from_jacobian(const Eigen::Matrix<Scalar, -1, -1>& J,  const Eigen::Matrix<Scalar, -1, 1>& area, int norm_p, double percent, bool soft_max, double t){
                // Compute per-triangle energy vector
        Eigen::VectorXd energy_per_tri = symmetric_dirichlet_energy(J.col(0), J.col(1), J.col(2), J.col(3), 1, soft_max, t).array();
        // Sort indices based on energy values
        std::vector<int> indices(energy_per_tri.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::sort(indices.begin(), indices.end(),
                [&](int i1, int i2) { return energy_per_tri[i1] > energy_per_tri[i2]; });

        // Calculate how many triangles to include
        int num_tris = static_cast<int>(std::ceil(percent / 100.0 * indices.size()));
        
        // Return average of worst N% triangles
        return energy_per_tri[num_tris - 1];

    }
    template <typename Scalar>
    Eigen::ArrayXd get_sym_dirich_per_triangle_from_jacobian(const Eigen::Matrix<Scalar, -1, -1> &J)
    {
        return symmetric_dirichlet_energy(J.col(0), J.col(1), J.col(2), J.col(3), 1.0).array();
    }
    
    template <typename Scalar>
    Scalar grad_and_hessian_from_jacobian(const Eigen::Matrix<Scalar, -1, 1> &area, const Eigen::Matrix<Scalar, -1, -1> &jacobian,
                                      Eigen::Matrix<Scalar, -1, -1> &total_grad, Eigen::SparseMatrix<Scalar> &hessian, bool with_hessian, double norm_p, bool projected_newton, bool soft_max = false, double t = 1.0, double E_min = 1.0)
    {
        int f_num = area.rows();
        total_grad.resize(f_num, 4);
        total_grad.setZero();
        Scalar energy = 0;
        hessian.resize(4 * f_num, 4 * f_num);
        std::vector<Eigen::Triplet<Scalar>> IJV;
        IJV.reserve(16 * f_num);
        Scalar total_area = area.sum();

        std::vector<Eigen::Matrix<Scalar, 4, 4>> all_hessian(f_num);

        for (int i = 0; i < f_num; i++)
        {
            Eigen::Matrix<Scalar, 1, 4> J = jacobian.row(i);
            Eigen::Matrix<Scalar, 4, 4> local_hessian;
            Eigen::Matrix<Scalar, 1, 4> local_grad;
            energy += jakob::gradient_and_hessian_from_J(J, local_grad, local_hessian, norm_p, soft_max, t, E_min) * area(i) / total_area;
            SPDLOG_TRACE("total area is {}", total_area);
            SPDLOG_TRACE("jacobian is {}, {}, {}, {}", J[0], J[1], J[2], J[3]);
            SPDLOG_TRACE("local gradient is {}, {},...", local_grad[0], local_grad[1]);

            local_grad *= area(i) / total_area;
            total_grad.row(i) = local_grad;
            if (with_hessian)
            {
                local_hessian *= area(i) / total_area;
                if (projected_newton)
                    projected_local_hessian(local_hessian);
                all_hessian[i] = local_hessian;
            }
        }

        if (soft_max) {
            // Compute weights w_i = exp(E_i / t) / sum_j exp(E_j / t)
            std::vector<Scalar> weights(f_num);
            for (int i = 0; i < f_num; i++){
                Eigen::Matrix<Scalar, 1, 4> J = jacobian.row(i);
                weights[i] = SymDir::symmetric_dirichlet_energy_t(J(0), J(1), J(2), J(3), norm_p, soft_max, t, E_min);
            }
            // Apply weights to compute proper gradient and hessian
            for (int i = 0; i < f_num; i++){
                Scalar w_i = weights[i] / energy;
                total_grad.row(i) *= w_i;
                if (with_hessian){
                    all_hessian[i] *= w_i;
                }
            }
            energy = std::log(std::max(energy, Scalar(1e-10))) * t;
        }
    
        if (with_hessian)
        {
            hessian.reserve(Eigen::VectorXi::Constant(4 * f_num, 4));
            for (int i = 0; i < f_num; i++)
            {
                Eigen::Matrix<Scalar, 4, 4> &local_hessian = all_hessian[i];
                // if (fabs(total_grad(i)) > 1e-3)
                    // project_hessian(local_hessian);
                for (int v1 = 0; v1 < 4; v1++)
                {
                    for (int v2 = 0; v2 < v1 + 1; v2++)
                    {
                        hessian.insert(v1 * f_num + i, v2 * f_num + i) = local_hessian(v1, v2);
                        if (v1 != v2)
                            hessian.insert(v2 * f_num + i, v1 * f_num + i) = local_hessian(v1, v2);
                    }
                }
            }
            hessian.makeCompressed();
        }
        return energy;
    }

    template <typename Scalar>
    Scalar get_grad_and_hessian(const Eigen::SparseMatrix<Scalar> &G,
                                const Eigen::Matrix<Scalar, -1, 1> &area,
                                const Eigen::Matrix<Scalar, -1, -1> &uv,
                                Eigen::Matrix<Scalar, -1, 1> &grad,
                                Eigen::SparseMatrix<Scalar> &hessian,
                                bool get_hessian,
                                double norm_p, bool projected_newton, bool soft_max, double t, double E_min)
    {
        int f_num = area.rows();
        Eigen::Matrix<Scalar, -1, -1> Ji, total_grad;
        jacobian_from_uv(G, uv, Ji);
        Scalar energy;
        energy = grad_and_hessian_from_jacobian(area, Ji, total_grad, hessian, get_hessian, norm_p, projected_newton, soft_max, t, E_min);

        Eigen::Matrix<Scalar, -1, 1> vec_grad = Eigen::Map<Eigen::Matrix<Scalar, -1, 1>>(total_grad.data(), total_grad.size());
        
        hessian = G.transpose() * hessian * G;
        grad = vec_grad.transpose() * G;

        return energy;
    }

    template void jacobian_from_uv<double>(const Eigen::SparseMatrix<double> &, const Eigen::Matrix<double, -1, -1> &, Eigen::Matrix<double, -1, -1> &);
    
    template double get_grad_and_hessian<double>(const Eigen::SparseMatrix<double> &,
                                             const Eigen::Matrix<double, -1, 1> &,
                                             const Eigen::Matrix<double, -1, -1> &,
                                             Eigen::Matrix<double, -1, 1> &,
                                             Eigen::SparseMatrix<double> &,
                                             bool,
                                             double, bool, bool, double, double);
    template double compute_energy_from_jacobian<double>(const Eigen::Matrix<double, -1, -1> &, const Eigen::Matrix<double, -1, 1> &, double, bool, double, double, bool);
    template double compute_worst_n_energy_from_jacobian<double>(
        const Eigen::Matrix<double, -1, -1>&,
        const Eigen::Matrix<double, -1, 1>&, 
        int,
        double, bool, double, double);
    template double compute_threshold_energy_from_jacobian<double>(        
        const Eigen::Matrix<double, -1, -1>&,
        const Eigen::Matrix<double, -1, 1>&, 
        int,
        double, bool, double);
    template Eigen::ArrayXd get_sym_dirich_per_triangle_from_jacobian(const Eigen::Matrix<double, -1, -1>&);
} // namespace SymDir
