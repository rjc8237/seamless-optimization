#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ExtremeOpt.h"
#include "MeshCutter.h"
#include "Parameters.h"
#include "pybind11_json.hpp"
#include "main_helper.h"
#include "energy.h"  // Include headers for your functions
#include "SYMDIR_NEW.h"  // Add this

using namespace SymDir;
namespace py = pybind11;

#ifdef PYBIND
#ifndef MULTIPRECISION

using json = nlohmann::json;

// Helper function wrapper for pybind11
inline Eigen::VectorXd compute_symmetric_dirichlet_energy(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXd& uv,
    const Eigen::MatrixXi& F)
{
    Eigen::VectorXd energy(F.rows());
    SymmetricDirichletEnergy sd_energy(SymmetricDirichletEnergy::EnergyType::Lp, 2);
    
    for (int i = 0; i < F.rows(); i++) {
        int v0 = F(i, 0), v1 = F(i, 1), v2 = F(i, 2);
        // Cast row blocks to Vector3d explicitly
        Eigen::Vector3d A = V.row(v0).transpose();
        Eigen::Vector3d B = V.row(v1).transpose();
        Eigen::Vector3d C = V.row(v2).transpose();
        Eigen::Vector2d a = uv.row(v0).head(2).transpose();
        Eigen::Vector2d b = uv.row(v1).head(2).transpose();
        Eigen::Vector2d c = uv.row(v2).head(2).transpose();
        
        energy(i) = sd_energy.symmetric_dirichlet_energy(A, B, C, a, b, c);
    }
    return energy;
}

// wrap as Python module
PYBIND11_MODULE(symdir, m)
{
    m.doc() = "pybind for optimization module";

    spdlog::set_level(spdlog::level::info);
    pybind11::call_guard<pybind11::scoped_ostream_redirect, pybind11::scoped_estream_redirect>
        default_call_guard;

    pybind11::class_<Mesh>(m, "Mesh")
        .def(py::init<>())
        .def(py::init<const Eigen::MatrixXd&, Eigen::MatrixXi&>());

    pybind11::class_<ExtremeOpt, Mesh>(m, "ExtremeOpt")
        .def(pybind11::init<const Eigen::MatrixXd&, const Eigen::MatrixXi&>())
        .def("create_mesh", &ExtremeOpt::create_mesh)
        .def("export_EE", &ExtremeOpt::export_EE)
        .def("export_FE", &ExtremeOpt::export_FE)
        .def("export_mesh", (void (ExtremeOpt::*)(Eigen::MatrixXd&, Eigen::MatrixXi&, Eigen::MatrixXd&))&ExtremeOpt::export_mesh)
        .def("export_mesh", (void (ExtremeOpt::*)(Eigen::MatrixXd&, Eigen::MatrixXi&, Eigen::MatrixXd&, std::vector<VertexAttributes>))&ExtremeOpt::export_mesh)
        .def("do_optimization", &ExtremeOpt::do_optimization)
        .def("init_constraints", &ExtremeOpt::init_constraints)
        .def("check_constraints", &ExtremeOpt::check_constraints)
        .def_readwrite("m_params", &ExtremeOpt::m_params)
        .def_readwrite("EE", &ExtremeOpt::EE)
        .def_readwrite("FE", &ExtremeOpt::FE);

    pybind11::class_<MeshCutter>(m, "MeshCutter")
        .def(pybind11::init<const Eigen::MatrixXd&,
	        const Eigen::MatrixXd&,
	        const Eigen::MatrixXi&,
	        const Eigen::MatrixXi&>())
        .def("cut_mesh", &MeshCutter::cut_mesh)
        .def("load_feature_edges", &MeshCutter::load_feature_edges)
        .def("reindex_feature_edges", &MeshCutter::reindex_feature_edges);

    pybind11::class_<Parameters>(m, "Parameters")
        .def(py::init<>())
        .def_readwrite("model_name", &Parameters::model_name)
        .def_readwrite("save_meshes", &Parameters::save_meshes)
        .def_readwrite("do_feature_alignment", &Parameters::do_feature_alignment)
        .def_readwrite("max_iters", &Parameters::max_iters)
        .def_readwrite("smooth_only_iters", &Parameters::smooth_only_iters)
        .def_readwrite("do_newton", &Parameters::do_newton)
        .def_readwrite("local_smooth", &Parameters::local_smooth)
        .def_readwrite("global_smooth", &Parameters::global_smooth)
        .def_readwrite("ls_iters", &Parameters::ls_iters)
        .def_readwrite("E_target", &Parameters::E_target)
        .def_readwrite("elen_alpha", &Parameters::elen_alpha)
        .def_readwrite("with_cons", &Parameters::with_cons)
        .def_readwrite("do_projection", &Parameters::do_projection)
        .def_readwrite("use_max_energy", &Parameters::use_max_energy)
        .def_readwrite("Lp", &Parameters::Lp);
    
    //m.def("check_constraints", &check_constraints);
    m.def("symmetric_dirichlet_energy", &compute_symmetric_dirichlet_energy,
          "Compute symmetric Dirichlet energy per triangle",
          py::arg("V"), py::arg("uv"), py::arg("F"));
    m.def("transform_EE", &transform_EE);
    m.def("transform_FE", &transform_FE);

    m.def("export_mesh", [](ExtremeOpt &extremeopt) {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        Eigen::MatrixXd uv;
        extremeopt.export_mesh(V, F, uv);
        return std::make_tuple(V, F, uv);
    });
}
#endif
#endif
