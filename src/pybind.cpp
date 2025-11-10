#include <pybind11/eigen.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ExtremeOpt.h"
#include "MeshCutter.h"
#include "Parameters.h"
#include "pybind11_json.hpp"
#include "main_helper.h"

using namespace SymDir;

using json = nlohmann::json;

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
        .def("export_mesh", &ExtremeOpt::export_mesh)
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
    
    m.def("check_constraints", &check_constraints);
    m.def("transform_EE", &transform_EE);
    m.def("symmetric_dirichlet_energy", &SymDir::symmetric_dirichlet_energy);
    m.def("transform_FE", &transform_FE);

    m.def("export_mesh", [](ExtremeOpt &extremeopt) {
        Eigen::MatrixXd V;
        Eigen::MatrixXi F;
        Eigen::MatrixXd uv;
        extremeopt.export_mesh(V, F, uv);
        return std::make_tuple(V, F, uv);
    });
}
