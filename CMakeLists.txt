cmake_minimum_required(VERSION 3.25)
project(SymmetricDirichlet)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-fPIC)
add_compile_options(-Wno-register)

option(USE_SUITESPARSE "Use suitesparse for QR factorization" ON)
option(USE_COMISO "Use Comiso for field generation" ON)
set ( SUITESPARSE_ENABLE_PROJECTS "suitesparse_config;cholmod;spqr;umfpack" )
option ( SUITESPARSE_USE_CUDA OFF )

if (USE_COMISO)
  add_compile_definitions(USE_COMISO)
  option(LIBIGL_COPYLEFT_COMISO "Use COSIMO" ON)
  set(COMISO_LIBS
    igl_copyleft::comiso
  )
endif()

if(USE_SUITESPARSE)
  add_compile_definitions(USE_SUITESPARSE)
  set(SUITESPARSE_LIBS
    SuiteSparse::SuiteSparseConfig
    SuiteSparse::SPQR
    SuiteSparse::CHOLMOD
    SuiteSparse::UMFPACK
  )
endif()

# Install executables to bin directory
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(cli11)
include(nlohmann_json)
include(libigl)
include(eigen)
include(spdlog)
include(polyscope)
include(suitesparse)
include(pybind11)
igl_include(predicates)


add_library(SymmetricDirichletLib
    src/energy.cpp
    src/ExtremeOpt.cpp
    src/opt.cpp
    src/rref.cpp
    src/smooth.cpp
    src/SYMDIR_NEW.cpp
    src/MeshCutter.cpp
)
target_include_directories(SymmetricDirichletLib PUBLIC include)
target_link_libraries(SymmetricDirichletLib PUBLIC
    igl::core
    igl::predicates
    ${COMISO_LIBS}
    ${SUITESPARSE_LIBS}
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
    spdlog::spdlog
    polyscope
)


add_executable(symmetric_dirichlet
    app/main.cpp
)
target_link_libraries(symmetric_dirichlet
    SymmetricDirichletLib
    CLI11::CLI11
)

add_executable(view_mesh
    app/view_mesh.cpp
)
target_link_libraries(view_mesh
    SymmetricDirichletLib
    CLI11::CLI11
)

add_executable(cut_mesh
    app/cut_mesh.cpp
)
target_link_libraries(cut_mesh
    SymmetricDirichletLib
    CLI11::CLI11
)

add_library(symdir MODULE
  src/pybind.cpp
)

# Link libraries
target_link_libraries(symdir PUBLIC
  SymmetricDirichletLib
  pybind11::module
)

# Set pybinding settings
set_target_properties(symdir PROPERTIES LIBRARY_OUTPUT_DIRECTORY
  ${PROJECT_SOURCE_DIR}/py
)
set_target_properties(symdir PROPERTIES PREFIX
  "${PYTHON_MODULE_PREFIX}"
)