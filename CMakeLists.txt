cmake_minimum_required(VERSION 3.25)
project(SymmetricDirichlet)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-fPIC)

# Install executables to bin directory
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(cli11)
include(nlohmann_json)
include(libigl)
include(eigen)
include(spdlog)
include(polyscope)
igl_include(predicates)

add_library(SymmetricDirichletLib
    src/energy.cpp
    src/ExtremeOpt.cpp
    src/opt.cpp
    src/rref.cpp
    src/smooth.cpp
    src/SYMDIR_NEW.cpp
)
target_include_directories(SymmetricDirichletLib PUBLIC include)
target_link_libraries(SymmetricDirichletLib PUBLIC
    igl::core
    igl::predicates
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
