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
igl_include(predicates)

add_library(SymmetricDirichletLib
    src/opt.cpp
    src/energy.cpp
    src/smooth.cpp
    src/ExtremeOpt.cpp
    src/SYMDIR_NEW.cpp
)
target_include_directories(SymmetricDirichletLib PUBLIC include)
target_link_libraries(SymmetricDirichletLib PUBLIC
    igl::core
    igl::predicates
    nlohmann_json::nlohmann_json
    Eigen3::Eigen
    spdlog::spdlog
)


add_executable(symmetric_dirichlet
    app/main.cpp
)
target_link_libraries(symmetric_dirichlet
    SymmetricDirichletLib
    CLI11::CLI11
)
