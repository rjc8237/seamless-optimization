cmake_minimum_required(VERSION 3.25)
project(SymmetricDirichlet)
# --- FORCE OPENBLAS ---
set(BLA_VENDOR "OpenBLAS")

# Tell CMake where Homebrew installed OpenBLAS
list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/openblas")
find_package(BLAS REQUIRED)

# Check what we found (This is for your sanity)
message(STATUS ">> BLAS Libraries Found: ${BLAS_LIBRARIES}")
# ----------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-fPIC)
add_compile_options(-Wno-register)
add_compile_options(-O3)  # Enable optimizations

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
include_directories(${CMAKE_SOURCE_DIR}/external/spectra/include)
igl_include(predicates)

find_package(Boost REQUIRED)
if(APPLE)
    # 1. flags to tell Clang to use OpenMP
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "omp")
    
    # 2. (CRITICAL) Tell CMake where 'omp.h' is hidden
    include_directories("/opt/homebrew/opt/libomp/include")
    
    # 3. Tell CMake where the library binary is hidden
    link_directories("/opt/homebrew/opt/libomp/lib")
    
    # 4. Help find_package locate the config files
    list(APPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/libomp")
endif()

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})

add_library(SymmetricDirichletLib
    src/energy.cpp
    src/ExtremeOpt.cpp
    src/opt.cpp
    src/rref.cpp
    src/smooth.cpp
    src/smooth_utils.cpp
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
    OpenMP::OpenMP_CXX
    ${Boost_LIBRARIES}
    ${BLAS_LIBRARIES}
)

add_executable(symmetric_dirichlet
    app/main.cpp
)
target_link_libraries(symmetric_dirichlet
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

target_compile_definitions(symdir PRIVATE PYBIND)
set_target_properties(symdir PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/py
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
)
# Set pybinding settings
set_target_properties(symdir PROPERTIES LIBRARY_OUTPUT_DIRECTORY
  ${PROJECT_SOURCE_DIR}/py
)
set_target_properties(symdir PROPERTIES PREFIX
  "${PYTHON_MODULE_PREFIX}"
)

if (ENABLE_VISUALIZATION)
add_executable(view_mesh
    app/view_mesh.cpp
)
target_link_libraries(view_mesh
    SymmetricDirichletLib
    CLI11::CLI11
)

add_executable(view_output
    app/view_output.cpp
)
target_link_libraries(view_output
    SymmetricDirichletLib
    CLI11::CLI11
)
endif()

set(ENV{OPENBLAS_NUM_THREADS} 8)
