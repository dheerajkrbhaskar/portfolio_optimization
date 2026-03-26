cmake_minimum_required(VERSION 3.10)

project(portfolio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

# =========================
# OpenMP
# =========================

find_package(OpenMP REQUIRED)

# =========================
# Core SEQ
# =========================

add_library(core_seq
    src/data_loader.cpp
    src/portfolio.cpp
    src/stats.cpp
    src/montecarlo.cpp
)

# =========================
# Core OMP
# =========================

add_library(core_omp
    src/data_loader.cpp
    src/portfolio.cpp
    src/stats.cpp
    src/montecarlo.cpp
)

target_link_libraries(core_omp OpenMP::OpenMP_CXX)

# =========================
# Sequential executable
# =========================

add_executable(seq main_seq.cpp)

target_link_libraries(seq core_seq)

# =========================
# OpenMP executable
# =========================

add_executable(omp main_omp.cpp)

target_link_libraries(omp
    core_omp
    OpenMP::OpenMP_CXX
)


# =========================
# MPI executable
# =========================

find_package(MPI REQUIRED)

add_library(core_mpi
    src/data_loader.cpp
    src/portfolio.cpp
    src/stats.cpp
    src/montecarlo.cpp
)

target_link_libraries(core_mpi
    PUBLIC MPI::MPI_CXX
    PUBLIC OpenMP::OpenMP_CXX
)

add_executable(mpi main_mpi.cpp)

target_link_libraries(mpi core_mpi)