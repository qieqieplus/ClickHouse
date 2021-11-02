if (NOT EXISTS "${ClickHouse_SOURCE_DIR}/contrib/eigen/signature_of_eigen3_matrix_library")
    message (ERROR "submodule contrib/eigen is missing. to fix try run: \n git submodule update --init --recursive")
    return()
endif()

set (EIGEN_INCLUDE_DIR "${ClickHouse_SOURCE_DIR}/contrib/eigen")
message(STATUS "Using eigen=${USE_EIGEN}: ${EIGEN_INCLUDE_DIR}")
