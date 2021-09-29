if (NOT USE_INTERNAL_EIGEN)
    return()
endif()

set(EIGEN_LIBRARY_DIR "${ClickHouse_SOURCE_DIR}/contrib/eigen")

# Imported target support
add_library (eigen INTERFACE)
target_include_directories (eigen SYSTEM INTERFACE ${EIGEN_LIBRARY_DIR})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message("OpenMP version: ${OpenMP_VERSION}")
    target_link_libraries(eigen INTERFACE OpenMP::OpenMP_CXX)
endif()
