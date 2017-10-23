project(voxel_map)
cmake_minimum_required(VERSION 2.8)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

FIND_PACKAGE(PythonInterp)
FIND_PACKAGE(PythonLibs)

if (${PYTHON_VERSION_MAJOR} EQUAL 3)
    FIND_PACKAGE(Boost COMPONENTS python3 chrono system thread)
else()
    FIND_PACKAGE(Boost COMPONENTS python chrono system thread)
endif()

include_directories(${PYTHON_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
include_directories(include)

add_library(py_voxel_map SHARED python_wrapper/voxel_map.cpp)
set_target_properties(py_voxel_map PROPERTIES
    OUTPUT_NAME voxel_map
    PREFIX "")
target_link_libraries(py_voxel_map ${PYTHON_LIBRARIES} ${Boost_LIBRARIES})
