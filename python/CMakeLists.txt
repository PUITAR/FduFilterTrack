# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

cmake_minimum_required(VERSION 3.18...3.22)

set(CMAKE_CXX_STANDARD 17)

if (PYTHON_EXECUTABLE)
    set(Python3_EXECUTABLE ${PYTHON_EXECUTABLE})
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module NumPy REQUIRED)

execute_process(COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE _tmp_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)
list(APPEND CMAKE_PREFIX_PATH "${_tmp_dir}")

# Now we can find pybind11
find_package(pybind11 CONFIG REQUIRED)

execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE _numpy_include
        OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ECHO STDOUT)

set(POSSIBLE_OMP_PATHS "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin/libiomp5.so;/usr/lib/x86_64-linux-gnu/libiomp5.so;/opt/intel/lib/intel64_lin/libiomp5.so")
foreach(POSSIBLE_OMP_PATH ${POSSIBLE_OMP_PATHS})
    if (EXISTS ${POSSIBLE_OMP_PATH})
        get_filename_component(OMP_PATH ${POSSIBLE_OMP_PATH} DIRECTORY)
    endif()
endforeach()

if(NOT OMP_PATH)
    message(FATAL_ERROR "Could not find Intel OMP in standard locations; use -DOMP_PATH to specify the install location for your environment")
endif()


set(POSSIBLE_MKL_LIB_PATHS "/opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.so;/usr/lib/x86_64-linux-gnu/libmkl_core.so;/opt/intel/mkl/lib/intel64/libmkl_core.so")
foreach(POSSIBLE_MKL_LIB_PATH ${POSSIBLE_MKL_LIB_PATHS})
    if (EXISTS ${POSSIBLE_MKL_LIB_PATH})
        get_filename_component(MKL_PATH ${POSSIBLE_MKL_LIB_PATH} DIRECTORY)
    endif()
endforeach()

set(POSSIBLE_MKL_INCLUDE_PATHS "/opt/intel/oneapi/mkl/latest/include;/usr/include/mkl;/opt/intel/mkl/include/;")
foreach(POSSIBLE_MKL_INCLUDE_PATH ${POSSIBLE_MKL_INCLUDE_PATHS})
    if (EXISTS ${POSSIBLE_MKL_INCLUDE_PATH})
        set(MKL_INCLUDE_PATH ${POSSIBLE_MKL_INCLUDE_PATH})
    endif()
endforeach()
if(NOT MKL_PATH)
    message(FATAL_ERROR "Could not find Intel MKL in standard locations; use -DMKL_PATH to specify the install location for your environment")
elseif(NOT MKL_INCLUDE_PATH)
    message(FATAL_ERROR "Could not find Intel MKL in standard locations; use -DMKL_INCLUDE_PATH to specify the install location for headers for your environment")
endif()
if (EXISTS ${MKL_PATH}/libmkl_def.so.2)
    set(MKL_DEF_SO ${MKL_PATH}/libmkl_def.so.2)
elseif(EXISTS ${MKL_PATH}/libmkl_def.so)
    set(MKL_DEF_SO ${MKL_PATH}/libmkl_def.so)
else()
    message(FATAL_ERROR "Despite finding MKL, libmkl_def.so was not found in expected locations.")
endif()

# pybind11_add_module(diskannpy MODULE src/diskann_bindings.cpp)
# the following is fairly synonymous with pybind11_add_module, but we need more target_link_libraries
# see https://pybind11.readthedocs.io/en/latest/compiling.html#advanced-interface-library-targets for more details
add_library(_diskannpy MODULE
        src/module.cpp
        src/builder.cpp
        src/dynamic_memory_index.cpp
        src/static_memory_index.cpp
        src/static_disk_index.cpp
)

target_include_directories(_diskannpy 
                            AFTER 
                            PRIVATE 
                            include
                            ${OMP_PATH}
                            ${Boost_INCLUDE_DIR}
                            ${MKL_PATH}
                            ${MKL_INCLUDE_PATH}
                            )

if (MSVC)
    target_compile_options(_diskannpy PRIVATE /U_WINDLL)
endif()

message("${MKL_PATH}")

target_link_libraries(
        _diskannpy
        PRIVATE
        OpenMP::OpenMP_CXX
        ${MKL_PATH}/libmkl_intel_ilp64.a
        ${MKL_PATH}/libmkl_intel_thread.a
        ${MKL_PATH}/libmkl_core.a
        ${MKL_DEF_SO}
        iomp5
        pthread
        m
        dl
        pybind11::module
        pybind11::lto
        pybind11::windows_extras
        ${PROJECT_NAME}
        ${DISKANN_TOOLS_TCMALLOC_LINK_OPTIONS}
        ${DISKANN_ASYNC_LIB}
)

pybind11_extension(_diskannpy)
if(NOT MSVC AND NOT ${CMAKE_BUILD_TYPE} MATCHES Debug|RelWithDebInfo)
    # Strip unnecessary sections of the binary on Linux/macOS
    pybind11_strip(_diskannpy)
endif()

set_target_properties(_diskannpy PROPERTIES CXX_VISIBILITY_PRESET "hidden"
        CUDA_VISIBILITY_PRESET "hidden")

# generally, the VERSION_INFO flag is set by pyproject.toml, by way of setup.py.
# attempts to locate the version within CMake fail because the version has to be available
# to pyproject.toml for the sdist to work after we build it.

if(NOT VERSION_INFO)
    set(VERSION_INFO "0.0.0dev")
endif()
target_compile_definitions(_diskannpy PRIVATE VERSION_INFO="${VERSION_INFO}")
