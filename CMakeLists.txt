cmake_minimum_required(VERSION 3.12)
project(cmake_wrapper)

include(conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

find_package(Python 3.8.0 REQUIRED)
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})

find_package(Eigen3 REQUIRED)
set(EIGEN3_FOUND 1)
set(EIGEN3_INCLUDE_DIR ${Eigen3_INCLUDE_DIR})

if(CONAN_LIBTORCH_USE_PTHREADPOOL)
    add_library(pthreadpool INTERFACE IMPORTED)
    set_property(TARGET pthreadpool PROPERTY INTERFACE_LINK_LIBRARIES CONAN_PKG::pthreadpool)
endif()

if(CONAN_LIBTORCH_USE_CPUINFO)
    add_library(cpuinfo INTERFACE IMPORTED)
    add_library(clog INTERFACE IMPORTED)
    set_property(TARGET cpuinfo PROPERTY INTERFACE_LINK_LIBRARIES CONAN_PKG::cpuinfo)
    set_property(TARGET clog PROPERTY INTERFACE_LINK_LIBRARIES CONAN_PKG::cpuinfo)
endif()

if(CONAN_LIBTORCH_USE_FXDIV)
    add_library(fxdiv INTERFACE IMPORTED)
    set_property(TARGET fxdiv PROPERTY INTERFACE_LINK_LIBRARIES CONAN_PKG::fxdiv)
endif()

if(CONAN_LIBTORCH_USE_PSIMD)
    add_library(psimd INTERFACE IMPORTED)
    set_property(TARGET psimd PROPERTY INTERFACE_LINK_LIBRARIES CONAN_PKG::psimd)
endif()

if(CONAN_LIBTORCH_USE_FP16)
    add_library(fp16 INTERFACE IMPORTED)
    set_property(TARGET fp16 PROPERTY INTERFACE_LINK_LIBRARIES CONAN_PKG::fp16)
endif()

if(CONAN_LIBTORCH_USE_SLEEF)
    add_library(sleef INTERFACE IMPORTED)
    set_property(TARGET sleef PROPERTY INTERFACE_LINK_LIBRARIES CONAN_PKG::sleef)
endif()

add_subdirectory(source_subfolder)
