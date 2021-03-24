cmake_minimum_required(VERSION 3.1)
project(test_package)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

find_package(Torch REQUIRED CONFIG)

add_executable(${PROJECT_NAME} test_package.cpp)
target_include_directories(${PROJECT_NAME} PRIVATE ${Torch_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
