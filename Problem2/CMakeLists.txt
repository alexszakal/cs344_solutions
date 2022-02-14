cmake_minimum_required(VERSION 3.20)
project(cs344HW2 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(hw2 main.cpp compare.cpp reference_calc.cpp student_func.cu)
#target_link_libraries(hw1 PRIVATE CUDA::cudart)
target_include_directories(hw2 SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(hw2 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(hw2 SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( hw2 PRIVATE ${OpenCV_LIBS} )
