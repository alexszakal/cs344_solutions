cmake_minimum_required(VERSION 3.20)
project(cs344HW5 LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(hw5 main.cpp
                   reference_calc.cpp
                   student.cu)
#target_link_libraries(hw5 PRIVATE CUDA::cudart)
target_include_directories(hw5 SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(hw5 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(hw5 SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( hw5 PRIVATE ${OpenCV_LIBS} )
