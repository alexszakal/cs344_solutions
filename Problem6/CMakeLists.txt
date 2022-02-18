cmake_minimum_required(VERSION 3.20)
project(cs344HW6 LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(hw6 main.cpp
                   reference_calc.cpp
                   student_func.cu
                   HW6.cpp
                   compare.cpp
                   loadSaveImage.cpp)
#target_link_libraries(hw6 PRIVATE CUDA::cudart)
target_include_directories(hw6 SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(hw6 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(hw6 SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( hw6 PRIVATE ${OpenCV_LIBS} )
