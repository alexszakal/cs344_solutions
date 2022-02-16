cmake_minimum_required(VERSION 3.20)
project(cs344HW3 LANGUAGES CUDA CXX)
  
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(hw3 main.cpp 
                   compare.cpp 
                   loadSaveImage.cpp
                   reference_calc.cpp 
                   HW3.cu 
                   student_func.cu)
#target_link_libraries(hw1 PRIVATE CUDA::cudart)
target_include_directories(hw3 SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(hw3 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(hw3 SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( hw3 PRIVATE ${OpenCV_LIBS} )
