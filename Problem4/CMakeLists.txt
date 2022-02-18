cmake_minimum_required(VERSION 3.20)
project(cs344HW4 LANGUAGES CUDA CXX)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(hw4 main.cpp 
                   compare.cpp 
                   loadSaveImage.cpp
                   reference_calc.cpp 
                   HW4.cu 
                   student_func.cu)
set_target_properties(hw4 PROPERTIES 
                          CXX_STANDARD 14
                          CXX_STANDARD_REQUIRED YES
                          CXX_EXTENSIONS NO)
                          
#target_link_libraries(hw4 PRIVATE CUDA::cudart)
target_include_directories(hw4 SYSTEM PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(hw4 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(hw4 SYSTEM PRIVATE ${OpenCV_INCLUDE_DIRS} )
target_link_libraries( hw4 PRIVATE ${OpenCV_LIBS} )

