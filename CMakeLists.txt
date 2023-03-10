# Compile with OpenGL or not
option(BUILD_OPENGL "Build OpenGL" OFF)

cmake_minimum_required( VERSION 3.10 FATAL_ERROR)

project(holoCu CXX CUDA)

# define directory paths
set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(PROJECT_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/include")

# find OpenGL 
if(BUILD_OPENGL)
  find_package(OpenGL REQUIRED)
  find_package(GLUT REQUIRED)
  find_package(glfw3 REQUIRED)
  find_package(GLEW REQUIRED)
  include_directories( ${GLEW_INCLUDE_DIRS} )
  include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

  set(SOURCES
    src/kernels.cu
    src/helpers.cpp 
    ${PROJECT_INCLUDE_DIR}/kernels.cuh
    ${PROJECT_INCLUDE_DIR}/optics.h
    ${PROJECT_INCLUDE_DIR}/common.h
    ${PROJECT_INCLUDE_DIR}/stb_image_write.h
    ${PROJECT_INCLUDE_DIR}/helpers.hpp
    ${PROJECT_INCLUDE_DIR}/helper_cuda.h
    ${PROJECT_INCLUDE_DIR}/helper_string.h
  )

endif()

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo -lcudadevrt -lcublas -lcurand")

include_directories(
  ${PROJECT_INCLUDE_DIR}  
  kernels
)

# run headless.cpp
if(NOT BUILD_OPENGL)
  set(SOURCES
      src/kernels.cu
      ${PROJECT_INCLUDE_DIR}/kernels.cuh
      ${PROJECT_INCLUDE_DIR}/optics.h
      ${PROJECT_INCLUDE_DIR}/common.h
      ${PROJECT_INCLUDE_DIR}/stb_image_write.h
  )
    add_executable(
      holoCu
      src/headless.cpp
      ${SOURCES}
    )
endif()

# run full.cpp
if(BUILD_OPENGL)
  add_executable(
    holoCu
    src/full.cpp
    ${SOURCES}
  )
endif()


target_compile_options(holoCu PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_75,code=sm_75>)
target_link_libraries(holoCu -lcublas -lcurand -lcufft)
target_link_libraries(holoCu ${CUDA_LIBRARIES})
if(BUILD_OPENGL)
  target_link_libraries(holoCu ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
  target_link_libraries(holoCu glfw)
  target_link_libraries(holoCu ${GLEW_LIBRARIES})
endif()










