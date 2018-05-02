project(medianfilter_test)
cmake_minimum_required(VERSION 3.5)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED) 


set(MAIN_PROJECT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/../..)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${MAIN_PROJECT_SOURCE_DIR}/bin)

set(WAVE_FORMAT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/../waveformat)
set(CPU_MEDIANFILTER_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/../medianfilter)
set(GPU_MEDIANFILTER_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/../medianfilter)

include_directories(
${OpenCV_INCLUDE_DIRS}
${WAVE_FORMAT_INCLUDE_DIRS}
${CPU_MEDIANFILTER_INCLUDE_DIRS}
${GPU_MEDIANFILTER_INCLUDE_DIRS}
)

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61)
add_executable(cpu_medianfilter_test cpu_medianfilter_test.c)

cuda_add_executable(gpu_medianfilter_test gpu_medianfilter_test.cu)

set(WAVE_FORMAT_LIBS ${MAIN_PROJECT_SOURCE_DIR}/bin/lib/waveformat/libwaveformat.so)
set(CPU_MEDIANFILTER_LIBS ${MAIN_PROJECT_SOURCE_DIR}/bin/lib/medianfilter/libcpu_medianfilter.a)
set(GPU_MEDIANFILTER_LIBS ${MAIN_PROJECT_SOURCE_DIR}/bin/lib/medianfilter/libgpu_medianfilter.a)

target_link_libraries(cpu_medianfilter_test ${CPU_MEDIANFILTER_LIBS} ${OpenCV_LIBS} ${WAVE_FORMAT_LIBS})
target_link_libraries(gpu_medianfilter_test ${GPU_MEDIANFILTER_LIBS} ${OpenCV_LIBS})
