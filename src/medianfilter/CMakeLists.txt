find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

set(WAVE_FORMAT_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/src/waveformat)
message(${WAVE_FORMAT_INCLUDE_DIRS})

include_directories(
${WAVE_FORMAT_INCLUDE_DIRS}
${OpenCV_INCLUDE_DIRS}
)

set(WAVE_FORMAT_LIBS ${PROJECT_SOURCE_DIRS}/bin/lib/libwaveformat.so)
add_library(cpu_medianfilter STATIC cpu_medianfilter.c)

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61)
cuda_add_library(gpu_medianfilter STATIC gpu_medianfilter.cu)

target_link_libraries(cpu_medianfilter ${OpenCV_LIBS} ${WAVE_FORMAT_LIBS})
target_link_libraries(gpu_medianfilter ${OpenCV_LIBS} ${WAVE_FORMAT_LIBS})
