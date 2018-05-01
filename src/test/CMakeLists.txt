project(medianfilter_test)
cmake_minimum_required(VERSION 3.5)

find_package(OpenCV REQUIRED)

set(MAIN_PROJECT_SOURCE_DIR ${PROJECT_SURCE_DIR}/../..)

set(WAVE_FORMAT_INCLUDE_DIRS ${PROJECT_SURCE_DIR}/../waveformat)
set(CPU_MEDIANFILTER_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/../medianfilter)
set(GPU_MEDIANFILTER_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/../medianfilter)

include_directories(
${OpenCV_INCLUDE_DIRS}
${WAVE_FORMAT_INCLUDE_DIRS}
${CPU_MEDIANFILTER_INCLUDE_DIRS}
${GPU_MEDIANFILTER_INCLUDE_DIRS}
)

add_executable(cpu_medianfilter_test cpu_medianfilter_test.c)

set(WAVE_FORMAT_LIBS ${MAIN_PROJECT_SOURCE_DIR}/bin/lib/wave_format/libwave_format.so)
set(CPU_MEDIANFILTER_LIBS ${MAIN_PROJECT_SOURCE_DIR}/bin/lib/medianfilter/libcpu_medianfilter.a)
set(GPU_MEDIANFILTER_LIBS ${MAIN_PROJECT_SOURCE_DIR}/bin/lib/medianfilter/libgpu_medianfilter.a)

target_link_libraries(cpu_medianfilter_test ${CPU_MEDIANFILTER_LIBS} ${OpenCV_LIBS})
