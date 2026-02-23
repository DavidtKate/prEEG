cmake_minimum_required(VERSION 3.20)
project(eeg_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(eeg_infer
    src/main.cpp
    src/features.cpp
    src/onnx_infer.cpp
)

set(ONNXRUNTIME_ROOT "C:/libs/onnxruntime-win-x64-1.24.2" CACHE PATH "ONNX Runtime root")

target_include_directories(eeg_infer PRIVATE
    ${ONNXRUNTIME_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_directories(eeg_infer PRIVATE
    ${ONNXRUNTIME_ROOT}/lib
)

target_link_libraries(eeg_infer PRIVATE onnxruntime)

# Copy DLL next to exe on build (Windows)
add_custom_command(TARGET eeg_infer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${ONNXRUNTIME_ROOT}/bin/onnxruntime.dll
    $<TARGET_FILE_DIR:eeg_infer>
)