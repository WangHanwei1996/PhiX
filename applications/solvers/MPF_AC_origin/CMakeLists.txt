add_executable(MPF_AC "MPF_AC.cu")

target_link_libraries(MPF_AC PRIVATE phix)

set_target_properties(MPF_AC PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
