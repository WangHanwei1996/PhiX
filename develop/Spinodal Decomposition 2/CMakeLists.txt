add_executable(spinodal_decomposition "Cahn-Hillard.cu")

target_link_libraries(spinodal_decomposition PRIVATE phix)

set_target_properties(spinodal_decomposition PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
