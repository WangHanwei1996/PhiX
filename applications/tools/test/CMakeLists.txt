add_executable(div_field div_field.cu)

target_link_libraries(div_field PRIVATE phix)

set_target_properties(div_field PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
