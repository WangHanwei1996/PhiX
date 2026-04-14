add_executable(test_fieldops test_fieldops.cu)

target_link_libraries(test_fieldops PRIVATE phix)

set_target_properties(test_fieldops PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
