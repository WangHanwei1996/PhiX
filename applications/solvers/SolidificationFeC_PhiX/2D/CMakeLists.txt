add_executable(SolidificationFeC_PhiX "SolidificationFeC_PhiX.cu")

target_link_libraries(SolidificationFeC_PhiX PRIVATE phix)

set_target_properties(SolidificationFeC_PhiX PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
