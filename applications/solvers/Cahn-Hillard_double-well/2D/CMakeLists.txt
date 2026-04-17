add_executable(CH_2D "Cahn-Hillard_double-well.cu")

target_link_libraries(CH_2D PRIVATE phix)

set_target_properties(CH_2D PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
