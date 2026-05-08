add_executable(DG_2D "dendrite_growth.cu")

target_link_libraries(DG_2D PRIVATE phix)

set_target_properties(DG_2D PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
