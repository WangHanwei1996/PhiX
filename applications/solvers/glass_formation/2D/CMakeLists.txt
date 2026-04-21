add_executable(GFA_2D "GFA.cu")

target_link_libraries(GFA_2D PRIVATE phix)

set_target_properties(GFA_2D PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
