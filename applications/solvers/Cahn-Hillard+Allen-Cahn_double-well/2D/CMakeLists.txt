add_executable(CH_AC_2D "CH+AC_double-well.cu")

target_link_libraries(CH_AC_2D PRIVATE phix)

set_target_properties(CH_AC_2D PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
