# [makePhi generated]

add_executable(dendrite_growth_flag dendrite_growth_flag.cu)

target_link_libraries(dendrite_growth_flag PRIVATE phix)

set_target_properties(dendrite_growth_flag PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
