# [makePhi generated]

add_executable(GFA_4ph GFA_4ph.cu)

target_link_libraries(GFA_4ph PRIVATE phix)

set_target_properties(GFA_4ph PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
