# [makePhi generated]

add_executable(GFA_binary GFA_binary.cu)

target_link_libraries(GFA_binary PRIVATE phix)

set_target_properties(GFA_binary PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

add_executable(GFA_evo GFA_evo.cu)

target_link_libraries(GFA_evo PRIVATE phix)

set_target_properties(GFA_evo PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
