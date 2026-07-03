# [GFA_FeB — Fe-B (Fe2B) glass-formation solver, adapted from GFA_evo]

add_executable(GFA_FeB GFA_FeB.cu)

target_link_libraries(GFA_FeB PRIVATE phix)

set_target_properties(GFA_FeB PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
