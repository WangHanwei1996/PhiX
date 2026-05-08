add_executable(GFA_eta_only "GFA_eta_only.cu")

target_link_libraries(GFA_eta_only PRIVATE phix curand)

set_target_properties(GFA_eta_only PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
