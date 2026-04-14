add_executable(ostwald_ripening "AC-CH.cu")

target_link_libraries(ostwald_ripening PRIVATE phix)

set_target_properties(ostwald_ripening PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
