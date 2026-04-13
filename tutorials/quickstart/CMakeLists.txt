add_executable(quickstart main.cu)

target_link_libraries(quickstart PRIVATE phix)

set_target_properties(quickstart PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    # Place the binary in the tutorial's source directory so that
    # running ./quickstart writes output/ next to main.cu, not inside build/.
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
