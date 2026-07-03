add_executable(module_perf test_perf.cu)
target_link_libraries(module_perf PRIVATE phix)
set_target_properties(module_perf PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME module_perf COMMAND module_perf)
