add_executable(module_field test_field.cpp)
target_link_libraries(module_field PRIVATE phix)
add_test(NAME module_field COMMAND module_field)

add_executable(module_reduce test_reduce.cu)
target_link_libraries(module_reduce PRIVATE phix)
set_target_properties(module_reduce PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_test(NAME module_reduce COMMAND module_reduce)
