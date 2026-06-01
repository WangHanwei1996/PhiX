add_executable(module_field test_field.cpp)
target_link_libraries(module_field PRIVATE phix)
add_test(NAME module_field COMMAND module_field)
