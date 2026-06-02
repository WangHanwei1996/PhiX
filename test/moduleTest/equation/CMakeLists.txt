add_executable(module_expr test_expr.cpp)
target_link_libraries(module_expr PRIVATE phix)
add_test(NAME module_expr COMMAND module_expr)
