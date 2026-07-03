# [makePhi style] implicit/semi-implicit example cases (v2.16.0+ solver layer)
foreach(app heat_implicit allen_cahn_semi spinodal_ch_semi)
    add_executable(${app} ${app}.cu)
    target_link_libraries(${app} PRIVATE phix)
    set_target_properties(${app} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endforeach()
