
add_custom_command(
    OUTPUT _core.c
    COMMAND Python::Interpreter -m cython
          "${CMAKE_CURRENT_SOURCE_DIR}/_core.pyx" --output-file _core.c
    DEPENDS _core.pyx
)

python_add_library(_core MODULE _core.c WITH_SOABI)

target_link_libraries(_core
    PUBLIC
    tinn
    genann
    floatfann
    doublefann
)

target_include_directories(_core
    PUBLIC
    ${CMAKE_SOURCE_DIR}/thirdparty/tinn
    ${CMAKE_SOURCE_DIR}/thirdparty/genann
    ${CMAKE_SOURCE_DIR}/thirdparty/fann
    ${CMAKE_SOURCE_DIR}/thirdparty/fann/src/include
    ${CMAKE_BINARY_DIR}/thirdparty/fann/src/include
)


install(TARGETS _core DESTINATION cynn)
