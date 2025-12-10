# _core module (tinn, genann, fann, nn1)
add_custom_command(
    OUTPUT _core.c
    COMMAND Python::Interpreter -m cython
          "${CMAKE_CURRENT_SOURCE_DIR}/_core.pyx" --output-file _core.c
    DEPENDS _core.pyx cnn.pxd
)

python_add_library(_core MODULE _core.c WITH_SOABI)

target_link_libraries(_core
    PUBLIC
    tinn
    genann
    floatfann
    doublefann
    nn1
)

target_include_directories(_core
    PUBLIC
    ${CMAKE_SOURCE_DIR}/thirdparty/tinn
    ${CMAKE_SOURCE_DIR}/thirdparty/genann
    ${CMAKE_SOURCE_DIR}/thirdparty/fann
    ${CMAKE_SOURCE_DIR}/thirdparty/fann/src/include
    ${CMAKE_BINARY_DIR}/thirdparty/fann/src/include
    ${CMAKE_SOURCE_DIR}/thirdparty/nn1
)

install(TARGETS _core DESTINATION cynn)

# kann module (KANN neural network library)
add_custom_command(
    OUTPUT kann.c
    COMMAND Python::Interpreter -m cython
          -I "${CMAKE_CURRENT_SOURCE_DIR}"
          "${CMAKE_CURRENT_SOURCE_DIR}/kann.pyx" --output-file kann.c
    DEPENDS kann.pyx kann.pxd
)

python_add_library(kann MODULE kann.c WITH_SOABI)

target_link_libraries(kann
    PUBLIC
    kann_lib
)

target_include_directories(kann
    PUBLIC
    ${CMAKE_SOURCE_DIR}/thirdparty/kann
)

install(TARGETS kann DESTINATION cynn)
