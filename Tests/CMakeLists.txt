set(test_libs
        ${GTEST_LIBRARIES}     
        pthread
)

add_executable(matrix_test
        ${algebra_src}
        MatrixTests.cpp)

target_link_libraries(matrix_test ${test_libs})

add_executable(synkhorn_test
        ${algebra_src}
        SynkhornTests.cpp)

target_link_libraries(synkhorn_test ${test_libs})