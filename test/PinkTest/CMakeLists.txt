include_directories(
    ${PROJECT_SOURCE_DIR}/src
    ${GTEST_INCLUDE_DIR}
)

add_executable(
    PinkTest
    main.cpp
    PinkTest.cpp
)
    
target_link_libraries(
    PinkTest
    ${GTEST_BOTH_LIBRARIES}
)

if(PINK_USE_CUDA)
    target_link_libraries(
        PinkTest
        CudaLib
    )
endif()

ADD_TEST(
    PinkTest
    ${CMAKE_BINARY_DIR}/bin/PinkTest
    --gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/PinkTest.xml
)
