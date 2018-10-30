include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/src
    ${GTEST_INCLUDE_DIR}
)

add_executable(
    OpenCVTest
    main.cpp
    image_resize.cpp
    #UpdateNeuronsTest.cpp
)
    
target_link_libraries(
    OpenCVTest
    ${GTEST_BOTH_LIBRARIES}
    ${CONAN_LIBS}
)

add_test(
    OpenCVTest
    ${CMAKE_BINARY_DIR}/bin/OpenCVTest
    --gtest_output=xml:${CMAKE_BINARY_DIR}/Testing/OpenCVTest.xml
)
