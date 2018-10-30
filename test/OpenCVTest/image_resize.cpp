/**
 * @file   CudaTest/mixed_precision.cpp
 * @date   Apr 16, 2018
 * @author Bernd Doser <bernd.doser@h-its.org>
 */

#include <cmath>
#include <iostream>
#include <vector>

#include "gtest/gtest.h"
#include "opencv2/opencv.hpp"

TEST(opencv, image_resize)
{
//    const std::vector<std::vector<int>> src{{1, 2}, {0, 8}};
//    std::vector<std::vector<int>> dst(4, {4});

    cv::Mat src(3, 3, CV_32F), dst;

    //cv::resize(src, dst, cv::Size2i(2, 2), 2.0, 2.0, cv::INTER_LINEAR);
    cv::resize(src, dst, cv::Size(), 0.5, 0.5, cv::INTER_LINEAR);
}
