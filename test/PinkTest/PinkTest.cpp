/**
 * @file   ImageProcessingTest/ImageProcessingTest.cpp
 * @brief  Regression tests for main pink executable
 * @date   Jul 3, 2019
 * @author Bernd Doser, HITS gGmbH
 */

#include <gtest/gtest.h>

#include "UtilitiesLib/InputData.h"
#include "CudaLib/main_gpu.h"

using namespace pink;

TEST(PinkTest, help)
{
	char *argv[] = {strdup("pink"), strdup("-h")};
	int argc = sizeof(argv) / sizeof(char*) - 1;

	InputData input_data(argc, argv);
	main_gpu(input_data);
}
