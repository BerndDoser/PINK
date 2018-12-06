/**
 * @file   UtilitiesLib/InputData.h
 * @date   Nov 3, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <string>

#include "ImageProcessingLib/ImageProcessing.h"
#include "ImageProcessingLib/Interpolation.h"
#include "IntermediateStorageType.h"
#include "SOMInitializationType.h"
#include "UtilitiesLib/DataType.h"
#include "UtilitiesLib/DistributionFunction.h"
#include "UtilitiesLib/ExecutionPath.h"
#include "UtilitiesLib/Layout.h"
#include "Version.h"

namespace pink {

#define DEFAULT_SIGMA     1.1
#define DEFAULT_DAMPING   0.2

struct InputData
{
    /// Default constructor.
    InputData();

    /// Constructor reading input data from arguments.
    InputData(int argc, char **argv);

    /// Print program header.
    void print_header() const;

    /// Print input data.
    void print_parameters() const;

    /// Print usage output for input arguments.
    void print_usage() const;

    std::string imagesFilename;
    std::string resultFilename;
    std::string somFilename;
    std::string rot_flip_filename;

    /// SOM layout
    Layout layout;

    /// List of SOM dimensions in each direction
    std::vector<uint32_t> som_dimensions;

    /// Number of neurons
    uint32_t som_size;

    /// List of neuron dimensions in each direction
    std::vector<uint32_t> neuron_dimensions;

    /// Be more talkative
    bool verbose;

    /// Random number seed
    int seed;

    uint32_t number_of_rotations;

    uint32_t number_of_threads;

    SOMInitialization init;
    uint32_t number_of_iterations;

    float progressFactor;
    bool use_flip;
    bool use_gpu;
    uint32_t number_of_images;
    uint32_t number_of_channels;
    uint32_t image_dim;
    int image_size;
    int neuron_size;
    int som_total_size;
    Interpolation interpolation;
    ExecutionPath executionPath;
    IntermediateStorageType intermediate_storage;
    DistributionFunction function;
    float sigma;
    float damping;
    int block_size_1;
    float max_update_distance;
    int useMultipleGPUs;
    int usePBC;
    int dimensionality;
    bool write_rot_flip;
    DataType euclidean_distance_type;
};

void stringToUpper(char* s);

} // namespace pink
