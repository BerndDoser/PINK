/**
 * @file   CudaLib/generate_euclidean_distance_matrix_first_step_multi_gpu.h
 * @date   Oct 30, 2014
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <omp.h>
#include <thread>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <vector>

#include "euclidean_distance_kernel.h"

namespace pink {

/// Calculate euclidean distance on multiple GPU devices
template <typename DataType, typename EuclideanType>
void generate_euclidean_distance_matrix_first_step_multi_gpu(std::vector<thrust::device_vector<EuclideanType>> const& d_som,
	std::vector<thrust::device_vector<EuclideanType>> const& d_rotated_images, std::vector<thrust::device_vector<DataType>>& d_first_step,
    uint32_t number_of_spatial_transformations, uint32_t som_size, uint32_t neuron_size, uint16_t block_size,
	std::vector<int> const& size, std::vector<int> const& offset)
{
    auto&& gpu_ids = cuda_get_gpu_ids();
    int number_of_gpus = gpu_ids.size();

    for (int i = 1; i < number_of_gpus; ++i)
    {
        // Set GPU device
        cudaSetDevice(gpu_ids[i]);

        // Copy data
        gpuErrchk(cudaMemcpyPeer(thrust::raw_pointer_cast(d_som[i].data()), i,
                            thrust::raw_pointer_cast(d_som[0].data()) + offset[i] * neuron_size, 0,
                            size[i] * neuron_size * sizeof(EuclideanType)));

        gpuErrchk(cudaMemcpyPeer(thrust::raw_pointer_cast(d_rotated_images[i].data()), i,
                            thrust::raw_pointer_cast(d_rotated_images[0].data()), 0,
                            number_of_spatial_transformations * neuron_size * sizeof(EuclideanType)));

        gpuErrchk(cudaDeviceSynchronize());
    }

    std::vector<std::thread> workers;
    for (int i = 1; i < number_of_gpus; ++i)
    {
        workers.push_back(std::thread([&, i]()
        {
            // Set GPU device
            cudaSetDevice(gpu_ids[i]);

            // Setup execution parameters
            dim3 dim_block(block_size);
            dim3 dim_grid(number_of_spatial_transformations, size[i]);

            switch (block_size)
            {
                case  512: euclidean_distance_kernel< 512><<<dim_grid, dim_block>>>(
                    thrust::raw_pointer_cast(d_som[i].data()), thrust::raw_pointer_cast(d_rotated_images[i].data()),
                    thrust::raw_pointer_cast(d_first_step[i].data()), neuron_size); break;
                case  256: euclidean_distance_kernel< 256><<<dim_grid, dim_block>>>(
                    thrust::raw_pointer_cast(d_som[i].data()), thrust::raw_pointer_cast(d_rotated_images[i].data()),
                    thrust::raw_pointer_cast(d_first_step[i].data()), neuron_size); break;
                case  128: euclidean_distance_kernel< 128><<<dim_grid, dim_block>>>(
                    thrust::raw_pointer_cast(d_som[i].data()), thrust::raw_pointer_cast(d_rotated_images[i].data()),
                    thrust::raw_pointer_cast(d_first_step[i].data()), neuron_size); break;
                case   64: euclidean_distance_kernel<  64><<<dim_grid, dim_block>>>(
                    thrust::raw_pointer_cast(d_som[i].data()), thrust::raw_pointer_cast(d_rotated_images[i].data()),
                    thrust::raw_pointer_cast(d_first_step[i].data()), neuron_size); break;
                default:
                    throw pink::exception("generate_euclidean_distance_matrix_first_step_multi_gpu: block size not supported");
            }
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        }));
    }

    // Set GPU device to master
    cudaSetDevice(gpu_ids[0]);

    // Setup execution parameters
    dim3 dim_block(block_size);
    dim3 dim_grid(number_of_spatial_transformations, size[0]);

    switch (block_size)
    {
        case  512: euclidean_distance_kernel< 512><<<dim_grid, dim_block>>>(
            thrust::raw_pointer_cast(d_som[0].data()), thrust::raw_pointer_cast(d_rotated_images[0].data()),
            thrust::raw_pointer_cast(d_first_step[0].data()), neuron_size); break;
        case  256: euclidean_distance_kernel< 256><<<dim_grid, dim_block>>>(
            thrust::raw_pointer_cast(d_som[0].data()), thrust::raw_pointer_cast(d_rotated_images[0].data()),
            thrust::raw_pointer_cast(d_first_step[0].data()), neuron_size); break;
        case  128: euclidean_distance_kernel< 128><<<dim_grid, dim_block>>>(
            thrust::raw_pointer_cast(d_som[0].data()), thrust::raw_pointer_cast(d_rotated_images[0].data()),
            thrust::raw_pointer_cast(d_first_step[0].data()), neuron_size); break;
        case   64: euclidean_distance_kernel<  64><<<dim_grid, dim_block>>>(
            thrust::raw_pointer_cast(d_som[0].data()), thrust::raw_pointer_cast(d_rotated_images[0].data()),
            thrust::raw_pointer_cast(d_first_step[0].data()), neuron_size); break;
        default:
            throw pink::exception("generate_euclidean_distance_matrix_first_step_multi_gpu: block size not supported");
    }
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Wait for all workers
    for (auto&& w : workers) w.join();

    for (int i = 1; i < number_of_gpus; ++i)
    {
        // Copy data
		gpuErrchk(cudaMemcpyPeer(thrust::raw_pointer_cast(d_first_step[0].data()) + offset[i] * number_of_spatial_transformations, 0,
			thrust::raw_pointer_cast(d_first_step[i].data()), i, size[i] * number_of_spatial_transformations * sizeof(DataType)));
	}

    gpuErrchk(cudaDeviceSynchronize());
}

} // namespace pink
