/**
 * @file   SelfOrganizingMapLib/Trainer.h
 * @date   Sep 10, 2018
 * @author Bernd Doser, HITS gGmbH
 */

#pragma once

#include <iostream>

#include "ImageProcessingLib/ImageRotator.h"

namespace pink {

class Trainer
{
public:

	Trainer(int verbosity = 0, int number_of_rotations = 360, float progress_factor = 0.1,
        bool use_flip = true, bool use_cuda = true, bool write_rot_flip = false)
     : verbosity(verbosity),
	   number_of_rotations(number_of_rotations),
	   progress_factor(progress_factor),
	   use_flip(use_flip),
	   use_cuda(use_cuda),
	   write_rot_flip(write_rot_flip)
    {}

    template <typename SOMType>
	void operator () (SOMType& som, typename SOMType::value_type const& image) const
	{
//		auto&& list_of_spatial_transformed_images = SpatialTransformer(Rotation<0,1>(number_of_rotations), use_flip)(image);
//		auto&& [euclidean_distance] generate_euclidean_distance_matrix(som, list_of_spatial_transformed_images);
//
//		auto&& best_match = find_best_match();
//
//		update_counter(best_match);
//		update_neurons(best_match);
	}

private:

	int verbosity;
	int number_of_rotations;
	float progress_factor;
	bool use_flip;
	bool use_cuda;
	bool write_rot_flip;

};

} // namespace pink