#include <simulation/structs/mutation.cuh>
#include<thrust/iterator/zip_iterator.h>

thrust::tuple<thrust::zip_iterator<Mutation::EntityIter>, thrust::zip_iterator<Mutation::EntityIter>> Mutation::Device::iter() {
	auto begin = thrust::make_zip_iterator(
		thrust::make_tuple(
			energyAlteration.begin(),
			sightAlteration.begin(),
			velocityAlteration.begin()
		));
	auto end = thrust::make_zip_iterator(
		thrust::make_tuple(
			energyAlteration.end(),
			sightAlteration.end(),
			velocityAlteration.end()
		));

	return thrust::make_tuple(begin, end);
}

