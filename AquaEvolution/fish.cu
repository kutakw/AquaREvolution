#include <simulation/structs/fish.cuh>

#include <thrust/iterator/zip_iterator.h>

thrust::tuple<thrust::zip_iterator<Fish::EntityIter>, thrust::zip_iterator<Fish::EntityIter>> Fish::Device::iter() {
	auto begin = thrust::make_zip_iterator(
		thrust::make_tuple(
		positions.begin(),
		directionVecs.begin(),
		alives.begin(),
		currentEnergy.begin(),
		nextDecisions.begin(),
		eatenAlgaeId.begin(),
		energyParams.begin(),
		sightParams.begin(),
		velocity.begin()
	));
	auto end = thrust::make_zip_iterator(
		thrust::make_tuple(
		positions.end(),
		directionVecs.end(),
		alives.end(),
		currentEnergy.end(),
		nextDecisions.end(),
		eatenAlgaeId.end(),
		energyParams.end(),
		sightParams.end(),
		velocity.end()
	));

	return thrust::make_tuple(begin, end);
}
