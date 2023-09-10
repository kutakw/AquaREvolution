#include <simulation/structs/algae.cuh>

thrust::tuple<thrust::zip_iterator<Algae::EntityIter>, thrust::zip_iterator<Algae::EntityIter>> Algae::Device::iter() {
	auto begin = thrust::make_zip_iterator(
		thrust::make_tuple(
		positions.begin(),
		directionVecs.begin(),
		alives.begin(),
		currentEnergy.begin()
	));
	auto end = thrust::make_zip_iterator(
		thrust::make_tuple(
		positions.end(),
		directionVecs.end(),
		alives.end(),
		currentEnergy.end()
	));

	return thrust::make_tuple(begin, end);
}
