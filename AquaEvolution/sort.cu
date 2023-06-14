#include <simulation/structs/sort.cuh>

#include <cuda/error.cuh>

#include <device_launch_parameters.h>

#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>

__global__ void linear_sum(uint64_t* dest, uint64_t* src, uint64_t size) {
	uint32_t id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id) return;
	for (uint32_t i = 1; i < size; ++i) {
		dest[i] = src[i - 1] + dest[i - 1];
	}
}

struct Accumulator {
	__device__ uint64_t operator()(const bool& x) {
		return x ? 1 : 0;
	}
};

void Aquarium::sort_algae()
{
	int N = algae->device.positions.size();

	thrust::transform(
		thrust::device, 
		algae->device.positions.begin(), 
		algae->device.positions.end(), 
		algaeKeys.begin(), 
		AlgaeKeysFunctor());

	auto begin = thrust::make_zip_iterator(thrust::make_tuple(
		algaeKeys.begin(),
		algae->device.alives.begin()
	));
	auto end = thrust::make_zip_iterator(thrust::make_tuple(
		algaeKeys.begin() + N,
		algae->device.alives.begin() + N
	));

	thrust::sort_by_key(begin, end,
		thrust::make_zip_iterator(thrust::make_tuple(
			algae->device.alives.begin(),
			algae->device.currentEnergy.begin(),
			algae->device.directionVecs.begin(),
			algae->device.positions.begin(),
			algaeKeys.begin()
		)), 
		SortAlgaePredic());

	uint64_t alive = thrust::transform_reduce(algae->device.alives.begin(), algae->device.alives.end(), Accumulator(), (uint64_t)0, thrust::plus<uint64_t>());
	algae->resize(algae->device, alive);

	auto constIter = thrust::make_constant_iterator<int32_t>(1);
	auto countIter = thrust::make_counting_iterator<uint64_t>(0);

	N = alive;

	thrust::device_vector<uint64_t> out_keys(algaeBucketIds.size());
	thrust::device_vector<uint64_t> out_vals(algaeBucketIds.size());
	auto new_end = thrust::reduce_by_key(algaeKeys.begin(), algaeKeys.begin() + N, constIter, out_keys.begin(), out_vals.begin());
	int dist = thrust::distance(out_keys.begin(), new_end.first);

	thrust::fill(algaeBucketIds.begin(), algaeBucketIds.end(), 0);
	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(out_keys.begin(), countIter)),
		thrust::make_zip_iterator(thrust::make_tuple(new_end.first, countIter + dist)),
		BucketIndexFunctor(out_vals, algaeBucketIds));

	thrust::device_vector<uint64_t> result(algaeBucketIds.size(), 0);

	linear_sum << <1, 1 >> > (result.data().get(), algaeBucketIds.data().get(), algaeBucketIds.size());
	checkCudaErrors(cudaDeviceSynchronize());
	algaeBucketIds = result;
}

