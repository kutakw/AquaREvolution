#ifndef SORT_CUH
#define SORT_CUH

#include <simulation/structs/aquarium.cuh>

struct AlgaeKeysFunctor {

	__device__ __host__
		uint64_t operator()(const float2& f) {
		return uint64_t(f.x * CELL.x / WIDTH) + uint64_t(f.y * CELL.y / HEIGHT) * CELL.x;
	}
};

struct SortAlgaePredic {
	using tup = thrust::tuple<uint64_t, bool>;

	__device__
		bool operator()(const tup& a, const tup& b) {
		if (a.tail != b.tail) return a.tail > b.tail;
		return a.head < b.head;
	}
};

struct BucketIndexFunctor {
	using tup = thrust::tuple<uint64_t, uint64_t>;

	uint64_t* out_vals;
	uint64_t* bucket;

	BucketIndexFunctor(
		thrust::device_vector<uint64_t>& out_vals_,
		thrust::device_vector<uint64_t>& bucket_
		) :
		out_vals(out_vals_.data().get()),
		bucket(bucket_.data().get()) 
	{}

	__device__
		void operator()(const tup& t) {
		bucket[t.get<0>()] = out_vals[t.get<1>()];
	}
};

__global__
void linear_sum(uint64_t* dest, uint64_t* src, uint64_t size);

#endif // !SORT_CUH
