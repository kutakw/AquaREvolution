#ifndef MUTATION_CUH
#define MUTATION_CUH

#include <simulation/structs/allocator.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct Mutation {
	using Entity = entity<float, float, float, float, float>;
	using Entity = entityIter<float, float, float, float, float>;
};


#endif // MUTATION_CUH