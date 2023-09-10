#ifndef GENERATORS_CUH
#define GENERATORS_CUH

#include <cuda_runtime.h>
#include <cuda/helper_math.cuh>
#include <thrust/random.h>
#include "aquarium.cuh"

struct GeneratePos {
	__host__ __device__
		float2 operator()(const unsigned int n) const
	{
		thrust::default_random_engine rng;
		thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
		rng.discard(n);

		float2 res = make_float2(dist(rng) * (WIDTH - 2.0f) + 1.0f, dist(rng) * (HEIGHT - 2.0f) + 1.0f);
		return res;
	}
};

struct GenerateVector {
	__host__ __device__
		float2 operator()(const unsigned int n) const
	{
		thrust::default_random_engine rng;
		thrust::uniform_real_distribution<float> dist(0.0f, 1.0f);
		rng.discard(n);

		float2 res = normalize(make_float2( dist(rng) * 2.0f - 1.0f, dist(rng) * 2.0f - 1.0f ));
		return res;
	}
};

#endif // !GENERATORS_CUH
