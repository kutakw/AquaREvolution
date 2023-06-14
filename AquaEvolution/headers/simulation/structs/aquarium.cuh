#ifndef AQUARIUM_CUH
#define AQUARIUM_CUH

#include <simulation/structs/fish.cuh>
#include <simulation/structs/algae.cuh>
#include <cuda/helper_math.cuh>


#ifdef THRUST_IMPL

//#include <thrust/random.h>
//#include <thrust/generate.h>
//#include <thrust/sequence.h>
//#include <thrust/execution_policy.h>
//#include <thrust/async/transform.h>
//#include <thrust/async/sort.h>
//
//struct Float2Generator {
//	
//	float a, b;
//
//	__host__ __device__
//		Float2Generator(float _a = 0.f, float _b = 1.f) : a(_a), b(_b) {};
//
//	__host__ __device__
//		float2 operator()(const unsigned int n) const
//	{
//		thrust::default_random_engine rng;
//		thrust::uniform_real_distribution<float> dist(a, b);
//		rng.discard(n);
//
//		float2 res = normalize(float2{ dist(rng), dist(rng) });
//		return res;
//	}
//};

struct Aquarium {
	static constexpr uint64_t FISH_MAX_COUNT = 10000;
	static constexpr uint64_t ALGAE_MAX_COUNT = 50000;
	static constexpr float WIDTH = 100.f;
	static constexpr float HEIGHT = 100.f;
	static constexpr uint64_t FISH_START = 1;
	static constexpr uint64_t ALGAE_START = 1;

	Fish fish;
	Algae algae;

	Aquarium() :
		fish(FISH_MAX_COUNT),
		algae(ALGAE_MAX_COUNT)
	{}

	void generateLife();
	void simulateGeneration();
};

#else
struct Aquarium {
	static constexpr uint64_t FISH_MAX_COUNT = 10000;
	static constexpr uint64_t ALGAE_MAX_COUNT = 50000;
	static constexpr float WIDTH = 100.f;
	static constexpr float HEIGHT = 100.f;

	Fish<Device> d_fish;
	Fish<Host> h_fish;
	Algae<Device> d_algae;
	Algae<Host> h_algae;

	Aquarium() :
		d_fish(Fish<Device>::make(FISH_MAX_COUNT)),
		h_fish(Fish<Host>::make(FISH_MAX_COUNT)),
		d_algae(Algae<Device>::make(ALGAE_MAX_COUNT)),
		h_algae(Algae<Host>::make(ALGAE_MAX_COUNT))
	{
		*h_fish.count = 1;
		h_fish.positions[0] = make_float2(50.0f, 50.0f);
		h_fish.directionVecs[0] = make_float2(1.0f, 0.0f);
		h_fish.alives[0] = true;
	}

	~Aquarium() {
		Fish<Device>::drop(d_fish);
		Fish<Host>::drop(h_fish);
		Fish<Device>::drop(d_fish);
		Fish<Host>::drop(h_fish);
	}
};
#endif

#endif // !AQUARIUM_CUH
