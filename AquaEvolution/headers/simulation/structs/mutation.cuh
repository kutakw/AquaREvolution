#ifndef MUTATION_CUH
#define MUTATION_CUH

#include <simulation/structs/allocator.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct Mutation {

	static constexpr int MUTATION_COUNT = 5;

	using Entity = entity<float2, float2, float>;
	using EntityIter = entityIter<float2, float2, float>;

	struct Host {
		thrust::host_vector<float2> energyAlteration;
		thrust::host_vector<float2> sightAlteration;
		thrust::host_vector<float> velocityAlteration;
	} host;	
	struct Device {
		thrust::device_vector<float2> energyAlteration; // max, decay
		thrust::device_vector<float2> sightAlteration; // dist, angle
		thrust::device_vector<float> velocityAlteration;

		thrust::tuple<thrust::zip_iterator<EntityIter>, thrust::zip_iterator<EntityIter>>iter();
	} device;
	const uint64_t capacity;

	Mutation(uint64_t capacity) :capacity(capacity)
	{
		reserve(host, capacity);
		reserve(device, capacity);
	}

	template <typename S, typename D>
	void update(D& dest, S& src) {
		dest.energyAlteration = src.energyAlteration;
		dest.sightAlteration = src.sightAlteration;
		dest.velocityAlteration = src.velocityAlteration;
	}

public:
	template <class T>
	void reserve(T& t, uint64_t capacity)
	{
		t.energyAlteration.reserve(capacity);
		t.sightAlteration.reserve(capacity);
		t.velocityAlteration.reserve(capacity);
	}

	template <class T>
	void resize(T& t, uint64_t size)
	{
		t.energyAlteration.resize(size);
		t.sightAlteration.resize(size);
		t.velocityAlteration.resize(size);
	}
};


#endif // MUTATION_CUH