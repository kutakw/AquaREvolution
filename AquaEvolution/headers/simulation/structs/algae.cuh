#ifndef ALGAE_CUH
#define ALGAE_CUH

#include <simulation/structs/allocator.cuh>
#include <cstdint>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct Algae {

	using Entity = entity<float2, float2, bool, float>;
	using EntityIter = entityIter<float2, float2, bool, float>;
	
	struct Host {
		thrust::host_vector<float2> positions;
		thrust::host_vector<float2> directionVecs;
		thrust::host_vector<bool> alives;
		thrust::host_vector<float> currentEnergy;
	} host;
	struct Device {
		thrust::device_vector<float2> positions;
		thrust::device_vector<float2> directionVecs;
		thrust::device_vector<bool> alives;
		thrust::device_vector<float> currentEnergy;

		thrust::tuple<thrust::zip_iterator<EntityIter>, thrust::zip_iterator<EntityIter>> iter();
	} device;
	const uint64_t capacity;

	Algae(uint64_t capacity) : capacity(capacity) {
		reserve(host, capacity);
		reserve(device, capacity);
	}

	template <typename S, typename D>
	void update(D& dest, S& src) {
		dest.positions = src.positions;
		dest.directionVecs = src.directionVecs;
		dest.alives = src.alives;
		dest.currentEnergy = src.currentEnergy;
	}

public:
	template <class T>
	void reserve(T& t, uint64_t capacity)
	{
		t.positions.reserve(capacity);
		t.directionVecs.reserve(capacity);
		t.alives.reserve(capacity);
		t.currentEnergy.reserve(capacity);
	}

	template <class T>
	void resize(T& t, uint64_t size)
	{
		t.positions.resize(size);
		t.directionVecs.resize(size);
		t.alives.resize(size);
		t.currentEnergy.resize(size);
	}
};

#endif // !ALGAE_CUH
