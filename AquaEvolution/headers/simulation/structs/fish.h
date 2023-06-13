#ifndef FISH_CUH
#define FISH_CUH

#include <simulation/structs/allocator.h>

enum class FishDecision {
	NONE, MOVE, EAT,
};

#ifdef THRUST_IMPL

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct Fish {
	struct Host {
		thrust::host_vector<uint64_t> count;
		thrust::host_vector<float2> positions;
		thrust::host_vector<float2> directionVecs;
		thrust::host_vector<bool> alives;
		thrust::host_vector<float> currentEnergy;
		thrust::host_vector<FishDecision> nextDecisions;
		thrust::host_vector<uint64_t> eatenAlgaeId;
	} host;
	struct Device {
		thrust::device_vector<uint64_t> count;
		thrust::device_vector<float2> positions;
		thrust::device_vector<float2> directionVecs;
		thrust::device_vector<bool> alives;
		thrust::device_vector<float> currentEnergy;
		thrust::device_vector<FishDecision> nextDecisions;
		thrust::device_vector<uint64_t> eatenAlgaeId;
	} device;
	const uint64_t capacity;


	Fish(uint64_t capacity) : capacity(capacity) {
		reserve(host, capacity);
		host.count.push_back(0);
		reserve(device, capacity);
		device.count = host.count;
	}

private:
	template <class T>
	void reserve(T& t, uint64_t capacity)
	{
		t.count.reserve(1);
		t.positions.reserve(capacity);
		t.directionVecs.reserve(capacity);
		t.alives.reserve(capacity);
		t.currentEnergy.reserve(capacity);
		t.nextDecisions.reserve(capacity);
		t.eatenAlgaeId.reserve(capacity);
	}

};
#else
#include "cuda/helper_math.cuh"
#include <cstdint>

template <class Alloc>
struct Fish {
	const uint64_t capacity;

	uint64_t* count{ nullptr };
	float2* positions{ nullptr };
	float2* directionVecs{ nullptr };
	bool* alives{ nullptr };
	float* currentEnergy{ nullptr };
	FishDecision* nextDecisions{ nullptr };
	uint64_t* eatenAlgaeId{ nullptr };

	static Fish make(uint64_t capacity){
		Fish<Alloc> f = Fish<Alloc>{ capacity };
		Alloc::make((void**)&f.count, sizeof(*f.count) * 1);
		Alloc::make((void**)&f.positions, sizeof(*f.positions) * capacity);
		Alloc::make((void**)&f.directionVecs, sizeof(*f.directionVecs) * capacity);
		Alloc::make((void**)&f.alives, sizeof(*f.alives) * capacity);
		Alloc::make((void**)&f.currentEnergy, sizeof(*f.currentEnergy) * capacity);
		Alloc::make((void**)&f.nextDecisions, sizeof(*f.nextDecisions) * capacity);
		Alloc::make((void**)&f.eatenAlgaeId, sizeof(*f.eatenAlgaeId) * capacity);

		return f;
	}

	static void drop(Fish& f) {
		Alloc::drop(f.count);
		Alloc::drop(f.positions);
		Alloc::drop(f.directionVecs);
		Alloc::drop(f.alives);
		Alloc::drop(f.currentEnergy);
		Alloc::drop(f.nextDecisions);
		Alloc::drop(f.eatenAlgaeId);
	}

private:
	Fish(uint64_t capacity) : capacity(capacity) {}

};
#endif

#endif // !FISH_CUH
