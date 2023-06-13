#ifndef ALGAE_CUH
#define ALGAE_CUH

#include <simulation/structs/allocator.h>
#include <cstdint>

#ifdef THRUST_IMPL

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct Algae {
	struct Host {
		thrust::host_vector<uint64_t> count;
		thrust::host_vector<float2> positions;
		thrust::host_vector<float2> directionVecs;
		thrust::host_vector<bool> alives;
		thrust::host_vector<float> currentEnergy;
	} host;
	struct Device {
		thrust::device_vector<uint64_t> count;
		thrust::device_vector<float2> positions;
		thrust::device_vector<float2> directionVecs;
		thrust::device_vector<bool> alives;
		thrust::device_vector<float> currentEnergy;
	} device;
	const uint64_t capacity;

	Algae(uint64_t capacity) : capacity(capacity) {
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
	}
};
#else
#include "cuda/error.cuh"

enum class AlgaeDecision {
	NONE, MOVE, EAT,
};

template <class Alloc>
struct Algae {
	const uint64_t capacity;

	uint64_t* count{ nullptr };
	float2* positions{ nullptr };
	float2* directionVecs{ nullptr };
	bool* alives{ nullptr };
	float* currentEnergy{ nullptr };

	static Algae make(uint64_t capacity) {
		Algae<Alloc> f = Algae<Alloc>{capacity};
		Alloc::make((void**)&f.count, sizeof(*f.count) * 1);
		Alloc::make((void**)&f.positions, sizeof(*f.positions) * capacity);
		Alloc::make((void**)&f.directionVecs, sizeof(*f.directionVecs) * capacity);
		Alloc::make((void**)&f.alives, sizeof(*f.alives) * capacity);
		Alloc::make((void**)&f.currentEnergy, sizeof(*f.currentEnergy) * capacity);

		return f;
	}

	static void drop(Algae& f) {
		Alloc::drop(f.count);
		Alloc::drop(f.positions);
		Alloc::drop(f.directionVecs);
		Alloc::drop(f.alives);
		Alloc::drop(f.currentEnergy);
	}



private:
	Algae(uint64_t capacity) : capacity(capacity) {}
};
#endif

#endif // !ALGAE_CUH
