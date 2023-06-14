#ifndef FISH_CUH
#define FISH_CUH

#include <simulation/structs/allocator.cuh>

enum class FishDecisionEnum {
	NONE, MOVE, EAT,
};

#ifdef THRUST_IMPL

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct Fish {

	using Entity = entity<float2, float2, bool, float, FishDecisionEnum, uint64_t>;
	using EntityIter = entityIter<float2, float2, bool, float, FishDecisionEnum, uint64_t>;

	struct Host {
		thrust::host_vector<float2> positions;
		thrust::host_vector<float2> directionVecs;
		thrust::host_vector<bool> alives;
		thrust::host_vector<float> currentEnergy;
		thrust::host_vector<FishDecisionEnum> nextDecisions;
		thrust::host_vector<uint64_t> eatenAlgaeId;

	} host;
	struct Device {
		thrust::device_vector<float2> positions;
		thrust::device_vector<float2> directionVecs;
		thrust::device_vector<bool> alives;
		thrust::device_vector<float> currentEnergy;
		thrust::device_vector<FishDecisionEnum> nextDecisions;
		thrust::device_vector<uint64_t> eatenAlgaeId;

		thrust::tuple<thrust::zip_iterator<EntityIter>, thrust::zip_iterator<EntityIter>> iter();
	} device;
	const uint64_t capacity;


	Fish(uint64_t capacity) : capacity(capacity) {
		reserve(host, capacity);
		reserve(device, capacity);
	}

	template <typename S, typename D>
	void update(D& dest, S& src) {
		dest.positions = src.positions;
		dest.directionVecs = src.directionVecs;
		dest.alives = src.alives;
		dest.currentEnergy = src.currentEnergy;
		dest.nextDecisions = src.nextDecisions;
		dest.eatenAlgaeId = src.eatenAlgaeId;
	}

public:
	template <class T>
	void reserve(T& t, uint64_t capacity)
	{
		t.positions.reserve(capacity);
		t.directionVecs.reserve(capacity);
		t.alives.reserve(capacity);
		t.currentEnergy.reserve(capacity);
		t.nextDecisions.reserve(capacity);
		t.eatenAlgaeId.reserve(capacity);
	}

	template <class T>
	void resize(T& t, uint64_t size)
	{
		t.positions.resize(size);
		t.directionVecs.resize(size);
		t.alives.resize(size);
		t.currentEnergy.resize(size);
		t.nextDecisions.resize(size);
		t.eatenAlgaeId.resize(size);
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
	FishDecisionEnum* nextDecisions{ nullptr };
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
