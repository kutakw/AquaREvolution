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
	static constexpr float MAX_ENERGY = 50.0f;
	static constexpr float INITAL_ENERGY = 25.0f;
	static constexpr float ENERGY_PER_KID = 10.0f;
	static constexpr float ENERGY_MINIMUM_TO_REPRODUCT = 15.0f;
	static constexpr float ENERGY_PER_ALGA_EATEN = 1.0f;
	static constexpr float SIGHT_DIST = 10.0f;
	static constexpr float SIGHT_ANGLE = 0.0f;
	static constexpr float VELOCITY = 2e-3f;
	static constexpr float ENERGY_DECAY_RATE = 0.1f;

	using Entity = entity<float2, float2, bool, float, FishDecisionEnum, uint64_t, float2, float2, float>;
	using EntityIter = entityIter<float2, float2, bool, float, FishDecisionEnum, uint64_t, float2, float2, float>;

	struct Host {
		thrust::host_vector<float2> positions;
		thrust::host_vector<float2> directionVecs;
		thrust::host_vector<bool> alives;
		thrust::host_vector<float> currentEnergy;
		thrust::host_vector<FishDecisionEnum> nextDecisions;
		thrust::host_vector<uint64_t> eatenAlgaeId;
		thrust::host_vector<float2> energyParams; // max, decay
		thrust::host_vector<float2> sightParams; // dist, angle
		thrust::host_vector<float> velocity;
	} host;
	struct Device {
		thrust::device_vector<float2> positions;
		thrust::device_vector<float2> directionVecs;
		thrust::device_vector<bool> alives;
		thrust::device_vector<float> currentEnergy;
		thrust::device_vector<FishDecisionEnum> nextDecisions;
		thrust::device_vector<uint64_t> eatenAlgaeId;
		thrust::device_vector<float2> energyParams; // max, decay
		thrust::device_vector<float2> sightParams; // dist, angle
		thrust::device_vector<float> velocity;

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
		dest.energyParams = src.energyParams;
		dest.sightParams = src.sightParams;
		dest.velocity = src.velocity;
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
		t.energyParams.reserve(capacity);
		t.sightParams.reserve(capacity);
		t.velocity.reserve(capacity);
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
		t.energyParams.resize(size);
		t.sightParams.resize(size);
		t.velocity.resize(size);
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
