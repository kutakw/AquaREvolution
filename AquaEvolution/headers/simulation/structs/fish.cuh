#ifndef FISH_CUH
#define FISH_CUH

#include <simulation/structs/allocator.cuh>

enum class FishDecisionEnum {
	NONE, MOVE, EAT,
};

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct Fish {
	static constexpr float MAX_ENERGY = 50.0f;
	static constexpr float INITAL_ENERGY = 30.0f;
	static constexpr float ENERGY_PER_KID = 10.0f;
	static constexpr float ENERGY_MINIMUM_TO_REPRODUCT = 5.0f;
	static constexpr float ENERGY_PER_ALGA_EATEN = 3.0f;
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
	friend std::ostream& operator<<(std::ostream& stream, const Fish& fish);

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
#endif // !FISH_CUH
