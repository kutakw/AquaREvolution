#ifndef AQUARIUM_CUH
#define AQUARIUM_CUH

#include <simulation/structs/fish.cuh>
#include <simulation/structs/algae.cuh>
#include <simulation/structs/mutation.cuh>
#include <cuda/helper_math.cuh>


#include <thrust/device_vector.h>

struct Aquarium {
	static constexpr uint64_t FISH_MAX_COUNT = 10000;
	static constexpr uint64_t ALGAE_MAX_COUNT = 30000;

	static constexpr float WIDTH = 100.f;
	static constexpr float HEIGHT = 100.f;
	
	static constexpr uint64_t FISH_START = 100;
	static constexpr uint64_t ALGAE_START = 1000;

	static constexpr ulonglong2 CELL = { 100, 100 };

	static constexpr int32_t ITER_PER_GENERATION = 1000;

	int currentAlgaeBuffer = 0;
	int currentFishBuffer = 0;
	Fish fishBuffer[2]{ Fish(FISH_MAX_COUNT), Fish(FISH_MAX_COUNT) };
	Algae algaeBuffer[2]{ Algae(ALGAE_MAX_COUNT), Algae(ALGAE_MAX_COUNT) };

	Fish* fish;
	Algae* algae;
	Mutation mutation;

	thrust::device_vector<uint64_t> algaeKeys;
	thrust::device_vector<uint64_t> algaeBucketIds;

	Aquarium();

	void generateLife();
	void generateMutations();
	void simulateGeneration();
private:
	void decision();
	void move();

	void sort_algae();

	void reproduction_algae();
	void reproduction_fish();
		
};
#endif // !AQUARIUM_CUH
