#ifndef AQUARIUM_CUH
#define AQUARIUM_CUH

#include <simulation/structs/fish.cuh>
#include <simulation/structs/algae.cuh>
#include <simulation/structs/mutation.cuh>
#include <cuda/helper_math.cuh>


#include <thrust/device_vector.h>
#include "config.h"

struct Aquarium {

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
