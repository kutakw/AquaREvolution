#ifndef CONFIG_H
#define CONFIG_H

#include <cuda/helper_math.cuh>
#include <thrust/device_vector.h>

// Graphical display of current simulation state
constexpr bool DISPLAY = true;
// Number of generations between displays
constexpr uint32_t DISPLAY_FREQ = 1;
// Data file dumping frequency (one dump per DUMP_FREQ number of generations)
static constexpr int32_t DUMP_FREQ = 1;

// Environment dimensions
constexpr float WIDTH = 100.f;
constexpr float HEIGHT = 100.f;

// Number of objects present during initialization
constexpr uint64_t FISH_START = 100;
constexpr uint64_t ALGAE_START = 1000;

// Cell dimensions
constexpr ulonglong2 CELL = { 100, 100 };

// Duration single generation simulation
constexpr int32_t ITER_PER_GENERATION = 1000;

// Alage parameters
constexpr uint64_t ALGAE_MAX_COUNT = 30000;
constexpr float ALGAE_INIT_ENERGY = 25.0f;
constexpr float ALGAE_MAX_ENERGY = 50.0f;
constexpr float ALGAE_ENERGY_LOSS = 0.1f;
constexpr float ALGAE_ENERGY_MINIMUM_TO_REPRODUCT = 10.0f;
constexpr float ALGAE_ENERGY_PER_KID = 10.0f;
constexpr float ALGAE_VELOCITY = 1e-3f;

// Fish parameters
constexpr uint64_t FISH_MAX_COUNT = 10000;
constexpr float FISH_MAX_ENERGY = 50.0f;
constexpr float FISH_INITAL_ENERGY = 30.0f;
constexpr float FISH_ENERGY_PER_KID = 10.0f;
constexpr float FISH_ENERGY_MINIMUM_TO_REPRODUCT = 5.0f;
constexpr float FISH_ENERGY_PER_ALGA_EATEN = 3.0f;
constexpr float FISH_SIGHT_DIST = 10.0f;
constexpr float FISH_SIGHT_ANGLE = 0.0f;
constexpr float FISH_VELOCITY = 2e-3f;
constexpr float FISH_ENERGY_DECAY_RATE = 0.1f;
constexpr float FISH_MINIMUM_ENERGY_CAPACITY = 15.0f;
constexpr float FISH_MINIMUM_ENERGY_DECAY = 0.001f;

// Mutations
constexpr uint64_t MUTATION_COUNT = 6;
inline void initMutationsFromConfig(Mutation& mutation) {
	
	mutation.resize(mutation.host, MUTATION_COUNT);

	// M0 -> faster but less energy-capacity
	mutation.host.energyAlteration[0] = make_float2(0.995f, 1.0f);
	mutation.host.sightAlteration[0] = make_float2(1.0f, 0.f); // (dist -> multiplier, angle -> addition)
	mutation.host.velocityAlteration[0] = 1.005f;
	// M1 -> more enrgy capacity but slower
	mutation.host.energyAlteration[1] = make_float2(1.005f, 1.0f);
	mutation.host.sightAlteration[1] = make_float2(1.0f, 0.0f); // (dist -> multiplier, angle -> addition)
	mutation.host.velocityAlteration[1] = 0.995f;
	// M2 -> faster but more energy usage
	mutation.host.energyAlteration[2] = make_float2(1.0f, 1.005f);
	mutation.host.sightAlteration[2] = make_float2(1.0f, 0.0f); // (dist -> multiplier, angle -> addition)
	mutation.host.velocityAlteration[2] = 1.005f;
	// M3 -> slower but less energy usage
	mutation.host.energyAlteration[3] = make_float2(1.0f, 0.995f);
	mutation.host.sightAlteration[3] = make_float2(1.0f, 0.0f); // (dist -> multiplier, angle -> addition)
	mutation.host.velocityAlteration[3] = 0.995f;
	// M4 -> further sight distance but less angle
	mutation.host.energyAlteration[4] = make_float2(1.0f, 1.0f);
	mutation.host.sightAlteration[4] = make_float2(1.005f, -0.001f); // (dist -> multiplier, angle -> addition)
	mutation.host.velocityAlteration[4] = 1.0f;
	// M5 -> closer sight distance but bigger anlge
	mutation.host.energyAlteration[5] = make_float2(1.0f, 1.0f);
	mutation.host.sightAlteration[5] = make_float2(0.995f, 0.001f); // (dist -> multiplier, angle -> addition)
	mutation.host.velocityAlteration[5] = 1.0f;
}

#endif // !CONFIG_H