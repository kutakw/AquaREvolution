#ifndef CONFIG_H
#define CONFIG_H

#include <cuda/helper_math.cuh>
#include <thrust/device_vector.h>

// Environment dimensions
static constexpr float WIDTH = 100.f;
static constexpr float HEIGHT = 100.f;

// Number of objects present during initialization
static constexpr uint64_t FISH_START = 100;
static constexpr uint64_t ALGAE_START = 1000;

// Cell dimensions
static constexpr ulonglong2 CELL = { 100, 100 };

// Duration single generation simulation
static constexpr int32_t ITER_PER_GENERATION = 1000;

// Alage parameters
static constexpr uint64_t ALGAE_MAX_COUNT = 30000;
static constexpr float ALGAE_INIT_ENERGY = 25.0f;
static constexpr float ALGAE_MAX_ENERGY = 50.0f;
static constexpr float ALGAE_ENERGY_LOSS = 0.1f;
static constexpr float ALGAE_ENERGY_MINIMUM_TO_REPRODUCT = 10.0f;
static constexpr float ALGAE_ENERGY_PER_KID = 10.0f;
static constexpr float ALGAE_VELOCITY = 1e-3f;

// Fish parameters
static constexpr uint64_t FISH_MAX_COUNT = 10000;
static constexpr float FISH_MAX_ENERGY = 50.0f;
static constexpr float FISH_INITAL_ENERGY = 30.0f;
static constexpr float FISH_ENERGY_PER_KID = 10.0f;
static constexpr float FISH_ENERGY_MINIMUM_TO_REPRODUCT = 5.0f;
static constexpr float FISH_ENERGY_PER_ALGA_EATEN = 3.0f;
static constexpr float FISH_SIGHT_DIST = 10.0f;
static constexpr float FISH_SIGHT_ANGLE = 0.0f;
static constexpr float FISH_VELOCITY = 2e-3f;
static constexpr float FISH_ENERGY_DECAY_RATE = 0.1f;

static constexpr int32_t DUMP_FREQ = 10;

#endif // !CONFIG_H