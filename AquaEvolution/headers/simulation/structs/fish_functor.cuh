#ifndef FISH_FUNCTOR_CUH
#define FISH_FUNCTOR_CUH

#include <cstdint>
#include <cuda/helper_math.cuh>
#include <simulation/structs/fish.cuh>
#include <thrust/random.h>
#include "aquarium.cuh"

struct GenerateFishFunctor {
	__device__
		Fish::Entity operator()(const uint32_t n) const
	{
		thrust::default_random_engine rng;
		thrust::uniform_real_distribution<float> dist(0, 1);
		rng.discard(n);

		float2 vec = normalize(make_float2(dist(rng), dist(rng)));
		float2 pos = make_float2(dist(rng) * (Aquarium::WIDTH - 2.0f) + 1.0f, dist(rng) * (Aquarium::HEIGHT - 2.0f) + 1.0f);
		float2 energyParams = make_float2(Fish::MAX_ENERGY, Fish::ENERGY_DECAY_RATE);
		float2 sightParams = make_float2(Fish::SIGHT_DIST, Fish::SIGHT_ANGLE);
		float energy = Fish::INITAL_ENERGY;
		float velocity = Fish::VELOCITY;
		return thrust::make_tuple(
			pos, 
			vec, 
			true, 
			energy, 
			FishDecisionEnum::NONE, 
			(uint64_t)-1,
			energyParams,
			sightParams,
			velocity);
	}
};

struct FishDecisionFunctor {
	uint64_t algae_size;
	float2* algae_pos;
	bool* algae_alive;

	uint64_t fish_size;
	FishDecisionEnum* fish_next;
	float2* fish_pos;
	float2* fish_new_vec;
	uint64_t* fish_eatenAlgaeId;

	uint64_t* bucket;

	__host__ __device__
		FishDecisionFunctor(
			uint64_t algae_size_,
			thrust::device_vector<float2>& algae_pos_, 
			thrust::device_vector<bool>& algae_alive_,
			uint64_t fish_size_,
			thrust::device_vector<FishDecisionEnum>& next_, 
			thrust::device_vector<float2>& fish_pos_, 
			thrust::device_vector<float2>& new_vec_, 
			thrust::device_vector<uint64_t>& eatenAlgaeId_,
			thrust::device_vector<uint64_t>& bucket_
			) : 
		algae_size(algae_size_),
		algae_pos(algae_pos_.data().get()),  
		algae_alive(algae_alive_.data().get()),
		fish_size(fish_size_),
		fish_next(next_.data().get()),
		fish_pos(fish_pos_.data().get()),
		fish_new_vec(new_vec_.data().get()),
		fish_eatenAlgaeId(eatenAlgaeId_.data().get()),
		bucket(bucket_.data().get())
	{}

	__device__ 
	void operator()(thrust::tuple<Fish::Entity, uint64_t> tup) {
		auto& e = tup.get_head();
		uint64_t id = tup.get<1>();
		auto& is_alive = e.get<2>();
		if (!is_alive) {
			fish_next[id] = FishDecisionEnum::NONE;
			return;
		}

		auto& pos = e.get<0>();

		uint64_t algae_id = -1;
		float dist = FLT_MAX;

#if 0
		for (uint64_t i = 0; i < algae_size; ++i) {
			if (!algae_alive[i]) continue;

			auto& alg_pos = algae_pos[i];
			float new_dist = length(pos - alg_pos);

			if (algae_id != -1 && new_dist > dist) continue;
			dist = new_dist;
			algae_id = i;
		}

#else 
		algae_id = findClosestAlga(e, &dist);
#endif

		if (algae_id == -1) {
			fish_next[id] = FishDecisionEnum::NONE;
			return;
		}

		bool eat_available = dist < 0.01;
		if (eat_available)
		{
			algae_alive[algae_id] = false;
			fish_eatenAlgaeId[id] = algae_id;
			fish_next[id] = FishDecisionEnum::EAT;
			return;
		}

		float2 a_pos = algae_pos[algae_id];
		float2 vec = a_pos - pos;
		float denom = sqrtf(dot(vec, vec));
		if (denom > 0.01f)
		{
			vec.x /= denom;
			vec.y /= denom;
		}

		fish_new_vec[id] = vec;
		fish_next[id] = FishDecisionEnum::MOVE;
		fish_eatenAlgaeId[id] = algae_id;
		return;
	}

	__device__ int findClosestAlga(Fish::Entity& e, float* distToBeat)
	{
		auto& pos = e.get<0>();
		uint2 fishCell = { (uint)(pos.x * Aquarium::CELL.x / Aquarium::WIDTH), (uint)(pos.y * Aquarium::CELL.y / Aquarium::HEIGHT) };

		//check same cell
		int closest_algae_id = findClosestAlgaInCell(e, fishCell, distToBeat);
		int tmp;

		// check cells above
		if (fishCell.y > 0)
		{
			// directly above
			tmp = findClosestAlgaInCell(e, { fishCell.x, fishCell.y - 1 }, distToBeat);
			if (tmp != -1) closest_algae_id = tmp;

			// left above
			if (fishCell.x > 0)
			{
				tmp = findClosestAlgaInCell(e, { fishCell.x - 1, fishCell.y - 1 }, distToBeat);
				if (tmp != -1) closest_algae_id = tmp;
			}
			// right above
			if (fishCell.x < Aquarium::CELL.x - 1)
			{
				tmp = findClosestAlgaInCell(e, { fishCell.x + 1, fishCell.y - 1 }, distToBeat);
				if (tmp != -1) closest_algae_id = tmp;
			}
		}
		// check cell below
		if (fishCell.y < Aquarium::CELL.y - 1)
		{
			// directly below
			tmp = findClosestAlgaInCell(e, { fishCell.x, fishCell.y + 1 }, distToBeat);
			if (tmp != -1) closest_algae_id = tmp;

			// left below
			if (fishCell.x > 0)
			{
				tmp = findClosestAlgaInCell(e, { fishCell.x - 1, fishCell.y + 1 }, distToBeat);
				if (tmp != -1) closest_algae_id = tmp;
			}
			// right below
			if (fishCell.x < Aquarium::CELL.x - 1)
			{
				tmp = findClosestAlgaInCell(e, { fishCell.x + 1, fishCell.y + 1 }, distToBeat);
				if (tmp != -1) closest_algae_id = tmp;
			}
		}
		// check cell on the left
		if (fishCell.x > 0)
		{
			tmp = findClosestAlgaInCell(e, { fishCell.x - 1, fishCell.y }, distToBeat);
			if (tmp != -1) closest_algae_id = tmp;
		}
		// check cell on the right
		if (fishCell.x < Aquarium::CELL.x - 1)
		{
			tmp = findClosestAlgaInCell(e, { fishCell.x + 1, fishCell.y }, distToBeat);
			if (tmp != -1) closest_algae_id = tmp;
		}

		return closest_algae_id;
	}

	__device__ int findClosestAlgaInCell(Fish::Entity& e, uint2 cell, float* distToBeat)
	{
		int closest_alga_id = -1;
		int cellArrayIdx = cell.x + cell.y * Aquarium::CELL.x;

		auto alga_id = bucket[cellArrayIdx];
		auto end = 
			cellArrayIdx == (Aquarium::CELL.x * Aquarium::CELL.y - 1) 
			? algae_size 
			: bucket[cellArrayIdx + 1];

		while (alga_id != end)
		{
			float curr_dist = algae_in_sight_dist(e, alga_id);
			if (curr_dist != -1 && curr_dist < *distToBeat)
			{
				closest_alga_id = alga_id;
				*distToBeat = curr_dist;
			}
			
			alga_id++;
		}
		return closest_alga_id;
	}

	// returns algae distance or -1 if fish cannot see the algae
	__device__ float algae_in_sight_dist(Fish::Entity& e, size_t algaId)
	{
		float2 algaPos = algae_pos[algaId];
		float2 fishPos = e.get<0>();
		float2 fishVec = e.get<1>();
		float2 fishSightParams = e.get<7>(); // dist, angle

		float2 vecToAlga = algaPos - fishPos;

		float dist = length(vecToAlga);

		//check distance
		if (dist > fishSightParams.x)
			return -1.f;

		//// check angle
		float cosine = dot(fishVec, vecToAlga/dist);
		if (cosine < fishSightParams.y)
			return -1.f;

		return dist;
	}
};

struct FishMoveFunctor {

	uint64_t fish_size;
	float2* pos;
	float2* vec;
	float* energy;
	bool* alives;

	__host__ __device__
		FishMoveFunctor(
			uint64_t fish_size_,
			thrust::device_vector<float2>& pos_,
			thrust::device_vector<float2>& vec_,
			thrust::device_vector<float>& energy_,
			thrust::device_vector<bool>& alives_) :
		fish_size(fish_size_),
		pos(pos_.data().get()),
		vec(vec_.data().get()),
		energy(energy_.data().get()),
		alives(alives_.data().get())
	{}

	__device__ 
	void operator()(thrust::tuple<Fish::Entity, uint64_t> tup) const {
		auto& e = tup.get_head();
		uint64_t id = tup.get<1>();

		auto is_alive = e.get<2>();
		if (!is_alive) return;

		auto decision = e.get<4>();
		auto en = e.get<3>();
		float velocity = e.get<8>();
		float2 energyParams = e.get<6>(); // max, decay
		
		switch (decision)
		{
		case FishDecisionEnum::NONE:
		{
			//printf("fish[%u] rotates\n", id);
			float2 v = vec[id];
			vec[id] = make_float2(v.y, -v.x);
			break;
		}
		case FishDecisionEnum::MOVE:
		{
			float2 p = pos[id];
			float2 v = vec[id];

			pos[id] = p + v * velocity;
			//pos[id] = p + v * 0.02f;
			break;
		}
		case FishDecisionEnum::EAT:
		{
			en += Fish::ENERGY_PER_ALGA_EATEN;
			break;
		}
		}

		en -= energyParams.y;

		// Check if fish alive
		if (energy <= 0)
		{
			printf("fish[%u] is dead\n", id);
			alives[id] = false;
		}

		energy[id] = min(en, energyParams.x);
	}
};

#endif // !FISH_FUNCTOR_CUH
