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
		bool alive = true;
		float currentEnergy = 25.0f;
		FishDecisionEnum next = FishDecisionEnum::NONE;
		uint64_t eatenAlgaeId = -1;
		return thrust::make_tuple(pos, vec, alive, currentEnergy, next, eatenAlgaeId);
	}
};

struct FishDecisionFunctor {
	uint64_t algae_size;
	float2* algae_pos;
	bool* algae_alive;

	uint64_t fish_size;
	FishDecisionEnum* next;
	float2* new_vec;
	uint64_t* eatenAlgaeId;

	
	__host__ __device__
		FishDecisionFunctor(
			uint64_t algae_size_,
			thrust::device_vector<float2>& algae_pos_, 
			thrust::device_vector<bool>& algae_alive_,
			uint64_t fish_size_,
			thrust::device_vector<FishDecisionEnum>& next_, 
			thrust::device_vector<float2>& new_vec_, 
			thrust::device_vector<uint64_t>& eatenAlgaeId_
			) : 
		algae_size(algae_size_),
		algae_pos(algae_pos_.data().get()),  
		algae_alive(algae_alive_.data().get()),
		fish_size(fish_size_),
		next(next_.data().get()),
		new_vec(new_vec_.data().get()),
		eatenAlgaeId(eatenAlgaeId_.data().get())
	{}

	__device__ 
	void operator()(thrust::tuple<Fish::Entity, uint64_t> tup) const {
		auto& e = tup.get_head();
		uint64_t id = tup.get<1>();
		auto& is_alive = e.get<2>();
		if (!is_alive) {
			next[id] = FishDecisionEnum::NONE;
			return;
		}

		auto& pos = e.get<0>();

		uint64_t algae_id = -1;
		float dist = FLT_MAX;
		for (uint64_t i = 0; i < algae_size; ++i) {
			if (!algae_alive[i]) continue;

			auto& alg_pos = algae_pos[i];
			float new_dist = length(pos - alg_pos);

			if (algae_id != -1 && new_dist > dist) continue;
			dist = new_dist;
			algae_id = i;
		}
		if (algae_id == -1) {
			next[id] = FishDecisionEnum::NONE;
			return;
		}

		bool eat_available = dist < 0.01;
		if (eat_available)
		{
			algae_alive[algae_id] = false;
			eatenAlgaeId[id] = algae_id;
			next[id] = FishDecisionEnum::EAT;
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

		new_vec[id] = vec;
		next[id] = FishDecisionEnum::MOVE;
		eatenAlgaeId[id] = algae_id;
		return;
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

			//pos[id] = p + v * 0.002f;
			pos[id] = p + v * 0.02f;
			break;
		}
		case FishDecisionEnum::EAT:
		{
			en += 0.15f;
			break;
		}
		}

		en -= 0.01f;

		// Check if fish alive
		if (energy <= 0)
		{
			printf("fish[%u] is dead\n", id);
			alives[id] = false;
		}

		energy[id] = min(en, 50.0f);
	}
};

#endif // !FISH_FUNCTOR_CUH
