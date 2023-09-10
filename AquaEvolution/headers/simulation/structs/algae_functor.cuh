#ifndef ALGAE_FUNCTOR_CUH
#define ALGAE_FUNCTOR_CUH

#include <cstdint>
#include <cuda/helper_math.cuh>
#include <simulation/structs/algae.cuh>
#include <thrust/random.h>
#include "aquarium.cuh"

struct GenerateAlgaeFunctor {
	__device__
		Algae::Entity operator()(const uint32_t n) const
	{
		thrust::default_random_engine rng;
		thrust::uniform_real_distribution<float> dist(0, 1);
		rng.discard(n);

		float2 pos = make_float2(dist(rng) * Aquarium::WIDTH, dist(rng) * Aquarium::HEIGHT);
		float2 vec = normalize(make_float2(dist(rng), dist(rng)));
		bool alive = true;
		float currentEnergy = Algae::INIT_ENERGY;
		return thrust::make_tuple(pos, vec, alive, currentEnergy);
	}
};

struct AlgaeDecisionFunctor {
	uint64_t algae_size;
	float2* algae_vec;

	
	__host__ __device__
		AlgaeDecisionFunctor(
			uint64_t algae_size_,
			thrust::device_vector<float2>& algae_vec_
			) : 
		algae_size(algae_size_),
		algae_vec(algae_vec_.data().get())
	{}

	__device__ 
	void operator()(thrust::tuple<Algae::Entity, uint64_t> tup) const {
		auto& e = tup.get_head();
		auto is_alive = e.get<2>();
		if (!is_alive) return;
		auto id = tup.get<1>();

		auto pos = e.get<0>();
		auto vec = e.get<1>();

		float2 new_pos = pos + vec * Algae::VELOCITY;
		if (new_pos.x < 0.0f || new_pos.x >= Aquarium::WIDTH)
			vec.x *= -1.0f;
		if (new_pos.y < 0.0f || new_pos.y >= Aquarium::HEIGHT)
			vec.y *= -1.0f;

		algae_vec[id] = vec;
	}
};

struct AlgaeMoveFunctor {

	uint64_t fish_size;
	float2* pos;
	float2* vec;
	float* energy;
	bool* alives;

	__host__ __device__
		AlgaeMoveFunctor(
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
	void operator()(thrust::tuple<Algae::Entity, uint64_t> tup) const {
		auto& e = tup.get_head();
		auto is_alive = e.get<2>();
		if (!is_alive) return;

		auto id = tup.get<1>();
		auto p = pos[id];
		auto v = vec[id];
		float2 new_pos = p + v * Algae::VELOCITY;
		pos[id] = new_pos;

		auto en = energy[id];
		float height = new_pos.y / Aquarium::HEIGHT;
		float energyLoss = Algae::ENERGY_LOSS;
		float energyGain = lerp(0.0f, 1.5f * Algae::ENERGY_LOSS, height);
		en -= energyLoss;
		en += energyGain;

		if (en < 0.0f)
		{
			alives[id] = false;
			return;
		}

		en = fminf(en, 50.0f);
		energy[id] = en;
	}
};

#endif // !ALGAE_FUNCTOR_CUH
