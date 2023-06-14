#include <simulation/structs/aquarium.cuh>

#include <thrust/async/for_each.h>
#include <thrust/async/transform.h>
#include <thrust/for_each.h>
#include <thrust/random.h>

#include <thrust/iterator/zip_iterator.h>

#include <cuda_runtime.h>

struct GenerateFishFunctor {
	__device__
		Fish::Entity operator()(const uint32_t n) const
	{
		thrust::default_random_engine rng(69);
		thrust::uniform_real_distribution<float> dist(0, 1);
		rng.discard(n);

		float2 pos = make_float2(dist(rng) * Aquarium::WIDTH, dist(rng) * Aquarium::HEIGHT);
		float2 vec = normalize(make_float2(dist(rng), dist(rng)));
		bool alive = true;
		float currentEnergy = 25.0f;
		FishDecisionEnum next = FishDecisionEnum::NONE;
		uint64_t eatenAlgaeId = -1;
		return thrust::make_tuple(pos, vec, alive, currentEnergy, next, eatenAlgaeId);
	}
};

struct GenerateAlgaeFunctor {
	__device__
		Algae::Entity operator()(const uint32_t n) const
	{
		thrust::default_random_engine rng(420);
		thrust::uniform_real_distribution<float> dist(0, 1);
		rng.discard(n);

		float2 pos = make_float2(dist(rng) * Aquarium::WIDTH, dist(rng) * Aquarium::HEIGHT);
		float2 vec = normalize(make_float2(dist(rng), dist(rng)));
		bool alive = true;
		float currentEnergy = 25.0f;
		return thrust::make_tuple(pos, vec, alive, currentEnergy);
	}
};

void Aquarium::generateLife()
{
	int n = Aquarium::FISH_START;
	auto countIter = thrust::counting_iterator<uint32_t>(0);
	fish.resize(fish.device, n);
	auto it = fish.device.iter();
	auto res = thrust::async::transform(thrust::device, countIter, countIter + n, it.get_head(), GenerateFishFunctor());

	int n2 = Aquarium::ALGAE_START;
	auto countIter2 = thrust::counting_iterator<uint32_t>(0);
	algae.resize(algae.device, n2);
	auto it2 = algae.device.iter();
	auto res2 = thrust::async::transform(thrust::device, countIter2, countIter2 + n2, it2.get_head(), GenerateAlgaeFunctor());

	res.wait();
	res2.wait();

	algae.update(algae.host, algae.device);
	fish.update(fish.host, fish.device);
}

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
		float2 vec = pos - a_pos;
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

void Aquarium::simulateGeneration() {
	int N = fish.device.positions.size();
	auto it = fish.device.iter();
	auto count = thrust::make_counting_iterator<uint64_t>(0);

	auto begin = thrust::make_zip_iterator(
		thrust::make_tuple(
			it.get<0>(),
			count
		));
	auto end = thrust::make_zip_iterator(
		thrust::make_tuple(
			it.get<1>(),
			count + N
		));

	thrust::for_each(
		thrust::device, 
		begin, end,
		FishDecisionFunctor(
			algae.device.positions.size(),
			algae.device.positions, 
			algae.device.alives,
			fish.device.positions.size(),
			fish.device.nextDecisions,
			fish.device.directionVecs,
			fish.device.eatenAlgaeId)
	);
}
