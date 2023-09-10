#include <simulation/structs/aquarium.cuh>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/sort.h>
#include <thrust/random.h>
//#include <../config.h>

struct ChildrenCountFunctorAlgae {
	__device__
	int32_t operator()(const Algae::Entity& e) const {
		auto is_alive = e.get<2>();
		if (!is_alive) return 0;

		auto energy = e.get<3>();

		int32_t children = fmaxf(energy - ALGAE_ENERGY_MINIMUM_TO_REPRODUCT, 0.f) / ALGAE_ENERGY_PER_KID;
		return children;
	}
};

struct ChildrenCountFunctorFish {
	__device__
		int32_t operator()(const Fish::Entity& e) const {
		auto is_alive = e.get<2>();
		if (!is_alive) return 0;

		auto energy = e.get<3>();

		int32_t children = fmaxf(energy - FISH_ENERGY_MINIMUM_TO_REPRODUCT, 0.f) / FISH_ENERGY_PER_KID;
		return children;
	}
};

struct ChildrenPerIterFunctor {
	__device__ int32_t operator()(const int32_t& x) const {
		return int32_t(x > 0);
	}
};

struct Decrement {
	__device__ void operator()(int32_t& x) {
		x--;
	}
};

struct GeneratorAlgae {
	using tup = thrust::tuple<float2, uint32_t>;
	__device__
		Algae::Entity operator()(const tup& n) const
	{
		thrust::default_random_engine rng;
		thrust::uniform_real_distribution<float> dist(-1, 1);
		rng.discard(n.get<1>());

		float2 pos = n.get<0>();
		float2 vec = normalize(make_float2(dist(rng), dist(rng)));
		bool alive = true;
		float currentEnergy = ALGAE_INIT_ENERGY;
		return thrust::make_tuple(pos, vec, alive, currentEnergy);
	}
};

struct GeneratorFish {

	float2* energyAlterations;
	float2* sightAlterations;
	float* velocityAlterations;

	__host__ __device__
		GeneratorFish(
			uint64_t mutationID,
			thrust::device_vector<float2>& energyAlteration_,
			thrust::device_vector<float2>& sightAlteration_,
			thrust::device_vector<float>& velocityAlteration_
		) 
	{
		/*float2**/ energyAlterations = energyAlteration_.data().get();
		/*float2**/ sightAlterations = sightAlteration_.data().get();
		/*float**/ velocityAlterations = velocityAlteration_.data().get();

		//energyAlteration = energyAlteration_[mutationID];
		//sightAlteration = sightAlteration_[mutationID];
		//velocityAlteration = velocityAlteration_[mutationID];
	}

	using tup = thrust::tuple<Fish::Entity, uint32_t>;
	__device__
		Fish::Entity operator()(const tup& n) const
	{
		thrust::default_random_engine rng;
		thrust::uniform_real_distribution<float> dist(-1, 1);
		thrust::uniform_real_distribution<uint64_t> mutation(0, Mutation::MUTATION_COUNT);
		uint64_t mutationId = mutation(rng);
		rng.discard(n.get<1>());

		auto fishEntity = n.get<0>();
		float2 pos = fishEntity.get<0>();
		float2 vec = normalize(make_float2(dist(rng), dist(rng)));
		float2 energyParams = fishEntity.get<6>() * energyAlterations[mutationId];
		float2 sightParams = fishEntity.get<7>();
		sightParams.x *= sightAlterations[mutationId].x;
		sightParams.y == clamp(sightParams.y + sightAlterations[mutationId].y,-1.f,1.f);
		float velocity = fishEntity.get<8>() * velocityAlterations[mutationId];
		bool alive = true;
		float currentEnergy = fminf(FISH_INITAL_ENERGY, energyParams.x);
		FishDecisionEnum next = FishDecisionEnum::NONE;
		uint64_t eatenAlgaeId = -1;

		return thrust::make_tuple(pos, vec, alive, currentEnergy, next, eatenAlgaeId,energyParams,sightParams,velocity);
	}
};

void Aquarium::reproduction_algae() {

	thrust::device_vector<int32_t> dc(algae->device.positions.size());
	auto it = algae->device.iter();
	
	thrust::transform(thrust::device, it.get<0>(), it.get<1>(), dc.begin(), ChildrenCountFunctorAlgae());

	//thrust::host_vector<int32_t> h = dc;
	//for (int32_t i = 0; i < h.size(); i++) {
	//	std::cout << i << ": " << h[i] << std::endl;
	//}


	auto begin = thrust::make_zip_iterator(thrust::make_tuple(
		algae->device.positions.begin(),
		dc.begin()
	));
	thrust::sort_by_key(thrust::device, dc.begin(), dc.end(), begin, thrust::greater<int32_t>());

	int32_t childrenLeft = thrust::reduce(dc.begin(), dc.end());
	int32_t minLeft = 0;
	int32_t childrenInLoop = thrust::transform_reduce(
		dc.begin(), dc.end(), ChildrenPerIterFunctor(), 0, thrust::plus<int32_t>());
	//std::cout << "reduction: " << childrenLeft << std::endl;

	Algae* back = &algaeBuffer[1 - currentAlgaeBuffer];
	if (childrenLeft > back->capacity) {
		minLeft = childrenLeft - back->capacity;
	}
	
	back->resize(back->device, childrenLeft - minLeft);

	auto& backIter = back->device.iter().get_head();
	auto countIter = thrust::make_counting_iterator<uint32_t>(0);
	while (childrenLeft > minLeft) {

		if (childrenInLoop > childrenLeft - minLeft)
			childrenInLoop = childrenLeft - minLeft;

		thrust::transform(
			thrust::make_zip_iterator(thrust::make_tuple(algae->device.positions.begin(), countIter)),
			thrust::make_zip_iterator(thrust::make_tuple(algae->device.positions.begin() + childrenInLoop, countIter + childrenInLoop)),
			backIter,
			GeneratorAlgae()
		);

		backIter += childrenInLoop;
		
		thrust::for_each(dc.begin(), dc.end(), Decrement());
		//childrenLeft = thrust::reduce(dc.begin(), dc.end());
		childrenLeft -= childrenInLoop;
		childrenInLoop = thrust::transform_reduce(
			dc.begin(), dc.end(), ChildrenPerIterFunctor(), 0, thrust::plus<int32_t>());
	}

	algae = back;
	currentAlgaeBuffer = 1 - currentAlgaeBuffer;
}

void Aquarium::reproduction_fish()
{
	thrust::device_vector<int32_t> dc(fish->device.positions.size());
	auto it = fish->device.iter();

	thrust::transform(thrust::device, it.get<0>(), it.get<1>(), dc.begin(), ChildrenCountFunctorFish());

	//thrust::host_vector<int32_t> h = dc;
	//for (int32_t i = 0; i < h.size(); i++) {
	//	std::cout << i << ": " << h[i] << std::endl;
	//}


	auto begin = thrust::make_zip_iterator(thrust::make_tuple(
		fish->device.positions.begin(),
		dc.begin()
	));
	thrust::sort_by_key(thrust::device, dc.begin(), dc.end(), begin, thrust::greater<int32_t>());

	int32_t childrenLeft = thrust::reduce(dc.begin(), dc.end());
	int32_t minLeft = 0;
	int32_t childrenInLoop = thrust::transform_reduce(
		dc.begin(), dc.end(), ChildrenPerIterFunctor(), 0, thrust::plus<int32_t>());
	//std::cout << "fish reduction: " << childrenLeft << std::endl;

	Fish* back = &fishBuffer[1 - currentFishBuffer];
	if (childrenLeft > back->capacity) {
		minLeft = childrenLeft - back->capacity;
	}

	back->resize(back->device, childrenLeft - minLeft);

	auto& backIter = back->device.iter().get_head();
	auto countIter = thrust::make_counting_iterator<uint32_t>(0);
	uint64_t i = 0;
	while (childrenLeft > minLeft) {

		if (childrenInLoop > childrenLeft - minLeft)
			childrenInLoop = childrenLeft - minLeft;

		auto it = fish->device.iter();
		uint64_t mutationId = i % fish->capacity;
		thrust::transform(
			thrust::device,
			thrust::make_zip_iterator(thrust::make_tuple(it.get_head(), countIter)),
			thrust::make_zip_iterator(thrust::make_tuple(it.get_head() + childrenInLoop, countIter + childrenInLoop)),
			backIter,
			GeneratorFish(mutationId,mutation.device.energyAlteration, mutation.device.sightAlteration, mutation.device.velocityAlteration)
		);

		backIter += childrenInLoop;

		thrust::for_each(dc.begin(), dc.end(), Decrement());
		//childrenLeft = thrust::reduce(dc.begin(), dc.end());
		childrenLeft -= childrenInLoop;
		childrenInLoop = thrust::transform_reduce(
			dc.begin(), dc.end(), ChildrenPerIterFunctor(), 0, thrust::plus<int32_t>());
		i++;
	}

	fish = back;
	currentFishBuffer = 1 - currentFishBuffer;
}
