#include <simulation/structs/reproduction.cuh>

#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/transform_reduce.h>
#include <thrust/sort.h>
#include <thrust/random.h>

struct ChildrenCountFunctor {
	__device__
	int32_t operator()(const Algae::Entity& e) const {
		auto is_alive = e.get<2>();
		if (!is_alive) return 0;

		auto energy = e.get<3>();

		int32_t children = fmaxf(energy - Algae::ENERGY_MINIMUM_TO_REPRODUCT, 0.f) / Algae::ENERGY_PER_KID;
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
		float currentEnergy = Algae::INIT_ENERGY;
		return thrust::make_tuple(pos, vec, alive, currentEnergy);
	}
};

void Aquarium::reproduction_algae() {

	thrust::device_vector<int32_t> dc(algae->device.positions.size());
	auto it = algae->device.iter();
	
	thrust::transform(thrust::device, it.get<0>(), it.get<1>(), dc.begin(), ChildrenCountFunctor());

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
	std::cout << "reduction: " << childrenLeft << std::endl;

	Algae* back = &algaeBuffer[1 - currentBuffer];
	if (childrenLeft > back->capacity) {
		minLeft = childrenLeft - back->capacity;
	}
	
	back->resize(back->device, childrenLeft - minLeft);

	auto& backIter = back->device.iter().get_head();
	auto countIter = thrust::make_counting_iterator<uint32_t>(0);
	while (childrenLeft > minLeft) {

		thrust::transform(
			thrust::make_zip_iterator(thrust::make_tuple(algae->device.positions.begin(), countIter)),
			thrust::make_zip_iterator(thrust::make_tuple(algae->device.positions.begin() + childrenInLoop, countIter + childrenInLoop)),
			backIter,
			GeneratorAlgae()
		);

		backIter += childrenInLoop;
		
		thrust::for_each(dc.begin(), dc.end(), Decrement());
		childrenLeft = thrust::reduce(dc.begin(), dc.end());
		childrenInLoop = thrust::transform_reduce(
			dc.begin(), dc.end(), ChildrenPerIterFunctor(), 0, thrust::plus<int32_t>());
	}

	algae = back;
	currentBuffer = 1 - currentBuffer;
}
