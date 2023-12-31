#include <simulation/structs/aquarium.cuh>
#include <simulation/structs/generators.cuh>
#include <simulation/structs/fish_functor.cuh>
#include <simulation/structs/algae_functor.cuh>
#include <cuda/error.cuh>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/sort.h>
#include <thrust/random.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "config.h"

Aquarium::Aquarium() :
	fish(fishBuffer),
	algae(algaeBuffer),
	algaeBucketIds(CELL.x * CELL.y),
	algaeKeys(ALGAE_MAX_COUNT),
	mutation(MUTATION_COUNT)
{}

void Aquarium::generateLife()
{
	int n = FISH_START;
	auto countIter = thrust::counting_iterator<uint32_t>(rand());
	fish->resize(fish->device, n);
	auto it = fish->device.iter();
	auto res = thrust::transform(thrust::device, countIter, countIter + n, it.get_head(), GenerateFishFunctor());

	int n2 = ALGAE_START;
	auto countIter2 = thrust::counting_iterator<uint32_t>(rand());
	algae->resize(algae->device, n2);
	auto it2 = algae->device.iter();
	auto res2 = thrust::transform(thrust::device, countIter2, countIter2 + n2, it2.get_head(), GenerateAlgaeFunctor());

	algae->update(algae->host, algae->device);
	fish->update(fish->host, fish->device);
}

void Aquarium::generateMutations() {
	initMutationsFromConfig(mutation);
	mutation.resize(mutation.device, MUTATION_COUNT);
	mutation.update(mutation.device, mutation.host);
}

void Aquarium::simulateGeneration() {
	sort_algae();

	//auto start = std::chrono::high_resolution_clock().now();
	for (int i = 0; i < ITER_PER_GENERATION; i++)
	{
		decision();
		move();
	}
	//auto end = std::chrono::high_resolution_clock().now();
	//auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
	//std::cout << time.count() << std::endl;
	//std::cout << static_cast<long long>(100) * 1000 / time.count() << std::endl;

	reproduction_algae();
	reproduction_fish();
}

void Aquarium::decision()
{
	int N = fish->device.positions.size();
	auto it = fish->device.iter();
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

	auto res = thrust::for_each(
		thrust::device,
		begin, end,
		FishDecisionFunctor(
			algae->device.positions.size(),
			algae->device.positions,
			algae->device.alives,
			fish->device.positions.size(),
			fish->device.nextDecisions,
			fish->device.positions,
			fish->device.directionVecs,
			fish->device.eatenAlgaeId,
			algaeBucketIds
			)
	);

	int N2 = algae->device.positions.size();
	auto it2 = algae->device.iter();
	auto count2 = thrust::make_counting_iterator<uint64_t>(0);

	auto begin2 = thrust::make_zip_iterator(
		thrust::make_tuple(
			it2.get<0>(),
			count2
		));
	auto end2 = thrust::make_zip_iterator(
		thrust::make_tuple(
			it2.get<1>(),
			count2 + N2
		));

	auto res2 = thrust::for_each(
		thrust::device,
		begin2, end2,
		AlgaeDecisionFunctor(
			algae->device.positions.size(),
			algae->device.directionVecs)
	);

	//res.wait();
	//res2.wait();
}

void Aquarium::move()
{
	int N = fish->device.positions.size();
	auto it = fish->device.iter();
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

	auto res = thrust::for_each(
		thrust::device,
		begin, end,
		FishMoveFunctor(
			fish->device.positions.size(),
			fish->device.positions,
			fish->device.directionVecs,
			fish->device.currentEnergy,
			fish->device.alives)
	);

	int N2 = algae->device.positions.size();
	auto it2 = algae->device.iter();
	auto count2 = thrust::make_counting_iterator<uint64_t>(0);

	auto begin2 = thrust::make_zip_iterator(
		thrust::make_tuple(
			it2.get<0>(),
			count2
		));
	auto end2 = thrust::make_zip_iterator(
		thrust::make_tuple(
			it2.get<1>(),
			count2 + N2
		));

	auto res2 = thrust::for_each(
		thrust::device,
		begin2, end2,
		AlgaeMoveFunctor(
			algae->device.positions.size(),
			algae->device.positions,
			algae->device.directionVecs,
			algae->device.currentEnergy,
			algae->device.alives)
	);
}


