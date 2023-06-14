#include <simulation/structs/aquarium.cuh>

#include <simulation/structs/generators.cuh>
#include <simulation/structs/fish_functor.cuh>
#include <simulation/structs/algae_functor.cuh>

#include <thrust/async/for_each.h>
#include <thrust/async/transform.h>
#include <thrust/for_each.h>
#include <thrust/random.h>

#include <thrust/iterator/zip_iterator.h>

#include <cuda_runtime.h>

void Aquarium::generateLife()
{
	// TODO: zrobic tak zeby na raz wygenerowac wszystkie losowe pos + vectory
	int n = Aquarium::FISH_START;
	auto countIter = thrust::counting_iterator<uint32_t>(0);
	fish.resize(fish.device, n);
	auto it = fish.device.iter();
	auto res = thrust::async::transform(thrust::device, countIter, countIter + n, it.get_head(), GenerateFishFunctor());
	//thrust::transform(thrust::device, countIter, countIter + n, fish.device.positions.begin(), GeneratePos());
	//thrust::transform(thrust::device, countIter, countIter + n, fish.device.directionVecs.begin(), GenerateVector());
	//thrust::fill(thrust::device, fish.device.alives.begin(), fish.device.alives.end(), true);
	//thrust::fill(thrust::device, fish.device.currentEnergy.begin(), fish.device.currentEnergy.end(), 25.0f);

	int n2 = Aquarium::ALGAE_START;
	auto countIter2 = thrust::counting_iterator<uint32_t>(0);
	algae.resize(algae.device, n2);
	auto it2 = algae.device.iter();
	auto res2 = thrust::async::transform(thrust::device, countIter2, countIter2 + n2, it2.get_head(), GenerateAlgaeFunctor());
	//thrust::transform(thrust::device, countIter, countIter + n2, algae.device.positions.begin(), GeneratePos());
	//thrust::transform(thrust::device, countIter, countIter + n2, algae.device.directionVecs.begin(), GenerateVector());
	//thrust::fill(thrust::device, algae.device.alives.begin(), algae.device.alives.end(), true);
	//thrust::fill(thrust::device, algae.device.currentEnergy.begin(), algae.device.currentEnergy.end(), 25.0f);

	res.wait();
	res2.wait();

	algae.update(algae.host, algae.device);
	fish.update(fish.host, fish.device);
}

void Aquarium::simulateGeneration() {
	for (int i = 0; i < 100; i++)
	{
		decision();
		move();
	}
}


void Aquarium::decision()
{
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

	auto res = thrust::async::for_each(
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

	int N2 = algae.device.positions.size();
	auto it2 = algae.device.iter();
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

	auto res2 = thrust::async::for_each(
		thrust::device,
		begin2, end2,
		AlgaeDecisionFunctor(
			algae.device.positions.size(),
			algae.device.directionVecs)
	);

	res.wait();
	res2.wait();
}

void Aquarium::move()
{
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

	auto res = thrust::async::for_each(
		thrust::device,
		begin, end,
		FishMoveFunctor(
			fish.device.positions.size(),
			fish.device.positions,
			fish.device.directionVecs,
			fish.device.currentEnergy,
			fish.device.alives)
	);

	int N2 = algae.device.positions.size();
	auto it2 = algae.device.iter();
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

	auto res2 = thrust::async::for_each(
		thrust::device,
		begin2, end2,
		AlgaeMoveFunctor(
			algae.device.positions.size(),
			algae.device.positions,
			algae.device.directionVecs,
			algae.device.currentEnergy,
			algae.device.alives)
	);

	res.wait();
	res2.wait();
}
