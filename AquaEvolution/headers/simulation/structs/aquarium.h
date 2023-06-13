#ifndef AQUARIUM_CUH
#define AQUARIUM_CUH

#include <simulation/structs/fish.h>
#include <simulation/structs/algae.h>
#include <cuda/helper_math.cuh>


#ifdef THRUST_IMPL


struct Aquarium {
	static constexpr uint64_t FISH_MAX_COUNT = 10000;
	static constexpr uint64_t ALGAE_MAX_COUNT = 50000;
	static constexpr float WIDTH = 100.f;
	static constexpr float HEIGHT = 100.f;

	Fish fish;
	Algae algae;

	Aquarium() :
		fish(FISH_MAX_COUNT),
		algae(ALGAE_MAX_COUNT)
	{
		fish.host.count.front()++;
		fish.host.positions.push_back(make_float2(50.0f, 50.0f));
		fish.host.directionVecs.push_back(make_float2(1.0f, 0.0f));
		fish.host.alives.push_back(true);

		algae.host.count.front()++;
		algae.host.positions.push_back(make_float2(20.0f, 50.0f));
		algae.host.directionVecs.push_back(make_float2(1.0f, 0.0f));
		algae.host.alives.push_back(true);
	}

	~Aquarium() {
		//Fish<Device>::drop(d_fish);
		//Fish<Host>::drop(h_fish);
		//Fish<Device>::drop(d_fish);
		//Fish<Host>::drop(h_fish);
	}
};

#else
struct Aquarium {
	static constexpr uint64_t FISH_MAX_COUNT = 10000;
	static constexpr uint64_t ALGAE_MAX_COUNT = 50000;
	static constexpr float WIDTH = 100.f;
	static constexpr float HEIGHT = 100.f;

	Fish<Device> d_fish;
	Fish<Host> h_fish;
	Algae<Device> d_algae;
	Algae<Host> h_algae;

	Aquarium() :
		d_fish(Fish<Device>::make(FISH_MAX_COUNT)),
		h_fish(Fish<Host>::make(FISH_MAX_COUNT)),
		d_algae(Algae<Device>::make(ALGAE_MAX_COUNT)),
		h_algae(Algae<Host>::make(ALGAE_MAX_COUNT))
	{
		*h_fish.count = 1;
		h_fish.positions[0] = make_float2(50.0f, 50.0f);
		h_fish.directionVecs[0] = make_float2(1.0f, 0.0f);
		h_fish.alives[0] = true;
	}

	~Aquarium() {
		Fish<Device>::drop(d_fish);
		Fish<Host>::drop(h_fish);
		Fish<Device>::drop(d_fish);
		Fish<Host>::drop(h_fish);
	}
};
#endif

#endif // !AQUARIUM_CUH
