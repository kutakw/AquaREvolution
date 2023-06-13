#ifndef ALGAE_CUH
#define ALGAE_CUH

#include "cuda/error.cuh"
#include <cstdint>
#include <simulation/structs/allocator.h>

enum class AlgaeDecision {
	NONE, MOVE, EAT,
};

template <class Alloc>
struct Algae {
	const uint64_t capacity;

	uint64_t* count{ nullptr };
	float2* positions{ nullptr };
	float2* directionVecs{ nullptr };
	bool* alives{ nullptr };
	float* currentEnergy{ nullptr };

	static Algae make(uint64_t capacity) {
		Algae<Alloc> f = Algae<Alloc>{capacity};
		Alloc::make((void**)&f.count, sizeof(*f.count) * 1);
		Alloc::make((void**)&f.positions, sizeof(*f.positions) * capacity);
		Alloc::make((void**)&f.directionVecs, sizeof(*f.directionVecs) * capacity);
		Alloc::make((void**)&f.alives, sizeof(*f.alives) * capacity);
		Alloc::make((void**)&f.currentEnergy, sizeof(*f.currentEnergy) * capacity);

		return f;
	}

	static void drop(Algae& f) {
		Alloc::drop(f.count);
		Alloc::drop(f.positions);
		Alloc::drop(f.directionVecs);
		Alloc::drop(f.alives);
		Alloc::drop(f.currentEnergy);
	}



private:
	Algae(uint64_t capacity) : capacity(capacity) {}
};

#endif // !ALGAE_CUH
