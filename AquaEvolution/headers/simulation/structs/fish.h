#ifndef FISH_CUH
#define FISH_CUH

#include "cuda/helper_math.cuh"
#include <cstdint>
#include <simulation/structs/allocator.h>

enum class FishDecision {
	NONE, MOVE, EAT,
};

template <class Alloc>
struct Fish {
	const uint64_t capacity;

	uint64_t* count{ nullptr };
	float2* positions{ nullptr };
	float2* directionVecs{ nullptr };
	bool* alives{ nullptr };
	float* currentEnergy{ nullptr };
	FishDecision* nextDecisions{ nullptr };
	uint64_t* eatenAlgaeId{ nullptr };

	static Fish make(uint64_t capacity){
		Fish<Alloc> f = Fish<Alloc>{ capacity };
		Alloc::make((void**)&f.count, sizeof(*f.count) * 1);
		Alloc::make((void**)&f.positions, sizeof(*f.positions) * capacity);
		Alloc::make((void**)&f.directionVecs, sizeof(*f.directionVecs) * capacity);
		Alloc::make((void**)&f.alives, sizeof(*f.alives) * capacity);
		Alloc::make((void**)&f.currentEnergy, sizeof(*f.currentEnergy) * capacity);
		Alloc::make((void**)&f.nextDecisions, sizeof(*f.nextDecisions) * capacity);
		Alloc::make((void**)&f.eatenAlgaeId, sizeof(*f.eatenAlgaeId) * capacity);

		return f;
	}

	static void drop(Fish& f) {
		Alloc::drop(f.count);
		Alloc::drop(f.positions);
		Alloc::drop(f.directionVecs);
		Alloc::drop(f.alives);
		Alloc::drop(f.currentEnergy);
		Alloc::drop(f.nextDecisions);
		Alloc::drop(f.eatenAlgaeId);
	}

private:
	Fish(uint64_t capacity) : capacity(capacity) {}

};

#endif // !FISH_CUH
