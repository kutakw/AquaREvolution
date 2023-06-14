#ifndef ALLOCATOR_CUH
#define ALLOCATOR_CUH

#define THRUST_IMPL

#ifdef THRUST_IMPL

#include <thrust/device_ptr.h>
#include <thrust/iterator/detail/normal_iterator.h>

template <typename T>
using iter = thrust::detail::normal_iterator<thrust::device_ptr<T>>;

template <typename... Args>
using entity = thrust::tuple<Args...>;

template <typename... Args>
using entityIter = thrust::tuple<iter<Args>...>;

#else
#include <memory>
#include "cuda/error.cuh"
#include <cuda_runtime.h>

class Host {
public:
	// T - type of data
	static void make(void** ptr, size_t capacity) {
		*ptr = malloc(capacity);
	}

	static void drop(void* ptr) {
		free(ptr);
	}
};

class Device {
public:
	static void make(void** ptr, size_t capacity) {
		checkCudaErrors(cudaMalloc(ptr, capacity));
	}

	static void drop(void* ptr) {
		checkCudaErrors(cudaFree(ptr));
	}
};
#endif

#endif // !ALLOCATOR_CUH
