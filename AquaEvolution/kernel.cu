//#define TEST

#ifndef TEST
#include <window.cuh>

int main(int argc, char* argv[])
{
	Window& instance = Window::instance();
	Aquarium aquarium;
	instance.renderLoop(aquarium);
	instance.free();

	return 0;
}
#else

#include <iostream>
#include <thrust/for_each.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/async/reduce.h>
#include <thrust/async/transform.h>
#include <thrust/async/sort.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

struct lala {
	__host__ __device__ 
		uint32_t operator()(const uint32_t x) const {
		return x - 1;
	}
};

int main() {
	constexpr uint64_t N = (uint64_t)1 << 10;
	//thrust::device_vector<uint64_t> d(N, 1);

	//auto result = thrust::async::reduce(d.begin(), d.end());
	//std::cout << result.extract();

	thrust::counting_iterator<uint32_t> index_sequence_begin(1);
	thrust::device_vector<uint32_t> t(N);
	thrust::device_vector<uint32_t> s(N);
	thrust::sequence(s.begin(), s.end(), 0, 1);

	auto a = thrust::async::transform(thrust::device, index_sequence_begin, index_sequence_begin + N, t.begin(), lala());
	auto a2 = thrust::async::sort(thrust::device, s.begin(), s.end(), thrust::greater<uint32_t>());
	a.wait();
	a2.wait();

	std::cout << t.back() << std::endl;
	std::cout << s.back() << std::endl;

	return 0;
}

#endif //!TEST