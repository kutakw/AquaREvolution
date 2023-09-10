#ifndef ALLOCATOR_CUH
#define ALLOCATOR_CUH

#include <thrust/device_ptr.h>
#include <thrust/iterator/detail/normal_iterator.h>

template <typename T>
using iter = thrust::detail::normal_iterator<thrust::device_ptr<T>>;

template <typename... Args>
using entity = thrust::tuple<Args...>;

template <typename... Args>
using entityIter = thrust::tuple<iter<Args>...>;
#endif // !ALLOCATOR_CUH
