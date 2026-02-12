//
// Created by Sanger Steel on 2/10/26.
//

#ifndef TRAINWEIGHTS_TYPES_H
#define TRAINWEIGHTS_TYPES_H
#include <nanobind/ndarray.h>
#include <type_traits>

namespace nb = nanobind;

enum class Device {
    CPU,
    CUDA,
    Unknown
};

const char* to_string(Device e);

enum class Datatype : std::uint32_t {
    uint8,
    uint16,
    uint32,
    uint64,
    int8,
    int16,
    int32,
    int64,
    float32,
    float64,
    count,
};


const char* to_string(Datatype e);

#define DTYPE_MAP(X) \
    X(Datatype::uint8, std::uint8_t) \
    X(Datatype::uint16, std::uint16_t) \
    X(Datatype::uint32, std::uint32_t) \
    X(Datatype::uint64, std::uint64_t) \
    X(Datatype::int8, std::int8_t) \
    X(Datatype::int16, std::int16_t) \
    X(Datatype::int32, std::int32_t) \
    X(Datatype::int64, std::int64_t) \
    X(Datatype::float32, float) \
    X(Datatype::float64, double)

template<Datatype D>
struct dtype_cpp;

#define X(name, cpp_type) \
    template<> struct dtype_cpp<name> { using type = cpp_type; };

DTYPE_MAP(X)

template<typename T, std::uint64_t Dim>
using View = decltype(
    std::declval<nb::ndarray<> >().view<T, nb::ndim<Dim> >()
);

static_assert(std::is_same_v<
    dtype_cpp<Datatype::float32>::type,
    float>);

template<typename Closure>
void dispatch_dtype(Datatype dtype, Closure&& f) {
    switch (dtype) {
#define CASES(name, cpp_type) \
case name: f(std::integral_constant<Datatype, name>{}); break;

        DTYPE_MAP(CASES)
        default: throw std::runtime_error("unsupported dtype");
#undef CASES
    }
}

template<Datatype D, std::uint64_t Dim>
struct TensorView {
    using type = typename dtype_cpp<D>::type;

    template<typename... Args>
    static auto view(const nb::ndarray<>& arr, Args&&... args) {
        return arr.view<type, nb::ndim<Dim>>()(std::forward<Args>(args)...);
    }
};


#endif //TRAINWEIGHTS_TYPES_H
