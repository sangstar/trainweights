//
// Created by Sanger Steel on 2/5/26.
//

#ifndef TRAINWEIGHTS_TENSOR_H
#define TRAINWEIGHTS_TENSOR_H

#include <nanobind/ndarray.h>
#include <string_view>
#include <vector>
#include "types.h"
#include "logger.h"

namespace nb = nanobind;

#define SIZE_MAP(dtype, type) sizeof(type),

constexpr std::array<std::size_t, static_cast<std::size_t>(Datatype::count)> DatatypeSizes = {
    DTYPE_MAP(SIZE_MAP)
};

template<typename T>
using View2D = decltype(
    std::declval<nb::ndarray<> >().view<T, nb::ndim<2> >()
);


template<Datatype D>
struct QuantBuf {
};

template<>
struct QuantBuf<Datatype::float32> {
    using type = float;
};

template<>
struct QuantBuf<Datatype::int8> {
    using type = std::int8_t;
    type* data;

    explicit QuantBuf(size_t elems) {
        data = static_cast<type *>(malloc(elems * sizeof(type)));
    }

    ~QuantBuf() {
        free(data);
    }
};


struct QuantBlockMetadata {
    size_t block_size;
    size_t start_idx;
    size_t end_idx;
    Datatype type;
    float scale;

    void write(std::ofstream& f) const;

    static QuantBlockMetadata from_stream(std::istream& f);
};


struct TensorDataView {
    std::string name;
    std::vector<std::size_t> dims{};


    Device device = Device::Unknown;
    Datatype dtype = Datatype::count;

    std::optional<std::vector<QuantBlockMetadata> > blocks{};
    std::optional<nb::ndarray<> > arr;

    // Optional metadata when serializing/deserializing
    std::optional<uint64_t> offset{};
    std::optional<float> quant_scale{};
    std::optional<uint64_t> nbytes{};

    [[nodiscard]] std::uint64_t get_nbytes() const;

    TensorDataView() = default;

    explicit TensorDataView(const nb::ndarray<>& ar, std::optional<Datatype> type = std::nullopt);

    TensorDataView(std::string& name, const nb::ndarray<>& ar);

    [[nodiscard]] std::string as_str() const;

    std::uint64_t write_metadata(std::ofstream& f) const;

    void write_tensor_data(std::ofstream& f, std::uint64_t offset) const;

    // TODO: Fix to make sure the int8 and float types are templated
    void dequantize(Datatype dest) {
        if (!blocks.has_value()) {
            return;
        }

        dispatch_dtype(dest, [&](auto tag) {
            using T = typename dtype_cpp<tag.value>::type;

            Logger::log("Found block. Quantizing...");

            size_t n = arr.value().size();

            auto q = nb::ndarray<nb::numpy, std::int8_t, nb::c_contig>(*arr);

            size_t ndim = q.ndim();

            // Build shape vector
            std::vector<size_t> shape(q.ndim());
            for (size_t i = 0; i < q.ndim(); ++i)
                shape[i] = q.shape(i);

            T* buf = static_cast<T *>(malloc(n * sizeof(T)));
            // Allocate float output
            nb::ndarray<nb::numpy, T, nb::c_contig> out(
                buf,
                ndim,
                shape.data(),
                nb::capsule(buf, [](void* p) noexcept { free(p); })
            );

            const std::int8_t* original = q.data();
            T* data = out.data();

            for (const auto& block: *blocks)
                for (size_t i = block.start_idx; i < block.end_idx; ++i)
                    data[i] = static_cast<T>(original[i]) * block.scale;

            *arr = nb::ndarray<>(out);

            dtype = dest;
        });
    }
};

template<Datatype D, std::uint64_t Dim, typename... Args>
auto get_tensor_value(const TensorDataView* tens, Args&&... args) {
    using T = typename dtype_cpp<D>::type;

    return TensorView<D, Dim>::view(
        tens->arr.value(),
        std::forward<Args>(args)...
    );
}


template<Datatype D>
void add_tensor_str_values(const TensorDataView* tens, std::string& s) {

    dispatch_dtype(D, [&](auto tag) {
        using T = typename dtype_cpp<tag.value>::type;

        s.append("[");
        const size_t max_elems = 16;
        size_t count = 0;
        if (tens->dims.size() == 1) {
            for (size_t i = 0; i < tens->dims[0] && count < max_elems; ++i) {
                auto v = get_tensor_value<D, 1>(tens, i);
                s.append(std::to_string(static_cast<int>(v)));
                s.append(", ");
                ++count;
            }
        } else if (tens->dims.size() == 2) {
            for (size_t i = 0; i < tens->dims[0] && count < max_elems; ++i) {
                for (size_t j = 0; j < tens->dims[1] && count < max_elems; ++j) {
                    auto v = get_tensor_value<D, 2>(tens, i, j);
                    s.append(std::to_string(static_cast<T>(v)));
                    s.append(", ");
                    ++count;
                }
            }
        }

        if (count > 0)
            s.resize(s.size() - 2); // remove last ", "

        s.append("...]");
    });
}

#define STR_ENTRY(name, type) add_tensor_str_values<name>,


constexpr auto TensorStrFnDispatcher = std::array{
    DTYPE_MAP(STR_ENTRY)
};

static_assert(
    TensorStrFnDispatcher.size() == static_cast<size_t>(Datatype::count)
);

void write_tensors(std::ofstream& f, std::vector<TensorDataView>& tensors);

std::vector<TensorDataView> read_tensors(std::istream& f, std::uint64_t tensor_count);

template<Datatype Inp, Datatype Out, std::uint64_t Dim>
void handle(void* buf, std::vector<QuantBlockMetadata>& blocks, std::uint64_t row_idx, std::uint64_t col_idx,
            std::uint64_t block_size, std::uint64_t cols,
            TensorView<Inp, Dim>* view_fn, const nb::ndarray<>& arr) {
    float max = 0.0f;
    auto ptr = static_cast<typename QuantBuf<Out>::type *>(buf);
    auto viewer = [&](const nb::ndarray<>& array, std::uint64_t r, std::uint64_t c) -> auto {
        if constexpr (Dim == 1) {
            return view_fn->view(array, c);
        }
        if constexpr (Dim == 2) {
            return view_fn->view(array, r, c);
        }
    };
    for (std::uint64_t i = col_idx; i < col_idx + block_size; ++i) {
        auto v = viewer(arr, row_idx, i);
        if (std::abs(static_cast<float>(v)) > std::abs(static_cast<float>(max))) {
            max = v;
        }
    }
    float scale = std::fabs(static_cast<float>(max)) / 127.0;
    if (scale == 0.0) {
        scale = 1.0;
    }
    blocks.emplace_back(
        QuantBlockMetadata{
            block_size, row_idx * cols, row_idx * cols + col_idx + block_size, Out, scale
        }
    );
    for (std::uint64_t i = col_idx; i < col_idx + block_size; ++i) {
        auto v = viewer(arr, row_idx, i);
        auto q = static_cast<int8_t>(
            std::clamp(std::roundf(v / scale), -127.0f, 127.0f)
        );
        ptr[row_idx * cols + i] = q;
    }
}


void quantize_block(void* buf, Datatype dtype, std::uint64_t row_idx, std::uint64_t col_idx, std::uint64_t cols,
                    std::uint64_t block_size, const nb::ndarray<>& arr);

struct QuantizedTensor {
    std::unique_ptr<void, void(*)(void*)> data;
    Datatype dtype;
    size_t count;
    size_t rows;
    size_t cols;
    std::string name;
    bool is_1d;
    std::vector<QuantBlockMetadata> blocks;

    nb::ndarray<> to_ndarray();

    TensorDataView to_TensorDataView();
};


template<Datatype Inp, Datatype Out, std::uint64_t Dim>
void quantize_block(void* buf, std::vector<QuantBlockMetadata>& blocks, TensorView<Inp, Dim> view_fn,
                    std::uint64_t row_idx, std::uint64_t col_idx, std::uint64_t cols,
                    std::uint64_t block_size, const nb::ndarray<>& arr) {
    handle<Inp, Out, Dim>(buf, blocks, row_idx, col_idx, block_size, cols, &view_fn, arr);
}


template<Datatype Inp, Datatype Out, std::uint64_t Dim>
QuantizedTensor quantize_impl(std::uint64_t rows, std::uint64_t cols, const TensorDataView& tens) {
    using OutT = typename QuantBuf<Out>::type;
    std::uint64_t block_size = 64;
    auto view_fn = TensorView<Inp, Dim>{};
    void* quantbuf = malloc(rows * cols * sizeof(OutT));
    std::vector<QuantBlockMetadata> blocks;
    blocks.reserve(rows * cols);
    // TODO: This part can be parallelized, or just parallelize per tensor may be better
    for (std::uint64_t i = 0; i < rows; ++i) {
        for (std::uint64_t j = 0; j < cols; j += block_size) {
            size_t bw = std::min(block_size, cols - j);
            quantize_block<Inp, Out, Dim>(quantbuf, blocks, view_fn, i, j, cols, bw, tens.arr.value());
        }
    }
    return QuantizedTensor{
        {quantbuf, free},
        Out, rows * cols, rows, cols, "", false, blocks
    };
}


template<std::uint64_t Dim>
std::optional<QuantizedTensor> quantize_i8(const TensorDataView& tens) {
    std::uint64_t rows, cols;
    rows = static_cast<std::uint64_t>(tens.dims[0]);
    auto is_1d = false;
    if (tens.dims.size() == 1) {
        cols = rows;
        rows = 1;
        is_1d = true;
    } else {
        cols = static_cast<std::uint64_t>(tens.dims[1]);
    }


    nanobind::gil_scoped_release release{};

    std::optional<QuantizedTensor> out;

    dispatch_dtype(tens.dtype, [&](auto tag) {
        using T = typename dtype_cpp<tag.value>::type;

        if constexpr (std::is_floating_point_v<T>) {
            out = quantize_impl<tag.value, Datatype::int8, Dim>(rows, cols, tens);
        }
    });

    if (out.has_value()) {
        out.value().name = tens.name;
        if (is_1d) {
            out.value().is_1d = true;
        }
    }

    return out;
}


#endif //TRAINWEIGHTS_TENSOR_H
