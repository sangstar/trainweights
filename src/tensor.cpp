//
// Created by Sanger Steel on 2/5/26.
//
#include "tensor.h"
#include <complex>
#include <optional>

#include "logger.h"
#include "serialize.h"


void QuantBlockMetadata::write(std::ofstream& f) const {
    auto b_size = static_cast<std::uint64_t>(block_size);
    write_exact(f, &b_size, sizeof(std::uint64_t));

    auto s_idx = static_cast<std::uint64_t>(start_idx);
    write_exact(f, &s_idx, sizeof(std::uint64_t));

    auto e_idx = static_cast<std::uint64_t>(end_idx);
    write_exact(f, &e_idx, sizeof(std::uint64_t));

    write_exact(f, &type, sizeof(Datatype));

    write_exact(f, &scale, sizeof(float));
}

QuantBlockMetadata QuantBlockMetadata::from_stream(std::istream& f) {
    auto block = QuantBlockMetadata{};
    std::uint64_t b_size;
    read_exact(f, &b_size, sizeof(std::uint64_t));
    block.block_size = static_cast<std::size_t>(b_size);

    std::uint64_t s_idx;
    read_exact(f, &s_idx, sizeof(std::uint64_t));
    block.start_idx = static_cast<std::size_t>(s_idx);

    std::uint64_t e_idx;
    read_exact(f, &e_idx, sizeof(std::uint64_t));
    block.end_idx = static_cast<std::size_t>(e_idx);

    Datatype d;
    read_exact(f, &d, sizeof(Datatype));
    block.type = d;

    float scale_constant;
    read_exact(f, &scale_constant, sizeof(float));
    block.scale = scale_constant;

    return block;
}

std::uint64_t TensorDataView::get_nbytes() const {
    std::size_t n_elems = 1;
    for (const auto& d: dims) {
        n_elems *= d;
    }
    return n_elems * DatatypeSizes[static_cast<std::size_t>(dtype)];
}


TensorDataView::TensorDataView(const nb::ndarray<>& ar, std::optional<Datatype> type) : arr(ar) {
    // Logger::log("Trying to parse array...");
    std::vector<std::size_t> parse_dims;
    parse_dims.reserve(ar.ndim());
    for (int i = 0; i < ar.ndim(); ++i) {
        parse_dims.emplace_back(ar.shape(i));
    }
    if (parse_dims.size() == 2 && parse_dims[1] == 1) {
        dims = {parse_dims[0]};
    } else {
        dims = parse_dims;
    }
    if (type.has_value()) {
        this->dtype = type.value();
    } else {
        auto arr_dtype = ar.dtype();
        // Logger::log("got dtype code", arr_dtype.code, arr_dtype.bits, arr_dtype.lanes);
        switch (arr_dtype.code) {
            case 0: {
                switch (arr_dtype.bits) {
                    case 8: dtype = Datatype::int8;
                        break;
                    case 16: dtype = Datatype::int16;
                        break;
                    case 32: dtype = Datatype::int32;
                        break;
                    case 64: dtype = Datatype::int64;
                        break;
                    default: throw std::runtime_error("unrecognized dtype");
                        break;
                }
            }
            break;
            case 1: {
                switch (arr_dtype.bits) {
                    case 8: dtype = Datatype::uint8;
                        break;
                    case 16: dtype = Datatype::uint16;
                        break;
                    case 32: dtype = Datatype::uint32;
                        break;
                    case 64: dtype = Datatype::uint64;
                        break;
                    default: throw std::runtime_error("unrecognized dtype");
                        break;
                }
            }
            break;
            case 2: {
                switch (arr_dtype.bits) {
                    case 32: dtype = Datatype::float32;
                        break;
                    case 64: dtype = Datatype::float64;
                        break;
                    default: throw std::runtime_error("unrecognized dtype");
                        break;
                }
            }
            break;
            case 4: {
                switch (arr_dtype.bits) {
                    case 16: throw std::runtime_error("bf16 not yet supported");
                        break;
                    default: throw std::runtime_error("unrecognized dtype");
                        break;
                }
            }
            break;
            default: throw std::runtime_error("unrecognized dtype");
                break;
        }
    }
    switch (ar.device_type()) {
        case nb::device::cpu::value: device = Device::CPU;
            break;
        case nb::device::cuda::value: device = Device::CUDA;
            break;
        default: device = Device::Unknown;
    }
}

TensorDataView::TensorDataView(std::string& name, const nb::ndarray<>& ar) : TensorDataView(ar) {
    this->name = std::move(name);
}


std::string TensorDataView::as_str() const {
    std::string s;
    s.append("TensorData(name=");
    s.append(name);
    s.append(", shape=[");
    auto size = dims.size();
    for (int i = 0; i < size; ++i) {
        s.append(std::to_string(dims[i]));
        if (i != size - 1) {
            s.append(", ");
        }
    }
    s.append("], device=");
    s.append(to_string(device));
    s.append(", dtype=");
    s.append(to_string(dtype));
    s.append(", values=");
    auto str_adder_fn = TensorStrFnDispatcher[static_cast<std::size_t>(dtype)];
    str_adder_fn(this, s);
    s.append(")");
    return s;
}

std::uint64_t TensorDataView::write_metadata(std::ofstream& f) const {
    write_str(f, name);

    static_assert(sizeof(Datatype) == sizeof(std::uint32_t));
    write_exact(f, &dtype, sizeof(Datatype));

    std::uint64_t ndim = dims.size();
    write_exact(f, &ndim, sizeof(std::uint64_t));
    write_vec<std::size_t>(f, dims);

    std::uint64_t num_bytes = this->get_nbytes();
    write_exact(f, &num_bytes, sizeof(std::uint64_t));

    std::uint64_t num_blocks = 0;
    if (blocks.has_value()) {
        num_blocks = blocks.value().size();
    }
    write_exact(f, &num_blocks, sizeof(std::uint64_t));

    for (const auto& block: blocks.value()) {
        block.write(f);
    }

    auto dummy_offset_pos = f.tellp();
    std::uint64_t dummy_offset = 0;
    write_exact(f, &dummy_offset, sizeof(std::uint64_t));
    return dummy_offset_pos;
}


void TensorDataView::write_tensor_data(std::ofstream& f, std::uint64_t offset) const {
    auto blob = arr->data();
    auto correct_offset = f.tellp();
    std::size_t n_bytes = 0;
    std::size_t n_elems = 1;
    for (const auto& d: dims) {
        n_elems *= d;
    }
    n_bytes = n_elems * DatatypeSizes[static_cast<std::size_t>(dtype)];
    write_exact(f, blob, n_bytes);
    auto end_pos = f.tellp();

    f.seekp(static_cast<std::streamoff>(offset), std::ios::beg);

    std::uint64_t fixed_offset = correct_offset;
    write_exact(f, &fixed_offset, sizeof(std::uint64_t));

    f.seekp(static_cast<std::streamoff>(end_pos), std::ios::beg);
    return;
}


void write_tensors(std::ofstream& f, std::vector<TensorDataView>& tensors) {
    std::vector<std::uint64_t> offsets;
    std::uint64_t len = tensors.size();
    offsets.reserve(len);
    for (const auto& t: tensors) {
        offsets.emplace_back(t.write_metadata(f));
    }

    for (int i = 0; i < len; ++i) {
        tensors[i].write_tensor_data(f, offsets[i]);
    }
}

std::vector<TensorDataView> read_tensors(std::istream& f, std::uint64_t tensor_count) {
    std::vector<TensorDataView> tensors;
    for (auto i = 0; i < tensor_count; ++i) {
        auto name = read_str(f);

        Datatype dtype;
        read_exact(f, &dtype, sizeof(Datatype));

        std::uint64_t ndim;
        read_exact(f, &ndim, sizeof(std::uint64_t));


        auto tens = TensorDataView();
        tens.name = name;
        tens.dtype = dtype;
        tens.dims = read_vec<std::size_t>(f);
        std::uint64_t nbytes;
        std::uint64_t offset;
        read_exact(f, &nbytes, sizeof(std::uint64_t));

        std::uint64_t num_blocks;
        read_exact(f, &num_blocks, sizeof(std::uint64_t));

        std::vector<QuantBlockMetadata> block_vec;
        block_vec.reserve(num_blocks);
        if (num_blocks != 0) {
            for (int j = 0; j < num_blocks; ++j) {
                block_vec.emplace_back(QuantBlockMetadata::from_stream(f));
            }
            tens.blocks = block_vec;
        }


        read_exact(f, &offset, sizeof(std::uint64_t));

        tens.nbytes = nbytes;
        tens.offset = offset;

        tensors.emplace_back(tens);
    }

    for (auto i = 0; i < tensor_count; ++i) {
        f.seekg(tensors[i].offset.value(), std::ios::beg);

        void* data = malloc(tensors[i].nbytes.value());
        read_exact(f, data, tensors[i].nbytes.value());

        tensors[i].arr = nb::ndarray<>(
            nb::ndarray<nb::numpy, std::int8_t>(
                data,
                tensors[i].dims.size(),
                tensors[i].dims.data(),
                nb::capsule(data, [](void* p) noexcept { free(p); })));
    }
    return tensors;
}


nb::ndarray<> QuantizedTensor::to_ndarray() {
    void* buf = data.release();
    nb::ndarray<> result;
    if (is_1d) {
        dispatch_dtype(dtype, [&](auto tag) {
            using T = typename dtype_cpp<tag.value>::type;
            result = nb::ndarray<>(
                nb::ndarray<nb::numpy, T, nb::c_contig>(
                    buf,
                    {cols},
                    nb::capsule(buf, [](void* p) noexcept {
                        free(p);
                    })
                )
            );
        });
    } else {
        dispatch_dtype(dtype, [&](auto tag) {
            using T = typename dtype_cpp<tag.value>::type;
            result = nb::ndarray<>(
                nb::ndarray<nb::numpy, T, nb::c_contig>(
                    buf,
                    {rows, cols},
                    nb::capsule(buf, [](void* p) noexcept {
                        free(p);
                    })
                )
            );
        });
    }

    return result;
}

TensorDataView QuantizedTensor::to_TensorDataView() {
    auto as_array = to_ndarray();
    auto tens = TensorDataView(name, as_array);
    tens.blocks = blocks;
    return tens;
}

