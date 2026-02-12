#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/pair.h>
#include <string_view>
#include <utility>

#include "logger.h"
#include "tensor.h"
#include "serialize.h"
namespace nb = nanobind;

// TODO: We're relying on Python to pass diff weights, we won't compute diff weights here.
//       Just focusing on quantizing and serializing

void parse_blob(std::string name, nb::ndarray<> a) {
    TensorDataView tens = TensorDataView(name, a);
    printf("%s\n", tens.as_str().c_str());
    return;
}

void quantize_and_save(std::string filename, std::vector<std::string> names, std::vector<nb::ndarray<> > tensors) {
    if (names.size() != tensors.size()) {
        throw std::runtime_error("names and tensor lists must match in size");
    }
    auto len = names.size();
    Logger::log("Got ", len, " tensors.");
    std::vector<TensorDataView> tensor_views;
    for (int i = 0; i < len; ++i) {
        auto name = names[i];
        const auto& tensor = tensors[i];
        auto as_view = TensorDataView(name, tensor);
        std::optional<QuantizedTensor> quantized;
        switch (as_view.dims.size()) {
            case 1: quantized = quantize_i8<1>(as_view);
                break;
            case 2: quantized = quantize_i8<2>(as_view);
                break;
        }
        if (quantized.has_value()) {
            auto qt = quantized.value().to_TensorDataView();
            Logger::log("Added and quantized tensor: ", qt.as_str());
            tensor_views.emplace_back(std::move(qt));
        } else {
            throw std::runtime_error("could not quantize tensor");
        }
    }
    serialize(std::move(filename), tensor_views);
}

nb::tuple load(std::string filename) {
    auto read_stream = std::ifstream(filename);
    auto header = Header::from_stream(read_stream);
    auto read_tens = read_tensors(read_stream, header.tensor_count);

    Logger::log("Read tensors. Dequantizing...");
    for (auto& t: read_tens) {
        t.dequantize<float>();
    }

    nb::list names;
    nb::list tensors;

    for (const auto& t: read_tens) {
        names.append(t.name);

        if (!t.arr.has_value())
            throw std::runtime_error("tensor missing data");

        tensors.append(t.arr.value());
    }

    return nb::make_tuple(names, tensors);
}

NB_MODULE(trainweights, m) {
    m.def("parse_blob", &parse_blob);
    m.def("quantize_and_save", &quantize_and_save);
    m.def("load", &load);
}
