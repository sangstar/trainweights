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


struct TrainWeightsLoader {
    std::ifstream stream;
    Header header;
    std::vector<TensorDataView> tensors;

    nb::tuple load_tensor(std::string name) const;

    TrainWeightsLoader(std::string filename)
        : stream(filename),
          header(Header::from_stream(stream)),
          tensors(read_tensors(stream, header.tensor_count))
    {}
};

nb::tuple TrainWeightsLoader::load_tensor(std::string name) const {
    auto specific_tensor = read_tensor(tensors, name.c_str());
    specific_tensor.dequantize(Datatype::float32);
    return nb::make_tuple(name, specific_tensor.arr.value());
}

void parse_blob(std::string name, nb::ndarray<> a) {
    TensorDataView tens = TensorDataView(name, a);
    printf("%s\n", tens.as_str().c_str());
    return;
}

void save(std::string filename, std::vector<std::string> names, std::vector<nb::ndarray<> > tensors) {
    if (names.size() != tensors.size()) {
        throw std::runtime_error("names and tensor lists must match in size");
    }
    logger.debug("Trying to save...");
    auto len = names.size();
    std::vector<TensorDataView> tensor_views;
    for (int i = 0; i < len; ++i) {
        auto name = names[i];
        const auto &tensor = tensors[i];
        logger.debugf("trying to instantiate tensor %s", name.c_str());
        auto tens = TensorDataView(name, tensor);
        logger.debug("got tensor ", tens.as_str().c_str());
        tensor_views.emplace_back(std::move(tens));
    }
    serialize(filename, tensor_views);
}

void quantize_and_save(std::string filename, std::vector<std::string> names, std::vector<nb::ndarray<> > tensors) {
    if (names.size() != tensors.size()) {
        throw std::runtime_error("names and tensor lists must match in size");
    }
    auto len = names.size();
    // Logger::log("Got ", len, " tensors.");
    std::vector<TensorDataView> tensor_views;
    for (int i = 0; i < len; ++i) {
        auto name = names[i];
        const auto &tensor = tensors[i];
        auto as_view = TensorDataView(name, tensor);
        std::optional<QuantizedTensor> quantized;
        if (as_view.dtype == Datatype::float32 || as_view.dtype == Datatype::float64) {
            switch (as_view.dims.size()) {
                case 1: quantized = quantize_i8<1>(as_view);
                    break;
                case 2: quantized = quantize_i8<2>(as_view);
                    break;
                default: throw std::runtime_error("quantization for this shape is unsupported");
            }
            if (quantized.has_value()) {
                auto qt = quantized.value().to_TensorDataView();
                // Logger::log("Added and quantized tensor: ", qt.as_str());
                tensor_views.emplace_back(std::move(qt));
            } else {
                throw std::runtime_error("could not quantize tensor");
            }
        } else {
            tensor_views.emplace_back(as_view);
        }

    }
    serialize(filename, tensor_views);
}

nb::tuple load(std::string filename) {
    auto read_stream = std::ifstream(filename);
    auto header = Header::from_stream(read_stream);
    logger.debugf("Header{version=%llu, tensor_count=%llu}", header.version, header.tensor_count);
    logger.debug("reading tensors...");
    auto read_tens = read_tensors(read_stream, header.tensor_count);

    // Logger::log("Read tensors. Dequantizing...");
    for (auto &t: read_tens) {
        if (t.dtype == Datatype::int8) {
            logger.debugf("dequantizing tensor %s", t.name.c_str());
            t.dequantize(Datatype::float32);
        }
    }

    nb::list names;
    nb::list tensors;

    for (const auto &t: read_tens) {
        names.append(t.name);

        if (!t.arr.has_value())
            throw std::runtime_error("tensor missing data");

        tensors.append(t.arr.value());
    }

    return nb::make_tuple(names, tensors);
}

NB_MODULE(_C, m) {
    m.def("quantize_and_save", &quantize_and_save);
    m.def("save", &save);
    m.def("load", &load);
    nb::class_<TrainWeightsLoader>(m, "TrainWeightsLoader")
        .def(nb::init<std::string>())
        .def("load_tensor", &TrainWeightsLoader::load_tensor);

}
