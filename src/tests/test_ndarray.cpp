#include <iostream>

#include "vector"
#include "../tensor.h"
#include "../serialize.h"


// Messy code just for basic validation checks


int test_create_ndarray() {
    constexpr int ndim = 2;
    size_t shape[ndim] = {1, 2};
    char data[] = "this is a test";
    auto array = nb::ndarray(data, ndim, shape);
    return 0;
}

int test_quantize_tensor() {
    constexpr int ndim = 1;
    size_t shape[ndim] = {6};
    float data[] = {1, 2, 3, 4, 5, 6};
    auto array = nb::ndarray(data, ndim, shape);
    auto tens = TensorDataView(array, Datatype::float32);
    auto qt = quantize_i8<1>(tens);
    if (!qt.has_value()) {
        throw std::runtime_error("tensor not quantized");
    }
    auto as_view = qt.value().to_TensorDataView();
    auto buf = (std::int8_t *) as_view.arr.value().data();
    printf("%s\n", as_view.as_str().c_str());
}

int test_write_str() {
    auto stream = std::ofstream("test.txt");
    auto text = std::string("foo bar");
    write_str(stream, text);

    stream.close();
    auto read = std::ifstream("test.txt");
    auto str = read_str(read);
    if (strcmp(str.c_str(), "foo bar") != 0) {
        throw std::runtime_error("test failed to read the written value");
    }
}

int test_read_write_header() {
    auto stream = std::ofstream("header.txt");
    Header header = Header(1, 3);

    header.write(stream);
    stream.close();

    auto read = std::ifstream("header.txt");
    auto read_header = Header::from_stream(read);
    assert(read_header.magic == header.magic);
    assert(read_header.version == header.version);
    assert(read_header.tensor_count == header.tensor_count);
}

int test_quantize_tensor_big() {
    constexpr int ndim = 2;
    size_t shape[ndim] = {4, 8};

    float data[] = {
        -10, -5, -2, -1, 0, 1, 2, 5,
        10, 20, 30, 40, 50, 60, 80, 100,
        -100, -80, -60, -40, -20, -10, -5, -1,
        3, 7, 15, 25, 55, 75, 90, 120
    };

    auto array = nb::ndarray(data, ndim, shape);
    auto tens = TensorDataView(array, Datatype::float32);

    auto qt = quantize_i8<2>(tens);
    if (qt.has_value()) {
        void* raw = qt->data.get();
        auto buf = reinterpret_cast<QuantBuf<Datatype::int8>::type *>(raw);
        printf("Quantized values:\n");
        for (size_t i = 0; i < 32; ++i) {
            printf("%4d ", buf[i]);
            if ((i + 1) % 8 == 0) printf("\n");
        }
    }
    auto qt_tens = qt->to_TensorDataView();
}

int test_write_and_read_tensors() {
    std::vector<TensorDataView> tensors;

    constexpr int ndim = 2;
    size_t shape[ndim] = {4, 8};

    float data[] = {
        -10, -5, -2, -1, 0, 1, 2, 5,
        10, 20, 30, 40, 50, 60, 80, 100,
        -100, -80, -60, -40, -20, -10, -5, -1,
        3, 7, 15, 25, 55, 75, 90, 120
    };

    auto array = nb::ndarray(data, ndim, shape);
    auto tens_a = TensorDataView(array, Datatype::float32);
    tens_a.name = std::string("tensor_a");
    tensors.emplace_back(tens_a);

    constexpr int ndim_b = 2;
    size_t shape_b[ndim_b] = {1, 6};
    float data_b[] = {1, 2, 3, 4, 5, 6};
    auto array_b = nb::ndarray(data_b, ndim_b, shape_b);
    auto tens_b = TensorDataView(array_b, Datatype::float32);
    tens_b.name = std::string("tensor_b");

    tensors.emplace_back(tens_b);
    auto len = tensors.size();
    std::vector<std::string> names = {"tensor_a", "tensor_b"};
    std::vector<TensorDataView> quantized_tensors;
    for (int i = 0; i < len; ++i) {
        auto name = names[i];
        const auto& tensor = tensors[i];
        std::optional<QuantizedTensor> quantized;
        switch (tensor.dims.size()) {
            case 1: quantized = quantize_i8<1>(tensor);
                break;
            case 2: quantized = quantize_i8<2>(tensor);
                break;
        }
        if (quantized.has_value()) {
            auto qt = quantized.value().to_TensorDataView();
            quantized_tensors.emplace_back(std::move(qt));
        } else {
            throw std::runtime_error("could not quantize tensor");
        }
    }
    serialize("tensor_test.tws", quantized_tensors);


    auto read_stream = std::ifstream("tensor_test.tws");
    auto header = Header::from_stream(read_stream);
    auto read_tens = read_tensors(read_stream, header.tensor_count);
    for (auto& t: read_tens) {
        t.dequantize(Datatype::float32);
    }

    for (auto& t: read_tens) {
        printf("%s\n", t.as_str().c_str());
    }
}

void test_dequantize_double() {
    std::vector<TensorDataView> tensors;

    constexpr int ndim = 2;
    size_t shape[ndim] = {4, 8};

    double data[] = {
        -10.35, -5.235, -2, -1, 0, 1, 2, 5,
        10.532151, 20.523, 3.50, 40.123, 50.235, 60.5, 80, 100,
        -100, -80.5312, -60, -40.2, -20, -10, -5, -1,
        3, 7, 15, 25, 55, 75.52351, 90, 120.1235
    };

    auto array = nb::ndarray(data, ndim, shape);
    auto tens_a = TensorDataView(array, Datatype::float64);
    tens_a.name = std::string("tensor_a");
    tensors.emplace_back(tens_a);

    constexpr int ndim_b = 2;
    size_t shape_b[ndim_b] = {1, 6};
    double data_b[] = {1, 2, 3, 4, 5, 6};
    auto array_b = nb::ndarray(data_b, ndim_b, shape_b);
    auto tens_b = TensorDataView(array_b, Datatype::float64);
    tens_b.name = std::string("tensor_b");

    tensors.emplace_back(tens_b);
    auto len = tensors.size();
    std::vector<std::string> names = {"tensor_a", "tensor_b"};
    std::vector<TensorDataView> quantized_tensors;
    for (int i = 0; i < len; ++i) {
        auto name = names[i];
        const auto& tensor = tensors[i];
        std::optional<QuantizedTensor> quantized;
        switch (tensor.dims.size()) {
            case 1: quantized = quantize_i8<1>(tensor);
                break;
            case 2: quantized = quantize_i8<2>(tensor);
                break;
        }
        if (quantized.has_value()) {
            auto qt = quantized.value().to_TensorDataView();
            quantized_tensors.emplace_back(std::move(qt));
        } else {
            throw std::runtime_error("could not quantize tensor");
        }
    }
    serialize("tensor_test.tws", quantized_tensors);


    auto read_stream = std::ifstream("tensor_test.tws");
    auto header = Header::from_stream(read_stream);
    auto read_tens = read_tensors(read_stream, header.tensor_count);
    for (auto& t: read_tens) {
        t.dequantize(Datatype::float64);
    }

    for (auto& t: read_tens) {
        printf("%s\n", t.as_str().c_str());
    }
}

void test_print_1D_tensor() {
    constexpr int ndim = 1;
    size_t shape[ndim] = {6};

    float data[] = {
        -10, -5, -2, -1, 0, 1,
    };

    auto array = nb::ndarray(data, ndim, shape);
    auto tens = TensorDataView(array, Datatype::float32);
    printf("%s\n", tens.as_str().c_str());
}

int main() {
    Py_Initialize();
    test_dequantize_double();
    test_quantize_tensor();
    test_print_1D_tensor();
    test_quantize_tensor_big();
    test_write_str();
    test_read_write_header();
    test_write_and_read_tensors();
    Py_Finalize();
    return 0;
}
