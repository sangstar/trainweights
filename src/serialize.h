//
// Created by Sanger Steel on 2/9/26.
//

#ifndef TRAINWEIGHTS_SERIALIZE_H
#define TRAINWEIGHTS_SERIALIZE_H

#include <vector>
#include <fstream>
#include "tensor.h"

/*
 * Serialization strategy:
 *
 * Header
 *  1. Magic bytes
 *  2. Version
 *  3. Tensor count
 *
 * Tensor data (tensor_count of them):
 *  1. name
 *  2. dtype (allows mixed types)
 *  3. ndim
 *  3. dimensions
 *  4. offset
 *
 * After the last tensor data, we note the position, go to each offset, and fill the values
*/

constexpr std::uint64_t version = 1;

void write_exact(std::ofstream& f, const void* data, const std::size_t n);

void write_str(std::ofstream& f, const std::string& s);

void read_exact(std::istream& f, void* data, std::size_t n);


template<typename T>
void write_vec(std::ofstream& f, const std::vector<T>& vec) {
    // Writes num elements, bytes per element, then elements
    std::uint64_t len = vec.size();
    write_exact(f, &len, sizeof(std::uint64_t));

    Datatype type;

    std::uint64_t bytes_per_elem = sizeof(T);
    write_exact(f, &bytes_per_elem, sizeof(std::uint64_t));

    for (const auto& e : vec) {
        write_exact(f, &e, bytes_per_elem);
    }
}

template <typename T>
std::vector<T> read_vec(std::istream& f) {
    // Writes num elements, bytes per element, then elements
    std::uint64_t len;
    read_exact(f, &len, sizeof(std::uint64_t));

    std::uint64_t bytes_per_elem;

    read_exact(f, &bytes_per_elem, sizeof(std::uint64_t));

    std::vector<T> values;
    values.reserve(len);

    for (int i = 0; i < len; ++i) {
        T value;
        read_exact(f, static_cast<void *>(&value), bytes_per_elem);
        values.emplace_back(value);
    }
    return values;
}


std::string read_str(std::istream& f);


constexpr char TrainWeightsMagic[4] = "TWS";

struct Header {
    const char* magic;
    std::uint64_t version;
    std::uint64_t tensor_count;
    explicit Header(std::uint64_t version, std::size_t num_tensors) : magic(TrainWeightsMagic), version(version), tensor_count(num_tensors) {}

    void write(std::ofstream& f) const;

    static Header from_stream(std::istream& f);
};


struct TrainWeightsFile {
    Header header;
    std::vector<TensorDataView> tensors;
};

void write_header(std::ofstream& f, std::uint8_t version, std::size_t num_tensors);

void serialize(
    const std::string& outfile,
    std::vector<TensorDataView> tensors);

#endif //TRAINWEIGHTS_SERIALIZE_H
