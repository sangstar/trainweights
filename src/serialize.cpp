//
// Created by Sanger Steel on 2/9/26.
//

#include "serialize.h"

void write_exact(std::ofstream& f, const void* data, const std::size_t n) {
    f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(n));
    if (!f) throw std::runtime_error("write failed");
}

void read_exact(std::istream& f, void* data, const std::size_t n) {
    f.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(n));
    if (!f) throw std::runtime_error("read failed");
}

void write_str(std::ofstream& f, const std::string& s) {
    std::size_t len = s.size();
    write_exact(f, &len, sizeof(std::size_t));
    write_exact(f, s.c_str(), sizeof(char) * len);
}


std::string read_str(std::istream& f) {
    std::size_t len = 0;
    read_exact(f, &len, sizeof(std::size_t));
    std::string s("", len);
    read_exact(f, s.data(), len);
    return s;
}

void Header::write(std::ofstream& f) const {
    write_exact(f, magic, 4);
    write_exact(f, &version, sizeof(std::uint64_t));
    write_exact(f, &tensor_count, sizeof(std::uint64_t));
}

Header Header::from_stream(std::istream& f) {
    char magic[4];
    read_exact(f, magic, 4);
    if (strcmp(magic, TrainWeightsMagic) != 0) {
        throw std::runtime_error("invalid magic");
    }
    std::uint64_t version;
    read_exact(f, &version, sizeof(std::uint64_t));

    std::uint64_t tensor_count;
    read_exact(f, &tensor_count, sizeof(std::uint64_t));

    return Header(version, tensor_count);
}

void serialize(const std::string& outfile, std::vector<TensorDataView> tensors) {
    std::ofstream stream(outfile);
    Header header = Header(version, tensors.size());
    header.write(stream);
    write_tensors(stream, tensors);
}
