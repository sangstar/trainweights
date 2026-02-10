//
// Created by Sanger Steel on 2/10/26.
//
#include "types.h"

const char* to_string(Device e) {
    switch (e) {
        case Device::CPU: return "CPU";
        case Device::CUDA: return "CUDA";
        case Device::Unknown: return "Unknown";
        default: return "unknown";
    }
}

const char* to_string(Datatype e) {
    switch (e) {
        case Datatype::uint8: return "uint8";
        case Datatype::uint16: return "uint16";
        case Datatype::uint32: return "uint32";
        case Datatype::uint64: return "uint64";
        case Datatype::int8: return "int8";
        case Datatype::int16: return "int16";
        case Datatype::int32: return "int32";
        case Datatype::int64: return "int64";
        case Datatype::float32: return "float32";
        case Datatype::float64: return "float64";
        case Datatype::count: return "unknown";
        default: return "unknown";
    }
}