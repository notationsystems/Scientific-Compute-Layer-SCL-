#pragma once
// Little-endian IEEE-754 byte helpers shared by every operation's
// decoder/encoder. Extracted verbatim from native/src/main.cpp when the
// second operation arrived -- the functions themselves are unchanged, they
// simply stopped being private to the CLI translation unit once more than
// one operation needed them.
//
// x86_64 is little-endian with IEEE-754 doubles, matching Python's
// struct.pack("<d", ...) byte-for-byte -- a documented environment
// assumption (docs/SCL_CONTRACT.md), not a portable deserializer.

#include <cstdint>
#include <cstring>
#include <vector>

namespace scl {

inline double read_double_le(const std::vector<uint8_t>& bytes, std::size_t offset) {
    double value;
    std::memcpy(&value, bytes.data() + offset, sizeof(double));
    return value;
}

inline void write_double_le(std::vector<uint8_t>& out, double value) {
    uint8_t buf[sizeof(double)];
    std::memcpy(buf, &value, sizeof(double));
    out.insert(out.end(), buf, buf + sizeof(double));
}

inline int32_t read_int32_le(const std::vector<uint8_t>& bytes, std::size_t offset) {
    int32_t value;
    std::memcpy(&value, bytes.data() + offset, sizeof(int32_t));
    return value;
}

}  // namespace scl
