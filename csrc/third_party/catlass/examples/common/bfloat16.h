/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 * Copyright 2017 The TensorFlow Authors. All Rights Reserved.
 * Copyright (c) 2016-     Facebook, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef OP_API_BFLOAT16_H
#define OP_API_BFLOAT16_H

#include <cmath>
#include <iostream>
#include "fp16_t.h"

namespace op {

// see framework/bfloat16.h for description.
struct bfloat16 {
    struct from_bits_t {};
    static constexpr from_bits_t from_bits()
    {
        return from_bits_t();
    }

    constexpr bfloat16(uint16_t bits, [[maybe_unused]] from_bits_t fromBits) : value(bits)
    {
    }

    // The default constructor must yield a zero value, not an uninitialized
    // value; some TF kernels use T() as a zero value.
    bfloat16() : value(ZERO_VALUE)
    {}

    bfloat16(float v)
    {
        value = round_to_bfloat16(v).value;
    }

    template<class T>
    bfloat16(const T &val)
        : bfloat16(static_cast<float>(val))
    {}

    template<typename T>
    bfloat16 &operator=(T other)
    {
        value = round_to_bfloat16(static_cast<float>(other)).value;
        return *this;
    }

    operator float() const
    {
        float result = 0;

        uint16_t *q = reinterpret_cast<uint16_t *>(&result);

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
        q[0] = value;
#else
        q[1] = value;
#endif
        return result;
    }

    operator bool() const
    {
        return std::abs(float(*this)) >= std::numeric_limits<float>::epsilon();
    }

    operator short() const
    {
        return static_cast<short>(float(*this));
    }

    operator int() const
    {
        return static_cast<int>(float(*this));
    }

    operator long() const
    {
        return static_cast<long>(float(*this));
    }

    operator char() const
    {
        return static_cast<char>(float(*this));
    }

    operator signed char() const
    {
        return static_cast<signed char>(float(*this));
    }

    operator unsigned char() const
    {
        return static_cast<unsigned char>(float(*this));
    }

    operator unsigned short() const
    {
        return static_cast<unsigned short>(float(*this));
    }

    operator unsigned int() const
    {
        return static_cast<unsigned int>(float(*this));
    }

    operator unsigned long() const
    {
        return static_cast<unsigned long>(float(*this));
    }

    operator unsigned long long() const
    {
        return static_cast<unsigned long long>(float(*this));
    }

    operator long long() const
    {
        return static_cast<long long>(float(*this));
    }

    operator double() const
    {
        return static_cast<double>(float(*this));
    }

    union FP32 {
        unsigned int u;
        float f;
    };

    // Converts a float point to bfloat16, with round-nearest-to-even as rounding
    // method.
    // There is a slightly faster implementation (8% faster on CPU)
    // than this (documented in cl/175987786), that is exponentially harder to
    // understand and document. Switch to the faster version when converting to
    // BF16 becomes compute-bound.
    static bfloat16 round_to_bfloat16(float v)
    {
        uint32_t input;
        FP32 f;
        f.f = v;
        input = f.u;
        bfloat16 output;

        if (float_isnan(v)) {
            // If the value is a NaN, squash it to a qNaN with msb of fraction set,
            // this makes sure after truncation we don't end up with an inf.
            //
            // qNaN magic: All exponent bits set + most significant bit of fraction
            // set.
            output.value = 0x7fc0;
        } else {
            constexpr uint32_t bfloat16_bit_len = 16;
            uint32_t lsb = (input >> bfloat16_bit_len) & 1;
            uint32_t rounding_bias = 0x7fff + lsb;
            input += rounding_bias;
            output.value = static_cast<uint16_t>(input >> bfloat16_bit_len);
        }
        return output;
    }

    constexpr static bfloat16 epsilon()
    {
        return bfloat16(0x3c00, from_bits());
    }

    constexpr static bfloat16 highest()
    {
        return bfloat16(0x7F7F, from_bits());
    }

    constexpr static bfloat16 lowest()
    {
        return bfloat16(0xFF7F, from_bits());
    }

    constexpr static bfloat16 min_positive_normal()
    {
        return bfloat16(0x0080, from_bits());
    }

    bool IsZero() const
    { return (value & 0x7FFF) == ZERO_VALUE; }

    uint16_t value;

    // A value that represents "not a number".
    static constexpr uint16_t NAN_VALUE = 0x7FC0;

private:
    // A value that represents "zero".
    static constexpr uint16_t ZERO_VALUE = 0;

    static bool float_isnan(const float &x)
    {
        return std::isnan(x);
    }
};

inline std::ostream &operator<<(std::ostream &os,
                                const bfloat16 &dt)
{
    os << static_cast<float>(dt);
    return os;
}

inline bfloat16 operator+(bfloat16 a, bfloat16 b)
{
    return bfloat16(static_cast<float>(a) + static_cast<float>(b));
}
inline bfloat16 operator+(bfloat16 a, int b)
{
    return bfloat16(static_cast<float>(a) + static_cast<float>(b));
}
inline bfloat16 operator+(int a, bfloat16 b)
{
    return bfloat16(static_cast<float>(a) + static_cast<float>(b));
}
inline bfloat16 operator-(bfloat16 a, bfloat16 b)
{
    return bfloat16(static_cast<float>(a) - static_cast<float>(b));
}
inline bfloat16 operator*(bfloat16 a, bfloat16 b)
{
    return bfloat16(static_cast<float>(a) * static_cast<float>(b));
}
inline bfloat16 operator/(bfloat16 a, bfloat16 b)
{
    return bfloat16(static_cast<float>(a) / static_cast<float>(b));
}
inline bfloat16 operator-(bfloat16 a)
{
    a.value ^= 0x8000;
    return a;
}

inline bool operator<(bfloat16 a, bfloat16 b)
{
    return static_cast<float>(a) < static_cast<float>(b);
}

template<typename T>
bool operator<(T a, bfloat16 b)
{
    return static_cast<float>(a) < static_cast<float>(b);
}

inline bool operator<=(bfloat16 a, bfloat16 b)
{
    return static_cast<float>(a) <= static_cast<float>(b);
}
inline bool operator==(bfloat16 a, bfloat16 b)
{
    return a.value == b.value;
}
inline bool operator!=(bfloat16 a, bfloat16 b)
{
    return a.value != b.value;
}

template<typename T>
bool operator>(T a, bfloat16 b)
{
    return static_cast<float>(a) > static_cast<float>(b);
}
inline bool operator>=(bfloat16 a, bfloat16 b)
{
    return static_cast<float>(a) >= static_cast<float>(b);
}
inline bfloat16 &operator+=(bfloat16 &a, bfloat16 b)
{
    a = a + b;
    return a;
}
inline bfloat16 &operator-=(bfloat16 &a, bfloat16 b)
{
    a = a - b;
    return a;
}
inline bfloat16 operator++(bfloat16 &a)
{
    a += bfloat16(1);
    return a;
}
inline bfloat16 operator--(bfloat16 &a)
{
    a -= bfloat16(1);
    return a;
}
inline bfloat16 operator++(bfloat16 &a, int)
{
    bfloat16 original_value = a;
    ++a;
    return original_value;
}
inline bfloat16 operator--(bfloat16 &a, int)
{
    bfloat16 original_value = a;
    --a;
    return original_value;
}
inline bfloat16 &operator*=(bfloat16 &a, bfloat16 b)
{
    a = a * b;
    return a;
}
inline bfloat16 &operator/=(bfloat16 &a, bfloat16 b)
{
    a = a / b;
    return a;
}
} // end namespace op

namespace std {
template<>
struct hash<op::bfloat16> {
    size_t operator()(const op::bfloat16 &v) const
    {
        return hash<float>()(static_cast<float>(v));
    }
};

using op::bfloat16;
inline bool isinf(const bfloat16 &a)
{ return std::isinf(float(a)); }
inline bool isnan(const bfloat16 &a)
{ return std::isnan(float(a)); }
inline bool isfinite(const bfloat16 &a)
{ return std::isfinite(float(a)); }
inline bfloat16 abs(const bfloat16 &a)
{ return bfloat16(std::abs(float(a))); }
inline bfloat16 exp(const bfloat16 &a)
{ return bfloat16(std::exp(float(a))); }
inline bfloat16 log(const bfloat16 &a)
{ return bfloat16(std::log(float(a))); }
inline bfloat16 log10(const bfloat16 &a)
{
    return bfloat16(std::log10(float(a)));
}
inline bfloat16 sqrt(const bfloat16 &a)
{
    return bfloat16(std::sqrt(float(a)));
}
inline bfloat16 pow(const bfloat16 &a, const bfloat16 &b)
{
    return bfloat16(std::pow(float(a), float(b)));
}
inline bfloat16 sin(const bfloat16 &a)
{ return bfloat16(std::sin(float(a))); }
inline bfloat16 cos(const bfloat16 &a)
{ return bfloat16(std::cos(float(a))); }
inline bfloat16 tan(const bfloat16 &a)
{ return bfloat16(std::tan(float(a))); }
inline bfloat16 tanh(const bfloat16 &a)
{
    return bfloat16(std::tanh(float(a)));
}
inline bfloat16 floor(const bfloat16 &a)
{
    return bfloat16(std::floor(float(a)));
}
inline bfloat16 ceil(const bfloat16 &a)
{
    return bfloat16(std::ceil(float(a)));
}

template<>
class numeric_limits<op::bfloat16> {
public:
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;
    static constexpr bool has_signaling_NaN = true;
    static constexpr bool is_bounded = true;
    static constexpr bool is_exact = false;
    static constexpr bool is_integer = false;
    static constexpr bool is_iec559 = false;
    static constexpr bool is_modulo = false;
    static constexpr bool is_signed = true;
    static constexpr bool is_specialized = true;
    static constexpr int digits = 8;
    static constexpr int digits10 = 2;
    static constexpr int max_digits10 = 4;
    static constexpr int min_exponent = -125;
    static constexpr int min_exponent10 = -37;
    static constexpr int max_exponent = 128;
    static constexpr int max_exponent10 = 38;
    static constexpr int radix = 2;
    static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
    static constexpr auto has_denorm_loss = numeric_limits<float>::has_denorm_loss;
    static constexpr auto round_style = numeric_limits<float>::round_style;
    static constexpr auto traps = numeric_limits<float>::traps;
    static constexpr auto tinyness_before = numeric_limits<float>::tinyness_before;

    static constexpr op::bfloat16 min()
    {
        return op::bfloat16::min_positive_normal();
    }
    static constexpr op::bfloat16 lowest()
    {
        return op::bfloat16::lowest();
    }
    static constexpr op::bfloat16 max()
    {
        return op::bfloat16::highest();
    }
    static constexpr op::bfloat16 epsilon()
    {
        return op::bfloat16::epsilon();
    }
    static constexpr op::bfloat16 round_error()
    {
        return op::bfloat16(0x3F00, op::bfloat16::from_bits());
    }
    static constexpr op::bfloat16 infinity()
    {
        return op::bfloat16(0x7F80, op::bfloat16::from_bits());
    }
    static constexpr op::bfloat16 quiet_NaN()
    {
        return op::bfloat16(op::bfloat16::NAN_VALUE, op::bfloat16::from_bits());
    }
    static constexpr op::bfloat16 signaling_NaN()
    {
        return op::bfloat16(0x7F80, op::bfloat16::from_bits());
    }
    static constexpr op::bfloat16 denorm_min()
    {
        return op::bfloat16(0x0001, op::bfloat16::from_bits());
    }
};

} // namespace std


#endif // OP_API_BFLOAT16_H
