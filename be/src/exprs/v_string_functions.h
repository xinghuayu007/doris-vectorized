// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#ifndef BE_V_STRING_FUNCTIONS_H
#define BE_V_STRING_FUNCTIONS_H

#include <stdint.h>
#include <unistd.h>
#include "runtime/string_value.hpp"

#ifdef __SSE2__
#include <emmintrin.h>
#endif

namespace doris {
class VStringFunctions {
public:
#ifdef __SSE2__
    /// n equals to 16 chars length
    static constexpr auto REGISTER_SIZE = sizeof(__m128i);
#endif
public:
    static StringVal rtrim(const StringVal& str) {
        if (str.is_null || str.len == 0) {
            return str;
        }
        auto begin = 0;
        auto end = str.len - 1;
#ifdef __SSE2__
        char blank = ' ';
        const auto pattern =  _mm_set1_epi8(blank);
        while (end - begin + 1 >= REGISTER_SIZE) {
            const auto v_haystack = _mm_loadu_si128(reinterpret_cast<const __m128i *>(str.ptr + end + 1 - REGISTER_SIZE));
            const auto v_against_pattern = _mm_cmpeq_epi8(v_haystack, pattern);
            const auto mask = _mm_movemask_epi8(v_against_pattern);
            int offset = __builtin_clz(~(mask << REGISTER_SIZE));
            /// means not found
            if (offset == 0)
            {
                return StringVal(str.ptr + begin, end - begin + 1);
            } else {
                end -= offset;
            }
        }
#endif
        while (end >= begin && str.ptr[end] == ' ') {
            --end;
        }
        if (end < 0) {
            return StringVal("");
        }
        return StringVal(str.ptr + begin, end - begin + 1);
    }

    static StringVal ltrim(const StringVal& str) {
        if (str.is_null || str.len == 0) {
            return str;
        }
        auto begin = 0;
        auto end = str.len - 1;
#ifdef __SSE2__
        char blank = ' ';
        const auto pattern =  _mm_set1_epi8(blank);
        while (end - begin + 1 >= REGISTER_SIZE) {
            const auto v_haystack = _mm_loadu_si128(reinterpret_cast<const __m128i *>(str.ptr + begin));
            const auto v_against_pattern = _mm_cmpeq_epi8(v_haystack, pattern);
            const auto mask = _mm_movemask_epi8(v_against_pattern);
            const auto offset = __builtin_ctz(mask ^ 0xffff);
            /// means not found
            if (offset == 0)
            {
                return StringVal(str.ptr + begin, end - begin + 1);
            } else if (offset > REGISTER_SIZE) {
                begin += REGISTER_SIZE;
            } else {
                begin += offset;
                return StringVal(str.ptr + begin, end - begin + 1);
            }
        }
#endif
        while (begin <= end && str.ptr[begin] == ' ') {
            ++begin;
        }
        return StringVal(str.ptr + begin, end - begin + 1);
    }

    static StringVal trim(const StringVal& str) {
        if (str.is_null || str.len == 0) {
            return str;
        }
        return rtrim(ltrim(str));
    }

    static bool is_ascii(StringVal str) {
    #ifdef __SSE2__
        size_t i = 0;
        __m128i binary_code = _mm_setzero_si128();
        if (str.len >= REGISTER_SIZE) {
            for (; i <= str.len - REGISTER_SIZE; i += REGISTER_SIZE) {
                __m128i chars = _mm_loadu_si128((const __m128i*)(str.ptr + i));
                binary_code = _mm_or_si128(binary_code, chars);
            }
        }
        int mask = _mm_movemask_epi8(binary_code);

        char or_code = 0;
        for (; i < str.len; i++) {
            or_code |= str.ptr[i];
        }
        mask |= (or_code & 0x80);

        return !mask;
    #else
        char or_code = 0;
        for (size_t i = 0; i < str.len; i++) {
            or_code |= str.ptr[i];
        }
        return !(or_code & 0x80);
    #endif
    }

    static void reverse(const StringVal& str, StringVal dst) {
        if (str.is_null) {
            dst.ptr = NULL;
            return;
        }
        const bool is_ascii = VStringFunctions::is_ascii(str);
        if (is_ascii) {
            int64_t begin = 0;
            int64_t end = str.len;
            int64_t result_end = dst.len;
    #if defined(__SSE2__)
            const auto shuffle_array = _mm_set_epi64((__m64)0x00'01'02'03'04'05'06'07ull, (__m64)0x08'09'0a'0b'0c'0d'0e'0full);
            for (; (begin + REGISTER_SIZE) < end; begin += REGISTER_SIZE) {
                result_end -= REGISTER_SIZE;
                _mm_storeu_si128((__m128i*)(dst.ptr + result_end),
                                 _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)(str.ptr + begin)), shuffle_array));
            }
    #endif
            for (; begin < end; ++begin) {
                --result_end;
                dst.ptr[result_end] = str.ptr[begin];
            }
        } else {
            for (size_t i = 0, char_size = 0; i < str.len; i += char_size) {
                char_size = get_utf8_byte_length((unsigned)(str.ptr)[i]);
                std::copy(str.ptr + i, str.ptr + i + char_size, dst.ptr + str.len - i - char_size);
            }
        }
    }

    static size_t get_utf8_byte_length(unsigned char byte) {
        size_t char_size = 0;
        if (byte >= 0xFC) {
            char_size = 6;
        } else if (byte >= 0xF8) {
            char_size = 5;
        } else if (byte >= 0xF0) {
            char_size = 4;
        } else if (byte >= 0xE0) {
            char_size = 3;
        } else if (byte >= 0xC0) {
            char_size = 2;
        } else {
            char_size = 1;
        }
        return char_size;
    }

    /**
     * Refer ClickHouse code: https://github.com/ClickHouse/ClickHouse/blob/master/src/Common/UTF8Helpers.cpp
     * Definition from WiKi:
     * 对于UTF-8编码中的任意字节B，如果B的第一位为0，则B独立的表示一个字符(ASCII码)；
     * 如果B的第一位为1，第二位为0，则B为一个多字节字符中的一个字节(非ASCII字符)；
     * 如果B的前两位为1，第三位为0，则B为两个字节表示的字符中的第一个字节；
     * 如果B的前三位为1，第四位为0，则B为三个字节表示的字符中的第一个字节；
     * 如果B的前四位为1，第五位为0，则B为四个字节表示的字符中的第一个字节；
     *
     * That means the UTF8 length is equal to the number of bytes which are not match to 10XX_XXXX
     * 0xxx_xxxx, 110x_xxxx, 1110_xxxx and 1111_0xxx
     * are greater than 1011_1111 when use int8_t(negative and positive)
     * Therefore it only needs to count the number of bytes which greater than 1011_1111
     * */
    static int utf8_length(const StringVal& str) {
        int len = 0;
        int begin = 0;
        int end = begin + str.len;
#if defined(__SSE2__)
        const auto threshold = _mm_set1_epi8(0xBF);
        for (; (begin + REGISTER_SIZE) < end; begin += REGISTER_SIZE) {
            len += __builtin_popcount(
                    _mm_movemask_epi8(_mm_cmpgt_epi8(
                            _mm_loadu_si128(reinterpret_cast<const __m128i*>(str.ptr + begin)), threshold)));
        }
#endif
        for (; begin < end; ++begin) {
            len += static_cast<int8_t>(*(str.ptr + begin)) > static_cast<int8_t>(0xBF);
        }
        return len;
    }
};
}

#endif //BE_V_STRING_FUNCTIONS_H