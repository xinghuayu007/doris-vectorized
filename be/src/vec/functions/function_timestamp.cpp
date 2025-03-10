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

#include "runtime/datetime_value.h"
#include "vec/columns/column_nullable.h"
#include "vec/columns/column_string.h"
#include "vec/columns/column_vector.h"
#include "vec/data_types/data_type_date.h"
#include "vec/data_types/data_type_date_time.h"
#include "vec/functions/function_totype.h"
#include "vec/functions/simple_function_factory.h"

namespace doris::vectorized {

struct StrToDate {
    static constexpr auto name = "str_to_date";
    using ReturnType = DataTypeDateTime;
    using ColumnType = ColumnVector<Int128>;

    static void vector_vector(const ColumnString::Chars& ldata,
                              const ColumnString::Offsets& loffsets,
                              const ColumnString::Chars& rdata,
                              const ColumnString::Offsets& roffsets, ColumnType::Container& res,
                              NullMap& null_map) {
        size_t size = loffsets.size();
        res.resize(size);
        for (size_t i = 0; i < size; ++i) {
            const char* l_raw_str = reinterpret_cast<const char*>(&ldata[loffsets[i - 1]]);
            int l_str_size = loffsets[i] - loffsets[i - 1] - 1;

            const char* r_raw_str = reinterpret_cast<const char*>(&rdata[roffsets[i - 1]]);
            int r_str_size = roffsets[i] - roffsets[i - 1] - 1;

            auto& ts_val = *reinterpret_cast<DateTimeValue*>(&res[i]);
            if (!ts_val.from_date_format_str(r_raw_str, r_str_size, l_raw_str, l_str_size)) {
                null_map[i] = 1;
            }
            ts_val.to_datetime();
        }
    }
};

struct NameMakeDate {
    static constexpr auto name = "makedate";
};

template <typename LeftDataType, typename RightDataType>
struct MakeDateImpl {
    using ResultDataType = DataTypeDateTime;
    using LeftDataColumnType = ColumnVector<typename LeftDataType::FieldType>;
    using RightDataColumnType = ColumnVector<typename RightDataType::FieldType>;
    using ColumnType = ColumnVector<Int128>;

    static void vector_vector(const typename LeftDataColumnType::Container& ldata,
                              const typename RightDataColumnType::Container& rdata,
                              ColumnType::Container& res, NullMap& null_map) {
        auto len = ldata.size();
        res.resize(len);

        for (size_t i = 0; i < len; ++i) {
            const auto& l = ldata[i];
            const auto& r = rdata[i];
            if (r <= 0 || l < 0 || l > 9999) {
                null_map[i] = 1;
                continue;
            }

            auto& res_val = *reinterpret_cast<DateTimeValue*>(&res[i]);

            DateTimeValue ts_value{l * 10000000000 + 101000000};
            ts_value.set_type(TIME_DATE);
            DateTimeVal ts_val;
            ts_value.to_datetime_val(&ts_val);
            if (ts_val.is_null) {
                null_map[i] = 1;
                continue;
            }

            TimeInterval interval(DAY, r - 1, false);
            res_val = DateTimeValue::from_datetime_val(ts_val);
            if (!res_val.date_add_interval(interval, DAY)) {
                null_map[i] = 1;
                continue;
            }
            res_val.cast_to_date();
        }
    }
};

using FunctionStrToDate = FunctionBinaryStringOperateToNullType<StrToDate>;
using FunctionMakeDate = FunctionBinaryToNullType<DataTypeInt32, DataTypeInt32, MakeDateImpl, NameMakeDate>;

void register_function_timestamp(SimpleFunctionFactory& factory) {
    factory.register_function<FunctionStrToDate>();
    factory.register_function<FunctionMakeDate>();
}

} // namespace doris::vectorized
