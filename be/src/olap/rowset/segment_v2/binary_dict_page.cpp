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

#include "olap/rowset/segment_v2/binary_dict_page.h"

#include "common/logging.h"
#include "gutil/strings/substitute.h" // for Substitute
#include "runtime/mem_pool.h"
#include "util/slice.h" // for Slice
#include "vec/columns/column_vector.h"
#include "vec/columns/column_string.h"

namespace doris {
namespace segment_v2 {

using strings::Substitute;

BinaryDictPageBuilder::BinaryDictPageBuilder(const PageBuilderOptions& options)
        : _options(options),
          _finished(false),
          _data_page_builder(nullptr),
          _dict_builder(nullptr),
          _encoding_type(DICT_ENCODING),
          _tracker(new MemTracker()),
          _pool(_tracker.get()) {
    // initially use DICT_ENCODING
    // TODO: the data page builder type can be created by Factory according to user config
    _data_page_builder.reset(new BitshufflePageBuilder<OLAP_FIELD_TYPE_INT>(options));
    PageBuilderOptions dict_builder_options;
    dict_builder_options.data_page_size = _options.dict_page_size;
    _dict_builder.reset(new BinaryPlainPageBuilder(dict_builder_options));
    reset();
}

bool BinaryDictPageBuilder::is_page_full() {
    if (_data_page_builder->is_page_full()) {
        return true;
    }
    if (_encoding_type == DICT_ENCODING && _dict_builder->is_page_full()) {
        return true;
    }
    return false;
}

Status BinaryDictPageBuilder::add(const uint8_t* vals, size_t* count) {
    if (_encoding_type == DICT_ENCODING) {
        DCHECK(!_finished);
        DCHECK_GT(*count, 0);
        const Slice* src = reinterpret_cast<const Slice*>(vals);
        size_t num_added = 0;
        uint32_t value_code = -1;

        if (_data_page_builder->count() == 0) {
            _first_value.assign_copy(reinterpret_cast<const uint8_t*>(src->get_data()),
                                     src->get_size());
        }

        for (int i = 0; i < *count; ++i, ++src) {
            auto iter = _dictionary.find(*src);
            if (iter != _dictionary.end()) {
                value_code = iter->second;
            } else {
                if (_dict_builder->is_page_full()) {
                    break;
                }
                Slice dict_item(src->data, src->size);
                if (src->size > 0) {
                    char* item_mem = (char*)_pool.allocate(src->size);
                    if (item_mem == nullptr) {
                        return Status::MemoryAllocFailed(
                                strings::Substitute("memory allocate failed, size:$0", src->size));
                    }
                    dict_item.relocate(item_mem);
                }
                value_code = _dictionary.size();
                _dictionary.emplace(dict_item, value_code);
                _dict_items.push_back(dict_item);
                _dict_builder->update_prepared_size(dict_item.size);
            }
            size_t add_count = 1;
            RETURN_IF_ERROR(_data_page_builder->add(reinterpret_cast<const uint8_t*>(&value_code),
                                                    &add_count));
            if (add_count == 0) {
                // current data page is full, stop processing remaining inputs
                break;
            }
            num_added += 1;
        }
        *count = num_added;
        return Status::OK();
    } else {
        DCHECK_EQ(_encoding_type, PLAIN_ENCODING);
        return _data_page_builder->add(vals, count);
    }
}

OwnedSlice BinaryDictPageBuilder::finish() {
    DCHECK(!_finished);
    _finished = true;

    OwnedSlice data_slice = _data_page_builder->finish();
    // TODO(gaodayue) separate page header and content to avoid this copy
    _buffer.append(data_slice.slice().data, data_slice.slice().size);
    encode_fixed32_le(&_buffer[0], _encoding_type);
    return _buffer.build();
}

void BinaryDictPageBuilder::reset() {
    _finished = false;
    _buffer.reserve(_options.data_page_size + BINARY_DICT_PAGE_HEADER_SIZE);
    _buffer.resize(BINARY_DICT_PAGE_HEADER_SIZE);

    if (_encoding_type == DICT_ENCODING && _dict_builder->is_page_full()) {
        _data_page_builder.reset(new BinaryPlainPageBuilder(_options));
        _encoding_type = PLAIN_ENCODING;
    } else {
        _data_page_builder->reset();
    }
    _finished = false;
}

size_t BinaryDictPageBuilder::count() const {
    return _data_page_builder->count();
}

uint64_t BinaryDictPageBuilder::size() const {
    return _pool.total_allocated_bytes() + _data_page_builder->size();
}

Status BinaryDictPageBuilder::get_dictionary_page(OwnedSlice* dictionary_page) {
    _dictionary.clear();
    _dict_builder->reset();
    size_t add_count = 1;
    // here do not check is_page_full of dict_builder
    // because it is checked in add
    for (auto& dict_item : _dict_items) {
        RETURN_IF_ERROR(
                _dict_builder->add(reinterpret_cast<const uint8_t*>(&dict_item), &add_count));
    }
    *dictionary_page = _dict_builder->finish();
    _dict_items.clear();
    return Status::OK();
}

Status BinaryDictPageBuilder::get_first_value(void* value) const {
    DCHECK(_finished);
    if (_data_page_builder->count() == 0) {
        return Status::NotFound("page is empty");
    }
    if (_encoding_type != DICT_ENCODING) {
        return _data_page_builder->get_first_value(value);
    }
    *reinterpret_cast<Slice*>(value) = Slice(_first_value);
    return Status::OK();
}

Status BinaryDictPageBuilder::get_last_value(void* value) const {
    DCHECK(_finished);
    if (_data_page_builder->count() == 0) {
        return Status::NotFound("page is empty");
    }
    if (_encoding_type != DICT_ENCODING) {
        return _data_page_builder->get_last_value(value);
    }
    uint32_t value_code;
    RETURN_IF_ERROR(_data_page_builder->get_last_value(&value_code));
    // TODO _dict_items is cleared in get_dictionary_page, which could cause
    // get_last_value to fail when it's called after get_dictionary_page.
    // the solution is to read last value from _dict_builder instead of _dict_items
    *reinterpret_cast<Slice*>(value) = _dict_items[value_code];
    return Status::OK();
}

BinaryDictPageDecoder::BinaryDictPageDecoder(Slice data, const PageDecoderOptions& options)
        : _data(data),
          _options(options),
          _data_page_decoder(nullptr),
          _parsed(false),
          _encoding_type(UNKNOWN_ENCODING) {}

Status BinaryDictPageDecoder::init() {
    CHECK(!_parsed);
    if (_data.size < BINARY_DICT_PAGE_HEADER_SIZE) {
        return Status::Corruption(strings::Substitute("invalid data size:$0, header size:$1",
                                                      _data.size, BINARY_DICT_PAGE_HEADER_SIZE));
    }
    size_t type = decode_fixed32_le((const uint8_t*)&_data.data[0]);
    _encoding_type = static_cast<EncodingTypePB>(type);
    _data.remove_prefix(BINARY_DICT_PAGE_HEADER_SIZE);
    if (_encoding_type == DICT_ENCODING) {
        // copy the codewords into a temporary buffer first
        // And then copy the strings corresponding to the codewords to the destination buffer
        TypeInfo* type_info = get_scalar_type_info(OLAP_FIELD_TYPE_INT);

        RETURN_IF_ERROR(ColumnVectorBatch::create(0, false, type_info, nullptr, &_batch));
        _data_page_decoder.reset(new BitShufflePageDecoder<OLAP_FIELD_TYPE_INT>(_data, _options));
    } else if (_encoding_type == PLAIN_ENCODING) {
        DCHECK_EQ(_encoding_type, PLAIN_ENCODING);
        _data_page_decoder.reset(new BinaryPlainPageDecoder(_data, _options));
    } else {
        LOG(WARNING) << "invalid encoding type:" << _encoding_type;
        return Status::Corruption(strings::Substitute("invalid encoding type:$0", _encoding_type));
    }

    RETURN_IF_ERROR(_data_page_decoder->init());
    _parsed = true;
    return Status::OK();
}

BinaryDictPageDecoder::~BinaryDictPageDecoder() {
    delete[] _start_offset_array;
    delete[] _len_array;
}

Status BinaryDictPageDecoder::seek_to_position_in_page(size_t pos) {
    return _data_page_decoder->seek_to_position_in_page(pos);
}

bool BinaryDictPageDecoder::is_dict_encoding() const {
    return _encoding_type == DICT_ENCODING;
}

void BinaryDictPageDecoder::set_dict_decoder(PageDecoder* dict_decoder) {
    _dict_decoder = (BinaryPlainPageDecoder*)dict_decoder;
    _start_offset_array = new uint32_t[_dict_decoder->_num_elems];
    _len_array = new uint32_t[_dict_decoder->_num_elems];
    for (int i = 0; i < _dict_decoder->_num_elems; i++) {
        const uint32_t start_offset = _dict_decoder->offset(i);
        uint32_t len = _dict_decoder->offset(i + 1) - start_offset;
        _start_offset_array[i] = start_offset;
        _len_array[i] = len;
    }
    _bit_shuffle_ptr = reinterpret_cast<BitShufflePageDecoder<OLAP_FIELD_TYPE_INT>*>(_data_page_decoder.get());
};

Status BinaryDictPageDecoder::next_batch(size_t* n, vectorized::MutableColumnPtr &dst) {
    if (_encoding_type == PLAIN_ENCODING) {
        return _data_page_decoder->next_batch(n, dst);
    }
    // dictionary encoding
    DCHECK(_parsed);
    DCHECK(_dict_decoder != nullptr) << "dict decoder pointer is nullptr";
 
    if (PREDICT_FALSE(*n == 0)) {
        return Status::OK();
    }
 
    if (_bit_shuffle_ptr->_cur_index > _bit_shuffle_ptr->_num_elements) {
        *n = 0;
        return Status::OK();
    }
 
     size_t max_fetch = std::min(*n, static_cast<size_t>(_bit_shuffle_ptr->_num_elements - _bit_shuffle_ptr->_cur_index));
 
    uint32_t total_len = 0;
    const int32_t* data_array = reinterpret_cast<const int32_t*>(_bit_shuffle_ptr->_decoded.data());
 
    size_t start_index = _bit_shuffle_ptr->_cur_index;
    for (int i = 0; i < max_fetch; i++, start_index++) {
        int32_t codeword = data_array[start_index];
        total_len += _len_array[codeword];
    }
 
    vectorized::ColumnString& column_str = reinterpret_cast<vectorized::ColumnString&>(*dst);
 
    size_t old_chars_size = column_str.get_chars().size();
    column_str.get_chars().resize(old_chars_size + total_len + max_fetch);
 
    start_index = _bit_shuffle_ptr->_cur_index;
    for (int i = 0; i < max_fetch; i++, start_index++) {
         int32_t codeword = data_array[start_index];
        const uint32_t start_offset = _start_offset_array[codeword];
        const uint32_t str_len = _len_array[codeword];
        if (str_len) {
            memcpy(column_str.get_chars().data() + old_chars_size, &_dict_decoder->_data[start_offset], str_len);
        }
        column_str.get_chars()[old_chars_size + str_len] = 0;
        old_chars_size = old_chars_size + str_len + 1;
        column_str.get_offsets().push_back(old_chars_size);
    }
 
    return Status::OK();
 
}

Status BinaryDictPageDecoder::next_batch(size_t* n, ColumnBlockView* dst) {
    if (_encoding_type == PLAIN_ENCODING) {
        return _data_page_decoder->next_batch(n, dst);
    }
    // dictionary encoding
    DCHECK(_parsed);
    DCHECK(_dict_decoder != nullptr) << "dict decoder pointer is nullptr";

    if (PREDICT_FALSE(*n == 0)) {
        return Status::OK();
    }
    Slice* out = reinterpret_cast<Slice*>(dst->data());

    _batch->resize(*n);

    ColumnBlock column_block(_batch.get(), dst->column_block()->pool());
    ColumnBlockView tmp_block_view(&column_block);
    RETURN_IF_ERROR(_data_page_decoder->next_batch(n, &tmp_block_view));
    const auto len = *n;

    size_t mem_len[len];
    for (int i = 0; i < len; ++i) {
        int32_t codeword = *reinterpret_cast<const int32_t*>(column_block.cell_ptr(i));
        // get the string from the dict decoder
        *out = _dict_decoder->string_at_index(codeword);
        mem_len[i] = out->size;
        out++;
    }

    // use SIMD instruction to speed up call function `RoundUpToPowerOfTwo`
    auto mem_size = 0;
    for (int i = 0; i < len; ++i) {
        mem_len[i] = BitUtil::RoundUpToPowerOf2Int32(mem_len[i], MemPool::DEFAULT_ALIGNMENT);
        mem_size += mem_len[i];
    }

    // allocate a batch of memory and do memcpy
    out = reinterpret_cast<Slice*>(dst->data());
    char* destination = (char*)dst->column_block()->pool()->allocate(mem_size);
    if (destination == nullptr) {
        return Status::MemoryAllocFailed(
                strings::Substitute("memory allocate failed, size:$0", mem_size));
    }
    for (int i = 0; i < len; ++i) {
        out->relocate(destination);
        destination += mem_len[i];
        ++out;
    }

    return Status::OK();
}

} // namespace segment_v2
} // namespace doris
