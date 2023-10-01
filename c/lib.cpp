#include <cassert>
#include <functional>

#include <usearch/index_punned_dense.hpp>
#include <usearch/index_punned_helpers.hpp>

#ifndef USEARCH_EXPORT
#define USEARCH_EXPORT
#endif

extern "C" {
#include "usearch.h"
}

using namespace unum::usearch;
using namespace unum;

using label_t = usearch_label_t;
using distance_t = usearch_distance_t;
using index_t = index_punned_dense_gt<label_t>;
using add_result_t = index_t::add_result_t;
using search_result_t = index_t::search_result_t;
using serialization_result_t = index_t::serialization_result_t;
using vector_view_t = span_gt<float>;

// helper functions that are not part of the C ABI
metric_kind_t to_native_metric(usearch_metric_kind_t kind) {
    switch (kind) {
    case usearch_metric_ip_k: return metric_kind_t::ip_k;
    case usearch_metric_l2sq_k: return metric_kind_t::l2sq_k;
    case usearch_metric_cos_k: return metric_kind_t::cos_k;
    case usearch_metric_haversine_k: return metric_kind_t::haversine_k;
    case usearch_metric_pearson_k: return metric_kind_t::pearson_k;
    case usearch_metric_jaccard_k: return metric_kind_t::jaccard_k;
    case usearch_metric_hamming_k: return metric_kind_t::hamming_k;
    case usearch_metric_tanimoto_k: return metric_kind_t::tanimoto_k;
    case usearch_metric_sorensen_k: return metric_kind_t::sorensen_k;
    default: return metric_kind_t::unknown_k;
    }
}

usearch_metric_kind_t to_usearch_metric(metric_kind_t kind) {
    switch (kind) {
    case metric_kind_t::ip_k: return usearch_metric_ip_k;
    case metric_kind_t::l2sq_k: return usearch_metric_l2sq_k;
    case metric_kind_t::cos_k: return usearch_metric_cos_k;
    case metric_kind_t::haversine_k: return usearch_metric_haversine_k;
    case metric_kind_t::pearson_k: return usearch_metric_pearson_k;
    case metric_kind_t::jaccard_k: return usearch_metric_jaccard_k;
    case metric_kind_t::hamming_k: return usearch_metric_hamming_k;
    case metric_kind_t::tanimoto_k: return usearch_metric_tanimoto_k;
    case metric_kind_t::sorensen_k: return usearch_metric_sorensen_k;
    default: return usearch_metric_unknown_k;
    }
}

scalar_kind_t to_native_scalar(usearch_scalar_kind_t kind) {
    switch (kind) {
    case usearch_scalar_f32_k: return scalar_kind_t::f32_k;
    case usearch_scalar_f64_k: return scalar_kind_t::f64_k;
    case usearch_scalar_f16_k: return scalar_kind_t::f16_k;
    case usearch_scalar_f8_k: return scalar_kind_t::f8_k;
    case usearch_scalar_b1_k: return scalar_kind_t::b1x8_k;
    default: return scalar_kind_t::unknown_k;
    }
}

add_result_t add_(index_t* index, usearch_label_t label, void const* vector, scalar_kind_t kind, int32_t level,
                  void* tape) {
    switch (kind) {
    case scalar_kind_t::f32_k: return index->add(label, (f32_t const*)vector, level, (byte_t*)tape);
    case scalar_kind_t::f64_k: return index->add(label, (f64_t const*)vector);
    case scalar_kind_t::f16_k: return index->add(label, (f16_t const*)vector);
    case scalar_kind_t::f8_k: return index->add(label, (f8_bits_t const*)vector);
    case scalar_kind_t::b1x8_k: return index->add(label, (b1x8_t const*)vector);
    default: return add_result_t{}.failed("Unknown scalar kind!");
    }
}

#if USEARCH_LOOKUP_LABEL
bool get_(index_t* index, label_t label, void* vector, scalar_kind_t kind) {
    switch (kind) {
    case scalar_kind_t::f32_k: return index->get(label, (f32_t*)vector);
    case scalar_kind_t::f64_k: return index->get(label, (f64_t*)vector);
    case scalar_kind_t::f16_k: return index->get(label, (f16_t*)vector);
    case scalar_kind_t::f8_k: return index->get(label, (f8_bits_t*)vector);
    case scalar_kind_t::b1x8_k: return index->get(label, (b1x8_t*)vector);
    default: return index->empty_search_result().failed("Unknown scalar kind!");
    }
}
#endif

search_result_t search_(index_t* index, void const* vector, scalar_kind_t kind, size_t n) {
    switch (kind) {
    case scalar_kind_t::f32_k: return index->search((f32_t const*)vector, n);
    case scalar_kind_t::f64_k: return index->search((f64_t const*)vector, n);
    case scalar_kind_t::f16_k: return index->search((f16_t const*)vector, n);
    case scalar_kind_t::f8_k: return index->search((f8_bits_t const*)vector, n);
    case scalar_kind_t::b1x8_k: return index->search((b1x8_t const*)vector, n);
    default: return index->empty_search_result().failed("Unknown scalar kind!");
    }
}

index_punned_dense_metric_t udf(metric_kind_t kind, usearch_metric_t raw_ptr) {
    index_punned_dense_metric_t result;
    result.kind_ = kind;
    result.func_ = [raw_ptr](punned_vector_view_t a, punned_vector_view_t b) -> distance_t {
        return raw_ptr((void const*)a.data(), (void const*)b.data());
    };
    return result;
}

extern "C" {

USEARCH_EXPORT usearch_index_t usearch_init(usearch_init_options_t* options, usearch_error_t* error) {

    assert(options && error);

    index_config_t config;
    config.connectivity = options->connectivity;
    config.vector_alignment = sizeof(float);
    index_t index =        //
        options->metric ?  //
            index_t::make( //
                options->dimensions, udf(to_native_metric(options->metric_kind), options->metric), config,
                to_native_scalar(options->quantization), options->expansion_add, options->expansion_search)
                        :  //
            index_t::make( //
                options->dimensions, to_native_metric(options->metric_kind), config,
                to_native_scalar(options->quantization), options->expansion_add, options->expansion_search);

    if (options->retriever != nullptr || options->retriever_mut != nullptr) {
        if (options->retriever == nullptr || options->retriever_mut == nullptr) {
            *error = "External mut and non-mut retrievers must be either both-set or both-null.";
        }
        index.set_node_retriever(options->retriever_ctx, options->retriever, options->retriever_mut);
    }
    index_t* result_ptr = new index_t(std::move(index));
    return result_ptr;
}

USEARCH_EXPORT void usearch_free(usearch_index_t index, usearch_error_t*) { delete reinterpret_cast<index_t*>(index); }

USEARCH_EXPORT void usearch_save(usearch_index_t index, char const* path, char** usearch_result_buf,
                                 usearch_error_t* error) {
    serialization_result_t result = reinterpret_cast<index_t*>(index)->save(path, usearch_result_buf);
    if (!result) {
        *error = result.error.what();
        result.error = nullptr;
    }
}

USEARCH_EXPORT void usearch_load(usearch_index_t index, char const* path, usearch_error_t* error) {
    serialization_result_t result = reinterpret_cast<index_t*>(index)->load(path);
    if (!result) {
        *error = result.error.what();
        result.error = nullptr;
    }
}

USEARCH_EXPORT void usearch_view(usearch_index_t index, char const* path, usearch_error_t* error) {
    serialization_result_t result = reinterpret_cast<index_t*>(index)->view(path);
    if (!result) {
        *error = result.error.what();
        result.error = nullptr;
    }
}

void usearch_view_mem(usearch_index_t index, char* data, usearch_error_t* error) {
    serialization_result_t result = reinterpret_cast<index_t*>(index)->view_mem(data);
    if (!result) {
        *error = result.error.what();
        result.error = nullptr;
    }
}

void usearch_view_mem_lazy(usearch_index_t index, char* data, usearch_error_t* error) {
    serialization_result_t result = reinterpret_cast<index_t*>(index)->view_mem_lazy(data);
    if (!result) {
        *error = result.error.what();
        // error needs to be reset. otherwise error_t destructor will raise.
        // todo:: fix for the rest of the interface
        result.error = nullptr;
    }
}

void usearch_update_header(usearch_index_t index, char* headerp, usearch_error_t* error) {
    serialization_result_t result = reinterpret_cast<index_t*>(index)->update_header(headerp);
    if (!result) {
        *error = result.error.what();
        result.error = nullptr;
    }
}

usearch_metadata_t usearch_metadata(usearch_index_t index, usearch_error_t*) {
    usearch_metadata_t res;
    precomputed_constants_t pre = reinterpret_cast<index_t*>(index)->metadata();

    res.inverse_log_connectivity = pre.inverse_log_connectivity;
    res.connectivity_max_base = pre.connectivity_max_base;
    res.neighbors_bytes = pre.neighbors_bytes;
    res.neighbors_base_bytes = pre.neighbors_base_bytes;
    res.dimensions = reinterpret_cast<index_t*>(index)->dimensions();
    res.expansion_add = reinterpret_cast<index_t*>(index)->expansion_add();
    res.expansion_search = reinterpret_cast<index_t*>(index)->expansion_search();
    res.connectivity = reinterpret_cast<index_t*>(index)->connectivity();
    res.metric_kind = to_usearch_metric(reinterpret_cast<index_t*>(index)->metric().kind());

    return res;
}

USEARCH_EXPORT size_t usearch_size(usearch_index_t index, usearch_error_t*) { //
    return reinterpret_cast<index_t*>(index)->size();
}

USEARCH_EXPORT size_t usearch_capacity(usearch_index_t index, usearch_error_t*) {
    return reinterpret_cast<index_t*>(index)->capacity();
}

USEARCH_EXPORT size_t usearch_dimensions(usearch_index_t index, usearch_error_t*) {
    return reinterpret_cast<index_t*>(index)->dimensions();
}

USEARCH_EXPORT size_t usearch_connectivity(usearch_index_t index, usearch_error_t*) {
    return reinterpret_cast<index_t*>(index)->connectivity();
}

USEARCH_EXPORT size_t usearch_expansion_add(usearch_index_t index, usearch_error_t*) {
    return reinterpret_cast<index_t*>(index)->expansion_add();
}

USEARCH_EXPORT size_t usearch_expansion_search(usearch_index_t index, usearch_error_t*) {
    return reinterpret_cast<index_t*>(index)->expansion_search();
}

USEARCH_EXPORT usearch_metric_kind_t usearch_metric_kind(usearch_index_t index, usearch_error_t*) {
    return to_usearch_metric(reinterpret_cast<index_t*>(index)->metric().kind());
}

USEARCH_EXPORT void usearch_reserve(usearch_index_t index, size_t capacity, usearch_error_t*) {
    // TODO: Consider returning the new capacity.
    reinterpret_cast<index_t*>(index)->reserve(capacity);
}

USEARCH_EXPORT void usearch_add(                                                                  //
    usearch_index_t index, usearch_label_t label, void const* vector, usearch_scalar_kind_t kind, //
    usearch_error_t* error) {
    add_result_t result = add_(reinterpret_cast<index_t*>(index), label, vector, to_native_scalar(kind), -1, nullptr);
    if (!result)
        *error = result.error.what();
}

int32_t usearch_newnode_level(usearch_index_t index, usearch_error_t*) {
    return reinterpret_cast<index_t*>(index)->newnode_level();
}

void usearch_add_external(                                                                                    //
    usearch_index_t index, usearch_label_t label, void const* vector, void* tape, usearch_scalar_kind_t kind, //
    int32_t level, usearch_error_t* error) {
    add_result_t result = add_(reinterpret_cast<index_t*>(index), label, vector, to_native_scalar(kind), level, tape);
    if (!result)
        *error = result.error.what();
}

void usearch_set_node_retriever(usearch_index_t index, void* retriever_ctx, usearch_node_retriever_t retriever,
                                usearch_node_retriever_t retriever_mut, usearch_error_t*) {
    reinterpret_cast<index_t*>(index)->set_node_retriever(retriever_ctx, retriever, retriever_mut);
}

#if USEARCH_LOOKUP_LABEL
USEARCH_EXPORT bool usearch_contains(usearch_index_t index, usearch_label_t label, usearch_error_t*) {
    return reinterpret_cast<index_t*>(index)->contains(label);
}
#endif

USEARCH_EXPORT size_t usearch_search(                                                            //
    usearch_index_t index, void const* vector, usearch_scalar_kind_t kind, size_t results_limit, //
    usearch_label_t* found_labels, usearch_distance_t* found_distances, usearch_error_t* error) {
    search_result_t result = search_(reinterpret_cast<index_t*>(index), vector, to_native_scalar(kind), results_limit);
    if (!result) {
        *error = result.error.what();
        return 0;
    }

    return result.dump_to(found_labels, found_distances);
}

USEARCH_EXPORT size_t usearch_search_with_tapes(                                                            //
    usearch_index_t index, void const* vector, usearch_scalar_kind_t kind, size_t results_limit, //
    usearch_label_t* found_labels, usearch_distance_t* found_distances, char** found_tapes, usearch_error_t* error) {
    search_result_t result = search_(reinterpret_cast<index_t*>(index), vector, to_native_scalar(kind), results_limit);
    if (!result) {
        *error = result.error.what();
        return 0;
    }

    return result.dump_to_with_tapes(found_labels, found_distances, found_tapes);
}

#if USEARCH_LOOKUP_LABEL
USEARCH_EXPORT bool usearch_get(                  //
    usearch_index_t index, usearch_label_t label, //
    void* vector, usearch_scalar_kind_t kind, usearch_error_t*) {
    return get_(reinterpret_cast<index_t*>(index), label, vector, to_native_scalar(kind));
}
#endif

USEARCH_EXPORT void usearch_remove(usearch_index_t, usearch_label_t, usearch_error_t* error) {
    if (error != nullptr)
        *error = "USearch does not support removal of elements yet.";
}

/** Cast types
 */
USEARCH_EXPORT void usearch_cast(usearch_scalar_kind_t from, void const* vector, usearch_scalar_kind_t to, void* result,
                                 size_t result_size, int dims, usearch_error_t* error) {

    scalar_kind_t from_native = to_native_scalar(from);
    scalar_kind_t to_native = to_native_scalar(to);
    if (from_native == scalar_kind_t::unknown_k) {
        *error = "Unknown \"from\" scalar kind.";
        return;
    }
    if (to_native == scalar_kind_t::unknown_k) {
        *error = "Unknown \"to\" scalar kind.";
        return;
    }

    if (result_size < dims * bytes_per_scalar(to_native_scalar(to))) {
        *error = "Result buffer is too small.";
        return;
    }

    switch (from_native) {
    case scalar_kind_t::f64_k:
        switch (to_native) {
        case scalar_kind_t::f64_k: cast_gt<f64_t, f64_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        case scalar_kind_t::f32_k: cast_gt<f64_t, f32_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        case scalar_kind_t::f16_k: cast_gt<f64_t, f16_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        case scalar_kind_t::b1x8_k: cast_gt<f64_t, b1x8_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        case scalar_kind_t::f8_k: cast_gt<f64_t, f8_bits_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        default: *error = "Unsupported \"to\" scalar kind."; return;
        }
        break;
    case scalar_kind_t::f32_k:
        switch (to_native) {
        case scalar_kind_t::f64_k: cast_gt<f32_t, f64_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        case scalar_kind_t::f32_k: cast_gt<f32_t, f32_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        case scalar_kind_t::f16_k: cast_gt<f32_t, f16_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        case scalar_kind_t::b1x8_k: cast_gt<f32_t, b1x8_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        case scalar_kind_t::f8_k: cast_gt<f32_t, f8_bits_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        default: *error = "Unsupported \"to\" scalar kind."; return;
        }
        break;
    case scalar_kind_t::f16_k:
        switch (to_native) {
        case scalar_kind_t::f64_k: cast_gt<f16_t, f64_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        case scalar_kind_t::f32_k: cast_gt<f16_t, f32_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        case scalar_kind_t::f16_k: cast_gt<f16_t, f16_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        case scalar_kind_t::b1x8_k: cast_gt<f16_t, b1x8_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        case scalar_kind_t::f8_k: cast_gt<f16_t, f8_bits_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        default: *error = "Unsupported \"to\" scalar kind."; return;
        }
        break;
    case scalar_kind_t::b1x8_k:
        switch (to_native) {
        case scalar_kind_t::f64_k: cast_gt<b1x8_t, f64_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        case scalar_kind_t::f32_k: cast_gt<b1x8_t, f32_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        case scalar_kind_t::f16_k: cast_gt<b1x8_t, f16_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        case scalar_kind_t::b1x8_k: cast_gt<b1x8_t, b1x8_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        case scalar_kind_t::f8_k: cast_gt<b1x8_t, f8_bits_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        default: *error = "Unsupported \"to\" scalar kind."; return;
        }
        break;
    case scalar_kind_t::f8_k:
        switch (to_native) {
        case scalar_kind_t::f64_k: cast_gt<f8_bits_t, f64_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        case scalar_kind_t::f32_k: cast_gt<f8_bits_t, f32_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        case scalar_kind_t::f16_k: cast_gt<f8_bits_t, f16_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        case scalar_kind_t::b1x8_k: cast_gt<f8_bits_t, b1x8_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        case scalar_kind_t::f8_k: cast_gt<f8_bits_t, f8_bits_t>{}((const byte_t*)vector, dims, (byte_t*)result); break;
        default: *error = "Unsupported \"to\" scalar kind."; return;
        }
        break;
    default: *error = "Unsupported \"from\" scalar kind."; return;
    }
}
}

USEARCH_EXPORT float usearch_dist(void const* a, void const* b, usearch_metric_kind_t metric, int dims,
                                  usearch_scalar_kind_t kind) {
    index_punned_dense_metric_t m = index_t::make_metric_(to_native_metric(metric), dims, to_native_scalar(kind));
    punned_vector_view_t av = {reinterpret_cast<byte_t const*>(a), dims * bytes_per_scalar(to_native_scalar(kind))};
    punned_vector_view_t bv = {reinterpret_cast<byte_t const*>(b), dims * bytes_per_scalar(to_native_scalar(kind))};
    return m(av, bv);
}
