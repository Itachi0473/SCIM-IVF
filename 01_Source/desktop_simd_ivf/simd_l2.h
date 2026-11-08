#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdint>
#include <cmath>
#include <limits>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

// AVX2/FMA implementation for modern Intel and AMD desktops.  The CMake
// target enables these instructions only for this desktop-specific variant.
inline float simd_squared_l2(const float* a, const float* b, std::size_t dimension) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 sum = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= dimension; i += 8) {
        const __m256 delta = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        sum = _mm256_fmadd_ps(delta, delta, sum);
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float result = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                   lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; i < dimension; ++i) {
        const float delta = a[i] - b[i];
        result += delta * delta;
    }
    return result;
#else
    float result = 0.0f;
    for (std::size_t i = 0; i < dimension; ++i) {
        const float delta = a[i] - b[i];
        result += delta * delta;
    }
    return result;
#endif
}

// Return negative inner product so callers that rank scores in ascending
// order can perform maximum-inner-product search without changing heap logic.
inline float simd_negative_dot(const float* a, const float* b,
                               std::size_t dimension) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 sum = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= dimension; i += 8)
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), sum);
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float result = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                   lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; i < dimension; ++i) result += a[i] * b[i];
    return -result;
#else
    float result = 0.0f;
    for (std::size_t i = 0; i < dimension; ++i) result += a[i] * b[i];
    return -result;
#endif
}

inline float simd_negative_dot_f32_u8(const float* a, const std::uint8_t* b,
                                      std::size_t dimension) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 sum = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= dimension; i += 8) {
        const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + i));
        const __m256 values = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
        sum = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), values, sum);
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float result = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                   lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; i < dimension; ++i) result += a[i] * static_cast<float>(b[i]);
    return -result;
#else
    float result = 0.0f;
    for (std::size_t i = 0; i < dimension; ++i)
        result += a[i] * static_cast<float>(b[i]);
    return -result;
#endif
}

// Cosine distance 1-cos(a,b). Inputs are not modified or implicitly
// normalized; the norm correction is part of the score.
inline float simd_cosine_distance(const float* a, const float* b,
                                  std::size_t dimension) {
#if defined(__AVX2__) && defined(__FMA__)
    __m256 dot = _mm256_setzero_ps();
    __m256 norm_a = _mm256_setzero_ps();
    __m256 norm_b = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= dimension; i += 8) {
        const __m256 av = _mm256_loadu_ps(a + i);
        const __m256 bv = _mm256_loadu_ps(b + i);
        dot = _mm256_fmadd_ps(av, bv, dot);
        norm_a = _mm256_fmadd_ps(av, av, norm_a);
        norm_b = _mm256_fmadd_ps(bv, bv, norm_b);
    }
    alignas(32) float d[8], na[8], nb[8];
    _mm256_store_ps(d, dot);
    _mm256_store_ps(na, norm_a);
    _mm256_store_ps(nb, norm_b);
    float dot_sum = 0.0f, norm_a_sum = 0.0f, norm_b_sum = 0.0f;
    for (int lane = 0; lane < 8; ++lane) {
        dot_sum += d[lane]; norm_a_sum += na[lane]; norm_b_sum += nb[lane];
    }
    for (; i < dimension; ++i) {
        dot_sum += a[i] * b[i];
        norm_a_sum += a[i] * a[i];
        norm_b_sum += b[i] * b[i];
    }
#else
    float dot_sum = 0.0f, norm_a_sum = 0.0f, norm_b_sum = 0.0f;
    for (std::size_t i = 0; i < dimension; ++i) {
        dot_sum += a[i] * b[i];
        norm_a_sum += a[i] * a[i];
        norm_b_sum += b[i] * b[i];
    }
#endif
    if (!(norm_a_sum > 0.0f) || !(norm_b_sum > 0.0f))
        return std::numeric_limits<float>::infinity();
    const float cosine = std::clamp(dot_sum / std::sqrt(norm_a_sum * norm_b_sum), -1.0f, 1.0f);
    return 1.0f - cosine;
}

// Cosine distance against the LVQ-8 reconstruction, evaluated directly from
// its byte codes without materializing a temporary FP32 vector.
inline float simd_cosine_distance_lvq8(const float* query, const float* centroid,
                                       const std::uint8_t* codes, float lower,
                                       float delta, std::size_t dimension) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256 lower_v = _mm256_set1_ps(lower);
    const __m256 delta_v = _mm256_set1_ps(delta);
    __m256 dot = _mm256_setzero_ps();
    __m256 norm_q = _mm256_setzero_ps();
    __m256 norm_x = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= dimension; i += 8) {
        const __m128i code_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes + i));
        const __m256 code_values = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(code_bytes));
        const __m256 center = _mm256_add_ps(_mm256_loadu_ps(centroid + i), lower_v);
        const __m256 x = _mm256_add_ps(center, _mm256_mul_ps(code_values, delta_v));
        const __m256 q = _mm256_loadu_ps(query + i);
        dot = _mm256_fmadd_ps(q, x, dot);
        norm_q = _mm256_fmadd_ps(q, q, norm_q);
        norm_x = _mm256_fmadd_ps(x, x, norm_x);
    }
    alignas(32) float d[8], nq[8], nx[8];
    _mm256_store_ps(d, dot); _mm256_store_ps(nq, norm_q); _mm256_store_ps(nx, norm_x);
    float dot_sum = 0.0f, norm_q_sum = 0.0f, norm_x_sum = 0.0f;
    for (int lane = 0; lane < 8; ++lane) {
        dot_sum += d[lane]; norm_q_sum += nq[lane]; norm_x_sum += nx[lane];
    }
    for (; i < dimension; ++i) {
        const float x = centroid[i] + lower + static_cast<float>(codes[i]) * delta;
        dot_sum += query[i] * x;
        norm_q_sum += query[i] * query[i];
        norm_x_sum += x * x;
    }
#else
    float dot_sum = 0.0f, norm_q_sum = 0.0f, norm_x_sum = 0.0f;
    for (std::size_t i = 0; i < dimension; ++i) {
        const float x = centroid[i] + lower + static_cast<float>(codes[i]) * delta;
        dot_sum += query[i] * x;
        norm_q_sum += query[i] * query[i];
        norm_x_sum += x * x;
    }
#endif
    if (!(norm_q_sum > 0.0f) || !(norm_x_sum > 0.0f))
        return std::numeric_limits<float>::infinity();
    const float cosine = std::clamp(dot_sum / std::sqrt(norm_q_sum * norm_x_sum), -1.0f, 1.0f);
    return 1.0f - cosine;
}

// Compute L2 directly from LVQ-8 residual codes without materializing a
// temporary FP32 vector. Reconstructed coordinate semantics match
// centroid[d] + lower + float(code[d]) * delta.
inline float simd_squared_l2_lvq8(const float* query, const float* centroid,
                                  const std::uint8_t* codes, float lower,
                                  float delta, std::size_t dimension) {
#if defined(__AVX2__)
    const __m256 lower_v = _mm256_set1_ps(lower);
    const __m256 delta_v = _mm256_set1_ps(delta);
    __m256 sum = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= dimension; i += 8) {
        const __m128i code_bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(codes + i));
        const __m256 code_values = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(code_bytes));
        const __m256 center = _mm256_add_ps(_mm256_loadu_ps(centroid + i), lower_v);
        const __m256 reconstructed = _mm256_add_ps(center, _mm256_mul_ps(code_values, delta_v));
        const __m256 difference = _mm256_sub_ps(_mm256_loadu_ps(query + i), reconstructed);
#if defined(__FMA__)
        sum = _mm256_fmadd_ps(difference, difference, sum);
#else
        sum = _mm256_add_ps(sum, _mm256_mul_ps(difference, difference));
#endif
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float result = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                   lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; i < dimension; ++i) {
        const float reconstructed = centroid[i] + lower + static_cast<float>(codes[i]) * delta;
        const float difference = query[i] - reconstructed;
        result += difference * difference;
    }
    return result;
#else
    float result = 0.0f;
    for (std::size_t i = 0; i < dimension; ++i) {
        const float reconstructed = centroid[i] + lower + static_cast<float>(codes[i]) * delta;
        const float difference = query[i] - reconstructed;
        result += difference * difference;
    }
    return result;
#endif
}

// Exact squared-L2 between original uint8 SIFT values and float centroids or
// queries. This preserves the dataset's integer values without storing them as
// float32 in IVF-Flat lists.
inline float simd_squared_l2_u8_f32(const std::uint8_t* a, const float* b,
                                    std::size_t dimension) {
#if defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= dimension; i += 8) {
        const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(a + i));
        const __m256 values = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
        const __m256 delta = _mm256_sub_ps(values, _mm256_loadu_ps(b + i));
#if defined(__FMA__)
        sum = _mm256_fmadd_ps(delta, delta, sum);
#else
        sum = _mm256_add_ps(sum, _mm256_mul_ps(delta, delta));
#endif
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float result = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                   lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; i < dimension; ++i) {
        const float delta = static_cast<float>(a[i]) - b[i];
        result += delta * delta;
    }
    return result;
#else
    float result = 0.0f;
    for (std::size_t i = 0; i < dimension; ++i) {
        const float delta = static_cast<float>(a[i]) - b[i];
        result += delta * delta;
    }
    return result;
#endif
}

// Symmetric operand order for query-time scans with a floating query and an
// exact uint8 vector payload.
inline float simd_squared_l2_f32_u8(const float* a, const std::uint8_t* b,
                                    std::size_t dimension) {
#if defined(__AVX2__)
    __m256 sum = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= dimension; i += 8) {
        const __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + i));
        const __m256 values = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes));
        const __m256 delta = _mm256_sub_ps(_mm256_loadu_ps(a + i), values);
#if defined(__FMA__)
        sum = _mm256_fmadd_ps(delta, delta, sum);
#else
        sum = _mm256_add_ps(sum, _mm256_mul_ps(delta, delta));
#endif
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float result = lanes[0] + lanes[1] + lanes[2] + lanes[3] +
                   lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; i < dimension; ++i) {
        const float delta = a[i] - static_cast<float>(b[i]);
        result += delta * delta;
    }
    return result;
#else
    float result = 0.0f;
    for (std::size_t i = 0; i < dimension; ++i) {
        const float delta = a[i] - static_cast<float>(b[i]);
        result += delta * delta;
    }
    return result;
#endif
}

// Maintain S=sum(x) and Q=sum(||x||^2) using four double-precision lanes.
// Float inputs are widened before arithmetic to preserve the core's double
// accumulator semantics.
inline void simd_apply_vector_delta(double* vector_sum, double* squared_norm_sum,
                                   const float* vector, std::size_t dimension, double sign) {
#if defined(__AVX2__) && defined(__FMA__)
    const __m256d sign4 = _mm256_set1_pd(sign);
    __m256d norm = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; i + 4 <= dimension; i += 4) {
        const __m128 xf = _mm_loadu_ps(vector + i);
        const __m256d x = _mm256_cvtps_pd(xf);
        const __m256d old_sum = _mm256_loadu_pd(vector_sum + i);
        _mm256_storeu_pd(vector_sum + i, _mm256_fmadd_pd(x, sign4, old_sum));
        norm = _mm256_fmadd_pd(x, x, norm);
    }
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, norm);
    double norm_sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];
    for (; i < dimension; ++i) {
        const double x = vector[i];
        vector_sum[i] += sign * x;
        norm_sum += x * x;
    }
    *squared_norm_sum += sign * norm_sum;
#else
    double norm_sum = 0.0;
    for (std::size_t i = 0; i < dimension; ++i) {
        const double x = vector[i];
        vector_sum[i] += sign * x;
        norm_sum += x * x;
    }
    *squared_norm_sum += sign * norm_sum;
#endif
}

inline float simd_mean_residual_from_aggregates(const double* vector_sum,
                                                double squared_norm_sum,
                                                const float* centroid,
                                                std::size_t dimension,
                                                std::size_t count) {
    if (count == 0) return 0.0f;
#if defined(__AVX2__) && defined(__FMA__)
    __m256d dot = _mm256_setzero_pd();
    __m256d center_norm = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; i + 4 <= dimension; i += 4) {
        const __m128 cf = _mm_loadu_ps(centroid + i);
        const __m256d c = _mm256_cvtps_pd(cf);
        const __m256d s = _mm256_loadu_pd(vector_sum + i);
        dot = _mm256_fmadd_pd(c, s, dot);
        center_norm = _mm256_fmadd_pd(c, c, center_norm);
    }
    alignas(32) double dot_lanes[4], norm_lanes[4];
    _mm256_store_pd(dot_lanes, dot);
    _mm256_store_pd(norm_lanes, center_norm);
    double dot_sum = dot_lanes[0] + dot_lanes[1] + dot_lanes[2] + dot_lanes[3];
    double norm_sum = norm_lanes[0] + norm_lanes[1] + norm_lanes[2] + norm_lanes[3];
    for (; i < dimension; ++i) {
        const double c = centroid[i];
        dot_sum += c * vector_sum[i];
        norm_sum += c * c;
    }
    const double sse = std::max(0.0, squared_norm_sum - 2.0 * dot_sum + count * norm_sum);
    return static_cast<float>(sse / count);
#else
    double dot_sum = 0.0, norm_sum = 0.0;
    for (std::size_t i = 0; i < dimension; ++i) {
        const double c = centroid[i];
        dot_sum += c * vector_sum[i];
        norm_sum += c * c;
    }
    const double sse = std::max(0.0, squared_norm_sum - 2.0 * dot_sum + count * norm_sum);
    return static_cast<float>(sse / count);
#endif
}

inline void simd_copy_floats(float* destination, const float* source, std::size_t count) {
#if defined(__AVX2__)
    std::size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        _mm256_storeu_ps(destination + i, _mm256_loadu_ps(source + i));
    }
    for (; i < count; ++i) destination[i] = source[i];
#else
    std::copy_n(source, count, destination);
#endif
}

// Return a bit per ID indicating membership in the inclusive dense range.
// Frozen-IVF sliding windows delete consecutive IDs, making this batch test
// cheaper than a hash lookup for every record.
inline unsigned simd_id_range_mask(const std::int64_t* ids, std::size_t count,
                                  std::int64_t first, std::int64_t last) {
#if defined(__AVX2__)
    if (count == 4) {
        const __m256i values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ids));
        const __m256i low = _mm256_set1_epi64x(first);
        const __m256i high = _mm256_set1_epi64x(last);
        const __m256i ge = _mm256_or_si256(_mm256_cmpgt_epi64(values, low),
                                           _mm256_cmpeq_epi64(values, low));
        const __m256i le = _mm256_or_si256(_mm256_cmpgt_epi64(high, values),
                                           _mm256_cmpeq_epi64(values, high));
        const __m256i matches = _mm256_and_si256(ge, le);
        return static_cast<unsigned>(_mm256_movemask_pd(_mm256_castsi256_pd(matches)));
    }
#endif
    unsigned mask = 0;
    for (std::size_t i = 0; i < count; ++i)
        if (ids[i] >= first && ids[i] <= last) mask |= 1u << i;
    return mask;
}

inline unsigned simd_id_range_mask(const std::uint32_t* ids, std::size_t count,
                                  std::int64_t first, std::int64_t last) {
#if defined(__AVX2__)
    if (count == 8 && first <= last && last >= 0 &&
        first <= static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
        const std::uint32_t low = static_cast<std::uint32_t>(std::max<std::int64_t>(0, first));
        const std::uint32_t high = static_cast<std::uint32_t>(std::min<std::int64_t>(
                std::numeric_limits<std::uint32_t>::max(), last));
        const __m256i values = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ids));
        const __m256i sign = _mm256_set1_epi32(static_cast<int>(0x80000000u));
        const __m256i ordered = _mm256_xor_si256(values, sign);
        const __m256i lowv = _mm256_xor_si256(_mm256_set1_epi32(static_cast<int>(low)), sign);
        const __m256i highv = _mm256_xor_si256(_mm256_set1_epi32(static_cast<int>(high)), sign);
        const __m256i ge = _mm256_or_si256(_mm256_cmpgt_epi32(ordered, lowv), _mm256_cmpeq_epi32(ordered, lowv));
        const __m256i le = _mm256_or_si256(_mm256_cmpgt_epi32(highv, ordered), _mm256_cmpeq_epi32(ordered, highv));
        return static_cast<unsigned>(_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_and_si256(ge, le))));
    }
#endif
    unsigned mask = 0;
    for (std::size_t i = 0; i < count; ++i)
        if (static_cast<std::int64_t>(ids[i]) >= first && static_cast<std::int64_t>(ids[i]) <= last)
            mask |= 1u << i;
    return mask;
}
