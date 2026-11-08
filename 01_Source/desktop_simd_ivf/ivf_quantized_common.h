#pragma once

#include "simd_l2.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace ivf_quantized_detail {

constexpr std::size_t kTrainingSampleLimit = 20000;
constexpr std::size_t kCoarseIterations = 12;
constexpr std::size_t kPqIterations = 10;

inline std::vector<float> sample_vectors(const float* vectors, std::size_t count,
                                         std::size_t dimension, std::uint32_t seed) {
    if (vectors == nullptr || dimension == 0 || count < 2)
        throw std::invalid_argument("quantizer training requires at least two vectors");
    const std::size_t sample_count = std::min(count, kTrainingSampleLimit);
    std::vector<std::size_t> rows(count);
    std::iota(rows.begin(), rows.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(rows.begin(), rows.end(), rng);
    rows.resize(sample_count);
    std::vector<float> sample(sample_count * dimension);
    for (std::size_t i = 0; i < sample_count; ++i)
        std::copy_n(vectors + rows[i] * dimension, dimension, sample.data() + i * dimension);
    return sample;
}

inline std::vector<float> train_kmeans(const std::vector<float>& points,
                                       std::size_t rows, std::size_t dimension,
                                       std::size_t clusters, std::size_t iterations,
                                       std::uint32_t seed) {
    if (rows < clusters || points.size() != rows * dimension || dimension == 0 || iterations == 0)
        throw std::invalid_argument("invalid k-means training shape");
    std::vector<std::size_t> order(rows);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<float> centers(clusters * dimension);
    for (std::size_t c = 0; c < clusters; ++c)
        std::copy_n(points.data() + order[c] * dimension, dimension,
                    centers.data() + c * dimension);

    std::vector<std::size_t> assignments(rows);
    std::vector<float> sums(clusters * dimension);
    std::vector<std::size_t> sizes(clusters);
    for (std::size_t iteration = 0; iteration < iterations; ++iteration) {
        std::fill(sums.begin(), sums.end(), 0.0f);
        std::fill(sizes.begin(), sizes.end(), 0);
        for (std::size_t row = 0; row < rows; ++row) {
            const float* point = points.data() + row * dimension;
            std::size_t best = 0;
            float best_distance = simd_squared_l2(point, centers.data(), dimension);
            for (std::size_t c = 1; c < clusters; ++c) {
                const float distance = simd_squared_l2(point, centers.data() + c * dimension, dimension);
                if (distance < best_distance) { best = c; best_distance = distance; }
            }
            assignments[row] = best;
            ++sizes[best];
            float* sum = sums.data() + best * dimension;
            for (std::size_t d = 0; d < dimension; ++d) sum[d] += point[d];
        }
        for (std::size_t c = 0; c < clusters; ++c) {
            float* center = centers.data() + c * dimension;
            if (sizes[c] == 0) {
                const std::size_t replacement = order[rng() % rows];
                std::copy_n(points.data() + replacement * dimension, dimension, center);
            } else {
                const float inverse = 1.0f / static_cast<float>(sizes[c]);
                const float* sum = sums.data() + c * dimension;
                for (std::size_t d = 0; d < dimension; ++d) center[d] = sum[d] * inverse;
            }
        }
    }
    return centers;
}

inline std::size_t nearest(const float* point, const std::vector<float>& centers,
                           std::size_t count, std::size_t dimension) {
    std::size_t best = 0;
    float best_distance = simd_squared_l2(point, centers.data(), dimension);
    for (std::size_t c = 1; c < count; ++c) {
        const float distance = simd_squared_l2(point, centers.data() + c * dimension, dimension);
        if (distance < best_distance) { best = c; best_distance = distance; }
    }
    return best;
}

inline std::vector<std::size_t> nearest_lists(const float* query,
                                             const std::vector<float>& centers,
                                             std::size_t nlist, std::size_t dimension,
                                             std::size_t nprobe) {
    std::vector<std::pair<float, std::size_t>> distances;
    distances.reserve(nlist);
    for (std::size_t c = 0; c < nlist; ++c)
        distances.emplace_back(simd_squared_l2(query, centers.data() + c * dimension, dimension), c);
    const std::size_t wanted = std::min(nprobe, nlist);
    std::partial_sort(distances.begin(), distances.begin() + wanted, distances.end(),
        [](const auto& a, const auto& b) { return a.first == b.first ? a.second < b.second : a.first < b.first; });
    std::vector<std::size_t> result(wanted);
    for (std::size_t i = 0; i < wanted; ++i) result[i] = distances[i].second;
    return result;
}

inline std::vector<float> residual_sample(const std::vector<float>& sample,
                                          std::size_t rows, std::size_t dimension,
                                          const std::vector<float>& coarse,
                                          std::size_t nlist) {
    std::vector<float> residuals(sample.size());
    for (std::size_t row = 0; row < rows; ++row) {
        const float* point = sample.data() + row * dimension;
        const std::size_t list = nearest(point, coarse, nlist, dimension);
        const float* center = coarse.data() + list * dimension;
        float* residual = residuals.data() + row * dimension;
        for (std::size_t d = 0; d < dimension; ++d) residual[d] = point[d] - center[d];
    }
    return residuals;
}

} // namespace ivf_quantized_detail
