#pragma once

#include "ivf_flat.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Exact IVF-Flat for datasets whose stored coordinates are original uint8
// values (such as SIFT1M). Centroids remain float32; vectors stay uint8.
class IvfFlatU8 {
public:
    IvfFlatU8(std::size_t dimension, std::size_t nlist, std::uint32_t seed = 1234);

    void train_with_centroids(const float* centroids);
    void add(const std::uint8_t* vectors, std::size_t count);
    std::vector<SearchResult> search(const float* query, std::size_t k,
                                     std::size_t nprobe) const;
    // Scan nearest lists in order until exactly budget distance computations
    // are used, counting every centroid and vector comparison. The final list
    // may be partially scanned.
    std::vector<SearchResult> search_with_distance_budget(
        const float* query, std::size_t k, std::size_t budget,
        IvfBudgetSearchStats* stats = nullptr) const;
    void save(const std::string& path) const;

    std::size_t size() const noexcept { return active_count_; }
    std::size_t dimension() const noexcept { return dimension_; }
    std::size_t nlist() const noexcept { return nlist_; }

private:
    struct List {
        std::vector<std::uint32_t> ids;
        std::vector<std::uint8_t> vectors;
    };

    std::size_t nearest_centroid(const std::uint8_t* vector) const;
    std::vector<std::size_t> nearest_centroids(const float* query,
                                               std::size_t nprobe) const;

    std::size_t dimension_;
    std::size_t nlist_;
    std::uint32_t seed_;
    bool trained_ = false;
    std::int64_t next_id_ = 0;
    std::size_t active_count_ = 0;
    std::vector<float> centroids_;
    std::vector<List> lists_;
};
