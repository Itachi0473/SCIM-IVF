#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

struct SearchResult {
    std::int64_t id;
    float squared_l2;
};

struct IvfBudgetSearchStats {
    std::size_t total_distance_computations = 0;
    std::size_t center_distance_computations = 0;
    std::size_t scanned_vectors = 0;
    std::size_t scanned_lists = 0;
};

// CPU-only IVF-Flat for dense float vectors and squared-L2 distance.
// The algorithm is: train coarse k-means centroids -> assign vectors to
// inverted lists -> scan vectors in the nprobe nearest lists exactly.
class IvfFlat {
public:
    IvfFlat(std::size_t dimension, std::size_t nlist, std::uint32_t seed = 1234);

    void train(const float* vectors, std::size_t count, std::size_t iterations = 25);
    // Install externally trained coarse centroids, then call add() to populate
    // the lists. This supports apples-to-apples comparisons with quantized IVF.
    void train_with_centroids(const float* centroids);
    // Count assignments first and reserve exact additional list capacity.
    // This avoids vector reallocation peaks for high-dimensional bulk loads.
    void reserve_for_add(const float* vectors, std::size_t count);
    void add(const float* vectors, std::size_t count);
    // Physically remove vectors whose internal IDs appear in ids. The coarse
    // centroids are deliberately unchanged; only their inverted lists change.
    // Returns the number of vectors actually removed. IDs are never reused.
    std::size_t remove_ids(const std::vector<std::int64_t>& ids);
    std::vector<SearchResult> search(const float* query, std::size_t k, std::size_t nprobe) const;
    // Bulk offline construction: route once, reserve exact posting capacities,
    // then populate the lists. IDs follow source row order.
    void add_bulk(const float* vectors, std::size_t count);
    // Search nearest lists in order until exactly budget distance computations
    // are used (all centroid distances plus vector distances; final list may be partial).
    std::vector<SearchResult> search_with_distance_budget(
        const float* query, std::size_t k, std::size_t budget,
        IvfBudgetSearchStats* stats = nullptr) const;
    // Scan an explicit set of distinct inverted lists. Used by hierarchical
    // routing experiments that choose candidate lists outside the flat quantizer.
    std::vector<SearchResult> search_lists(
        const float* query, std::size_t k,
        const std::vector<std::size_t>& list_ids) const;

    // Binary persistence for this implementation. The file stores coarse
    // centroids, all list ids and all original float vectors.
    void save(const std::string& path) const;
    static IvfFlat load(const std::string& path);

    bool is_trained() const noexcept { return trained_; }
    std::size_t dimension() const noexcept { return dimension_; }
    std::size_t nlist() const noexcept { return nlist_; }
    std::size_t size() const noexcept { return active_count_; }
    std::size_t code_bytes() const noexcept;
    std::size_t estimated_index_bytes() const noexcept;
    const std::vector<float>& centroids() const noexcept { return centroids_; }
    std::vector<std::pair<std::int64_t, std::size_t>> vector_list_assignments() const;

private:
    struct InvertedList {
        std::vector<std::int64_t> ids;
        std::vector<float> vectors; // contiguous: ids.size() * dimension_
    };

    float squared_l2(const float* a, const float* b) const;
    std::size_t nearest_centroid(const float* vector) const;
    std::vector<std::size_t> nearest_centroids(const float* query, std::size_t nprobe) const;

    std::size_t dimension_;
    std::size_t nlist_;
    std::uint32_t seed_;
    bool trained_ = false;
    std::int64_t next_id_ = 0;
    std::size_t active_count_ = 0;
    std::vector<float> centroids_; // nlist_ * dimension_
    std::vector<InvertedList> lists_;
};
