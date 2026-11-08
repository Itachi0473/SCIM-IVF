#include "ivf_flat_u8.h"
#include "simd_l2.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>

namespace {
constexpr char kMagic[] = "IVFFLATU8";
constexpr std::uint32_t kVersion = 1;

template <typename T>
void write_scalar(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("failed writing uint8 IVF index");
}

template <typename T>
void write_values(std::ofstream& out, const T* values, std::size_t count) {
    if (count == 0) return;
    out.write(reinterpret_cast<const char*>(values),
              static_cast<std::streamsize>(count * sizeof(T)));
    if (!out) throw std::runtime_error("failed writing uint8 IVF payload");
}
}

IvfFlatU8::IvfFlatU8(std::size_t dimension, std::size_t nlist, std::uint32_t seed)
    : dimension_(dimension), nlist_(nlist), seed_(seed), lists_(nlist) {
    if (dimension_ == 0 || nlist_ == 0)
        throw std::invalid_argument("dimension and nlist must be positive");
}

void IvfFlatU8::train_with_centroids(const float* centroids) {
    if (trained_) throw std::logic_error("uint8 IVF is already trained");
    if (centroids == nullptr) throw std::invalid_argument("centroids must not be null");
    centroids_.assign(centroids, centroids + nlist_ * dimension_);
    trained_ = true;
}

std::size_t IvfFlatU8::nearest_centroid(const std::uint8_t* vector) const {
    std::size_t best = 0;
    float best_distance = simd_squared_l2_u8_f32(vector, centroids_.data(), dimension_);
    for (std::size_t c = 1; c < nlist_; ++c) {
        const float distance = simd_squared_l2_u8_f32(
            vector, centroids_.data() + c * dimension_, dimension_);
        if (distance < best_distance) {
            best = c;
            best_distance = distance;
        }
    }
    return best;
}

void IvfFlatU8::add(const std::uint8_t* vectors, std::size_t count) {
    if (!trained_) throw std::logic_error("train_with_centroids must precede uint8 IVF add");
    if (vectors == nullptr) throw std::invalid_argument("vectors must not be null");
    const std::uint64_t capacity = static_cast<std::uint64_t>(
        std::numeric_limits<std::uint32_t>::max()) + 1ULL;
    if (static_cast<std::uint64_t>(next_id_) > capacity ||
        static_cast<std::uint64_t>(count) > capacity - static_cast<std::uint64_t>(next_id_))
        throw std::overflow_error("vector IDs exceed compact 32-bit IVF storage capacity");
    for (std::size_t i = 0; i < count; ++i) {
        const auto* vector = vectors + i * dimension_;
        const std::size_t list_id = nearest_centroid(vector);
        List& list = lists_[list_id];
        list.ids.push_back(static_cast<std::uint32_t>(next_id_++));
        list.vectors.insert(list.vectors.end(), vector, vector + dimension_);
    }
    active_count_ += count;
}

std::vector<std::size_t> IvfFlatU8::nearest_centroids(const float* query,
                                                      std::size_t nprobe) const {
    std::vector<std::pair<float, std::size_t>> distances;
    distances.reserve(nlist_);
    for (std::size_t c = 0; c < nlist_; ++c)
        distances.emplace_back(simd_squared_l2(query, centroids_.data() + c * dimension_, dimension_), c);
    const std::size_t wanted = std::min(nprobe, nlist_);
    std::partial_sort(distances.begin(), distances.begin() + wanted, distances.end(),
        [](const auto& a, const auto& b) { return a.first == b.first ? a.second < b.second : a.first < b.first; });
    std::vector<std::size_t> result;
    result.reserve(wanted);
    for (std::size_t i = 0; i < wanted; ++i) result.push_back(distances[i].second);
    return result;
}

std::vector<SearchResult> IvfFlatU8::search(const float* query, std::size_t k,
                                            std::size_t nprobe) const {
    if (!trained_) throw std::logic_error("train_with_centroids must precede uint8 IVF search");
    if (query == nullptr || k == 0 || nprobe == 0) return {};
    auto better = [](const SearchResult& a, const SearchResult& b) {
        return a.squared_l2 < b.squared_l2 ||
               (a.squared_l2 == b.squared_l2 && a.id < b.id);
    };
    std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(better)> heap(better);
    for (const auto list_id : nearest_centroids(query, nprobe)) {
        const List& list = lists_[list_id];
        for (std::size_t i = 0; i < list.ids.size(); ++i) {
            const auto* vector = list.vectors.data() + i * dimension_;
            const SearchResult hit{static_cast<std::int64_t>(list.ids[i]),
                                   simd_squared_l2_u8_f32(vector, query, dimension_)};
            if (heap.size() < k) heap.push(hit);
            else if (better(hit, heap.top())) { heap.pop(); heap.push(hit); }
        }
    }
    std::vector<SearchResult> result;
    result.reserve(heap.size());
    while (!heap.empty()) { result.push_back(heap.top()); heap.pop(); }
    std::sort(result.begin(), result.end(), better);
    return result;
}

std::vector<SearchResult> IvfFlatU8::search_with_distance_budget(
        const float* query, std::size_t k, std::size_t budget,
        IvfBudgetSearchStats* stats) const {
    if (!trained_) throw std::logic_error("train_with_centroids must precede uint8 IVF search");
    if (query == nullptr || k == 0) return {};
    if (budget < nlist_)
        throw std::invalid_argument("distance budget must cover every centroid comparison");

    if (stats != nullptr) {
        *stats = {};
        stats->center_distance_computations = nlist_;
    }
    std::vector<std::pair<float, std::size_t>> ordered_centroids;
    ordered_centroids.reserve(nlist_);
    for (std::size_t i = 0; i < nlist_; ++i) {
        ordered_centroids.emplace_back(
            simd_squared_l2(query, centroids_.data() + i * dimension_, dimension_), i);
    }
    std::sort(ordered_centroids.begin(), ordered_centroids.end(),
              [](const auto& a, const auto& b) {
                  return a.first == b.first ? a.second < b.second : a.first < b.first;
              });

    auto better = [](const SearchResult& a, const SearchResult& b) {
        return a.squared_l2 < b.squared_l2 ||
               (a.squared_l2 == b.squared_l2 && a.id < b.id);
    };
    std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(better)> heap(better);
    std::size_t remaining = budget - nlist_;
    for (const auto& center : ordered_centroids) {
        if (remaining == 0) break;
        const List& list = lists_[center.second];
        const std::size_t take = std::min(remaining, list.ids.size());
        if (take == 0) continue;
        if (stats != nullptr) ++stats->scanned_lists;
        for (std::size_t i = 0; i < take; ++i) {
            const SearchResult candidate{
                static_cast<std::int64_t>(list.ids[i]),
                simd_squared_l2_u8_f32(list.vectors.data() + i * dimension_, query, dimension_)};
            if (heap.size() < k) heap.push(candidate);
            else if (better(candidate, heap.top())) { heap.pop(); heap.push(candidate); }
        }
        remaining -= take;
        if (stats != nullptr) stats->scanned_vectors += take;
    }
    if (remaining != 0)
        throw std::runtime_error("distance budget exceeds available indexed vectors");
    if (stats != nullptr) stats->total_distance_computations = budget;

    std::vector<SearchResult> result;
    result.reserve(std::min(k, heap.size()));
    while (!heap.empty()) { result.push_back(heap.top()); heap.pop(); }
    std::sort(result.begin(), result.end(), better);
    return result;
}

void IvfFlatU8::save(const std::string& path) const {
    if (!trained_) throw std::logic_error("cannot save an untrained uint8 IVF index");
    const std::filesystem::path destination(path);
    const std::filesystem::path temporary = destination.string() + ".tmp";
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create temporary uint8 IVF index file");
    out.write(kMagic, sizeof(kMagic) - 1);
    write_scalar(out, kVersion);
    write_scalar(out, static_cast<std::uint64_t>(dimension_));
    write_scalar(out, static_cast<std::uint64_t>(nlist_));
    write_scalar(out, seed_);
    write_scalar(out, next_id_);
    write_values(out, centroids_.data(), centroids_.size());
    for (const List& list : lists_) {
        write_scalar(out, static_cast<std::uint64_t>(list.ids.size()));
        write_values(out, list.ids.data(), list.ids.size());
        write_values(out, list.vectors.data(), list.vectors.size());
    }
    out.close();
    if (!out) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw std::runtime_error("failed to finish uint8 IVF index file");
    }
    std::error_code error;
    std::filesystem::rename(temporary, destination, error);
    if (error) {
        std::filesystem::remove(temporary);
        throw std::runtime_error("cannot activate uint8 IVF index file: " + error.message());
    }
}
