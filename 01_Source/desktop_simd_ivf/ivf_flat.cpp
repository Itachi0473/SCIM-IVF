#include "ivf_flat.h"
#include "simd_l2.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace {
constexpr char kFileMagic[] = "IVFFLAT1";
constexpr std::uint32_t kFileVersion = 1;

template <typename T>
void write_value(std::ofstream& output, const T& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!output) throw std::runtime_error("failed to write index file");
}

template <typename T>
void write_values(std::ofstream& output, const T* values, std::size_t count) {
    if (count == 0) return;
    output.write(reinterpret_cast<const char*>(values), static_cast<std::streamsize>(sizeof(T) * count));
    if (!output) throw std::runtime_error("failed to write index file");
}

template <typename T>
void read_value(std::ifstream& input, T& value) {
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) throw std::runtime_error("invalid or truncated index file");
}

template <typename T>
void read_values(std::ifstream& input, T* values, std::size_t count) {
    if (count == 0) return;
    input.read(reinterpret_cast<char*>(values), static_cast<std::streamsize>(sizeof(T) * count));
    if (!input) throw std::runtime_error("invalid or truncated index file");
}

std::size_t checked_size(std::uint64_t value, const char* label) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::runtime_error(std::string("index ") + label + " is too large");
    }
    return static_cast<std::size_t>(value);
}
} // namespace

IvfFlat::IvfFlat(std::size_t dimension, std::size_t nlist, std::uint32_t seed)
    : dimension_(dimension), nlist_(nlist), seed_(seed), lists_(nlist) {
    if (dimension == 0 || nlist == 0) {
        throw std::invalid_argument("dimension and nlist must be positive");
    }
}

float IvfFlat::squared_l2(const float* a, const float* b) const {
    return simd_squared_l2(a, b, dimension_);
}

std::size_t IvfFlat::nearest_centroid(const float* vector) const {
    std::size_t best = 0;
    float best_distance = squared_l2(vector, centroids_.data());
    for (std::size_t i = 1; i < nlist_; ++i) {
        const float distance = squared_l2(vector, centroids_.data() + i * dimension_);
        if (distance < best_distance) { // identical tie rule: lower list id wins
            best = i;
            best_distance = distance;
        }
    }
    return best;
}

std::vector<std::size_t> IvfFlat::nearest_centroids(const float* query, std::size_t nprobe) const {
    std::vector<std::size_t> ids(nlist_);
    std::iota(ids.begin(), ids.end(), 0);
    std::sort(ids.begin(), ids.end(), [this, query](std::size_t a, std::size_t b) {
        const float da = squared_l2(query, centroids_.data() + a * dimension_);
        const float db = squared_l2(query, centroids_.data() + b * dimension_);
        return da == db ? a < b : da < db;
    });
    ids.resize(std::min(nprobe, nlist_));
    return ids;
}

void IvfFlat::train(const float* vectors, std::size_t count, std::size_t iterations) {
    if (vectors == nullptr || count < nlist_ || iterations == 0) {
        throw std::invalid_argument("training needs at least nlist vectors and one iteration");
    }

    // FAISS also learns a coarse quantizer with k-means. This deterministic
    // initialization makes this standalone implementation reproducible.
    std::vector<std::size_t> choices(count);
    std::iota(choices.begin(), choices.end(), 0);
    std::mt19937 rng(seed_);
    std::shuffle(choices.begin(), choices.end(), rng);
    centroids_.resize(nlist_ * dimension_);
    for (std::size_t c = 0; c < nlist_; ++c) {
        std::copy_n(vectors + choices[c] * dimension_, dimension_, centroids_.data() + c * dimension_);
    }

    std::vector<std::size_t> assignments(count);
    for (std::size_t step = 0; step < iterations; ++step) {
        std::vector<float> sums(nlist_ * dimension_, 0.0f);
        std::vector<std::size_t> sizes(nlist_, 0);
        for (std::size_t i = 0; i < count; ++i) {
            const float* vector = vectors + i * dimension_;
            const std::size_t cluster = nearest_centroid(vector);
            assignments[i] = cluster;
            ++sizes[cluster];
            for (std::size_t j = 0; j < dimension_; ++j) {
                sums[cluster * dimension_ + j] += vector[j];
            }
        }
        for (std::size_t c = 0; c < nlist_; ++c) {
            float* centroid = centroids_.data() + c * dimension_;
            if (sizes[c] == 0) {
                const std::size_t replacement = choices[rng() % count];
                std::copy_n(vectors + replacement * dimension_, dimension_, centroid);
            } else {
                for (std::size_t j = 0; j < dimension_; ++j) {
                    centroid[j] = sums[c * dimension_ + j] / static_cast<float>(sizes[c]);
                }
            }
        }
    }
    trained_ = true;
}

void IvfFlat::train_with_centroids(const float* centroids) {
    if (trained_) throw std::logic_error("IVF is already trained");
    if (centroids == nullptr) throw std::invalid_argument("coarse centroids must not be null");
    centroids_.assign(centroids, centroids + nlist_ * dimension_);
    trained_ = true;
}

void IvfFlat::add(const float* vectors, std::size_t count) {
    if (!trained_) throw std::logic_error("train must be called before add");
    if (vectors == nullptr) throw std::invalid_argument("vectors must not be null");
    for (std::size_t i = 0; i < count; ++i) {
        const float* vector = vectors + i * dimension_;
        const std::size_t list_id = nearest_centroid(vector);
        InvertedList& list = lists_[list_id];
        list.ids.push_back(next_id_++);
        const std::size_t old_size = list.vectors.size();
        list.vectors.resize(old_size + dimension_);
        simd_copy_floats(list.vectors.data() + old_size, vector, dimension_);
        ++active_count_;
    }
}

void IvfFlat::add_bulk(const float* vectors, std::size_t count) {
    if (!trained_) throw std::logic_error("train must be called before add_bulk");
    if (vectors == nullptr) throw std::invalid_argument("vectors must not be null");
    if (count > std::numeric_limits<std::uint32_t>::max() ||
        static_cast<std::uint64_t>(next_id_) + count >
            static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error("bulk insertion IDs exceed supported range");
    }

    std::vector<std::uint32_t> assignments(count);
    std::vector<std::size_t> additions(nlist_, 0);
    for (std::size_t i = 0; i < count; ++i) {
        const std::size_t list_id = nearest_centroid(vectors + i * dimension_);
        assignments[i] = static_cast<std::uint32_t>(list_id);
        ++additions[list_id];
    }
    for (std::size_t list_id = 0; list_id < nlist_; ++list_id) {
        InvertedList& list = lists_[list_id];
        const std::size_t final_count = list.ids.size() + additions[list_id];
        if (final_count > std::numeric_limits<std::size_t>::max() / dimension_) {
            throw std::length_error("requested inverted list is too large");
        }
        list.ids.reserve(final_count);
        list.vectors.reserve(final_count * dimension_);
    }
    for (std::size_t i = 0; i < count; ++i) {
        InvertedList& list = lists_[assignments[i]];
        list.ids.push_back(next_id_++);
        const std::size_t old_size = list.vectors.size();
        list.vectors.resize(old_size + dimension_);
        simd_copy_floats(list.vectors.data() + old_size,
                         vectors + i * dimension_, dimension_);
    }
    active_count_ += count;
}

std::size_t IvfFlat::remove_ids(const std::vector<std::int64_t>& ids) {
    if (!trained_) throw std::logic_error("train must be called before remove_ids");
    if (ids.empty()) return 0;

    bool dense_range = true;
    for (std::size_t i = 1; i < ids.size(); ++i) {
        if (ids[i - 1] == std::numeric_limits<std::int64_t>::max() || ids[i] != ids[i - 1] + 1) {
            dense_range = false;
            break;
        }
    }
    const std::int64_t first = dense_range ? ids.front() : 0;
    const std::int64_t last = dense_range ? ids.back() : -1;
    const std::unordered_set<std::int64_t> targets = dense_range
            ? std::unordered_set<std::int64_t>{}
            : std::unordered_set<std::int64_t>(ids.begin(), ids.end());
    std::size_t removed = 0;
    for (InvertedList& list : lists_) {
        std::size_t position = 0;
        while (position < list.ids.size()) {
            if (dense_range && position + 4 <= list.ids.size() &&
                simd_id_range_mask(list.ids.data() + position, 4, first, last) == 0) {
                position += 4;
                continue;
            }
            const bool selected = dense_range
                    ? list.ids[position] >= first && list.ids[position] <= last
                    : targets.count(list.ids[position]) != 0;
            if (!selected) {
                ++position;
                continue;
            }

            // Swap the last record into the removed slot. This keeps vector
            // storage contiguous and avoids shifting the entire inverted list.
            const std::size_t last = list.ids.size() - 1;
            if (position != last) {
                list.ids[position] = list.ids[last];
                simd_copy_floats(list.vectors.data() + position * dimension_,
                                 list.vectors.data() + last * dimension_, dimension_);
            }
            list.ids.pop_back();
            list.vectors.resize(last * dimension_);
            ++removed;
            --active_count_;
            // Examine position again because a previously-last record moved here.
        }
    }
    return removed;
}

std::vector<std::pair<std::int64_t, std::size_t>> IvfFlat::vector_list_assignments() const {
    if (!trained_) throw std::logic_error("train must be called before vector_list_assignments");
    std::vector<std::pair<std::int64_t, std::size_t>> result;
    result.reserve(active_count_);
    for (std::size_t list_id = 0; list_id < lists_.size(); ++list_id)
        for (const std::int64_t id : lists_[list_id].ids) result.emplace_back(id, list_id);
    return result;
}

std::size_t IvfFlat::code_bytes() const noexcept {
    std::size_t bytes = 0;
    for (const InvertedList& list : lists_) bytes += list.vectors.size() * sizeof(float);
    return bytes;
}

std::size_t IvfFlat::estimated_index_bytes() const noexcept {
    std::size_t bytes = centroids_.size() * sizeof(float) + code_bytes();
    for (const InvertedList& list : lists_) bytes += list.ids.size() * sizeof(std::int64_t);
    return bytes;
}

void IvfFlat::reserve_for_add(const float* vectors, std::size_t count) {
    if (!trained_) throw std::logic_error("train must be called before reserve_for_add");
    if (vectors == nullptr) throw std::invalid_argument("vectors must not be null");
    std::vector<std::size_t> additions(nlist_, 0);
    for (std::size_t i = 0; i < count; ++i) {
        ++additions[nearest_centroid(vectors + i * dimension_)];
    }
    for (std::size_t list_id = 0; list_id < nlist_; ++list_id) {
        InvertedList& list = lists_[list_id];
        const std::size_t final_count = list.ids.size() + additions[list_id];
        if (final_count > std::numeric_limits<std::size_t>::max() / dimension_) {
            throw std::length_error("requested inverted list is too large");
        }
        list.ids.reserve(final_count);
        list.vectors.reserve(final_count * dimension_);
    }
}

std::vector<SearchResult> IvfFlat::search(
        const float* query, std::size_t k, std::size_t nprobe) const {
    if (!trained_) throw std::logic_error("train must be called before search");
    if (query == nullptr || k == 0 || nprobe == 0) return {};

    auto better = [](const SearchResult& a, const SearchResult& b) {
        return a.squared_l2 < b.squared_l2 ||
               (a.squared_l2 == b.squared_l2 && a.id < b.id);
    };
    std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(better)> heap(better);
    for (const std::size_t list_id : nearest_centroids(query, nprobe)) {
        const InvertedList& list = lists_[list_id];
        for (std::size_t i = 0; i < list.ids.size(); ++i) {
            SearchResult candidate{list.ids[i], squared_l2(query, list.vectors.data() + i * dimension_)};
            if (heap.size() < k) {
                heap.push(candidate);
            } else if (better(candidate, heap.top())) {
                heap.pop();
                heap.push(candidate);
            }
        }
    }
    std::vector<SearchResult> result;
    while (!heap.empty()) {
        result.push_back(heap.top());
        heap.pop();
    }
    std::sort(result.begin(), result.end(), better);
    return result;
}

std::vector<SearchResult> IvfFlat::search_with_distance_budget(
        const float* query, std::size_t k, std::size_t budget,
        IvfBudgetSearchStats* stats) const {
    if (!trained_) throw std::logic_error("train must be called before search");
    if (query == nullptr || k == 0) return {};
    if (budget < nlist_) {
        throw std::invalid_argument("distance budget must cover every centroid comparison");
    }

    if (stats != nullptr) {
        *stats = {};
        stats->center_distance_computations = nlist_;
    }
    std::vector<std::pair<float, std::size_t>> ordered_centroids;
    ordered_centroids.reserve(nlist_);
    for (std::size_t i = 0; i < nlist_; ++i) {
        ordered_centroids.emplace_back(
            squared_l2(query, centroids_.data() + i * dimension_), i);
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
        const InvertedList& list = lists_[center.second];
        const std::size_t take = std::min(remaining, list.ids.size());
        if (take == 0) continue;
        if (stats != nullptr) ++stats->scanned_lists;
        for (std::size_t i = 0; i < take; ++i) {
            const SearchResult candidate{
                list.ids[i], squared_l2(query, list.vectors.data() + i * dimension_)};
            if (heap.size() < k) {
                heap.push(candidate);
            } else if (better(candidate, heap.top())) {
                heap.pop();
                heap.push(candidate);
            }
        }
        remaining -= take;
        if (stats != nullptr) stats->scanned_vectors += take;
    }
    if (remaining != 0) {
        throw std::runtime_error("distance budget exceeds available indexed vectors");
    }
    if (stats != nullptr) {
        stats->total_distance_computations = budget;
    }

    std::vector<SearchResult> result;
    result.reserve(std::min(k, heap.size()));
    while (!heap.empty()) {
        result.push_back(heap.top());
        heap.pop();
    }
    std::sort(result.begin(), result.end(), better);
    return result;
}

std::vector<SearchResult> IvfFlat::search_lists(
        const float* query, std::size_t k,
        const std::vector<std::size_t>& list_ids) const {
    if (!trained_) throw std::logic_error("train must be called before search_lists");
    if (query == nullptr || k == 0 || list_ids.empty()) return {};

    auto better = [](const SearchResult& a, const SearchResult& b) {
        return a.squared_l2 < b.squared_l2 ||
               (a.squared_l2 == b.squared_l2 && a.id < b.id);
    };
    std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(better)> heap(better);
    for (const std::size_t list_id : list_ids) {
        if (list_id >= nlist_) throw std::out_of_range("search list id is out of range");
        const InvertedList& list = lists_[list_id];
        for (std::size_t i = 0; i < list.ids.size(); ++i) {
            const SearchResult candidate{
                list.ids[i], squared_l2(query, list.vectors.data() + i * dimension_)};
            if (heap.size() < k) {
                heap.push(candidate);
            } else if (better(candidate, heap.top())) {
                heap.pop();
                heap.push(candidate);
            }
        }
    }
    std::vector<SearchResult> result;
    while (!heap.empty()) {
        result.push_back(heap.top());
        heap.pop();
    }
    std::sort(result.begin(), result.end(), better);
    return result;
}

void IvfFlat::save(const std::string& path) const {
    if (!trained_) throw std::logic_error("cannot save an untrained index");
    const std::filesystem::path destination(path);
    const std::filesystem::path temporary = destination.string() + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create temporary index file");

    output.write(kFileMagic, sizeof(kFileMagic) - 1);
    write_value(output, kFileVersion);
    write_value(output, static_cast<std::uint64_t>(dimension_));
    write_value(output, static_cast<std::uint64_t>(nlist_));
    write_value(output, seed_);
    write_value(output, next_id_);
    write_values(output, centroids_.data(), centroids_.size());
    for (const InvertedList& list : lists_) {
        const std::uint64_t count = static_cast<std::uint64_t>(list.ids.size());
        write_value(output, count);
        write_values(output, list.ids.data(), list.ids.size());
        write_values(output, list.vectors.data(), list.vectors.size());
    }
    output.close();
    if (!output) throw std::runtime_error("failed to finish index file");

    std::error_code error;
    std::filesystem::rename(temporary, destination, error);
    if (error) {
        // Windows cannot replace an existing destination with rename(). The
        // completed temporary file remains intact until replacement is safe.
        std::filesystem::remove(destination, error);
        if (error) throw std::runtime_error("cannot replace existing index file");
        std::filesystem::rename(temporary, destination, error);
        if (error) throw std::runtime_error("cannot activate saved index file");
    }
}

IvfFlat IvfFlat::load(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open index file");
    char magic[sizeof(kFileMagic) - 1];
    input.read(magic, sizeof(magic));
    if (!input || !std::equal(std::begin(magic), std::end(magic), std::begin(kFileMagic))) {
        throw std::runtime_error("not an IVFFlat index file");
    }
    std::uint32_t version = 0;
    std::uint64_t stored_dimension = 0, stored_nlist = 0;
    std::uint32_t seed = 0;
    std::int64_t next_id = 0;
    read_value(input, version);
    read_value(input, stored_dimension);
    read_value(input, stored_nlist);
    read_value(input, seed);
    read_value(input, next_id);
    if (version != kFileVersion || next_id < 0) throw std::runtime_error("unsupported index file version");

    const std::size_t dimension = checked_size(stored_dimension, "dimension");
    const std::size_t nlist = checked_size(stored_nlist, "nlist");
    IvfFlat index(dimension, nlist, seed);
    index.centroids_.resize(dimension * nlist);
    read_values(input, index.centroids_.data(), index.centroids_.size());
    for (InvertedList& list : index.lists_) {
        std::uint64_t stored_count = 0;
        read_value(input, stored_count);
        const std::size_t count = checked_size(stored_count, "list size");
        if (count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::runtime_error("index list is too large");
        }
        list.ids.resize(count);
        list.vectors.resize(count * dimension);
        read_values(input, list.ids.data(), list.ids.size());
        read_values(input, list.vectors.data(), list.vectors.size());
        index.active_count_ += count;
    }
    index.next_id_ = next_id;
    index.trained_ = true;
    return index;
}
