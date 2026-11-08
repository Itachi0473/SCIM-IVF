#include "adaptive_hierarchical_ivf.h"
#include "simd_l2.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {
#ifdef GLOVE100
constexpr std::size_t kRows = 1'183'514, kDim = 100, kInitialNlist = 1'100;
using BaseVector = float;
#else
constexpr std::size_t kRows = 1'000'000, kDim = 128, kInitialNlist = 1'000;
using BaseVector = std::uint8_t;
#endif
constexpr std::size_t kBatchCount = 40;
constexpr std::size_t kBatchRows = 20'000, kQueries = 1'000, kTopK = 10;
constexpr std::size_t kMaxPosting = 1'000, kReassignRange = 64;
constexpr std::uint32_t kSeed = 1234;
using Clock = std::chrono::steady_clock;
using TruthRow = std::array<std::int32_t, kTopK>;
struct Hit { float distance; std::uint32_t id; };
struct Move { std::size_t from, to; std::uint32_t id; };
struct Partition {
    std::vector<std::uint32_t> ids;
    std::vector<BaseVector> vectors;
    std::vector<float> center;
};
struct Maintenance { std::size_t splits = 0, candidates = 0, moved = 0; };
struct SearchResult { std::vector<std::uint32_t> ids; std::size_t centers = 0, vectors = 0; };

template <class T> T read_scalar(std::istream& in, const char* label) {
    T v{}; in.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!in) throw std::runtime_error(std::string("truncated ") + label);
    return v;
}

std::vector<BaseVector> read_base(const std::filesystem::path& path) {
#ifdef GLOVE100
    if (std::filesystem::file_size(path) != kRows * kDim * sizeof(float))
        throw std::runtime_error("GloVe base must be exactly 1,183,514 x 100 FP32 values");
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open GloVe base: " + path.string());
    std::vector<BaseVector> data(kRows * kDim);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(BaseVector)));
    if (!in) throw std::runtime_error("truncated GloVe base payload");
    return data;
#else
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open SIFT base: " + path.string());
    const auto rows = read_scalar<std::uint32_t>(in, "base rows");
    const auto dim = read_scalar<std::uint32_t>(in, "base dimension");
    if (rows != kRows || dim != kDim || std::filesystem::file_size(path) != 8ULL + kRows * kDim)
        throw std::runtime_error("expected 1M x 128 uint8 SIFT base");
    std::vector<BaseVector> data(kRows * kDim);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!in) throw std::runtime_error("truncated SIFT base payload");
    return data;
#endif
}

float vector_l2(const float* a, const float* b) { return simd_squared_l2(a, b, kDim); }
#ifndef GLOVE100
float vector_l2(const std::uint8_t* a, const float* b) { return simd_squared_l2_u8_f32(a, b, kDim); }
#endif

std::vector<float> read_queries(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open SIFT queries: " + path.string());
    std::vector<float> data(kQueries * kDim);
    for (std::size_t q = 0; q < kQueries; ++q) {
        if (read_scalar<std::int32_t>(in, "query dimension") != static_cast<std::int32_t>(kDim))
            throw std::runtime_error("query vectors must be 128D fvecs");
        in.read(reinterpret_cast<char*>(data.data() + q * kDim), static_cast<std::streamsize>(kDim * sizeof(float)));
        if (!in) throw std::runtime_error("truncated query vector");
    }
    return data;
}

std::vector<TruthRow> read_truth(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open exact truth: " + path.string());
    std::vector<TruthRow> data(kQueries);
    for (auto& row : data) {
        if (read_scalar<std::int32_t>(in, "truth row length") != static_cast<std::int32_t>(kTopK))
            throw std::runtime_error("expected top-10 truth rows");
        in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(sizeof(row)));
        if (!in) throw std::runtime_error("truncated truth row");
    }
    return data;
}

std::vector<std::vector<std::int64_t>> read_deletions(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open deletion schedule: " + path.string());
    std::string line;
    if (!std::getline(in, line) || line != "batch,position,original_vector_id")
        throw std::runtime_error("unexpected deletion schedule header");
    std::vector<std::vector<std::int64_t>> batches(kBatchCount);
    for (auto& b : batches) b.reserve(kBatchRows);
    std::vector<std::uint8_t> seen(kRows, 0);
    std::size_t n = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream row(line);
        std::size_t batch = 0, position = 0; std::int64_t id = -1; char c1 = 0, c2 = 0;
        if (!(row >> batch >> c1 >> position >> c2 >> id) || c1 != ',' || c2 != ',' ||
            batch == 0 || batch > kBatchCount || position >= kBatchRows || id < 0 || id >= static_cast<std::int64_t>(kRows))
            throw std::runtime_error("malformed deletion schedule row");
        auto& target = batches[batch - 1];
        if (position != target.size() || seen[static_cast<std::size_t>(id)])
            throw std::runtime_error("deletions must be unique and position ordered");
        target.push_back(id); seen[static_cast<std::size_t>(id)] = 1; ++n;
    }
    if (n != kBatchCount * kBatchRows) throw std::runtime_error("deletion schedule must contain 800,000 IDs");
    return batches;
}

bool better(const Hit& a, const Hit& b) { return a.distance == b.distance ? a.id < b.id : a.distance < b.distance; }
void offer(std::vector<Hit>& best, Hit x) {
    const auto at = std::lower_bound(best.begin(), best.end(), x, better);
    if (best.size() == kTopK && at == best.end()) return;
    best.insert(at, x);
    if (best.size() > kTopK) best.pop_back();
}

class SpfreshLire {
public:
    explicit SpfreshLire(const std::vector<BaseVector>& data) : base_(data.data()) {}

    void initialize(const std::vector<float>& centers,
                    const std::vector<std::vector<std::uint32_t>>& ids_by_leaf) {
        if (centers.size() != ids_by_leaf.size() * kDim) throw std::runtime_error("initial center/leaf mismatch");
        parts_.resize(ids_by_leaf.size());
        for (std::size_t i = 0; i < parts_.size(); ++i) {
            parts_[i].center.assign(centers.begin() + static_cast<std::ptrdiff_t>(i * kDim),
                                    centers.begin() + static_cast<std::ptrdiff_t>((i + 1) * kDim));
            parts_[i].ids = ids_by_leaf[i];
            rebuild_payload(parts_[i]);
            active_ += ids_by_leaf[i].size();
        }
    }

    std::size_t size() const { return active_; }
    std::size_t nlist() const { return parts_.size(); }
    std::size_t max_size() const {
        std::size_t m = 0; for (const auto& p : parts_) m = std::max(m, p.ids.size()); return m;
    }

    std::size_t remove_ids(const std::vector<std::int64_t>& ids) {
        std::unordered_set<std::uint32_t> deleting;
        deleting.reserve(ids.size() * 2);
        for (const auto id : ids) deleting.insert(static_cast<std::uint32_t>(id));
        std::size_t removed = 0;
        for (auto& part : parts_) {
            std::vector<std::uint32_t> keep;
            keep.reserve(part.ids.size());
            for (const auto id : part.ids) {
                const auto it = deleting.find(id);
                if (it == deleting.end()) keep.push_back(id);
                else { deleting.erase(it); ++removed; }
            }
            part.ids = std::move(keep);
            rebuild_payload(part);
        }
        if (!deleting.empty() || removed != ids.size() || removed > active_)
            throw std::runtime_error("SPFresh-LIRE did not remove the requested live IDs");
        active_ -= removed;
        remove_empty();
        return removed;
    }

    Maintenance maintain() {
        Maintenance stats;
        std::vector<std::size_t> pending;
        std::vector<std::uint8_t> queued(parts_.size(), 0);
        auto enqueue = [&]() {
            if (queued.size() < parts_.size()) queued.resize(parts_.size(), 0);
            for (std::size_t i = 0; i < parts_.size(); ++i)
                if (parts_[i].ids.size() > kMaxPosting && !queued[i]) { pending.push_back(i); queued[i] = 1; }
        };
        enqueue();
        while (!pending.empty()) {
            const std::size_t leaf = pending.back(); pending.pop_back(); queued[leaf] = 0;
            if (parts_[leaf].ids.size() <= kMaxPosting) continue;
            split_and_reassign(leaf, stats);
            enqueue();
            if (stats.splits % 25 == 0)
                std::cout << "LIRE_CAP_PROGRESS splits=" << stats.splits << " nlist=" << parts_.size()
                          << " pending=" << pending.size() << " candidates=" << stats.candidates
                          << " moved=" << stats.moved << '\n' << std::flush;
        }
        remove_empty();
        return stats;
    }

    SearchResult search_budget(const float* query, std::size_t total_budget) const {
        SearchResult out;
        std::vector<std::pair<float, std::size_t>> ranked;
        ranked.reserve(parts_.size());
        for (std::size_t i = 0; i < parts_.size(); ++i)
            ranked.emplace_back(simd_squared_l2(query, parts_[i].center.data(), kDim), i);
        out.centers = ranked.size();
        if (total_budget <= out.centers) throw std::runtime_error("query budget is smaller than centroid count");
        std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
            return a.first == b.first ? a.second < b.second : a.first < b.first;
        });
        const std::size_t vector_budget = total_budget - out.centers;
        std::vector<Hit> top; top.reserve(kTopK + 1);
        for (const auto& entry : ranked) {
            const auto& part = parts_[entry.second];
            for (std::size_t i = 0; i < part.ids.size(); ++i) {
                offer(top, Hit{vector_l2(part.vectors.data() + i * kDim, query), part.ids[i]});
                if (++out.vectors == vector_budget) break;
            }
            if (out.vectors == vector_budget) break;
        }
        if (out.vectors != vector_budget) throw std::runtime_error("LIRE did not consume the exact query budget");
        for (const auto& hit : top) out.ids.push_back(hit.id);
        return out;
    }

    void validate(std::size_t expected_active) const {
        if (active_ != expected_active) throw std::runtime_error("LIRE active count mismatch");
        std::vector<std::uint8_t> seen(kRows, 0);
        std::size_t count = 0;
        for (const auto& p : parts_) for (const auto id : p.ids) {
            if (id >= kRows || seen[id]++) throw std::runtime_error("LIRE has duplicate or invalid vector ownership");
            ++count;
        }
        if (count != expected_active) throw std::runtime_error("LIRE postings do not cover active vectors");
    }

private:
    float point_center_distance(std::uint32_t id, const float* center) const {
        return vector_l2(base_ + static_cast<std::size_t>(id) * kDim, center);
    }
    void remove_empty() {
        parts_.erase(std::remove_if(parts_.begin(), parts_.end(), [](const Partition& p) { return p.ids.empty(); }), parts_.end());
    }
    void rebuild_payload(Partition& p) const {
        p.vectors.resize(p.ids.size() * kDim);
        for (std::size_t i = 0; i < p.ids.size(); ++i)
            std::copy_n(base_ + static_cast<std::size_t>(p.ids[i]) * kDim, kDim, p.vectors.data() + i * kDim);
    }

    struct SplitResult { std::array<std::vector<std::uint32_t>, 2> ids; std::array<std::vector<float>, 2> centers; };
    SplitResult balanced_split(const std::vector<std::uint32_t>& ids, std::uint32_t seed) const {
        if (ids.size() < 2) throw std::runtime_error("cannot split posting with fewer than two vectors");
        std::mt19937 rng(seed);
        const std::size_t first = static_cast<std::size_t>(rng()) % ids.size();
        const auto first_id = ids[first];
        std::size_t second = first; double farthest = -1.0;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const auto* a = base_ + static_cast<std::size_t>(first_id) * kDim;
            const auto* b = base_ + static_cast<std::size_t>(ids[i]) * kDim;
            double d = 0.0;
            for (std::size_t j = 0; j < kDim; ++j) { const double x = static_cast<double>(a[j]) - b[j]; d += x * x; }
            if (d > farthest) { farthest = d; second = i; }
        }
        SplitResult result;
        result.centers[0].resize(kDim); result.centers[1].resize(kDim);
        for (std::size_t j = 0; j < kDim; ++j) {
            result.centers[0][j] = static_cast<float>(base_[static_cast<std::size_t>(first_id) * kDim + j]);
            result.centers[1][j] = static_cast<float>(base_[static_cast<std::size_t>(ids[second]) * kDim + j]);
        }
        std::vector<std::size_t> assignment(ids.size(), 0);
        for (std::size_t iteration = 0; iteration < 5; ++iteration) {
            std::vector<std::pair<float, float>> distances(ids.size());
            std::vector<std::pair<float, std::size_t>> regrets(ids.size());
            for (std::size_t i = 0; i < ids.size(); ++i) {
                const float d0 = point_center_distance(ids[i], result.centers[0].data());
                const float d1 = point_center_distance(ids[i], result.centers[1].data());
                distances[i] = {d0, d1};
                regrets[i] = {std::abs(d0 - d1), i};
            }
            std::sort(regrets.begin(), regrets.end(), [](const auto& a, const auto& b) {
                return a.first == b.first ? a.second < b.second : a.first > b.first;
            });
            result.ids[0].clear(); result.ids[1].clear();
            const std::array<std::size_t, 2> cap{(ids.size() + 1) / 2, ids.size() / 2};
            std::array<std::size_t, 2> sizes{};
            std::array<std::vector<double>, 2> sums{std::vector<double>(kDim, 0.0), std::vector<double>(kDim, 0.0)};
            for (const auto& item : regrets) {
                const std::size_t i = item.second;
                const std::size_t preferred = distances[i].first <= distances[i].second ? 0 : 1;
                const std::size_t chosen = sizes[preferred] < cap[preferred] ? preferred : 1 - preferred;
                assignment[i] = chosen; ++sizes[chosen];
                const auto* x = base_ + static_cast<std::size_t>(ids[i]) * kDim;
                for (std::size_t d = 0; d < kDim; ++d) sums[chosen][d] += x[d];
            }
            for (std::size_t c = 0; c < 2; ++c) {
                for (std::size_t d = 0; d < kDim; ++d)
                    result.centers[c][d] = static_cast<float>(sums[c][d] / static_cast<double>(sizes[c]));
            }
        }
        for (std::size_t i = 0; i < ids.size(); ++i) result.ids[assignment[i]].push_back(ids[i]);
        if (result.ids[0].empty() || result.ids[1].empty()) throw std::runtime_error("balanced split made an empty child");
        return result;
    }

    void split_and_reassign(std::size_t leaf, Maintenance& stats) {
        const std::vector<std::uint32_t> old_ids = parts_[leaf].ids;
        const std::vector<float> old_center = parts_[leaf].center;
        std::vector<std::pair<float, std::size_t>> rank;
        rank.reserve(parts_.size());
        for (std::size_t i = 0; i < parts_.size(); ++i) {
            if (i != leaf && !parts_[i].ids.empty())
                rank.emplace_back(simd_squared_l2(old_center.data(), parts_[i].center.data(), kDim), i);
        }
        const std::size_t neighbor_count = std::min(kReassignRange, rank.size());
        std::partial_sort(rank.begin(), rank.begin() + static_cast<std::ptrdiff_t>(neighbor_count), rank.end(),
            [](const auto& a, const auto& b) { return a.first == b.first ? a.second < b.second : a.first < b.first; });
        std::vector<std::size_t> neighbors; neighbors.reserve(neighbor_count);
        for (std::size_t i = 0; i < neighbor_count; ++i) neighbors.push_back(rank[i].second);

        const std::uint32_t split_seed = kSeed + static_cast<std::uint32_t>(active_) * 17u +
            static_cast<std::uint32_t>(leaf * 31u + stats.splits);
        SplitResult split = balanced_split(old_ids, split_seed);
        parts_[leaf].ids = std::move(split.ids[0]);
        parts_[leaf].center = std::move(split.centers[0]);
        rebuild_payload(parts_[leaf]);
        const std::size_t child = parts_.size();
        Partition new_child;
        new_child.ids = std::move(split.ids[1]);
        new_child.center = std::move(split.centers[1]);
        rebuild_payload(new_child);
        parts_.push_back(std::move(new_child));
        std::unordered_set<std::uint32_t> left_ids(parts_[leaf].ids.begin(), parts_[leaf].ids.end());

        std::vector<std::pair<std::size_t, std::uint32_t>> candidates;
        for (const auto id : old_ids) {
            const float d0 = point_center_distance(id, old_center.data());
            const float d1 = point_center_distance(id, parts_[leaf].center.data());
            const float d2 = point_center_distance(id, parts_[child].center.data());
            if (d0 <= d1 && d0 <= d2) candidates.emplace_back(left_ids.count(id) ? leaf : child, id);
        }
        for (const auto neighbor : neighbors) {
            const auto ids = parts_[neighbor].ids;
            for (const auto id : ids) {
                const float old_d = point_center_distance(id, old_center.data());
                const float d1 = point_center_distance(id, parts_[leaf].center.data());
                const float d2 = point_center_distance(id, parts_[child].center.data());
                if (std::min(d1, d2) <= old_d) candidates.emplace_back(neighbor, id);
            }
        }
        stats.candidates += candidates.size();
        std::vector<Move> moves; moves.reserve(candidates.size());
        for (const auto& candidate : candidates) {
            const std::size_t from = candidate.first;
            const std::uint32_t id = candidate.second;
            float best_distance = point_center_distance(id, parts_[from].center.data());
            std::size_t best = from;
            for (std::size_t c = 0; c < parts_.size(); ++c) {
                const float d = point_center_distance(id, parts_[c].center.data());
                if (d < best_distance) { best_distance = d; best = c; }
            }
            if (best != from) moves.push_back({from, best, id});
        }
        std::unordered_map<std::size_t, std::unordered_set<std::uint32_t>> moved_by_source;
        for (const auto& m : moves) moved_by_source[m.from].insert(m.id);
        for (const auto& entry : moved_by_source) {
            auto& ids = parts_[entry.first].ids;
            ids.erase(std::remove_if(ids.begin(), ids.end(), [&](std::uint32_t id) { return entry.second.count(id) != 0; }), ids.end());
            rebuild_payload(parts_[entry.first]);
        }
        for (const auto& m : moves) {
            auto& destination = parts_[m.to];
            destination.ids.push_back(m.id);
            const auto* vector = base_ + static_cast<std::size_t>(m.id) * kDim;
            destination.vectors.insert(destination.vectors.end(), vector, vector + kDim);
        }
        stats.moved += moves.size();
        ++stats.splits;
    }

    const BaseVector* base_;
    std::vector<Partition> parts_;
    std::size_t active_ = 0;
};

double p99(std::vector<double> x) {
    std::sort(x.begin(), x.end());
    return x.empty() ? 0.0 : x[std::min(x.size() - 1, static_cast<std::size_t>(std::ceil(0.99 * x.size()) - 1))];
}
std::string truth_name(std::size_t stage) {
    std::ostringstream s; s << "stage_" << std::setw(2) << std::setfill('0') << stage << "_top10_l2.ivecs"; return s.str();
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 6) {
            std::cerr << "Usage: sift1m_random_delete_spfresh_lire_runner BASE.u8bin QUERIES.fvecs PREPARED_DIR SHARED_INDEX.ivfidx OUTPUT_DIR\n";
            return 2;
        }
        const std::filesystem::path base_path(argv[1]), query_path(argv[2]), prepared(argv[3]), snapshot(argv[4]), output(argv[5]);
        if (!std::filesystem::exists(snapshot)) throw std::runtime_error("shared initial snapshot does not exist");
        if (std::filesystem::exists(output) && std::filesystem::directory_iterator(output) != std::filesystem::directory_iterator())
            throw std::runtime_error("output must be absent or empty: " + output.string());
        std::filesystem::create_directories(output);
        auto base = read_base(base_path);
        const auto queries = read_queries(query_path);
        const auto deletions = read_deletions(prepared / "deletion_batches.csv");

        std::vector<float> centers;
        std::vector<std::vector<std::uint32_t>> leaf_ids;
        {
            auto initial = AdaptiveHierarchicalIvf::load(snapshot.string());
            if (initial.size() != kRows || initial.leaf_nlist() != kInitialNlist)
                throw std::runtime_error("shared initial snapshot has unexpected dimensions or cluster count");
            centers = initial.leaf_centroids();
            leaf_ids = initial.leaf_vector_ids();
        }
        SpfreshLire index(base);
        index.initialize(centers, leaf_ids);
        if (index.size() != kRows || index.nlist() != kInitialNlist) throw std::runtime_error("LIRE init from shared index failed");
        centers.clear(); centers.shrink_to_fit(); leaf_ids.clear(); leaf_ids.shrink_to_fit();

        const std::size_t initial_max_before = index.max_size();
        std::cout << "LIRE_CAP_BEGIN initial_nlist=" << index.nlist() << " max_posting="
                  << initial_max_before << " cap=" << kMaxPosting << '\n' << std::flush;
        const auto initial_maintenance_start = Clock::now();
        const Maintenance initial_maintenance = index.maintain();
        const double initial_maintenance_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - initial_maintenance_start).count();
        const std::size_t initial_max_after = index.max_size();
        if (index.max_size() > kMaxPosting)
            throw std::runtime_error("LIRE initialization left an over-capacity posting");
        index.validate(kRows);
        std::cout << "SPFresh-LIRE initial_capacity_pass max=" << initial_max_before << "->" << initial_max_after
                  << " nlist_after=" << index.nlist() << " splits=" << initial_maintenance.splits
                  << " NPA_candidates=" << initial_maintenance.candidates << " moved=" << initial_maintenance.moved
                  << " maintenance_ms=" << std::fixed << std::setprecision(1) << initial_maintenance_ms << '\n' << std::flush;

        std::ofstream summary(output / "stage_metrics.csv", std::ios::trunc);
        std::ofstream query_rows(output / "query_metrics.csv", std::ios::trunc);
        std::ofstream initial_rows(output / "initial_maintenance.csv", std::ios::trunc);
        std::ofstream manifest(output / "manifest.txt", std::ios::trunc);
        if (!summary || !query_rows || !initial_rows || !manifest) throw std::runtime_error("cannot create LIRE outputs");
        summary << "method,stage,active_vectors,nlist,max_posting_size,initial_maintenance_ms,initial_splits,initial_npa_candidates,initial_reassigned_vectors,delete_ms,maintenance_ms,total_update_ms,splits,npa_candidates,reassigned_vectors,recall_at_10,query_mean_ms,query_p99_ms,mean_center_distances,mean_vector_distances,mean_total_distances\n";
        query_rows << "stage,query_id,active_vectors,total_distance_budget,center_distances,vector_distances,total_distances,top10_hits,query_ms\n";
        initial_rows << "nlist_before,max_posting_before,nlist_after,max_posting_after,maintenance_ms,splits,npa_candidates,reassigned_vectors\n"
                     << kInitialNlist << ',' << initial_max_before << ',' << index.nlist() << ',' << initial_max_after << ','
                     << initial_maintenance_ms << ',' << initial_maintenance.splits << ','
                     << initial_maintenance.candidates << ',' << initial_maintenance.moved << '\n';
        manifest << "method=SPFresh-LIRE deletion-only port\n"
                 << "initial_index_snapshot=" << std::filesystem::absolute(snapshot).string() << "\n"
                 << "initialization=reuse the shared index's " << kInitialNlist << " centroids and vector-to-leaf ownership; before stage 0, run LIRE capacity maintenance until every posting has <=1000 vectors\n"
                 << "initial_capacity_pass=max posting before=" << initial_max_before << "; after=" << initial_max_after
                 << "; nlist after=" << index.nlist() << "; elapsed_ms=" << initial_maintenance_ms
                 << "; splits=" << initial_maintenance.splits << "; NPA candidates=" << initial_maintenance.candidates
                 << "; reassigned vectors=" << initial_maintenance.moved << "\n"
                 << "initial_capacity_pass_cost_is_reported_separately_from_deletion_batches=true\n"
#ifdef GLOVE100
                 << "data=GloVe100-angular; 1,183,514 x 100 normalized FP32; metric=squared L2 (angular-equivalent ranking)\n"
#else
                 << "data=1M x 128 uint8 SIFT; exact vector storage; metric=squared L2\n"
#endif
                 << "deletions=40 batches x 20,000 unique IDs; same fixed-seed schedule and per-stage exact truths as Quake\n"
                 << "policy=split when posting size >1000; balanced 2-way split with 5 Lloyd-style refinement passes; candidate neighborhood=64 nearest leaves around old center; NPA candidate check then nearest-center reassignment; repeat until all postings <=1000; empty postings removed; no underfull merge\n"
                 << "query=1000 official queries per stage; recall@10; total distance budget=floor(active_vectors/20), including centroid distances; early stopping off; 32 warmups\n"
                 << "query timing=single-threaded steady_clock around query search\n";

        for (std::size_t stage = 0; stage <= kBatchCount; ++stage) {
            const std::size_t active = kRows - stage * kBatchRows;
            double delete_ms = 0.0, maintenance_ms = 0.0;
            Maintenance maintenance;
            if (stage > 0) {
                const auto ds = Clock::now();
                if (index.remove_ids(deletions[stage - 1]) != kBatchRows) throw std::runtime_error("LIRE deletion count mismatch");
                delete_ms = std::chrono::duration<double, std::milli>(Clock::now() - ds).count();
                const auto ms = Clock::now();
                maintenance = index.maintain();
                maintenance_ms = std::chrono::duration<double, std::milli>(Clock::now() - ms).count();
            }
            index.validate(active);
            QuerySearchOptions opts; opts.enable_early_stop = false; opts.max_total_distance_computations = active / 20;
            if (opts.max_total_distance_computations <= index.nlist()) throw std::runtime_error("query budget below centroid count");
            for (std::size_t q = 0; q < 32; ++q) (void)index.search_budget(queries.data() + q * kDim, opts.max_total_distance_computations);
            std::vector<double> latencies; latencies.reserve(kQueries);
            std::size_t correct = 0, center_sum = 0, vector_sum = 0;
            const auto truth = read_truth(prepared / "exact_truth" / truth_name(stage));
            for (std::size_t q = 0; q < kQueries; ++q) {
                const auto start = Clock::now();
                const auto found = index.search_budget(queries.data() + q * kDim, opts.max_total_distance_computations);
                const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                latencies.push_back(ms);
                std::size_t hits = 0;
                for (const auto id : found.ids)
                    for (const auto expected : truth[q]) if (static_cast<std::int32_t>(id) == expected) { ++hits; break; }
                correct += hits; center_sum += found.centers; vector_sum += found.vectors;
                query_rows << stage << ',' << q << ',' << active << ',' << opts.max_total_distance_computations << ','
                           << found.centers << ',' << found.vectors << ',' << found.centers + found.vectors << ','
                           << hits << ',' << ms << '\n';
            }
            query_rows.flush();
            const double recall = static_cast<double>(correct) / static_cast<double>(kQueries * kTopK);
            const double mean_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / kQueries;
            summary << "SPFresh-LIRE," << stage << ',' << active << ',' << index.nlist() << ',' << index.max_size() << ','
                    << (stage == 0 ? initial_maintenance_ms : 0.0) << ','
                    << (stage == 0 ? initial_maintenance.splits : 0) << ','
                    << (stage == 0 ? initial_maintenance.candidates : 0) << ','
                    << (stage == 0 ? initial_maintenance.moved : 0) << ','
                    << delete_ms << ',' << maintenance_ms << ',' << delete_ms + maintenance_ms << ',' << maintenance.splits << ','
                    << maintenance.candidates << ',' << maintenance.moved << ',' << recall << ',' << mean_ms << ',' << p99(latencies) << ','
                    << static_cast<double>(center_sum) / kQueries << ',' << static_cast<double>(vector_sum) / kQueries << ','
                    << static_cast<double>(center_sum + vector_sum) / kQueries << '\n';
            summary.flush();
            std::cout << "SPFresh-LIRE stage=" << std::setw(2) << std::setfill('0') << stage
                      << " active=" << active << " nlist=" << index.nlist() << " splits=" << maintenance.splits
                      << " NPA_candidates=" << maintenance.candidates << " moved=" << maintenance.moved
                      << " maintenance_ms=" << std::fixed << std::setprecision(1) << maintenance_ms
                      << " recall=" << std::setprecision(4) << recall << " query_ms=" << std::setprecision(3) << mean_ms << '\n' << std::flush;
        }
        std::cout << "COMPLETE output=" << std::filesystem::absolute(output).string() << '\n' << std::flush;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
