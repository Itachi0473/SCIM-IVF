#include "adaptive_hierarchical_ivf.h"
#include "simd_l2.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
#ifdef GLOVE100
constexpr std::size_t kRows = 1'183'514;
constexpr std::size_t kDim = 100;
constexpr std::size_t kInitialNlist = 1'100;
using BaseVector = float;
#else
constexpr std::size_t kRows = 1'000'000;
constexpr std::size_t kDim = 128;
constexpr std::size_t kInitialNlist = 1'000;
using BaseVector = std::uint8_t;
#endif
constexpr std::size_t kBatchCount = 40;
constexpr std::size_t kBatchRows = 20'000;
constexpr std::size_t kQueries = 1'000;
constexpr std::size_t kTopK = 10;
constexpr std::size_t kWindow = 1'000;
constexpr std::uint32_t kSeed = 1234;
constexpr double kAlpha = 0.9;
constexpr double kQuakeThresholdNs = 50.0; // Quake's SIFT1M maintenance-ablation configuration.
constexpr std::size_t kQuakeMinPartitionSize = 32;
constexpr std::size_t kQuakeRefinementRadius = 25;
constexpr std::size_t kQuakeRefinementIterations = 3;
constexpr std::size_t kLireMaxPartitionSize = 1'000; // User-requested LIRE posting cap.
constexpr std::size_t kLireRefinementRadius = 25;
using Clock = std::chrono::steady_clock;
using TruthRow = std::array<std::int32_t, kTopK>;

template <typename T>
T read_scalar(std::istream& in, const char* label) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error(std::string("truncated ") + label);
    return value;
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
    const auto rows = read_scalar<std::uint32_t>(in, "base row count");
    const auto dim = read_scalar<std::uint32_t>(in, "base dimension");
    if (rows != kRows || dim != kDim || std::filesystem::file_size(path) != 8ULL + kRows * kDim)
        throw std::runtime_error("base must be exactly 1M x 128 uint8 SIFT");
    std::vector<BaseVector> data(kRows * kDim);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
    if (!in) throw std::runtime_error("truncated SIFT base payload");
    return data;
#endif
}

float scan_l2(const BaseVector* a, const float* b) {
#ifdef GLOVE100
    return simd_squared_l2(a, b, kDim);
#else
    return simd_squared_l2_u8_f32(a, b, kDim);
#endif
}

std::vector<float> read_queries(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open queries: " + path.string());
    std::vector<float> queries(kQueries * kDim);
    for (std::size_t q = 0; q < kQueries; ++q) {
        if (read_scalar<std::int32_t>(in, "query dimension") != static_cast<std::int32_t>(kDim))
            throw std::runtime_error("expected 128D fvecs query");
        in.read(reinterpret_cast<char*>(queries.data() + q * kDim),
                static_cast<std::streamsize>(kDim * sizeof(float)));
        if (!in) throw std::runtime_error("truncated query vector");
    }
    return queries;
}

std::vector<TruthRow> read_truth(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open stage truth: " + path.string());
    std::vector<TruthRow> truth(kQueries);
    for (auto& row : truth) {
        if (read_scalar<std::int32_t>(in, "truth row length") != static_cast<std::int32_t>(kTopK))
            throw std::runtime_error("stage truth must contain top-10 rows");
        in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(sizeof(row)));
        if (!in) throw std::runtime_error("truncated stage truth");
    }
    if (in.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error("unexpected trailing bytes in stage truth");
    return truth;
}

std::vector<std::vector<std::int64_t>> read_deletion_batches(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open deletion sequence: " + path.string());
    std::string line;
    if (!std::getline(in, line) || line != "batch,position,original_vector_id")
        throw std::runtime_error("unexpected deletion_batches.csv header");
    std::vector<std::vector<std::int64_t>> batches(kBatchCount);
    for (auto& batch : batches) batch.reserve(kBatchRows);
    std::vector<std::uint8_t> seen(kRows, 0);
    std::size_t row_count = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream row(line);
        std::size_t batch = 0, position = 0;
        std::int64_t id = -1;
        char comma1 = 0, comma2 = 0;
        if (!(row >> batch >> comma1 >> position >> comma2 >> id) || comma1 != ',' || comma2 != ',' ||
            batch == 0 || batch > kBatchCount || position >= kBatchRows || id < 0 || id >= static_cast<std::int64_t>(kRows))
            throw std::runtime_error("malformed deletion batch row: " + line);
        auto& target = batches[batch - 1];
        if (position != target.size() || seen[static_cast<std::size_t>(id)])
            throw std::runtime_error("deletion IDs must be unique and ordered");
        target.push_back(id);
        seen[static_cast<std::size_t>(id)] = 1;
        ++row_count;
    }
    if (row_count != kBatchCount * kBatchRows)
        throw std::runtime_error("deletion schedule must contain exactly 800,000 IDs");
    return batches;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t rank = std::max<std::size_t>(1,
        static_cast<std::size_t>(std::ceil(p * static_cast<double>(values.size()))));
    return values[std::min(values.size() - 1, rank - 1)];
}

class ScanLatencyProfile {
public:
    ScanLatencyProfile(const std::vector<BaseVector>& base, const std::vector<float>& queries) {
        constexpr std::array<std::size_t, 10> grid{1, 2, 4, 16, 64, 256, 1024, 4096, 16384, 65536};
        for (const auto n : grid) {
            double total_ns = 0.0;
            for (std::size_t trial = 0; trial < 5; ++trial) {
                const float* q = queries.data() + (trial * 193 % kQueries) * kDim;
                std::priority_queue<std::pair<float, std::uint32_t>> top;
                const auto start = Clock::now();
                for (std::size_t i = 0; i < n; ++i) {
                    const float d = scan_l2(base.data() + i * kDim, q);
                    if (top.size() < kTopK) top.emplace(d, static_cast<std::uint32_t>(i));
                    else if (d < top.top().first) { top.pop(); top.emplace(d, static_cast<std::uint32_t>(i)); }
                }
                total_ns += std::chrono::duration<double, std::nano>(Clock::now() - start).count();
                if (!top.empty()) sink_ += top.top().first;
            }
            n_.push_back(n);
            ns_.push_back(total_ns / 5.0);
        }
    }

    double operator()(std::size_t n) const {
        if (n == 0) return 0.0;
        if (n <= n_.front()) return ns_.front() * static_cast<double>(n) / n_.front();
        auto upper = std::lower_bound(n_.begin(), n_.end(), n);
        std::size_t lo = 0, hi = 1;
        if (upper == n_.end()) { hi = n_.size() - 1; lo = hi - 1; }
        else {
            hi = static_cast<std::size_t>(upper - n_.begin());
            if (*upper == n) return ns_[hi];
            lo = hi - 1;
        }
        const double f = static_cast<double>(n - n_[lo]) / static_cast<double>(n_[hi] - n_[lo]);
        return ns_[lo] + f * (ns_[hi] - ns_[lo]);
    }

    void write_csv(const std::filesystem::path& path) const {
        std::ofstream out(path, std::ios::trunc);
        if (!out) throw std::runtime_error("cannot write Quake latency profile");
        out << "list_vectors,scan_latency_ns\n" << std::fixed << std::setprecision(3);
        for (std::size_t i = 0; i < n_.size(); ++i) out << n_[i] << ',' << ns_[i] << '\n';
    }
private:
    std::vector<std::size_t> n_;
    std::vector<double> ns_;
    mutable volatile float sink_ = 0.0f;
};

struct WindowQuery { std::vector<std::size_t> leaves; std::size_t scanned_size_sum = 0; };
enum class QuakeAction { None, Split, Delete, DeleteRejected };
struct QuakeCandidate {
    std::size_t leaf = 0, size = 0, hits = 0;
    double rate = 0.0, split_delta = 0.0, delete_delta = 0.0, reassign_delete_delta = 0.0;
    QuakeAction action = QuakeAction::None;
};

double reassign_delete_delta(const AdaptiveHierarchicalIvf& index,
        const std::vector<BaseVector>& base, const std::vector<std::size_t>& sizes,
        const std::vector<std::size_t>& hits, std::size_t leaf, double hit_rate,
        const ScanLatencyProfile& latency) {
    const std::size_t total = sizes.size(), partition_size = sizes[leaf];
    if (total <= 1 || partition_size == 0) return 0.0;
    const auto ids = index.leaf_vector_ids(leaf);
    const auto centers = index.leaf_centroids();
    std::vector<std::size_t> reassign_counts(total, 0);
    for (const auto id : ids) {
        const auto* x = base.data() + static_cast<std::size_t>(id) * kDim;
        std::size_t best = total;
        float best_distance = std::numeric_limits<float>::max();
        for (std::size_t candidate = 0; candidate < total; ++candidate) {
            if (candidate == leaf) continue;
            const float d = scan_l2(x, centers.data() + candidate * kDim);
            if (d < best_distance) { best_distance = d; best = candidate; }
        }
        if (best != total) ++reassign_counts[best];
    }
    const double overhead = latency(total - 1) - latency(total);
    const double removal = hit_rate * latency(partition_size);
    double reassignment = 0.0;
    for (std::size_t dest = 0; dest < total; ++dest) {
        if (reassign_counts[dest] == 0) continue;
        const double dest_rate = static_cast<double>(hits[dest]) / static_cast<double>(kWindow);
        const double old_cost = dest_rate * latency(sizes[dest]);
        const double new_cost = (dest_rate + hit_rate) * latency(sizes[dest] + partition_size);
        reassignment += new_cost - old_cost;
    }
    return overhead + removal + reassignment;
}

std::vector<QuakeCandidate> choose_quake_actions(const AdaptiveHierarchicalIvf& index,
        const std::vector<BaseVector>& base, const std::deque<WindowQuery>& window,
        const ScanLatencyProfile& latency) {
    const auto sizes = index.leaf_sizes();
    std::vector<std::size_t> hits(sizes.size(), 0);
    std::size_t scanned_sizes = 0;
    for (const auto& q : window) {
        scanned_sizes += q.scanned_size_sum;
        for (const auto leaf : q.leaves) if (leaf < hits.size()) ++hits[leaf];
    }
    std::vector<QuakeCandidate> result;
    if (window.size() < kWindow) return result;
    const double scan_fraction = static_cast<double>(scanned_sizes) /
        (static_cast<double>(window.size()) * static_cast<double>(kRows));
    const std::size_t count = sizes.size();
    const std::size_t avg_size = index.size() / count;
    const double base_cost = latency(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (sizes[i] == 0) continue;
        QuakeCandidate c;
        c.leaf = i; c.size = sizes[i]; c.hits = hits[i];
        c.rate = static_cast<double>(hits[i]) / static_cast<double>(kWindow);
        const double t_minus = count > 1 ? latency(count - 1) : base_cost;
        const double merged_size = static_cast<double>(avg_size) +
            static_cast<double>(sizes[i]) / static_cast<double>(std::max<std::size_t>(1, count - 1));
        const double merged_rate = scan_fraction +
            c.rate / static_cast<double>(std::max<std::size_t>(1, count - 1));
        const double old_cost = static_cast<double>(count - 1) * scan_fraction * latency(avg_size) +
            c.rate * latency(sizes[i]);
        double new_cost = 0.0;
        if (sizes[i] < count) {
            new_cost = static_cast<double>(sizes[i]) * merged_rate * latency(avg_size + 1) +
                static_cast<double>(count - sizes[i] - 1) * merged_rate * latency(avg_size);
        } else {
            new_cost = static_cast<double>(count - 1) * merged_rate *
                latency(static_cast<std::size_t>(std::ceil(merged_size)));
        }
        c.delete_delta = (t_minus - base_cost) + new_cost - old_cost;
        if (c.delete_delta < -kQuakeThresholdNs) {
            c.reassign_delete_delta = reassign_delete_delta(index, base, sizes, hits, i, c.rate, latency);
            c.action = c.reassign_delete_delta < -kQuakeThresholdNs ? QuakeAction::Delete : QuakeAction::DeleteRejected;
        } else if (sizes[i] > kQuakeMinPartitionSize) {
            c.split_delta = latency(count + 1) - base_cost +
                2.0 * kAlpha * c.rate * latency(sizes[i] / 2) - c.rate * latency(sizes[i]);
            if (c.split_delta < -kQuakeThresholdNs) c.action = QuakeAction::Split;
        }
        result.push_back(c);
    }
    return result;
}

struct QuerySummary {
    double recall = 0, mean_ms = 0, p99_ms = 0;
    double mean_centers = 0, mean_vectors = 0, mean_total = 0;
};

QuerySummary query_stage(const AdaptiveHierarchicalIvf& index,
        const std::vector<float>& queries, const std::vector<TruthRow>& truth,
        std::size_t active, std::size_t stage, std::ofstream& query_csv,
        std::deque<WindowQuery>* hit_window) {
    QuerySearchOptions options;
    options.enable_early_stop = false;
    options.max_total_distance_computations = active / 20;
    if (options.max_total_distance_computations < index.leaf_nlist())
        throw std::runtime_error("N/20 budget is smaller than live centroid count");
    for (std::size_t q = 0; q < 32; ++q)
        (void)index.search_with_options(queries.data() + q * kDim, kTopK, options);
    QuerySummary out;
    std::vector<double> latencies;
    latencies.reserve(kQueries);
    std::size_t hits = 0, center_sum = 0, vector_sum = 0, total_sum = 0;
    const auto sizes = index.leaf_sizes();
    for (std::size_t q = 0; q < kQueries; ++q) {
        QuerySearchStats stats;
        std::vector<std::size_t> probed;
        const auto start = Clock::now();
        const auto found = index.search_with_options(queries.data() + q * kDim, kTopK,
            options, &stats, hit_window ? &probed : nullptr);
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        latencies.push_back(ms);
        std::size_t qhits = 0;
        for (const auto& item : found)
            for (const auto expected : truth[q]) if (item.id == expected) { ++qhits; break; }
        hits += qhits;
        center_sum += stats.center_distance_computations;
        vector_sum += stats.scanned_vectors;
        total_sum += stats.total_distance_computations;
        query_csv << stage << ',' << q << ',' << active << ',' << options.max_total_distance_computations
                  << ',' << stats.center_distance_computations << ',' << stats.scanned_vectors << ','
                  << stats.total_distance_computations << ',' << qhits << ',' << ms << '\n';
        if (hit_window) {
            WindowQuery row;
            row.leaves = std::move(probed);
            for (const auto leaf : row.leaves) if (leaf < sizes.size()) row.scanned_size_sum += sizes[leaf];
            hit_window->push_back(std::move(row));
            if (hit_window->size() > kWindow) hit_window->pop_front();
        }
    }
    query_csv.flush();
    out.recall = static_cast<double>(hits) / static_cast<double>(kQueries * kTopK);
    out.mean_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / static_cast<double>(kQueries);
    out.p99_ms = percentile(latencies, 0.99);
    out.mean_centers = static_cast<double>(center_sum) / kQueries;
    out.mean_vectors = static_cast<double>(vector_sum) / kQueries;
    out.mean_total = static_cast<double>(total_sum) / kQueries;
    return out;
}

std::pair<std::size_t, std::size_t> size_range(const AdaptiveHierarchicalIvf& index) {
    const auto sizes = index.leaf_sizes();
    if (sizes.empty()) return {0, 0};
    const auto mm = std::minmax_element(sizes.begin(), sizes.end());
    return {*mm.first, *mm.second};
}

std::string truth_name(std::size_t stage) {
    std::ostringstream name;
    name << "stage_" << std::setw(2) << std::setfill('0') << stage << "_top10_l2.ivecs";
    return name.str();
}

void run_method(const std::string& name, const std::filesystem::path& snapshot,
        const std::vector<BaseVector>& base, const std::vector<float>& queries,
        const std::filesystem::path& prepared,
        const std::vector<std::vector<std::int64_t>>& deletion_batches,
        const std::filesystem::path& output,
        const ScanLatencyProfile& latency, bool is_quake) {
    const auto method_dir = output / name;
    std::filesystem::create_directories(method_dir);
    auto index = AdaptiveHierarchicalIvf::load(snapshot.string());
    if (index.size() != kRows || index.leaf_nlist() != kInitialNlist)
        throw std::runtime_error(name + " loaded a mismatched initial index snapshot");

    std::ofstream summary(method_dir / "stage_metrics.csv", std::ios::trunc);
    std::ofstream query_csv(method_dir / "query_metrics.csv", std::ios::trunc);
    std::ofstream decisions(method_dir / "maintenance_decisions.csv", std::ios::trunc);
    std::ofstream manifest(method_dir / "manifest.txt", std::ios::trunc);
    if (!summary || !query_csv || !decisions || !manifest)
        throw std::runtime_error("cannot create " + name + " output files");
    summary << "method,stage,active_vectors,indexed_vectors,deleted_this_batch,nlist_before,nlist_after,min_list_size,max_list_size,delete_ms,maintenance_ms,total_update_ms,split_candidates,splits,deleted_partitions,reassigned_vectors,empty_removed,recall_at_10,query_mean_ms,query_p99_ms,mean_center_distances,mean_vector_distances,mean_total_distances,total_sse\n";
    query_csv << "stage,query_id,active_vectors,total_distance_budget,center_distances,vector_distances,total_distances,top10_hits,query_ms\n";
    decisions << "stage,leaf,size,window_hits,hit_rate,split_delta_ns,delete_delta_ns,reassign_delete_delta_ns,decision\n";
    manifest << "method=" << (is_quake ? "Quake cost-aware maintenance port" : "Quake artifact LIRE size-threshold branch") << '\n'
             << "initial_index_snapshot=" << std::filesystem::absolute(snapshot).string() << "\n"
#ifdef GLOVE100
             << "initial_index=shared GloVe100-angular index; 1,183,514 vectors; nlist=1100; normalized FP32; KMeans 10 iterations; seed=1234\n"
#else
             << "initial_index=shared 1M SIFT1M index; nlist=1000; uint8 storage; KMeans 10 iterations; seed=1234\n"
#endif
             << "deletions=40 batches x 20,000 unique IDs; fixed seed=1234; same schedule and exact stage truths for both methods\n"
             << "query=1,000 official SIFT queries per stage; Recall@10; exact stage truth; budget=floor(active_vectors/20), total centroid+vector distance calculations; early stop off\n"
             << "query timing=32 warmups, then 1,000 measured queries; update timing excludes query evaluation\n"
             << "data-preservation=maintenance preserves all still-active vectors; partition deletions reassign their live vectors to surviving centroids\n";
    if (is_quake) {
        manifest << "policy=cost model; alpha=0.9; rolling query-hit window=1000; split/delete threshold=50ns; min partition size=32; delete rejection enabled\n"
#ifdef GLOVE100
                 << "cost profile=measured on this host using normalized FP32 GloVe100 L2 scans and top-10 heap; k=10; 5 trials per source grid point\n"
#else
                 << "cost profile=measured on this host using the uint8 SIFT L2 scan kernel and top-10 heap; k=10; 5 trials per source grid point\n"
#endif
                 << "split=two-way project IVF split; local refinement radius=25 including seed; iterations=3; centroid refresh before maintenance\n"
                 << "hit-rate scan-fraction denominator=initial 1M vector count, matching the Quake tracker\n";
    } else {
        manifest << "policy=Quake artifact LIRE fixed-size branch; split if list size > 2000; short-list deletion disabled to preserve the deletion workload's active population; empty lists removed\n"
                 << "split refinement radius=25 including seed; refinement_iterations=0 (one assignment pass in this project port); one maintenance pass after each deletion batch\n";
    }

    std::deque<WindowQuery> window;
    for (std::size_t stage = 0; stage <= kBatchCount; ++stage) {
        const std::size_t active = kRows - stage * kBatchRows;
        const auto truth = read_truth(prepared / "exact_truth" / truth_name(stage));
        const std::size_t nlist_before = index.leaf_nlist();
        double delete_ms = 0.0, maintenance_ms = 0.0;
        std::size_t split_candidates = 0, splits = 0, deleted_partitions = 0;
        std::size_t reassigned_vectors = 0, empty_removed = 0;
        if (stage > 0) {
            index.begin_update_batch();
            const auto delete_start = Clock::now();
            const auto removed = index.remove_ids(deletion_batches[stage - 1]);
            delete_ms = std::chrono::duration<double, std::milli>(Clock::now() - delete_start).count();
            if (removed != kBatchRows) throw std::runtime_error("deletion batch did not remove 20,000 vectors");
            (void)index.end_update_batch();
            const auto maintenance_start = Clock::now();
            if (is_quake) {
                index.quake_refresh_centroids();
                const auto candidates = choose_quake_actions(index, base, window, latency);
                std::vector<std::size_t> delete_ids, split_ids;
                for (const auto& c : candidates) {
                    const char* action = "none";
                    if (c.action == QuakeAction::Delete) { delete_ids.push_back(c.leaf); action = "delete_reassigned"; }
                    else if (c.action == QuakeAction::Split) { split_ids.push_back(c.leaf); action = "split"; }
                    else if (c.action == QuakeAction::DeleteRejected) action = "delete_rejected";
                    decisions << stage << ',' << c.leaf << ',' << c.size << ',' << c.hits << ','
                              << std::setprecision(10) << c.rate << ',' << c.split_delta << ','
                              << c.delete_delta << ',' << c.reassign_delete_delta << ',' << action << '\n';
                }
                decisions.flush();
                split_candidates = candidates.size();
                if (!delete_ids.empty()) {
                    const auto sizes = index.leaf_sizes();
                    for (const auto id : delete_ids) reassigned_vectors += sizes[id];
                    const auto map = index.quake_delete_partitions(delete_ids);
                    deleted_partitions = delete_ids.size();
                    std::vector<std::size_t> remapped;
                    for (const auto id : split_ids)
                        if (id < map.size() && map[id] != std::numeric_limits<std::size_t>::max()) remapped.push_back(map[id]);
                    split_ids = std::move(remapped);
                }
                if (!split_ids.empty()) {
                    const auto report = index.maintain_selected(split_ids, SelectedMaintenanceAction::Split,
                        kQuakeRefinementRadius - 1, kQuakeRefinementIterations);
                    splits = report.split_partitions;
                }
                const auto before_cleanup = index.leaf_nlist();
                (void)index.quake_delete_partitions({});
                empty_removed = before_cleanup - index.leaf_nlist();
                window.clear();
            } else {
                const auto report = index.maintain_quake_lire_policy(
                    kLireMaxPartitionSize, 0, kLireRefinementRadius, 0);
                splits = report.split_partitions;
                empty_removed = report.removed_empty_partitions;
            }
            maintenance_ms = std::chrono::duration<double, std::milli>(Clock::now() - maintenance_start).count();
            if (index.size() != active)
                throw std::runtime_error(name + " maintenance changed the active vector population");
        }
        const auto query = query_stage(index, queries, truth, active, stage, query_csv,
                                       is_quake ? &window : nullptr);
        const auto [min_size, max_size] = size_range(index);
        summary << name << ',' << stage << ',' << active << ',' << index.size() << ','
                << (stage == 0 ? 0 : kBatchRows) << ',' << nlist_before << ',' << index.leaf_nlist() << ','
                << min_size << ',' << max_size << ',' << delete_ms << ',' << maintenance_ms << ','
                << delete_ms + maintenance_ms << ',' << split_candidates << ',' << splits << ','
                << deleted_partitions << ',' << reassigned_vectors << ',' << empty_removed << ','
                << query.recall << ',' << query.mean_ms << ',' << query.p99_ms << ','
                << query.mean_centers << ',' << query.mean_vectors << ',' << query.mean_total << ','
                << index.total_sse() << '\n';
        summary.flush();
        std::cout << name << " stage=" << std::setw(2) << std::setfill('0') << stage
                  << " active=" << active << " nlist=" << index.leaf_nlist()
                  << " splits=" << splits << " deleted_partitions=" << deleted_partitions
                  << " maintenance_ms=" << std::fixed << std::setprecision(1) << maintenance_ms
                  << " recall=" << std::setprecision(4) << query.recall
                  << " query_ms=" << std::setprecision(3) << query.mean_ms << " p99=" << query.p99_ms << '\n' << std::flush;
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 5 || argc > 7) {
            std::cerr << "Usage: sift1m_random_delete_lire_quake_runner BASE.u8bin QUERIES.fvecs PREPARED_DIR OUTPUT_DIR [SHARED_INITIAL_INDEX] [both|lire|quake]\n";
            return 2;
        }
        const std::filesystem::path base_path(argv[1]), query_path(argv[2]), prepared(argv[3]), output(argv[4]);
        const std::filesystem::path supplied_snapshot = argc >= 6 ? std::filesystem::path(argv[5]) : std::filesystem::path();
        const std::string methods = argc >= 7 ? argv[6] : "both";
        if (methods != "both" && methods != "lire" && methods != "quake")
            throw std::invalid_argument("method selector must be both, lire, or quake");
        if (std::filesystem::exists(output) && std::filesystem::directory_iterator(output) != std::filesystem::directory_iterator())
            throw std::runtime_error("output directory must be absent or empty: " + output.string());
        std::filesystem::create_directories(output);
        std::cout << "Reading SIFT data and shared deletion schedule...\n" << std::flush;
        const auto base = read_base(base_path);
        const auto queries = read_queries(query_path);
        const auto deletion_batches = read_deletion_batches(prepared / "deletion_batches.csv");
        const auto snapshot = supplied_snapshot.empty() ? output / "shared_initial_index.ivfidx" : supplied_snapshot;
        double build_ms = 0.0;
        if (!std::filesystem::exists(snapshot)) {
            AdaptiveHierarchicalConfig config;
            config.dimension = kDim;
            config.initial_leaf_nlist = kInitialNlist;
            config.kmeans_iterations = 10;
            config.automatic_maintenance = false;
#ifdef GLOVE100
            config.uint8_storage = false;
#else
            config.uint8_storage = true;
#endif
            config.seed = kSeed;
            AdaptiveHierarchicalIvf initial(config);
            const auto start = Clock::now();
#ifdef GLOVE100
            initial.train(base.data(), kRows);
#else
            initial.train_u8(base.data(), kRows);
#endif
            build_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            if (initial.size() != kRows || initial.leaf_nlist() != kInitialNlist)
                throw std::runtime_error("initial shared index build validation failed");
            initial.save(snapshot.string());
        }
        std::ofstream root_manifest(output / "manifest.txt", std::ios::trunc);
        if (!root_manifest) throw std::runtime_error("cannot create root manifest");
        root_manifest << "experiment=random-delete comparison of LIRE and Quake\n"
                      << "base=" << std::filesystem::absolute(base_path).string() << '\n'
                      << "queries=" << std::filesystem::absolute(query_path).string() << " (first 1000 official queries)\n"
                      << "prepared_workload=" << std::filesystem::absolute(prepared).string() << '\n'
                      << "shared_initial_snapshot=" << std::filesystem::absolute(snapshot).string() << '\n'
                      << "initial_snapshot_reused=" << (!supplied_snapshot.empty()) << "\n"
                      << "initial_build_ms=" << std::setprecision(12) << build_ms << '\n'
                      << "initial_index=1,000,000 vectors; 1,000 clusters; KMeans 10 iterations; uint8; seed=1234\n"
                      << "methods=" << methods << "; each selected method loads this exact serialized initial index before replay\n"
                      << "LIRE max_partition_size=1000; Quake threshold=50ns from repository Quake SIFT1M maintenance-ablation config\n";
        root_manifest.flush();
        std::cout << "SHARED_INITIAL_INDEX=" << std::filesystem::absolute(snapshot).string()
                  << " bytes=" << std::filesystem::file_size(snapshot) << " build_ms=" << std::fixed
                  << std::setprecision(1) << build_ms << '\n' << std::flush;
        ScanLatencyProfile latency(base, queries);
        latency.write_csv(output / "quake_scan_latency_profile.csv");
        if (methods == "both" || methods == "lire")
            run_method("lire", snapshot, base, queries, prepared, deletion_batches, output, latency, false);
        if (methods == "both" || methods == "quake")
            run_method("quake", snapshot, base, queries, prepared, deletion_batches, output, latency, true);
        std::cout << "COMPLETE output=" << std::filesystem::absolute(output).string() << '\n' << std::flush;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
