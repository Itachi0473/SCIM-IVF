#include "../adaptive_hierarchical_ivf.h"
#include "../ivf_quantized_common.h"

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
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::size_t kDim = 128;
constexpr std::size_t kRows = 1'000'000;
constexpr std::size_t kQueries = 1'000;
constexpr std::size_t kTopK = 10;
constexpr std::size_t kNlist = 64;
constexpr std::size_t kNprobe = 4;
constexpr std::size_t kInitialRows = 50'000;
constexpr std::size_t kBatchRows = 50'000;
constexpr std::size_t kInsertBatches = 19;
constexpr std::size_t kWarmup = 32;
constexpr std::uint32_t kSeed = 1234;
constexpr float kPrdThreshold = 0.1f;
using Clock = std::chrono::steady_clock;
using TruthRow = std::array<std::int32_t, kTopK>;

template <typename T>
T read_scalar(std::istream& in, const char* label) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error(std::string("truncated ") + label);
    return value;
}

std::vector<std::uint8_t> read_base(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open SIFT base: " + path.string());
    const auto rows = read_scalar<std::uint32_t>(in, "base rows");
    const auto dim = read_scalar<std::uint32_t>(in, "base dimension");
    if (rows != kRows || dim != kDim ||
        std::filesystem::file_size(path) != 8ULL + kRows * kDim)
        throw std::runtime_error("expected headered 1M x 128 SIFT uint8 base");
    std::vector<std::uint8_t> base(kRows * kDim);
    in.read(reinterpret_cast<char*>(base.data()), static_cast<std::streamsize>(base.size()));
    if (!in) throw std::runtime_error("truncated SIFT base payload");
    return base;
}

std::vector<float> read_queries(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open SIFT queries: " + path.string());
    std::vector<float> queries(kQueries * kDim);
    for (std::size_t q = 0; q < kQueries; ++q) {
        if (read_scalar<std::int32_t>(in, "query dimension") != static_cast<std::int32_t>(kDim))
            throw std::runtime_error("SIFT query dimension is not 128");
        in.read(reinterpret_cast<char*>(queries.data() + q * kDim),
                static_cast<std::streamsize>(kDim * sizeof(float)));
        if (!in) throw std::runtime_error("truncated SIFT query vectors");
    }
    return queries;
}

std::vector<TruthRow> read_truth(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open SIFT ground truth: " + path.string());
    const auto bytes = std::filesystem::file_size(path);
    const std::size_t top10_1k_bytes = kQueries * (sizeof(std::int32_t) + kTopK * sizeof(std::int32_t));
    constexpr std::size_t kOfficialRows = 10'000;
    constexpr std::size_t kOfficialK = 100;
    const std::size_t official_bytes = kOfficialRows * (sizeof(std::int32_t) + kOfficialK * sizeof(std::int32_t));
    std::vector<TruthRow> truth(kQueries);
    if (bytes == top10_1k_bytes) {
        for (auto& row : truth) {
            if (read_scalar<std::int32_t>(in, "truth row length") != static_cast<std::int32_t>(kTopK))
                throw std::runtime_error("top-10 truth file has an unexpected row size");
            in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(sizeof(row)));
            if (!in) throw std::runtime_error("truncated top-10 truth file");
        }
    } else if (bytes == official_bytes) {
        for (auto& row : truth) {
            if (read_scalar<std::int32_t>(in, "official truth row length") != static_cast<std::int32_t>(kOfficialK))
                throw std::runtime_error("official SIFT truth is not top-100");
            std::array<std::int32_t, kOfficialK> ids{};
            in.read(reinterpret_cast<char*>(ids.data()), static_cast<std::streamsize>(sizeof(ids)));
            if (!in) throw std::runtime_error("truncated official SIFT truth");
            std::copy_n(ids.begin(), kTopK, row.begin());
        }
    } else {
        throw std::runtime_error("truth must be 1K top-10 or official 10K top-100 SIFT ivecs");
    }
    return truth;
}

struct FullIvf {
    std::vector<float> centers;
    std::vector<std::vector<std::uint32_t>> lists;
};

FullIvf build_full_ivf(const std::vector<float>& base) {
    const auto sample = ivf_quantized_detail::sample_vectors(base.data(), kRows, kDim, kSeed);
    const auto train_start = Clock::now();
    FullIvf index;
    index.centers = ivf_quantized_detail::train_kmeans(
        sample, sample.size() / kDim, kDim, kNlist, 12, kSeed);
    index.lists.resize(kNlist);
    for (std::size_t id = 0; id < kRows; ++id) {
        const std::size_t cluster = ivf_quantized_detail::nearest(
            base.data() + id * kDim, index.centers, kNlist, kDim);
        index.lists[cluster].push_back(static_cast<std::uint32_t>(id));
    }
    const double ms = std::chrono::duration<double, std::milli>(Clock::now() - train_start).count();
    std::cout << "FULL_IVF_BUILD_END nlist=" << kNlist << " training_sample="
              << sample.size() / kDim << " build_ms=" << ms << '\n' << std::flush;
    return index;
}

struct Hit { std::int64_t id = -1; float distance = 0.0f; };

std::vector<Hit> search_full_ivf(const FullIvf& index, const std::vector<float>& base,
                                const float* query, std::size_t& scanned_vectors) {
    const auto selected = ivf_quantized_detail::nearest_lists(
        query, index.centers, kNlist, kDim, kNprobe);
    auto worse = [](const Hit& a, const Hit& b) {
        return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
    };
    std::priority_queue<Hit, std::vector<Hit>, decltype(worse)> heap(worse);
    scanned_vectors = 0;
    for (const auto list_id : selected) {
        const auto& list = index.lists[list_id];
        scanned_vectors += list.size();
        for (const auto id : list) {
            const float distance = simd_squared_l2(
                query, base.data() + static_cast<std::size_t>(id) * kDim, kDim);
            const Hit candidate{static_cast<std::int64_t>(id), distance};
            if (heap.size() < kTopK) heap.push(candidate);
            else if (worse(candidate, heap.top())) { heap.pop(); heap.push(candidate); }
        }
    }
    std::vector<Hit> result;
    result.reserve(heap.size());
    while (!heap.empty()) { result.push_back(heap.top()); heap.pop(); }
    std::sort(result.begin(), result.end(), worse);
    return result;
}

AdaptiveHierarchicalIvf build_prd_only(const std::vector<std::uint8_t>& base,
                                       std::size_t& total_prd_splits,
                                       double& insert_maintenance_ms) {
    AdaptiveHierarchicalConfig config;
    config.dimension = kDim;
    config.initial_leaf_nlist = kNlist;
    config.kmeans_iterations = 10;
    config.leaf_nprobe_ratio = 8;
    config.prd_threshold = kPrdThreshold;
    config.drift_criterion = DriftCriterion::GlobalFitOnly;
    config.automatic_maintenance = false;
    config.uint8_storage = true;
    config.split_local_refinement_radius = 0;
    config.residual_tail_reassign_fraction = 0.0f;
    config.seed = kSeed;
    AdaptiveHierarchicalIvf index(config);
    index.train_u8(base.data(), kInitialRows);
    total_prd_splits = 0;
    insert_maintenance_ms = 0.0;
    for (std::size_t batch = 0; batch < kInsertBatches; ++batch) {
        const std::size_t first = kInitialRows + batch * kBatchRows;
        const auto start = Clock::now();
        index.begin_update_batch();
        index.add_u8(base.data() + first * kDim, kBatchRows);
        (void)index.end_update_batch();
        const auto report = index.maintain_with_policy(true, false, false, false);
        total_prd_splits += report.prd_split_partitions;
        insert_maintenance_ms += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        std::cout << "PRD_ONLY_UPDATE batch=" << (batch + 1) << '/' << kInsertBatches
                  << " vectors=" << index.size() << " nlist=" << index.leaf_nlist()
                  << " prd_splits=" << report.prd_split_partitions << '\n' << std::flush;
    }
    if (index.size() != kRows) throw std::runtime_error("PRD-only index does not contain 1M vectors");
    return index;
}

struct EvalResult {
    std::string method;
    std::size_t nlist = 0;
    double recall = 0.0;
    double mean_scanned = 0.0;
    double mean_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
};

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t rank = std::max<std::size_t>(1,
        static_cast<std::size_t>(std::ceil(p * values.size())));
    return values[std::min(values.size() - 1, rank - 1)];
}

template <typename SearchFn>
EvalResult evaluate(const std::string& method, std::size_t nlist,
                    const std::vector<float>& queries,
                    const std::vector<TruthRow>& truth,
                    const std::filesystem::path& per_query_path,
                    SearchFn search) {
    for (std::size_t q = 0; q < kWarmup; ++q) {
        std::size_t scanned = 0;
        (void)search(queries.data() + q * kDim, scanned);
    }
    std::ofstream per_query(per_query_path, std::ios::trunc);
    if (!per_query) throw std::runtime_error("cannot create per-query output: " + per_query_path.string());
    per_query << "method,query_id,nprobe,scanned_vectors,hits_at_10,query_ms\n";
    std::size_t total_hits = 0;
    std::uint64_t total_scanned = 0;
    std::vector<double> latencies;
    latencies.reserve(kQueries);
    for (std::size_t q = 0; q < kQueries; ++q) {
        std::size_t scanned = 0;
        const auto start = Clock::now();
        const auto found = search(queries.data() + q * kDim, scanned);
        const double query_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        std::size_t hits = 0;
        for (const auto& hit : found) {
            for (const auto expected : truth[q]) {
                if (hit.id == expected) { ++hits; break; }
            }
        }
        total_hits += hits;
        total_scanned += scanned;
        latencies.push_back(query_ms);
        per_query << method << ',' << q << ',' << kNprobe << ',' << scanned << ',' << hits << ','
                  << std::fixed << std::setprecision(8) << query_ms << '\n';
    }
    EvalResult result;
    result.method = method;
    result.nlist = nlist;
    result.recall = static_cast<double>(total_hits) / (kQueries * kTopK);
    result.mean_scanned = static_cast<double>(total_scanned) / kQueries;
    result.mean_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / kQueries;
    result.p95_ms = percentile(latencies, 0.95);
    result.p99_ms = percentile(latencies, 0.99);
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            std::cerr << "Usage: sift1m_fixed_nprobe4_full_ivf_vs_prd_runner BASE.u8bin QUERIES.fvecs TRUTH.ivecs OUTPUT_DIR\n";
            return 2;
        }
        const std::filesystem::path base_path(argv[1]);
        const std::filesystem::path query_path(argv[2]);
        const std::filesystem::path truth_path(argv[3]);
        const std::filesystem::path output_dir(argv[4]);
        if (std::filesystem::exists(output_dir) &&
            std::filesystem::directory_iterator(output_dir) != std::filesystem::directory_iterator())
            throw std::runtime_error("output directory must be empty: " + output_dir.string());
        std::filesystem::create_directories(output_dir);
        const auto base_u8 = read_base(base_path);
        const auto queries = read_queries(query_path);
        const auto truth = read_truth(truth_path);
        std::vector<float> base_float(base_u8.size());
        std::transform(base_u8.begin(), base_u8.end(), base_float.begin(),
                       [](std::uint8_t x) { return static_cast<float>(x); });

        const auto full_build_start = Clock::now();
        const FullIvf full_ivf = build_full_ivf(base_float);
        const double full_build_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - full_build_start).count();
        std::size_t prd_splits = 0;
        double prd_insert_maintenance_ms = 0.0;
        const auto prd_build_start = Clock::now();
        auto prd = build_prd_only(base_u8, prd_splits, prd_insert_maintenance_ms);
        const double prd_build_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - prd_build_start).count();

        const auto full_eval = evaluate("Full IVF-Flat", kNlist, queries, truth,
            output_dir / "full_ivf_per_query.csv",
            [&full_ivf, &base_float](const float* query, std::size_t& scanned) {
                return search_full_ivf(full_ivf, base_float, query, scanned);
            });
        const auto prd_eval = evaluate("PRD-only incremental", prd.leaf_nlist(), queries, truth,
            output_dir / "prd_only_per_query.csv",
            [&prd](const float* query, std::size_t& scanned) {
                QuerySearchStats stats;
                const auto found = prd.search_with_options(
                    query, kTopK, kNprobe, QuerySearchOptions{}, &stats, nullptr);
                scanned = stats.scanned_vectors;
                std::vector<Hit> converted;
                converted.reserve(found.size());
                for (const auto& item : found) converted.push_back({item.id, item.squared_l2});
                return converted;
            });

        std::ofstream summary(output_dir / "summary.csv", std::ios::trunc);
        if (!summary) throw std::runtime_error("cannot create summary CSV");
        summary << "method,nlist,nprobe,query_count,topk,recall_at_10,mean_scanned_vectors,mean_query_ms,p95_query_ms,p99_query_ms,build_or_insert_ms,prd_splits\n";
        summary << std::fixed << std::setprecision(8);
        summary << full_eval.method << ',' << full_eval.nlist << ',' << kNprobe << ',' << kQueries << ','
                << kTopK << ',' << full_eval.recall << ',' << full_eval.mean_scanned << ','
                << full_eval.mean_ms << ',' << full_eval.p95_ms << ',' << full_eval.p99_ms << ','
                << full_build_ms << ",0\n";
        summary << prd_eval.method << ',' << prd_eval.nlist << ',' << kNprobe << ',' << kQueries << ','
                << kTopK << ',' << prd_eval.recall << ',' << prd_eval.mean_scanned << ','
                << prd_eval.mean_ms << ',' << prd_eval.p95_ms << ',' << prd_eval.p99_ms << ','
                << prd_insert_maintenance_ms << ',' << prd_splits << '\n';
        summary.close();
        if (!summary) throw std::runtime_error("failed writing summary CSV");

        std::ofstream manifest(output_dir / "manifest.txt", std::ios::trunc);
        if (!manifest) throw std::runtime_error("cannot create experiment manifest");
        manifest << "dataset=SIFT1M original row order, 1M x 128 uint8\n"
                 << "queries=first 1000 official SIFT1M queries; exact L2 top-10 ground truth\n"
                 << "query_protocol=exactly 4 nearest coarse clusters per query; no total-vector scan cap\n"
                 << "full_ivf=all 1M vectors assigned to a 64-cluster IVF-Flat; training sample=20000, seed=1234, kmeans iterations=12\n"
                 << "prd_only=initial 50000 input-order vectors, nlist=64, seed=1234, 10 kmeans iterations; then 19 sequential 50000-vector insertions\n"
                 << "prd_threshold=" << kPrdThreshold << "; GlobalFitOnly; PRD split enabled; load split, capacity split, merge, local split reassignment, and global center mean refresh disabled\n"
                 << "warmup=32 queries per index, excluded from metrics\n"
                 << "full_ivf_build_ms=" << full_build_ms << "\n"
                 << "prd_build_plus_insert_maintenance_ms=" << prd_build_ms << "\n"
                 << "prd_insert_maintenance_only_ms=" << prd_insert_maintenance_ms << "\n"
                 << "prd_split_count=" << prd_splits << "\n"
                 << "note=fixed nprobe controls lists, not vectors; scanned vector count varies with list sizes\n";
        manifest.close();
        if (!manifest) throw std::runtime_error("failed writing experiment manifest");

        std::cout << "RESULT Full-IVF recall@10=" << full_eval.recall
                  << " mean_scanned=" << full_eval.mean_scanned << '\n'
                  << "RESULT PRD-only recall@10=" << prd_eval.recall
                  << " final_nlist=" << prd_eval.nlist
                  << " mean_scanned=" << prd_eval.mean_scanned
                  << " prd_splits=" << prd_splits << '\n'
                  << "COMPLETE output=" << output_dir.string() << '\n' << std::flush;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
