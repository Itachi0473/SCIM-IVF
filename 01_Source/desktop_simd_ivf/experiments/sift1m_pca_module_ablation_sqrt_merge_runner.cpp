#include "../adaptive_hierarchical_ivf.h"

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
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::size_t kRows = 1'000'000;
constexpr std::size_t kDim = 128;
constexpr std::size_t kInitialRows = 50'000;
constexpr std::size_t kBatchRows = 50'000;
constexpr std::size_t kInsertBatches = 19;
constexpr std::size_t kQueryCount = 1'000;
constexpr std::size_t kTopK = 10;
constexpr std::size_t kInitialNlist = 64;
constexpr std::size_t kWarmup = 32;
constexpr std::uint32_t kSeed = 1234;
using Clock = std::chrono::steady_clock;
using TruthRow = std::array<std::int32_t, kTopK>;

template <typename T>
T read_scalar(std::istream& in, const char* label) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error(std::string("truncated ") + label);
    return value;
}

std::vector<float> read_base(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open base: " + path);
    const auto bytes = static_cast<std::uintmax_t>(kRows) * kDim * sizeof(float);
    if (std::filesystem::file_size(path) != bytes)
        throw std::runtime_error("base must contain exactly 1M x 128 raw float32 values");
    std::vector<float> data(kRows * kDim);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error("truncated base");
    for (const float x : data)
        if (!std::isfinite(x)) throw std::runtime_error("base contains non-finite values");
    return data;
}

std::vector<float> read_queries(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open queries: " + path);
    std::vector<float> queries(kQueryCount * kDim);
    for (std::size_t q = 0; q < kQueryCount; ++q) {
        if (read_scalar<std::int32_t>(in, "query dimension") != static_cast<std::int32_t>(kDim))
            throw std::runtime_error("query vectors must be 128D fvecs");
        in.read(reinterpret_cast<char*>(queries.data() + q * kDim),
                static_cast<std::streamsize>(kDim * sizeof(float)));
        if (!in) throw std::runtime_error("truncated query vectors");
    }
    return queries;
}

std::array<TruthRow, kQueryCount> read_truth(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open exact truth: " + path);
    std::array<TruthRow, kQueryCount> truth{};
    for (auto& row : truth) {
        if (read_scalar<std::int32_t>(in, "truth row length") != static_cast<std::int32_t>(kTopK))
            throw std::runtime_error("exact truth must contain top-10 rows");
        in.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(sizeof(row)));
        if (!in) throw std::runtime_error("truncated exact truth");
    }
    if (in.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error("unexpected trailing bytes in exact truth");
    return truth;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto rank = std::max<std::size_t>(1,
        static_cast<std::size_t>(std::ceil(p * static_cast<double>(values.size()))));
    return values[std::min(values.size() - 1, rank - 1)];
}

struct Variant {
    const char* name;
    bool prd;
    bool load_splits;
    bool capacity_splits;
    bool load_merges;
};

struct Result {
    const char* name = "";
    std::size_t final_nlist = 0;
    double build_ms = 0.0;
    double insert_total_ms = 0.0;
    double query_mean_ms = 0.0;
    double query_p50_ms = 0.0;
    double query_p95_ms = 0.0;
    double query_p99_ms = 0.0;
    double recall = 0.0;
    std::size_t load_splits = 0;
    std::size_t capacity_splits = 0;
    std::size_t prd_splits = 0;
    std::size_t recentered = 0;
    std::size_t merged = 0;
};

Result run_variant(const Variant& variant, const std::vector<float>& base,
                   const std::vector<float>& queries,
                   const std::array<TruthRow, kQueryCount>& truth,
                   std::ofstream& stage_csv, std::ofstream& query_csv) {
    AdaptiveHierarchicalConfig config;
    config.dimension = kDim;
    config.initial_leaf_nlist = kInitialNlist;
    config.leaf_nprobe_ratio = 8;
    config.kmeans_iterations = 10;
    config.automatic_maintenance = false;
    config.prd_threshold = 0.05f;
    config.drift_criterion = DriftCriterion::GlobalFitOnly;
    config.short_list_fraction = 0.20f;
    config.short_list_sqrt_n_fraction = 0.20f;
    config.load_split_fraction = 2.20f;
    config.capacity_sqrt_n_split_multiplier = 1.05f;
    config.split_local_refinement_radius = 50;
    config.split_local_refinement_iterations = 1;
    config.seed = kSeed;

    Result result;
    result.name = variant.name;
    AdaptiveHierarchicalIvf index(config);
    const auto build_start = Clock::now();
    index.train(base.data(), kInitialRows);
    result.build_ms = std::chrono::duration<double, std::milli>(Clock::now() - build_start).count();
    if (index.size() != kInitialRows || index.leaf_nlist() != kInitialNlist)
        throw std::runtime_error("initial 64-cluster build validation failed");

    for (std::size_t batch = 0; batch < kInsertBatches; ++batch) {
        const std::size_t first = kInitialRows + batch * kBatchRows;
        const auto update_start = Clock::now();
        index.begin_update_batch();
        index.add(base.data() + first * kDim, kBatchRows);
        (void)index.end_update_batch();
        const MaintenanceReport maintenance = index.maintain_with_policy(
            variant.prd, variant.load_splits, variant.capacity_splits, variant.load_merges);
        const GlobalCenterRefreshReport center_refresh = index.refresh_all_centroids_to_means();
        if (center_refresh.vectors_covered != index.size())
            throw std::runtime_error("global center refresh did not cover all active vectors");
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - update_start).count();
        result.insert_total_ms += elapsed_ms;
        result.load_splits += maintenance.load_split_partitions;
        result.capacity_splits += maintenance.capacity_split_partitions;
        result.prd_splits += maintenance.prd_split_partitions;
        result.recentered += maintenance.recentered_partitions;
        result.merged += maintenance.merged_short_partitions;
        stage_csv << variant.name << ',' << (batch + 1) << ',' << index.size() << ','
                  << index.leaf_nlist() << ',' << elapsed_ms << ','
                  << maintenance.load_split_partitions << ','
                  << maintenance.capacity_split_partitions << ','
                  << maintenance.prd_split_partitions << ','
                  << maintenance.recentered_partitions << ','
                  << maintenance.merged_short_partitions << ','
                  << center_refresh.centers_updated << ','
                  << center_refresh.vectors_covered << '\n';
        stage_csv.flush();
        std::cout << "UPDATE variant=" << variant.name << " batch=" << (batch + 1)
                  << "/" << kInsertBatches << " vectors=" << index.size()
                  << " nlist=" << index.leaf_nlist() << " update_ms="
                  << std::fixed << std::setprecision(1) << elapsed_ms << '\n' << std::flush;
    }
    if (index.size() != kRows) throw std::runtime_error("final active vector count is not 1M");

    QuerySearchOptions options;
    options.enable_early_stop = false;
    options.max_total_distance_computations = kRows / 20;
    for (std::size_t q = 0; q < kWarmup; ++q)
        (void)index.search_with_options(queries.data() + q * kDim, kTopK, options, nullptr);

    std::vector<double> latencies;
    latencies.reserve(kQueryCount);
    std::size_t total_hits = 0;
    for (std::size_t q = 0; q < kQueryCount; ++q) {
        QuerySearchStats stats;
        const auto start = Clock::now();
        const auto found = index.search_with_options(
            queries.data() + q * kDim, kTopK, options, &stats);
        const double latency_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - start).count();
        if (stats.total_distance_computations != options.max_total_distance_computations ||
            stats.early_stopped || found.size() != kTopK)
            throw std::runtime_error("final search did not honor N/20 budget exactly");
        std::size_t hits = 0;
        for (const auto& candidate : found) {
            for (const auto expected : truth[q]) {
                if (candidate.id == expected) { ++hits; break; }
            }
        }
        total_hits += hits;
        latencies.push_back(latency_ms);
        query_csv << variant.name << ',' << q << ',' << options.max_total_distance_computations
                  << ',' << stats.center_distance_computations << ',' << stats.scanned_vectors
                  << ',' << std::setprecision(9) << latency_ms << ',' << hits << '\n';
    }
    query_csv.flush();
    result.final_nlist = index.leaf_nlist();
    result.recall = static_cast<double>(total_hits) / (kQueryCount * kTopK);
    result.query_mean_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    result.query_p50_ms = percentile(latencies, 0.50);
    result.query_p95_ms = percentile(latencies, 0.95);
    result.query_p99_ms = percentile(latencies, 0.99);
    return result;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5) {
            std::cerr << "Usage: sift1m_pca_module_ablation_sqrt_merge_runner BASE.f32 QUERIES.fvecs "
                         "FINAL_TOP10.ivecs OUTPUT_DIR\n";
            return 2;
        }
        const std::string output_dir = argv[4];
        if (std::filesystem::exists(output_dir) &&
            std::filesystem::directory_iterator(output_dir) != std::filesystem::directory_iterator())
            throw std::runtime_error("output directory is not empty: " + output_dir);
        std::filesystem::create_directories(output_dir);

        std::cout << "LOAD_BEGIN PCA-SIFT1M\n" << std::flush;
        const auto base = read_base(argv[1]);
        const auto queries = read_queries(argv[2]);
        const auto truth = read_truth(argv[3]);

        std::ofstream summary(output_dir + "/summary.csv");
        std::ofstream stages(output_dir + "/batch_metrics.csv");
        std::ofstream query(output_dir + "/query_metrics.csv");
        std::ofstream manifest(output_dir + "/manifest.txt");
        if (!summary || !stages || !query || !manifest)
            throw std::runtime_error("cannot create output files");
        summary << "variant,initial_nlist,final_nlist,initial_vectors,insert_batches,inserted_vectors,"
                   "query_count,topk,budget_total_distances,recall_at_10,query_mean_ms,query_p50_ms,"
                   "query_p95_ms,query_p99_ms,build_ms,total_insert_and_maintenance_ms,"
                   "load_splits,capacity_splits,prd_splits,recentered,merged\n";
        stages << "variant,batch,active_vectors,nlist,insert_maintenance_and_center_refresh_ms,load_splits,"
                   "capacity_splits,prd_splits,recentered,merged,centers_refreshed,vectors_covered_by_refresh\n";
        query << "variant,query_id,budget_total_distances,center_distances,scanned_vectors,"
                 "query_ms,recall_hits_at_10\n";
        manifest << "dataset=SIFT1M vectors sorted by full-dataset PC1 ascending\n"
                 << "base=" << argv[1] << "\nqueries=" << argv[2]
                 << "\ntruth=" << argv[3] << "\nmetric=squared_L2\ntopk=10\n"
                 << "initial_index=first 50,000 PCA-ordered vectors; KMeans nlist=64, seed=1234, 10 iterations\n"
                 << "updates=19 sequential batches x 50,000; insertion-only; no intermediate queries\n"
                 << "final_active_vectors=1,000,000\nquery_set=first 1,000 official SIFT1M queries\n"
                 << "query_budget=N/20=50,000 total distance computations, including all centroid distances\n"
                 << "warmup=32 queries per variant, excluded from latency statistics\n"
                 << "query_timing=steady_clock around search_with_options only; p99 nearest-rank percentile\n"
                 << "maintenance=after each complete insertion batch; PRD GlobalFitOnly threshold=0.05; "
                    "short-list merge cutoff=n<0.20*sqrt(N); load split threshold=n>2.20*T, T=N/K; "
                    "capacity split iff K<=sqrt(N) and n>1.05*sqrt(N)\n"
                 << "split_local_reassignment=radius 50 additional neighbors, 1 assignment pass; "
                    "all-center mean refresh after each maintenance batch; no full-vector reassignment\n"
                 << "short_partition_reassignment=route each source vector to nearest other live centroid; "
                    "load split remains n>2.20*T; merge destination checks and later capacity/PRD/load stages unchanged\n"
                 << "full=PRD+load split/merge+capacity split\n"
                 << "ablation_no_capacity=PRD+load split/merge, capacity split disabled\n"
                 << "ablation_no_prd=load split/merge+capacity split, PRD disabled\n";

        const std::array<Variant, 3> variants{{
            {"Full", true, true, true, true},
            {"No capacity split", true, true, false, true},
            {"No PRD", false, true, true, true},
        }};
        for (const auto& variant : variants) {
            std::cout << "VARIANT_BEGIN name=" << variant.name << '\n' << std::flush;
            const Result r = run_variant(variant, base, queries, truth, stages, query);
            summary << r.name << ',' << kInitialNlist << ',' << r.final_nlist << ','
                    << kInitialRows << ',' << kInsertBatches << ',' << (kRows - kInitialRows) << ','
                    << kQueryCount << ',' << kTopK << ',' << (kRows / 20) << ','
                    << std::setprecision(8) << r.recall << ','
                    << std::setprecision(9) << r.query_mean_ms << ',' << r.query_p50_ms << ','
                    << r.query_p95_ms << ',' << r.query_p99_ms << ',' << r.build_ms << ','
                    << r.insert_total_ms << ',' << r.load_splits << ',' << r.capacity_splits << ','
                    << r.prd_splits << ',' << r.recentered << ',' << r.merged << '\n';
            summary.flush();
            std::cout << "VARIANT_END name=" << r.name << " recall@10=" << std::fixed
                      << std::setprecision(4) << r.recall << " mean_ms=" << r.query_mean_ms
                      << " p99_ms=" << r.query_p99_ms << " final_nlist=" << r.final_nlist
                      << " splits(load/capacity/PRD)=" << r.load_splits << '/'
                      << r.capacity_splits << '/' << r.prd_splits << '\n' << std::flush;
        }
        std::cout << "COMPLETE output=" << output_dir << '\n' << std::flush;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
