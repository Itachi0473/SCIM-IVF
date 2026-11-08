#include "adaptive_hierarchical_ivf.h"

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
constexpr std::size_t kQueryCount = 1'000;
constexpr std::size_t kTopK = 10;
constexpr std::size_t kWarmupQueries = 32;
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

std::vector<BaseVector> read_base(const std::filesystem::path& path) {
#ifdef GLOVE100
    if (std::filesystem::file_size(path) != kRows * kDim * sizeof(float))
        throw std::runtime_error("GloVe base must be exactly 1,183,514 x 100 FP32 values");
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open GloVe base: " + path.string());
    std::vector<BaseVector> base(kRows * kDim);
    in.read(reinterpret_cast<char*>(base.data()), static_cast<std::streamsize>(base.size() * sizeof(BaseVector)));
    if (!in) throw std::runtime_error("truncated GloVe base payload");
    return base;
#else
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open SIFT base: " + path.string());
    const auto rows = read_scalar<std::uint32_t>(in, "base row count");
    const auto dim = read_scalar<std::uint32_t>(in, "base dimension");
    if (rows != kRows || dim != kDim ||
        std::filesystem::file_size(path) != 8ULL + kRows * kDim)
        throw std::runtime_error("base must be exactly 1M x 128 uint8 SIFT");
    std::vector<BaseVector> base(kRows * kDim);
    in.read(reinterpret_cast<char*>(base.data()), static_cast<std::streamsize>(base.size()));
    if (!in) throw std::runtime_error("truncated SIFT base payload");
    return base;
#endif
}

std::vector<float> read_queries(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open queries: " + path.string());
    std::vector<float> queries(kQueryCount * kDim);
    for (std::size_t q = 0; q < kQueryCount; ++q) {
        if (read_scalar<std::int32_t>(in, "query dimension") != static_cast<std::int32_t>(kDim))
            throw std::runtime_error("expected 128D fvecs query");
        in.read(reinterpret_cast<char*>(queries.data() + q * kDim),
                static_cast<std::streamsize>(kDim * sizeof(float)));
        if (!in) throw std::runtime_error("truncated query sample");
    }
    return queries;
}

std::vector<TruthRow> read_truth(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open stage truth: " + path.string());
    std::vector<TruthRow> truth(kQueryCount);
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

std::vector<std::vector<std::int64_t>> read_deletion_batches(
        const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open deletion sequence: " + path.string());
    std::string line;
    if (!std::getline(in, line) || line != "batch,position,original_vector_id")
        throw std::runtime_error("unexpected deletion_batches.csv header");
    std::vector<std::vector<std::int64_t>> batches(kBatchCount);
    for (auto& batch : batches) batch.reserve(kBatchRows);
    std::vector<std::uint8_t> seen(kRows, 0);
    std::size_t rows = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream row(line);
        std::size_t batch = 0, position = 0;
        std::int64_t id = -1;
        char comma1 = 0, comma2 = 0;
        if (!(row >> batch >> comma1 >> position >> comma2 >> id) ||
            comma1 != ',' || comma2 != ',' || batch == 0 || batch > kBatchCount ||
            position >= kBatchRows || id < 0 || id >= static_cast<std::int64_t>(kRows))
            throw std::runtime_error("malformed deletion batch row: " + line);
        auto& target = batches[batch - 1];
        if (position != target.size() || seen[static_cast<std::size_t>(id)])
            throw std::runtime_error("deletion IDs must be unique and ordered by batch/position");
        target.push_back(id);
        seen[static_cast<std::size_t>(id)] = 1;
        ++rows;
    }
    if (rows != kBatchCount * kBatchRows)
        throw std::runtime_error("deletion schedule must contain exactly 800,000 IDs");
    for (const auto& batch : batches)
        if (batch.size() != kBatchRows)
            throw std::runtime_error("each deletion batch must contain exactly 20,000 IDs");
    return batches;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto rank = std::max<std::size_t>(1,
        static_cast<std::size_t>(std::ceil(p * static_cast<double>(values.size()))));
    return values[std::min(values.size() - 1, rank - 1)];
}

struct QuerySummary {
    double recall = 0.0;
    double mean_ms = 0.0;
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double mean_centers = 0.0;
    double mean_vectors = 0.0;
    double mean_total = 0.0;
};

QuerySummary query_stage(const AdaptiveHierarchicalIvf& index,
                         const std::vector<float>& queries,
                         const std::vector<TruthRow>& truth,
                         std::size_t active_count,
                         std::size_t stage,
                         std::ofstream& query_rows) {
    QuerySearchOptions options;
    options.enable_early_stop = false;
    options.max_total_distance_computations = active_count / 20;
    if (options.max_total_distance_computations < index.leaf_nlist())
        throw std::runtime_error("N/20 scan budget is smaller than the live centroid count");

    for (std::size_t q = 0; q < kWarmupQueries; ++q)
        (void)index.search_with_options(queries.data() + q * kDim, kTopK, options);

    QuerySummary result;
    std::vector<double> latencies;
    latencies.reserve(kQueryCount);
    std::size_t hits = 0;
    double center_sum = 0.0, vector_sum = 0.0, total_sum = 0.0;
    for (std::size_t q = 0; q < kQueryCount; ++q) {
        QuerySearchStats stats;
        const auto start = Clock::now();
        const auto found = index.search_with_options(
            queries.data() + q * kDim, kTopK, options, &stats);
        const double elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        latencies.push_back(elapsed_ms);
        std::size_t query_hits = 0;
        for (const auto& item : found) {
            for (const auto expected : truth[q])
                if (item.id == expected) { ++query_hits; break; }
        }
        hits += query_hits;
        center_sum += stats.center_distance_computations;
        vector_sum += stats.scanned_vectors;
        total_sum += stats.total_distance_computations;
        query_rows << stage << ',' << q << ',' << active_count << ','
                   << options.max_total_distance_computations << ','
                   << stats.center_distance_computations << ',' << stats.scanned_vectors << ','
                   << stats.total_distance_computations << ',' << query_hits << ','
                   << elapsed_ms << '\n';
    }
    query_rows.flush();
    result.recall = static_cast<double>(hits) / static_cast<double>(kQueryCount * kTopK);
    result.mean_ms = std::accumulate(latencies.begin(), latencies.end(), 0.0) / kQueryCount;
    result.p50_ms = percentile(latencies, 0.50);
    result.p95_ms = percentile(latencies, 0.95);
    result.p99_ms = percentile(latencies, 0.99);
    result.mean_centers = center_sum / kQueryCount;
    result.mean_vectors = vector_sum / kQueryCount;
    result.mean_total = total_sum / kQueryCount;
    return result;
}

std::pair<std::size_t, std::size_t> list_size_range(const AdaptiveHierarchicalIvf& index) {
    const auto sizes = index.leaf_sizes();
    if (sizes.empty()) return {0, 0};
    const auto minmax = std::minmax_element(sizes.begin(), sizes.end());
    return {*minmax.first, *minmax.second};
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5 && argc != 6) {
            std::cerr << "Usage: sift1m_random_delete_full_runner BASE.u8bin QUERIES.fvecs "
                         "PREPARED_DATA_DIR OUTPUT_DIR [SHARED_INITIAL_INDEX.ivfidx]\n";
            return 2;
        }
        const std::filesystem::path base_path(argv[1]);
        const std::filesystem::path query_path(argv[2]);
        const std::filesystem::path prepared_dir(argv[3]);
        const std::filesystem::path output(argv[4]);
        const std::filesystem::path shared_index_path = argc == 6 ? std::filesystem::path(argv[5]) : std::filesystem::path();
        if (std::filesystem::exists(output) &&
            std::filesystem::directory_iterator(output) != std::filesystem::directory_iterator())
            throw std::runtime_error("output directory must be absent or empty: " + output.string());
        std::filesystem::create_directories(output);

        std::cout << "Reading base, queries, and prepared deletion/truth files...\n" << std::flush;
        const auto base = read_base(base_path);
        const auto queries = read_queries(query_path);
        const auto deletion_batches = read_deletion_batches(prepared_dir / "deletion_batches.csv");

        AdaptiveHierarchicalConfig config;
        config.dimension = kDim;
        config.initial_leaf_nlist = kInitialNlist;
        config.kmeans_iterations = 10;
        config.automatic_maintenance = false;
        config.prd_threshold = 0.075f;
        config.drift_criterion = DriftCriterion::GlobalFitOnly;
        config.short_list_sqrt_n_fraction = 0.20f;
        config.load_split_fraction = 2.20f;
        config.capacity_sqrt_n_split_multiplier = 1.05f;
        config.split_local_refinement_radius = 50;
        config.split_local_refinement_iterations = 1;
#ifdef GLOVE100
        config.uint8_storage = false;
        config.angular_metric = false; // normalized GloVe vectors: L2 ranking equals angular ranking
#else
        config.uint8_storage = true;
#endif
        config.seed = kSeed;

        std::ofstream summary(output / "stage_metrics.csv", std::ios::trunc);
        std::ofstream query_rows(output / "query_metrics.csv", std::ios::trunc);
        std::ofstream manifest(output / "manifest.txt", std::ios::trunc);
        if (!summary || !query_rows || !manifest)
            throw std::runtime_error("cannot create experiment outputs");
        summary << "stage,active_vectors,deleted_this_batch,nlist_before,nlist_after,min_list_size,max_list_size,"
                   "build_ms,delete_ms,maintenance_ms,insert_delete_maintenance_ms,marked_partitions,"
                   "load_splits,capacity_splits,prd_splits,merged_short,removed_empty,recentered,"
                   "recall_at_10,query_mean_ms,query_p50_ms,query_p95_ms,query_p99_ms,"
                   "mean_center_distances,mean_vector_distances,mean_total_distances,total_sse\n";
        query_rows << "stage,query_id,active_vectors,total_distance_budget,center_distances,vector_distances,"
                       "total_distances,top10_hits,query_ms\n";

        manifest << "dataset="
#ifdef GLOVE100
                 << "GloVe100-angular ANN-Benchmarks, 1,183,514 x 100 normalized FP32 base\n"
#else
                 << "SIFT1M original order, full 1,000,000 x 128 exact uint8 base\n"
#endif
                 << "base=" << std::filesystem::absolute(base_path).string() << "\n"
                 << "query_sample=" << std::filesystem::absolute(query_path).string() << " (first 1,000 official queries)\n"
                 << "deletion_schedule=" << std::filesystem::absolute(prepared_dir / "deletion_batches.csv").string() << "\n"
                 << "groundtruth_dir=" << std::filesystem::absolute(prepared_dir / "exact_truth").string() << "\n"
#ifdef GLOVE100
                 << "initial_index=shared 1,183,514-vector GloVe index; nlist=1100; KMeans 10 iterations; seed=1234; FP32 vectors\n"
#else
                 << "initial_build=all 1,000,000 vectors; KMeans nlist=1000; 10 iterations; seed=1234\n"
#endif
                 << "updates=40 deletion-only batches x 20,000 unique vector IDs; 800,000 total removed; no inserts\n"
                 << "maintenance=once after each deletion batch; PRD GlobalFitOnly threshold=0.075; load split threshold=2.20*T; short-list merge cutoff=0.20*sqrt(N); capacity split uses 1.05*sqrt(N) while K<=sqrt(N)\n"
                 << "split_local_refinement=radius 50, one assignment pass\n"
                 << "centers=refresh all live centroids to means after each maintenance batch\n"
#ifdef GLOVE100
                 << "storage=exact normalized FP32 vectors; centroids/statistics FP32; distance=squared L2 (angular-equivalent ranking)\n"
#else
                 << "storage=exact uint8 vectors; centroids/statistics FP32; distance=squared L2\n"
#endif
                 << "query=1000 queries after every stage, after maintenance; Recall@10 against exact stage top-10\n"
                 << "query_budget=floor(active_vectors/20) total distance computations, including centroid comparisons; early stopping disabled; 32 warmup queries excluded from latency\n"
                 << "update_timing=delete, maintenance, and all-center refresh reported separately and combined; excludes query measurement\n"
                 << "initial_training_time is reported separately in stage 0\n";

        AdaptiveHierarchicalIvf index(config);
        double build_ms = 0.0;
        if (!shared_index_path.empty()) {
            index = AdaptiveHierarchicalIvf::load(shared_index_path.string());
            index.set_prd_threshold(0.075f);
            index.set_short_list_sqrt_n_fraction(0.20f);
            index.set_capacity_sqrt_n_split_multiplier(1.05f);
            index.set_split_local_refinement(50, 1);
            index.set_automatic_maintenance(false);
        } else {
            const auto build_start = Clock::now();
#ifdef GLOVE100
            index.train(base.data(), kRows);
#else
            index.train_u8(base.data(), kRows);
#endif
            build_ms = std::chrono::duration<double, std::milli>(Clock::now() - build_start).count();
        }
        if (index.size() != kRows || index.leaf_nlist() != kInitialNlist)
            throw std::runtime_error("initial index validation failed");
        std::cout << "BUILD_COMPLETE vectors=" << index.size() << " nlist=" << index.leaf_nlist()
                  << " build_ms=" << std::fixed << std::setprecision(1) << build_ms << '\n' << std::flush;

        for (std::size_t stage = 0; stage <= kBatchCount; ++stage) {
            const std::size_t active = kRows - stage * kBatchRows;
            const auto truth = read_truth(prepared_dir / "exact_truth" /
                                          ("stage_" + (stage < 10 ? std::string("0") : std::string()) +
                                           std::to_string(stage) + "_top10_l2.ivecs"));
            const std::size_t nlist_before = index.leaf_nlist();
            double delete_ms = 0.0, maintenance_ms = 0.0;
            MaintenanceReport maintenance{};
            if (stage > 0) {
                index.begin_update_batch();
                const auto delete_start = Clock::now();
                const std::size_t removed = index.remove_ids(deletion_batches[stage - 1]);
                delete_ms = std::chrono::duration<double, std::milli>(Clock::now() - delete_start).count();
                if (removed != kBatchRows)
                    throw std::runtime_error("a deletion batch did not remove exactly 20,000 vectors");
                (void)index.end_update_batch();

                const auto maintenance_start = Clock::now();
                maintenance = index.maintain_with_policy(true, true, true, true);
                const auto refresh = index.refresh_all_centroids_to_means();
                if (refresh.vectors_covered != index.size())
                    throw std::runtime_error("center refresh missed active vectors");
                maintenance_ms = std::chrono::duration<double, std::milli>(Clock::now() - maintenance_start).count();
                if (index.size() != active)
                    throw std::runtime_error("active vector count mismatch after delete batch");
                if (maintenance.total_split_partitions !=
                    maintenance.load_split_partitions + maintenance.capacity_split_partitions +
                    maintenance.prd_split_partitions)
                    throw std::runtime_error("split counters do not add up");
            }

            const QuerySummary query = query_stage(index, queries, truth, active, stage, query_rows);
            const auto [min_size, max_size] = list_size_range(index);
            const double total_update_ms = delete_ms + maintenance_ms;
            summary << stage << ',' << active << ',' << (stage == 0 ? 0 : kBatchRows) << ','
                    << nlist_before << ',' << index.leaf_nlist() << ',' << min_size << ',' << max_size << ','
                    << (stage == 0 ? build_ms : 0.0) << ',' << delete_ms << ',' << maintenance_ms << ','
                    << total_update_ms << ',' << maintenance.marked_partitions << ','
                    << maintenance.load_split_partitions << ',' << maintenance.capacity_split_partitions << ','
                    << maintenance.prd_split_partitions << ',' << maintenance.merged_short_partitions << ','
                    << maintenance.removed_empty_partitions << ',' << maintenance.recentered_partitions << ','
                    << query.recall << ',' << query.mean_ms << ',' << query.p50_ms << ',' << query.p95_ms << ','
                    << query.p99_ms << ',' << query.mean_centers << ',' << query.mean_vectors << ','
                    << query.mean_total << ',' << index.total_sse() << '\n';
            summary.flush();
            std::cout << "STAGE=" << std::setw(2) << std::setfill('0') << stage
                      << " active=" << active << " nlist=" << index.leaf_nlist()
                      << " delete_ms=" << std::fixed << std::setprecision(1) << delete_ms
                      << " maint_ms=" << maintenance_ms
                      << " recall@10=" << std::setprecision(4) << query.recall
                      << " query_mean_ms=" << std::setprecision(3) << query.mean_ms
                      << " p99_ms=" << query.p99_ms << '\n' << std::flush;
        }
        std::cout << "COMPLETE output=" << std::filesystem::absolute(output).string() << '\n' << std::flush;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
