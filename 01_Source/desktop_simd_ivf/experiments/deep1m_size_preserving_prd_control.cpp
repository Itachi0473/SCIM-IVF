#include "../adaptive_hierarchical_ivf.h"
#include "../simd_l2.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

namespace {
constexpr std::size_t kRows = 1'000'000;
constexpr std::size_t kSourceRows = 10'000'000;
constexpr std::size_t kDimension = 96;
constexpr std::size_t kQueries = 1'000;
constexpr std::size_t kTopK = 10;
constexpr std::size_t kInitialNlist = 1'000;
constexpr std::uint32_t kBuildSeed = 1234;
constexpr std::uint32_t kPartitionSeed = 20261106;
using Clock = std::chrono::steady_clock;
using TruthRow = std::array<std::int64_t, kTopK>;

void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

template <class F>
void parallel_for(std::size_t count, std::size_t workers, F&& fn) {
    workers = std::max<std::size_t>(1, std::min(workers, count));
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (std::size_t worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&] {
            for (;;) {
                const std::size_t i = next.fetch_add(1, std::memory_order_relaxed);
                if (i >= count) break;
                fn(i);
            }
        });
    }
    for (auto& thread : threads) thread.join();
}

double seconds(Clock::time_point start, Clock::time_point end = Clock::now()) {
    return std::chrono::duration<double>(end - start).count();
}

std::vector<float> read_fbin_prefix(const std::string& path,
                                    std::size_t expected_rows,
                                    std::size_t take_rows) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open Deep fbin: " + path);
    std::int32_t rows = 0, dimension = 0;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
    require(in && rows == static_cast<std::int32_t>(expected_rows) &&
            dimension == static_cast<std::int32_t>(kDimension),
            "Deep fbin header is not 10M x 96: " + path);
    require(take_rows <= expected_rows, "requested fbin prefix exceeds source rows");
    const std::uintmax_t expected_bytes = 8ULL +
        static_cast<std::uintmax_t>(expected_rows) * kDimension * sizeof(float);
    require(std::filesystem::file_size(path) == expected_bytes,
            "Deep fbin file size does not match header: " + path);
    std::vector<float> data(take_rows * kDimension);
    in.read(reinterpret_cast<char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!in) throw std::runtime_error("truncated Deep fbin payload: " + path);
    for (const float value : data)
        if (!std::isfinite(value)) throw std::runtime_error("Deep fbin contains non-finite values");
    return data;
}

std::vector<float> read_query_prefix(const std::string& path, std::size_t count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open Deep query fbin: " + path);
    std::int32_t rows = 0, dimension = 0;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
    require(in && rows == 10'000 && dimension == static_cast<std::int32_t>(kDimension),
            "Deep query file must be 10K x 96 fbin");
    require(count <= static_cast<std::size_t>(rows), "query prefix exceeds official query rows");
    const auto expected_bytes = 8ULL + 10'000ULL * kDimension * sizeof(float);
    require(std::filesystem::file_size(path) == expected_bytes,
            "Deep query fbin has an unexpected size");
    std::vector<float> queries(count * kDimension);
    in.read(reinterpret_cast<char*>(queries.data()),
            static_cast<std::streamsize>(queries.size() * sizeof(float)));
    if (!in) throw std::runtime_error("truncated Deep query fbin");
    return queries;
}

std::vector<std::size_t> parse_positive_list(const std::string& text, const char* label) {
    std::vector<std::size_t> values;
    std::size_t start = 0;
    while (start < text.size()) {
        const std::size_t comma = text.find(',', start);
        const std::string token = text.substr(start,
            comma == std::string::npos ? std::string::npos : comma - start);
        require(!token.empty(), std::string("empty value in ") + label);
        const auto value = static_cast<std::size_t>(std::stoull(token));
        require(value > 0, std::string(label) + " values must be positive");
        values.push_back(value);
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    require(!values.empty(), std::string(label) + " list must not be empty");
    std::sort(values.begin(), values.end());
    require(std::adjacent_find(values.begin(), values.end()) == values.end(),
            std::string(label) + " values must be unique");
    return values;
}

struct SizeCandidates {
    std::size_t capacity = 0;
    std::size_t load = 0;
    std::size_t short_lists = 0;
    std::size_t prd = 0;
    std::size_t total_size_triggers() const { return capacity + load + short_lists; }
};

SizeCandidates count_candidates(const AdaptiveHierarchicalIvf& index) {
    SizeCandidates out;
    const double n = static_cast<double>(index.size());
    const double k = static_cast<double>(index.leaf_nlist());
    const double sqrt_n = std::sqrt(n);
    const double target = n / k;
    for (const auto& health : index.partition_health()) {
        const double count = static_cast<double>(health.vector_count);
        if (count == 0.0) continue;
        if (k <= sqrt_n && count > 1.05 * sqrt_n) ++out.capacity;
        if (count > 2.2 * target) ++out.load;
        if (count < 0.2 * sqrt_n) ++out.short_lists;
        if (health.marked) ++out.prd;
    }
    return out;
}

struct SizeSummary {
    std::size_t minimum = 0;
    std::size_t maximum = 0;
    double mean = 0.0;
};

SizeSummary summarize_sizes(const std::vector<std::size_t>& sizes) {
    SizeSummary out;
    if (sizes.empty()) return out;
    out.minimum = *std::min_element(sizes.begin(), sizes.end());
    out.maximum = *std::max_element(sizes.begin(), sizes.end());
    out.mean = std::accumulate(sizes.begin(), sizes.end(), 0.0) / sizes.size();
    return out;
}

std::vector<std::size_t> row_owners(const AdaptiveHierarchicalIvf& index,
                                    const std::vector<std::int64_t>& row_to_id,
                                    const std::vector<std::size_t>& id_to_row) {
    std::vector<std::size_t> owners(kRows, index.leaf_nlist());
    std::vector<std::uint8_t> seen(kRows, 0);
    for (const auto& assignment : index.vector_leaf_assignments()) {
        require(assignment.first >= 0 &&
                static_cast<std::size_t>(assignment.first) < id_to_row.size(),
                "active ID is outside the row mapping");
        const std::size_t row = id_to_row[static_cast<std::size_t>(assignment.first)];
        require(row < kRows, "active ID maps to an invalid source row");
        require(!seen[row], "duplicate active source row in assignment map");
        seen[row] = 1;
        owners[row] = assignment.second;
    }
    require(std::all_of(seen.begin(), seen.end(), [](std::uint8_t value) { return value != 0; }),
            "assignment map does not cover all active source rows");
    for (std::size_t row = 0; row < kRows; ++row)
        require(row_to_id[row] >= 0 && owners[row] < index.leaf_nlist(),
                "row-to-ID or row-to-owner map is invalid");
    for (std::size_t row = 0; row < kRows; ++row)
        require(static_cast<std::size_t>(row_to_id[row]) < id_to_row.size() &&
                id_to_row[static_cast<std::size_t>(row_to_id[row])] == row,
                "row-to-ID mapping disagrees with ID-to-row mapping");
    return owners;
}

double selected_residual_sse(const AdaptiveHierarchicalIvf& index,
                             const std::vector<float>& data,
                             const std::vector<std::vector<std::size_t>>& groups,
                             const std::vector<std::int64_t>& row_to_id,
                             const std::vector<std::size_t>& id_to_row) {
    const auto owners = row_owners(index, row_to_id, id_to_row);
    const auto centers = index.leaf_centroids();
    double sum = 0.0;
    for (const auto& group : groups)
        for (const std::size_t row : group)
            sum += simd_squared_l2(data.data() + row * kDimension,
                                   centers.data() + owners[row] * kDimension,
                                   kDimension);
    return sum;
}

std::vector<TruthRow> exact_top10(const std::vector<float>& data,
                                  const std::vector<std::int64_t>& row_to_id,
                                  const std::vector<float>& queries,
                                  std::size_t workers) {
    const std::size_t query_count = queries.size() / kDimension;
    std::vector<TruthRow> truth(query_count);
    parallel_for(query_count, workers, [&](std::size_t qid) {
        std::priority_queue<std::pair<float, std::int64_t>> best;
        const float* query = queries.data() + qid * kDimension;
        for (std::size_t row = 0; row < kRows; ++row) {
            const auto hit = std::make_pair(
                simd_squared_l2(data.data() + row * kDimension, query, kDimension), row_to_id[row]);
            if (best.size() < kTopK) best.push(hit);
            else if (hit < best.top()) { best.pop(); best.push(hit); }
        }
        for (std::size_t i = kTopK; i > 0; --i) {
            truth[qid][i - 1] = best.top().second;
            best.pop();
        }
    });
    return truth;
}

struct QueryMetrics {
    double recall = 0.0;
    double affected_recall = 0.0;
    double mean_latency_ms = 0.0;
    double mean_distances = 0.0;
    std::size_t affected_queries = 0;
};

QueryMetrics evaluate(const AdaptiveHierarchicalIvf& index,
                      const std::vector<float>& queries,
                      const std::vector<TruthRow>& truth,
                      const std::unordered_set<std::int64_t>& changed_ids,
                      std::size_t budget) {
    const std::size_t query_count = queries.size() / kDimension;
    std::vector<std::size_t> hits(query_count, 0), affected_hits(query_count, 0);
    std::vector<std::uint8_t> affected(query_count, 0);
    std::vector<std::uint8_t> full_budget(query_count, 0);
    std::vector<double> latency(query_count, 0.0), distances(query_count, 0.0);
    parallel_for(query_count, 16, [&](std::size_t qid) {
        QuerySearchOptions options;
        options.enable_early_stop = false;
        options.max_total_distance_computations = budget;
        QuerySearchStats stats;
        const auto start = Clock::now();
        const auto found = index.search_with_options(
            queries.data() + qid * kDimension, kTopK, options, &stats);
        latency[qid] = seconds(start) * 1000.0;
        full_budget[qid] = stats.total_distance_computations == budget;
        distances[qid] = static_cast<double>(stats.total_distance_computations);
        for (const auto id : truth[qid])
            if (changed_ids.count(id)) affected[qid] = 1;
        for (const auto& result : found)
            if (std::find(truth[qid].begin(), truth[qid].end(), result.id) != truth[qid].end())
                ++hits[qid];
        if (affected[qid]) affected_hits[qid] = hits[qid];
    });
    require(std::all_of(full_budget.begin(), full_budget.end(),
                        [](std::uint8_t value) { return value != 0; }),
            "query did not use the full fixed distance budget");

    QueryMetrics result;
    std::size_t all_hits = 0, affected_query_hits = 0;
    double latency_sum = 0.0, distance_sum = 0.0;
    for (std::size_t qid = 0; qid < query_count; ++qid) {
        all_hits += hits[qid];
        latency_sum += latency[qid];
        distance_sum += distances[qid];
        if (affected[qid]) {
            ++result.affected_queries;
            affected_query_hits += affected_hits[qid];
        }
    }
    result.recall = static_cast<double>(all_hits) / (query_count * kTopK);
    result.mean_latency_ms = latency_sum / query_count;
    result.mean_distances = distance_sum / query_count;
    if (result.affected_queries != 0)
        result.affected_recall = static_cast<double>(affected_query_hits) /
                                 (result.affected_queries * kTopK);
    return result;
}

struct PlannedVector {
    bool changed = false;
    std::size_t row = 0;
    std::size_t original_leaf = 0;
    double used_scale = 1.0;
    double old_residual = 0.0;
    double new_residual = 0.0;
    std::array<float, kDimension> value{};
};

std::size_t nearest_center(const float* vector, const std::vector<float>& centers,
                           std::size_t leaves) {
    std::size_t best = 0;
    float best_distance = simd_squared_l2(vector, centers.data(), kDimension);
    for (std::size_t leaf = 1; leaf < leaves; ++leaf) {
        const float distance = simd_squared_l2(
            vector, centers.data() + leaf * kDimension, kDimension);
        if (distance < best_distance) { best = leaf; best_distance = distance; }
    }
    return best;
}

std::vector<PlannedVector> plan_expansion(const std::vector<float>& data,
                                         const std::vector<std::size_t>& rows,
                                         const std::vector<std::size_t>& owners,
                                         const std::vector<float>& centers,
                                         std::size_t leaves,
                                         double requested_scale) {
    std::vector<PlannedVector> planned(rows.size());
    parallel_for(rows.size(), 16, [&](std::size_t i) {
        PlannedVector plan;
        plan.row = rows[i];
        plan.original_leaf = owners[rows[i]];
        const float* original = data.data() + plan.row * kDimension;
        const float* center = centers.data() + plan.original_leaf * kDimension;
        plan.old_residual = simd_squared_l2(original, center, kDimension);
        if (requested_scale <= 1.0) { planned[i] = plan; return; }

        constexpr std::size_t kBackoffSteps = 48;
        const double decrement = (requested_scale - 1.0) / kBackoffSteps;
        for (std::size_t attempt = 0; attempt < kBackoffSteps; ++attempt) {
            const double scale = requested_scale - decrement * attempt;
            if (scale <= 1.0) break;
            std::array<float, kDimension> candidate{};
            bool finite = true;
            for (std::size_t d = 0; d < kDimension; ++d) {
                const double value = static_cast<double>(center[d]) + scale *
                    (static_cast<double>(original[d]) - center[d]);
                if (!std::isfinite(value) || value < -1.0e30 || value > 1.0e30) {
                    finite = false;
                    break;
                }
                candidate[d] = static_cast<float>(value);
                if (!std::isfinite(candidate[d])) { finite = false; break; }
            }
            if (!finite) continue;
            bool changed = false;
            for (std::size_t d = 0; d < kDimension; ++d)
                changed = changed || candidate[d] != original[d];
            if (!changed) continue;
            const double new_residual = simd_squared_l2(candidate.data(), center, kDimension);
            if (!(new_residual > plan.old_residual)) continue;
            if (nearest_center(candidate.data(), centers, leaves) != plan.original_leaf) continue;
            plan.changed = true;
            plan.used_scale = scale;
            plan.new_residual = new_residual;
            plan.value = candidate;
            break;
        }
        planned[i] = plan;
    });
    return planned;
}

struct CaseStats {
    std::size_t selected_vectors = 0;
    std::size_t modified_vectors = 0;
    double mean_used_scale = 1.0;
    double residual_before = 0.0;
    double residual_after_update = 0.0;
};

CaseStats summarize_plan(const std::vector<PlannedVector>& plan) {
    CaseStats result;
    result.selected_vectors = plan.size();
    double accepted_scales = 0.0;
    for (const auto& item : plan) {
        result.residual_before += item.old_residual;
        result.residual_after_update += item.changed ? item.new_residual : item.old_residual;
        if (item.changed) {
            ++result.modified_vectors;
            accepted_scales += item.used_scale;
        }
    }
    if (result.modified_vectors != 0)
        result.mean_used_scale = accepted_scales / result.modified_vectors;
    return result;
}

void append_partition_diagnostics(const std::string& output_dir,
                                  const std::string& scale_label,
                                  const std::string& method_name,
                                  std::size_t target_count,
                                  const std::vector<std::size_t>& target_leaves,
                                  const std::vector<std::vector<std::size_t>>& target_groups,
                                  const std::vector<std::size_t>& size_before,
                                  const std::vector<std::size_t>& modified_by_group,
                                  const std::vector<double>& used_scale_sum,
                                  const std::vector<double>& residual_before,
                                  const std::vector<double>& residual_update,
                                  const std::vector<double>& residual_after_maintenance,
                                  const std::vector<std::size_t>& size_after_update) {
    const bool new_file = !std::filesystem::exists(output_dir + "/target_partition_diagnostics.csv");
    std::ofstream out(output_dir + "/target_partition_diagnostics.csv", std::ios::app);
    if (!out) throw std::runtime_error("cannot append target partition diagnostics");
    if (new_file)
        out << "target_count,scale,method,selection_order,source_leaf,size_before,size_after_update,selected_vectors,modified_vectors,modified_fraction,mean_accepted_scale,mean_residual_before,mean_residual_after_update,mean_residual_after_maintenance\n";
    out << std::setprecision(12);
    for (std::size_t i = 0; i < target_leaves.size(); ++i) {
        const std::size_t count = target_groups[i].size();
        const double denom = static_cast<double>(std::max<std::size_t>(1, count));
        const double scale_mean = modified_by_group[i]
            ? used_scale_sum[i] / modified_by_group[i] : 1.0;
        out << target_count << ',' << scale_label << ',' << method_name << ',' << i + 1 << ',' << target_leaves[i] << ','
            << size_before[target_leaves[i]] << ',' << size_after_update[target_leaves[i]] << ',' << count << ','
            << modified_by_group[i] << ',' << static_cast<double>(modified_by_group[i]) / denom << ','
            << scale_mean << ',' << residual_before[i] / denom << ',' << residual_update[i] / denom << ','
            << residual_after_maintenance[i] / denom << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 4 || argc > 7) {
            std::cerr << "Usage: deep1m_size_preserving_prd_control BASE_10M.fbin QUERY_10K.fbin OUTPUT_DIR [COMMON_START_INDEX] [TARGET_COUNTS_CSV] [BUDGET_DIVISORS_CSV]\n";
            return 2;
        }
        const std::string base_path = argv[1];
        const std::string query_path = argv[2];
        const std::string output_dir = argv[3];
        const std::string common_start_path = argc >= 5 ? argv[4] : "";
        const std::vector<std::size_t> target_counts = argc >= 6
            ? parse_positive_list(argv[5], "target count") : std::vector<std::size_t>{32, 128};
        const std::vector<std::size_t> budget_divisors = argc >= 7
            ? parse_positive_list(argv[6], "budget divisor") : std::vector<std::size_t>{100, 20};
        require(!std::filesystem::exists(output_dir) ||
                std::filesystem::is_empty(output_dir),
                "output directory must be absent or empty");
        std::filesystem::create_directories(output_dir);

        std::ofstream manifest(output_dir + "/manifest.txt");
        std::ofstream preprocess(output_dir + "/preprocessing_rounds.csv");
        std::ofstream summary(output_dir + "/summary.csv");
        std::ofstream partition_health_csv(output_dir + "/common_start_partitions.csv");
        std::ofstream target_csv(output_dir + "/target_partitions.csv");
        std::ofstream updates_csv(output_dir + "/accepted_updates.csv");
        std::ofstream case_health_csv(output_dir + "/case_premaintenance_partitions.csv");
        require(manifest && preprocess && summary && partition_health_csv && target_csv &&
                updates_csv && case_health_csv, "cannot create experiment output files");

        manifest << "experiment=Deep1M size-preserving radial expansion; PRD versus SCIM-Geom\n"
                 << "base_source=" << base_path << "\nbase_vectors=first 1,000,000 of validated 10,000,000 x 96 fbin\n"
                 << "base_storage=FP32 (native Deep data); no normalization or quantization\n"
                 << "queries=" << query_path << "\nquery_source=first 1,000 official Deep10M_2 query vectors\n"
                 << "metric=squared_L2\nseed_kmeans=" << kBuildSeed << "\nseed_target_partitions=" << kPartitionSeed << "\n"
                 << "initial_build=all first 1,000,000 vectors; kmeans nlist=" << kInitialNlist
                 << "; 10 iterations; one worker for deterministic reduction\n"
                 << "preprocessing=PRD disabled; load split, capacity split, and short-list merge enabled; refresh all centers to per-leaf means after each round; repeat to size-predicate fixed point\n"
                 << "capacity_predicate=K<=sqrt(N) and n_i>1.05*sqrt(N)\n"
                 << "load_predicate=n_i>2.2*N/K\nshort_list_predicate=0<n_i<0.2*sqrt(N)\n"
                 << "diagnostic_prd_reference_reset=after size preprocessing only; reset_prd_references_for_diagnostic(); verify no marked leaves\n"
                 << "targets=first " << target_counts.front();
        for (std::size_t i = 1; i < target_counts.size(); ++i)
            manifest << " then " << target_counts[i];
        manifest << " of fixed-seed shuffled eligible partitions; selection excludes query/recall feedback; eligible size >= max(0.75*N/K, 2*0.2*sqrt(N)+64)\n"
                 << "scales=1.00 no-change control,1.10,1.25,1.50; each case starts from common preprocessed snapshot\n"
                 << "expansion=x'=c+s*(x-c) in FP32; back off only when candidate is nonfinite, unchanged, residual does not increase, or nearest common-start center changes; no clipping\n"
                 << "updates=one batched remove/add transaction per scale; all accepted updates generated once, normal add() routing, no forced leaf assignment; then maintenance\n"
                 << "comparison=SCIM enables PRD; SCIM-Geom disables only PRD; both enable capacity/load split and short-list merge, PRD threshold=0.075, GlobalFitOnly, split-local refinement radius=50 and 1 iteration, then refresh all centers to means\n"
                 << "query=first 1,000 official queries; exact current top-10 recomputed by exhaustive scan per case; total-distance budgets=";
        for (std::size_t i = 0; i < budget_divisors.size(); ++i) {
            if (i) manifest << ',';
            manifest << "N/" << budget_divisors[i];
        }
        manifest << " include center distances; 16 query workers; no early stop\n"
                 << "affected_query_definition=at least one exact current top-10 ID is among the newly inserted IDs for accepted changed vectors\n"
                 << "results=summary, preprocessing rounds, shared-start and case partition diagnostics; no full neighbor result dump\n";

        std::cout << "READ_BASE_BEGIN rows=" << kRows << " dimension=" << kDimension << std::endl;
        auto base = read_fbin_prefix(base_path, kSourceRows, kRows);
        auto queries = read_query_prefix(query_path, kQueries);
        std::cout << "READ_DONE base_values=" << base.size() << " query_count="
                  << queries.size() / kDimension << std::endl;

        AdaptiveHierarchicalConfig config;
        config.dimension = kDimension;
        config.initial_leaf_nlist = kInitialNlist;
        config.kmeans_iterations = 10;
        config.worker_threads = 1;
        config.seed = kBuildSeed;
        config.uint8_storage = false;
        config.lvq8_storage = false;
        config.lvq4_storage = false;
        config.angular_metric = false;
        config.automatic_maintenance = false;
        config.prd_threshold = 0.075f;
        config.drift_criterion = DriftCriterion::GlobalFitOnly;
        config.short_list_fraction = 0.20f;
        config.short_list_sqrt_n_fraction = 0.20f;
        config.load_split_fraction = 2.20f;
        config.capacity_sqrt_n_split_multiplier = 1.05f;
        config.capacity_n_divisor = 0.0f;
        config.capacity_cluster_count_gate = 0;
        config.sqrt_load_target = false;
        config.load_target_size = 0.0f;
        config.split_local_refinement_radius = 50;
        config.split_local_refinement_iterations = 1;

        AdaptiveHierarchicalIvf common(config);
        double build_seconds = 0.0;
        if (common_start_path.empty()) {
            std::cout << "BUILD_BEGIN rows=" << kRows << " nlist=" << kInitialNlist << std::endl;
            const auto build_start = Clock::now();
            common.train(base.data(), kRows);
            build_seconds = seconds(build_start);
            require(common.size() == kRows && common.leaf_nlist() == kInitialNlist,
                    "initial full-1M build shape check failed");
            std::cout << "BUILD_DONE seconds=" << build_seconds << " K=" << common.leaf_nlist() << std::endl;
        } else {
            require(std::filesystem::is_regular_file(common_start_path),
                    "shared common-start index does not exist");
            common = AdaptiveHierarchicalIvf::load(common_start_path);
            const auto& loaded = common.config();
            require(common.size() == kRows && loaded.dimension == kDimension &&
                    loaded.initial_leaf_nlist == kInitialNlist &&
                    !loaded.uint8_storage && !loaded.lvq8_storage && !loaded.lvq4_storage &&
                    loaded.seed == kBuildSeed && std::abs(loaded.prd_threshold - 0.075f) < 1e-6f &&
                    std::abs(loaded.load_split_fraction - 2.20f) < 1e-6f &&
                    std::abs(loaded.short_list_sqrt_n_fraction - 0.20f) < 1e-6f &&
                    std::abs(loaded.capacity_sqrt_n_split_multiplier - 1.05f) < 1e-6f &&
                    loaded.split_local_refinement_radius == 50 &&
                    loaded.split_local_refinement_iterations == 1,
                    "loaded common-start snapshot has a mismatched Deep1M control configuration");
            std::cout << "COMMON_START_LOADED path=" << common_start_path
                      << " N=" << common.size() << " K=" << common.leaf_nlist() << std::endl;
            manifest << "common_start_source=" << common_start_path << "\n"
                     << "initial_build=reused validated preprocessed snapshot; no retraining or size-preprocessing replay\n";
        }

        preprocess << "round,K_before,capacity_candidates,load_candidates,short_candidates,load_splits,capacity_splits,merged_short,centers_updated,vectors_refreshed,K_after,min_size,max_size,mean_size,seconds\n";
        bool stable = false;
        for (std::size_t round = 0; common_start_path.empty() && round < 100; ++round) {
            const SizeCandidates before = count_candidates(common);
            if (before.total_size_triggers() == 0) { stable = true; break; }
            const std::size_t k_before = common.leaf_nlist();
            const auto start = Clock::now();
            const MaintenanceReport report = common.maintain_with_policy(false, true, true, true);
            const GlobalCenterRefreshReport refresh = common.refresh_all_centroids_to_means();
            const double elapsed = seconds(start);
            const auto sizes = common.leaf_sizes();
            const SizeSummary shape = summarize_sizes(sizes);
            const SizeCandidates after = count_candidates(common);
            preprocess << round + 1 << ',' << k_before << ',' << before.capacity << ',' << before.load << ','
                       << before.short_lists << ',' << report.load_split_partitions << ','
                       << report.capacity_split_partitions << ',' << report.merged_short_partitions << ','
                       << refresh.centers_updated << ',' << refresh.vectors_covered << ','
                       << common.leaf_nlist() << ',' << shape.minimum << ',' << shape.maximum << ','
                       << shape.mean << ',' << elapsed << '\n';
            preprocess.flush();
            std::cout << "PREPROCESS round=" << round + 1 << " K=" << common.leaf_nlist()
                      << " candidates(cap/load/short)=" << before.capacity << '/' << before.load << '/'
                      << before.short_lists << " actions(load/cap/merge)=" << report.load_split_partitions
                      << '/' << report.capacity_split_partitions << '/' << report.merged_short_partitions
                      << " remaining=" << after.capacity << '/' << after.load << '/'
                      << after.short_lists << " seconds=" << elapsed << std::endl;
            if (report.load_split_partitions == 0 && report.capacity_split_partitions == 0 &&
                report.merged_short_partitions == 0 && before.total_size_triggers() != 0) {
                manifest << "preprocessing_failure=positive size trigger count but maintenance made no structural progress at round "
                         << round + 1 << "\n";
                throw std::runtime_error("size preprocessing stalled; thresholds were not relaxed");
            }
        }
        if (common_start_path.empty() && !stable && count_candidates(common).total_size_triggers() == 0)
            stable = true;
        if (!common_start_path.empty())
            stable = count_candidates(common).total_size_triggers() == 0 &&
                     count_candidates(common).prd == 0;
        require(stable, "size preprocessing did not reach a stable fixed point within 100 rounds");
        require(common.size() == kRows, "preprocessing changed total live vector count");

        if (common_start_path.empty()) common.reset_prd_references_for_diagnostic();
        const SizeCandidates common_candidates = count_candidates(common);
        require(common_candidates.capacity == 0 && common_candidates.load == 0 &&
                common_candidates.short_lists == 0 && common_candidates.prd == 0,
                "common start is not clean after the requested reference reset");
        const std::vector<std::size_t> common_sizes = common.leaf_sizes();
        const auto common_health = common.partition_health();
        const SizeSummary common_shape = summarize_sizes(common_sizes);
        partition_health_csv << "leaf_id,size,reference_mean_residual,current_mean_residual,global_fit_drift,prd_marked\n"
                             << std::setprecision(12);
        for (const auto& health : common_health)
            partition_health_csv << health.list_id << ',' << health.vector_count << ','
                                 << health.reference_mean_residual << ',' << health.recent_mean_residual << ','
                                 << health.global_fit_drift << ',' << health.marked << '\n';
        if (common_start_path.empty()) common.save(output_dir + "/common_start_index.bin");
        auto common_groups = common.leaf_vector_ids();
        std::vector<std::size_t> common_owner(kRows, common.leaf_nlist());
        for (std::size_t leaf = 0; leaf < common_groups.size(); ++leaf)
            for (const std::uint32_t id : common_groups[leaf]) {
                require(id < kRows && common_owner[id] == common.leaf_nlist(),
                        "common-start vector IDs are not unique base row IDs");
                common_owner[id] = leaf;
            }

        std::vector<std::size_t> eligible;
        const double sqrt_n = std::sqrt(static_cast<double>(kRows));
        const double average = static_cast<double>(kRows) / common.leaf_nlist();
        const double minimum_target_size = std::max(0.75 * average, 0.4 * sqrt_n + 64.0);
        for (std::size_t leaf = 0; leaf < common_groups.size(); ++leaf)
            if (common_groups[leaf].size() >= minimum_target_size)
                eligible.push_back(leaf);
        std::mt19937 target_rng(kPartitionSeed);
        std::shuffle(eligible.begin(), eligible.end(), target_rng);
        const std::size_t max_target_count = *std::max_element(target_counts.begin(), target_counts.end());
        if (eligible.size() < max_target_count) {
            manifest << "preprocessing_K=" << common.leaf_nlist() << "\n"
                     << "preprocessing_size_min=" << common_shape.minimum << "\n"
                     << "preprocessing_size_max=" << common_shape.maximum << "\n"
                     << "eligible_target_partitions=" << eligible.size() << "\n"
                     << "eligible_size_cutoff=" << minimum_target_size << "\n"
                     << "status=stopped before perturbation; eligible partition count is below the requested maximum; selection cutoff not relaxed\n";
            throw std::runtime_error("not enough size-eligible target partitions; did not relax selection cutoff");
        }

        target_csv << "selection_order,leaf_id,size,eligibility_cutoff\n";
        for (std::size_t i = 0; i < max_target_count; ++i)
            target_csv << i + 1 << ',' << eligible[i] << ',' << common_sizes[eligible[i]] << ','
                       << minimum_target_size << '\n';
        manifest << "preprocessing_K=" << common.leaf_nlist() << "\npreprocessing_size_min="
                 << common_shape.minimum << "\npreprocessing_size_max=" << common_shape.maximum
                 << "\npreprocessing_size_mean=" << common_shape.mean
                 << "\ncommon_start_size_candidates=0\ncommon_start_prd_candidates=0\n"
                 << "target_eligibility_cutoff=" << minimum_target_size
                 << "\neligible_target_partitions=" << eligible.size()
                 << "\ncommon_start_snapshot="
                 << (common_start_path.empty() ? output_dir + "/common_start_index.bin" : common_start_path)
                 << "\nbuild_seconds=" << build_seconds << '\n';
        manifest.flush();

        updates_csv << "target_count,scale,selected_partition_count,selected_vectors,modified_vectors,modified_fraction,mean_accepted_scale,residual_sse_before,residual_sse_after_update,relative_residual_growth\n"
                    << std::setprecision(12);
        case_health_csv << "target_count,scale,leaf_id,size,reference_mean_residual,current_mean_residual,global_fit_drift,prd_marked,is_target_leaf\n"
                        << std::setprecision(12);
        summary << "target_count,scale,method,N,K_before,K_after,selected_vectors,modified_vectors,modified_fraction,mean_accepted_scale,capacity_candidates_before,load_candidates_before,short_candidates_before,prd_candidates_before,target_prd_candidates_before,load_splits,capacity_splits,prd_splits,prd_split_failures,merged_short,local_reassigned_vectors,centers_refreshed,capacity_candidates_after,load_candidates_after,short_candidates_after,global_sse_before,global_sse_after_update,global_sse_after_maintenance,global_relative_sse_growth,target_sse_before,target_sse_after_update,target_sse_after_maintenance,target_relative_sse_growth,update_seconds,maintenance_seconds,budget,recall10,affected_queries,affected_recall10,mean_total_distances,mean_query_ms\n"
                 << std::setprecision(12);

        std::vector<std::string> scales{"1.00", "1.10", "1.25", "1.50"};
        for (std::size_t target_index = 0; target_index < target_counts.size(); ++target_index) {
            const std::size_t target_count = target_counts[target_index];
            std::cout << "TARGET_BATCH_BEGIN count=" << target_count << std::endl;
            for (std::size_t scale_index = 0; scale_index < scales.size(); ++scale_index) {
                const std::string scale_label = scales[scale_index];
                const double requested_scale = std::stod(scale_label);
                std::vector<std::size_t> target_leaves(eligible.begin(), eligible.begin() + target_count);
                std::vector<std::vector<std::size_t>> target_groups(target_count);
                std::vector<std::size_t> selected_rows;
                for (std::size_t i = 0; i < target_count; ++i) {
                    auto& group = target_groups[i];
                    group.reserve(common_groups[target_leaves[i]].size());
                    for (const std::uint32_t id : common_groups[target_leaves[i]])
                        group.push_back(static_cast<std::size_t>(id));
                    std::sort(group.begin(), group.end());
                    selected_rows.insert(selected_rows.end(), group.begin(), group.end());
                }
                std::sort(selected_rows.begin(), selected_rows.end());
                require(std::adjacent_find(selected_rows.begin(), selected_rows.end()) == selected_rows.end(),
                        "target partition selection contains duplicate rows");

                const auto planned = plan_expansion(base, selected_rows, common_owner,
                    common.leaf_centroids(), common.leaf_nlist(), requested_scale);
                const CaseStats case_stats = summarize_plan(planned);
                require(case_stats.selected_vectors == selected_rows.size(), "selected vector count mismatch");

                std::vector<std::size_t> changed_rows;
                std::vector<std::int64_t> remove_ids;
                std::vector<float> replacements;
                std::vector<std::int64_t> row_to_id(kRows);
                std::iota(row_to_id.begin(), row_to_id.end(), std::int64_t{0});
                std::vector<std::size_t> id_to_row(kRows + case_stats.modified_vectors);
                std::iota(id_to_row.begin(), id_to_row.begin() + kRows, std::size_t{0});
                std::vector<float> current_data = base;
                std::vector<double> group_modified(target_count, 0.0), group_scale_sum(target_count, 0.0);
                std::vector<double> group_before(target_count, 0.0), group_update(target_count, 0.0);
                std::vector<std::size_t> group_size_after_update(target_count, 0);
                std::vector<std::size_t> row_to_group(kRows, target_count);
                for (std::size_t group = 0; group < target_count; ++group)
                    for (const std::size_t row : target_groups[group]) row_to_group[row] = group;

                for (const auto& plan : planned) {
                    const std::size_t group = row_to_group[plan.row];
                    group_before[group] += plan.old_residual;
                    group_update[group] += plan.changed ? plan.new_residual : plan.old_residual;
                    ++group_size_after_update[group];
                    if (!plan.changed) continue;
                    changed_rows.push_back(plan.row);
                    remove_ids.push_back(static_cast<std::int64_t>(plan.row));
                    replacements.insert(replacements.end(), plan.value.begin(), plan.value.end());
                    std::copy(plan.value.begin(), plan.value.end(),
                              current_data.begin() + plan.row * kDimension);
                    group_modified[group] += 1.0;
                    group_scale_sum[group] += plan.used_scale;
                }
                require(changed_rows.size() == case_stats.modified_vectors,
                        "planned modified vector count mismatch");
                for (std::size_t i = 0; i < changed_rows.size(); ++i) {
                    const std::size_t row = changed_rows[i];
                    const std::int64_t new_id = static_cast<std::int64_t>(kRows + i);
                    row_to_id[row] = new_id;
                    id_to_row[kRows + i] = row;
                }

                AdaptiveHierarchicalIvf updated = common;
                double update_seconds = 0.0;
                if (!changed_rows.empty()) {
                    updated.begin_update_batch();
                    const auto update_start = Clock::now();
                    const std::size_t removed = updated.remove_ids(remove_ids);
                    require(removed == remove_ids.size(), "normal delete did not remove every selected old ID");
                    updated.add(replacements.data(), changed_rows.size());
                    (void)updated.end_update_batch();
                    update_seconds = seconds(update_start);
                }
                require(updated.size() == kRows, "controlled update changed total vector count");
                require(updated.leaf_sizes() == common_sizes,
                        "normal insertion did not preserve every partition size before maintenance");
                const auto update_owners = row_owners(updated, row_to_id, id_to_row);
                for (std::size_t row = 0; row < kRows; ++row)
                    require(update_owners[row] == common_owner[row],
                            "post-update vector ownership differs from common-start ownership");

                const SizeCandidates pre = count_candidates(updated);
                require(pre.capacity == 0 && pre.load == 0 && pre.short_lists == 0,
                        "a size-based predicate became active after controlled update");
                std::size_t target_prd = 0;
                std::set<std::size_t> target_leaf_set(target_leaves.begin(), target_leaves.end());
                const auto pre_health = updated.partition_health();
                for (const auto& health : pre_health) {
                    if (health.marked && target_leaf_set.count(health.list_id)) ++target_prd;
                    case_health_csv << target_count << ',' << scale_label << ',' << health.list_id << ','
                                    << health.vector_count << ',' << health.reference_mean_residual << ','
                                    << health.recent_mean_residual << ',' << health.global_fit_drift << ','
                                    << health.marked << ',' << target_leaf_set.count(health.list_id) << '\n';
                }
                case_health_csv.flush();
                for (std::size_t group = 0; group < target_count; ++group)
                    require(group_size_after_update[group] == common_sizes[target_leaves[group]],
                            "target partition size changed during update transaction");

                auto truth_start = Clock::now();
                const auto truth = exact_top10(current_data, row_to_id, queries, 16);
                const double truth_seconds = seconds(truth_start);
                std::unordered_set<std::int64_t> changed_id_set;
                changed_id_set.reserve(changed_rows.size());
                for (const std::size_t row : changed_rows) changed_id_set.insert(row_to_id[row]);

                const double global_sse_before = common.total_sse();
                const double global_sse_update = updated.total_sse();
                std::vector<std::int64_t> common_row_to_id(kRows);
                std::iota(common_row_to_id.begin(), common_row_to_id.end(), std::int64_t{0});
                std::vector<std::size_t> common_id_to_row(kRows);
                std::iota(common_id_to_row.begin(), common_id_to_row.end(), std::size_t{0});
                const double target_sse_before = selected_residual_sse(
                    common, base, target_groups, common_row_to_id, common_id_to_row);
                const double target_sse_update = selected_residual_sse(
                    updated, current_data, target_groups, row_to_id, id_to_row);

                std::cout << "CASE_READY targets=" << target_count << " scale=" << scale_label
                          << " selected=" << selected_rows.size() << " modified=" << changed_rows.size()
                          << " size_candidates(cap/load/short)=" << pre.capacity << '/' << pre.load << '/'
                          << pre.short_lists << " PRD=" << pre.prd << " target_PRD=" << target_prd
                          << " update_s=" << update_seconds << " exact_truth_s=" << truth_seconds << std::endl;

                for (const bool enable_prd : {true, false}) {
                    const std::string method_name = enable_prd ? "SCIM" : "SCIM-Geom";
                    AdaptiveHierarchicalIvf method = updated;
                    const auto maintenance_start = Clock::now();
                    const MaintenanceReport report = method.maintain_with_policy(
                        enable_prd, true, true, true);
                    const GlobalCenterRefreshReport refresh = method.refresh_all_centroids_to_means();
                    const double maintenance_seconds = seconds(maintenance_start);
                    require(method.size() == kRows, "maintenance changed total active vector count");
                    const SizeCandidates post = count_candidates(method);
                    require(report.load_split_partitions == 0 &&
                            report.capacity_split_partitions == 0 &&
                            report.merged_short_partitions == 0,
                            "size-based maintenance action fired in controlled comparison");
                    if (!enable_prd)
                        require(report.prd_split_partitions == 0 &&
                                method.leaf_nlist() == common.leaf_nlist(),
                                "SCIM-Geom changed geometry despite PRD being disabled");
                    const double target_sse_after = selected_residual_sse(
                        method, current_data, target_groups, row_to_id, id_to_row);
                    std::vector<double> group_after(target_count, 0.0);
                    const auto final_owners = row_owners(method, row_to_id, id_to_row);
                    const auto final_centers = method.leaf_centroids();
                    for (std::size_t group = 0; group < target_count; ++group)
                        for (const std::size_t row : target_groups[group])
                            group_after[group] += simd_squared_l2(
                                current_data.data() + row * kDimension,
                                final_centers.data() + final_owners[row] * kDimension,
                                kDimension);

                    std::vector<std::size_t> modified_by_group(target_count, 0);
                    std::vector<double> used_scale_sum(target_count, 0.0);
                    for (const auto& plan : planned) if (plan.changed) {
                        const std::size_t group = row_to_group[plan.row];
                        ++modified_by_group[group];
                        used_scale_sum[group] += plan.used_scale;
                    }
                    append_partition_diagnostics(output_dir, scale_label, method_name, target_count,
                        target_leaves, target_groups, common_sizes, modified_by_group,
                        used_scale_sum, group_before, group_update, group_after,
                        common_sizes);

                    std::vector<std::size_t> budgets;
                    for (const std::size_t divisor : budget_divisors) {
                        require(divisor <= kRows, "query budget divisor must not exceed N");
                        budgets.push_back(kRows / divisor);
                    }
                    for (const std::size_t budget : budgets) {
                        const QueryMetrics metrics = evaluate(method, queries, truth,
                                                              changed_id_set, budget);
                        const double global_sse_after = method.total_sse();
                        const double global_growth = global_sse_before > 0.0
                            ? (global_sse_update - global_sse_before) / global_sse_before : 0.0;
                        const double target_growth = target_sse_before > 0.0
                            ? (target_sse_update - target_sse_before) / target_sse_before : 0.0;
                        summary << target_count << ',' << scale_label << ',' << method_name << ','
                                << kRows << ',' << common.leaf_nlist() << ',' << method.leaf_nlist() << ','
                                << selected_rows.size() << ',' << changed_rows.size() << ','
                                << static_cast<double>(changed_rows.size()) /
                                   std::max<std::size_t>(1, selected_rows.size()) << ','
                                << case_stats.mean_used_scale << ',' << pre.capacity << ',' << pre.load << ','
                                << pre.short_lists << ',' << pre.prd << ',' << target_prd << ','
                                << report.load_split_partitions << ',' << report.capacity_split_partitions << ','
                                << report.prd_split_partitions << ',' << report.prd_split_failures << ','
                                << report.merged_short_partitions << ','
                                << report.local_refinement_reassigned_vectors << ','
                                << refresh.centers_updated << ',' << post.capacity << ',' << post.load << ','
                                << post.short_lists << ',' << global_sse_before << ',' << global_sse_update << ','
                                << global_sse_after << ',' << global_growth << ',' << target_sse_before << ','
                                << target_sse_update << ',' << target_sse_after << ',' << target_growth << ','
                                << update_seconds << ',' << maintenance_seconds << ',' << budget << ',' << metrics.recall << ','
                                << metrics.affected_queries << ',' << metrics.affected_recall << ','
                                << metrics.mean_distances << ',' << metrics.mean_latency_ms << '\n';
                        summary.flush();
                        std::cout << "RESULT targets=" << target_count << " scale=" << scale_label
                                  << " method=" << method_name << " budget=" << budget
                                  << " recall10=" << metrics.recall << " affected_q="
                                  << metrics.affected_queries << " affected_recall10="
                                  << metrics.affected_recall << " PRD_splits="
                                  << report.prd_split_partitions << " K=" << method.leaf_nlist() << std::endl;
                    }
                }
                updates_csv << target_count << ',' << scale_label << ',' << target_count << ','
                            << case_stats.selected_vectors << ',' << case_stats.modified_vectors << ','
                            << static_cast<double>(case_stats.modified_vectors) /
                               std::max<std::size_t>(1, case_stats.selected_vectors) << ','
                            << case_stats.mean_used_scale << ',' << case_stats.residual_before << ','
                            << case_stats.residual_after_update << ','
                            << (case_stats.residual_before > 0.0
                                ? (case_stats.residual_after_update - case_stats.residual_before) /
                                  case_stats.residual_before : 0.0) << '\n';
                updates_csv.flush();
            }
            if (target_index + 1 < target_counts.size())
                std::cout << "TARGET_BATCH_VALID count=" << target_count
                          << "; proceeding to " << target_counts[target_index + 1]
                          << " targets from the same common start" << std::endl;
        }

        std::ofstream complete(output_dir + "/COMPLETE.txt");
        complete << "Completed Deep1M size-preserving PRD control for target counts ";
        for (std::size_t i = 0; i < target_counts.size(); ++i) {
            if (i) complete << ',';
            complete << target_counts[i];
        }
        complete << " and scales 1.00, 1.10, 1.25, 1.50.\n"
                 << "Every case was independently copied from the shared preprocessed starting index.\n"
                 << "All normal-routing, ownership, per-list-size, and fixed-query-budget validations passed.\n";
        std::cout << "COMPLETE output=" << output_dir << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << std::endl;
        return 1;
    }
}
