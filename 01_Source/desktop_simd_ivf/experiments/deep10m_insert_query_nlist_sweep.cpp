#include "../simd_l2.h"

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
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
constexpr std::size_t kDimension = 96;
constexpr std::size_t kBaseCount = 10000000;
constexpr std::size_t kQueryCount = 10000;
constexpr std::size_t kTopK = 10;
// The user-specified budget counts scanned base vectors only. Center routing
// work is reported separately and does not consume this budget.
constexpr std::size_t kVectorScanBudget = 100000;
constexpr std::size_t kTrainingIterations = 12;
constexpr std::size_t kInsertionTrials = 3;
constexpr std::size_t kInsertionBlock = 1000;
constexpr std::uint32_t kSeed = 1234;
constexpr std::array<std::size_t, 9> kNlists{
    100, 500, 1000, 2000, 3000, 4000, 5000, 8000, 10000};

struct Hit {
    float distance = 0.0f;
    std::uint32_t id = 0;
};

struct ListIds {
    std::vector<std::uint32_t> ids;
};

struct Config {
    std::size_t nlist = 0;
    std::size_t training_rows = 0;
    std::vector<float> centers;
    std::vector<ListIds> lists;
    double total_sse = 0.0;
    double recall_at_10 = 0.0;
    std::size_t scanned_vectors = 0;
    double insertion_mean_us = 0.0;
    double insertion_median_block_us = 0.0;
    double insertion_p95_block_us = 0.0;
    double insertion_stddev_block_us = 0.0;
    double query_mean_us = 0.0;
    double query_median_block_us = 0.0;
    double query_p95_block_us = 0.0;
    double query_stddev_block_us = 0.0;
};

template <typename T>
T read_scalar(std::ifstream& in, const std::string& what) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("truncated file while reading " + what);
    return value;
}

template <typename T>
void read_values(std::ifstream& in, T* destination, std::size_t count,
                 const std::string& what) {
    constexpr std::size_t kChunkBytes = 64 * 1024 * 1024;
    const std::size_t chunk_elements = std::max<std::size_t>(1, kChunkBytes / sizeof(T));
    while (count > 0) {
        const std::size_t chunk = std::min(count, chunk_elements);
        in.read(reinterpret_cast<char*>(destination),
                static_cast<std::streamsize>(chunk * sizeof(T)));
        if (!in) throw std::runtime_error("truncated file while reading " + what);
        destination += chunk;
        count -= chunk;
    }
}

template <typename Function>
void parallel_ranges(std::size_t count, std::size_t workers, Function function) {
    workers = std::max<std::size_t>(1, std::min(workers, count));
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (std::size_t worker = 0; worker < workers; ++worker) {
        const std::size_t begin = count * worker / workers;
        const std::size_t end = count * (worker + 1) / workers;
        threads.emplace_back([=, &function]() { function(begin, end, worker); });
    }
    for (auto& thread : threads) thread.join();
}

std::vector<float> read_base(const std::string& path) {
    const std::uint64_t expected = 8 + kBaseCount * kDimension * sizeof(float);
    if (std::filesystem::file_size(path) != expected)
        throw std::runtime_error("base must be Deep10M fbin with 10M x 96 values");
    std::vector<float> base(kBaseCount * kDimension);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open base vectors: " + path);
    const auto rows = read_scalar<std::int32_t>(in, "base row count");
    const auto dimension = read_scalar<std::int32_t>(in, "base dimension");
    if (rows != static_cast<std::int32_t>(kBaseCount) ||
        dimension != static_cast<std::int32_t>(kDimension))
        throw std::runtime_error("unexpected Deep10M base header");
    read_values(in, base.data(), base.size(), "base vectors");
    return base;
}

std::vector<float> read_fbin(const std::string& path, std::size_t expected_rows) {
    const std::uint64_t expected = 8 + expected_rows * kDimension * sizeof(float);
    if (std::filesystem::file_size(path) != expected)
        throw std::runtime_error("fbin size does not match the requested shape: " + path);
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open fbin vectors: " + path);
    const auto rows = read_scalar<std::int32_t>(in, "fbin row count");
    const auto dimension = read_scalar<std::int32_t>(in, "fbin dimension");
    if (rows != static_cast<std::int32_t>(expected_rows) ||
        dimension != static_cast<std::int32_t>(kDimension))
        throw std::runtime_error("unexpected fbin header: " + path);
    std::vector<float> values(expected_rows * kDimension);
    in.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!in || in.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error("truncated or oversized fbin payload: " + path);
    return values;
}

using TruthRow = std::array<std::int32_t, kTopK>;

std::vector<TruthRow> read_truth(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open ground truth: " + path);
    const auto rows = read_scalar<std::int32_t>(in, "ground-truth row count");
    const auto width = read_scalar<std::int32_t>(in, "ground-truth width");
    if (rows != static_cast<std::int32_t>(kQueryCount) || width != 100)
        throw std::runtime_error("ground truth must be Deep10M_2 10000 x 100 ibin");
    std::vector<TruthRow> truth(kQueryCount);
    for (std::size_t row = 0; row < kQueryCount; ++row) {
        std::array<std::int32_t, 100> ids{};
        in.read(reinterpret_cast<char*>(ids.data()),
                static_cast<std::streamsize>(ids.size() * sizeof(std::int32_t)));
        if (!in) throw std::runtime_error("truncated ground truth");
        std::copy_n(ids.begin(), kTopK, truth[row].begin());
    }
    if (in.peek() != std::ifstream::traits_type::eof())
        throw std::runtime_error("ground truth has unexpected trailing data");
    return truth;
}

std::size_t nearest_center(const float* point, const std::vector<float>& centers,
                           std::size_t nlist) {
    std::size_t best = 0;
    float best_distance = simd_squared_l2(point, centers.data(), kDimension);
    for (std::size_t center = 1; center < nlist; ++center) {
        const float distance = simd_squared_l2(
            point, centers.data() + center * kDimension, kDimension);
        if (distance < best_distance) {
            best = center;
            best_distance = distance;
        }
    }
    return best;
}

std::vector<float> train_centers(const std::vector<float>& sample, std::size_t rows,
                                 std::size_t nlist, std::size_t workers) {
    std::mt19937 rng(kSeed + static_cast<std::uint32_t>(nlist));
    std::vector<std::size_t> order(rows);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    std::vector<float> centers(nlist * kDimension);
    for (std::size_t c = 0; c < nlist; ++c)
        std::copy_n(sample.data() + order[c] * kDimension, kDimension,
                    centers.data() + c * kDimension);

    const std::size_t center_values = nlist * kDimension;
    std::vector<std::vector<float>> local_sums(
        workers, std::vector<float>(center_values));
    std::vector<std::vector<std::uint32_t>> local_counts(
        workers, std::vector<std::uint32_t>(nlist));
    std::vector<float> global_sums(center_values);
    std::vector<std::uint64_t> global_counts(nlist);

    for (std::size_t iteration = 0; iteration < kTrainingIterations; ++iteration) {
        for (auto& sums : local_sums) std::fill(sums.begin(), sums.end(), 0.0f);
        for (auto& counts : local_counts) std::fill(counts.begin(), counts.end(), 0);
        parallel_ranges(rows, workers, [&](std::size_t begin, std::size_t end,
                                           std::size_t worker) {
            auto& sums = local_sums[worker];
            auto& counts = local_counts[worker];
            for (std::size_t row = begin; row < end; ++row) {
                const float* point = sample.data() + row * kDimension;
                const std::size_t cluster = nearest_center(point, centers, nlist);
                ++counts[cluster];
                float* sum = sums.data() + cluster * kDimension;
                for (std::size_t d = 0; d < kDimension; ++d) sum[d] += point[d];
            }
        });

        std::fill(global_sums.begin(), global_sums.end(), 0.0f);
        std::fill(global_counts.begin(), global_counts.end(), 0);
        for (std::size_t worker = 0; worker < workers; ++worker) {
            for (std::size_t c = 0; c < nlist; ++c)
                global_counts[c] += local_counts[worker][c];
            for (std::size_t value = 0; value < center_values; ++value)
                global_sums[value] += local_sums[worker][value];
        }
        for (std::size_t c = 0; c < nlist; ++c) {
            if (global_counts[c] == 0) continue;
            const float inverse = 1.0f / static_cast<float>(global_counts[c]);
            float* center = centers.data() + c * kDimension;
            const float* sum = global_sums.data() + c * kDimension;
            for (std::size_t d = 0; d < kDimension; ++d) center[d] = sum[d] * inverse;
        }
        std::cout << "TRAIN_ITERATION nlist=" << nlist << " iteration="
                  << (iteration + 1) << '/' << kTrainingIterations << std::endl;
    }
    return centers;
}

Config build_config(std::size_t nlist, const std::vector<float>& base,
                    const std::vector<std::size_t>& shuffled_rows,
                    std::size_t workers) {
    Config config;
    config.nlist = nlist;
    config.training_rows = std::min<std::size_t>(
        kBaseCount, std::max<std::size_t>(20000, nlist * 8));
    std::vector<float> sample(config.training_rows * kDimension);
    for (std::size_t i = 0; i < config.training_rows; ++i)
        std::copy_n(base.data() + shuffled_rows[i] * kDimension, kDimension,
                    sample.data() + i * kDimension);

    std::cout << "TRAIN_BEGIN nlist=" << nlist
              << " sample=" << config.training_rows << std::endl;
    config.centers = train_centers(sample, config.training_rows, nlist, workers);
    std::vector<std::uint32_t> assignments(kBaseCount);
    std::vector<long double> worker_sse(workers, 0.0L);
    constexpr std::size_t kAssignmentBlock = 1'000'000;
    for (std::size_t block_begin = 0; block_begin < kBaseCount;
         block_begin += kAssignmentBlock) {
        const std::size_t block_count = std::min(kAssignmentBlock, kBaseCount - block_begin);
        std::vector<long double> block_sse(workers, 0.0L);
        parallel_ranges(block_count, workers, [&](std::size_t begin, std::size_t end,
                                                   std::size_t worker) {
            long double sse = 0.0L;
            for (std::size_t local = begin; local < end; ++local) {
                const std::size_t row = block_begin + local;
                const float* point = base.data() + row * kDimension;
                const std::size_t cluster = nearest_center(point, config.centers, nlist);
                assignments[row] = static_cast<std::uint32_t>(cluster);
                sse += simd_squared_l2(point,
                    config.centers.data() + cluster * kDimension, kDimension);
            }
            block_sse[worker] = sse;
        });
        for (std::size_t worker = 0; worker < workers; ++worker)
            worker_sse[worker] += block_sse[worker];
        std::cout << "ASSIGN_PROGRESS nlist=" << nlist << " rows="
                  << (block_begin + block_count) << '/' << kBaseCount << std::endl;
    }
    config.total_sse = static_cast<double>(
        std::accumulate(worker_sse.begin(), worker_sse.end(), 0.0L));

    config.lists.resize(nlist);
    std::vector<std::size_t> counts(nlist, 0);
    for (const auto cluster : assignments) ++counts[cluster];
    for (std::size_t c = 0; c < nlist; ++c)
        config.lists[c].ids.reserve(counts[c]);
    for (std::size_t row = 0; row < kBaseCount; ++row)
        config.lists[assignments[row]].ids.push_back(static_cast<std::uint32_t>(row));
    std::cout << "BUILD_END nlist=" << nlist
              << " mean_sse=" << config.total_sse / kBaseCount << std::endl;
    return config;
}

void insert_hit(std::array<Hit, kTopK>& best, std::size_t& size, Hit candidate) {
    std::size_t position = 0;
    while (position < size &&
           (best[position].distance < candidate.distance ||
            (best[position].distance == candidate.distance && best[position].id < candidate.id)))
        ++position;
    if (position >= kTopK) return;
    const std::size_t new_size = std::min(kTopK, size + 1);
    for (std::size_t i = new_size; i > position + 1; --i) best[i - 1] = best[i - 2];
    best[position] = candidate;
    size = new_size;
}

std::array<Hit, kTopK> search_one(const Config& config, const std::vector<float>& base,
                                 const float* query) {
    std::vector<std::pair<float, std::size_t>> ranked(config.nlist);
    for (std::size_t c = 0; c < config.nlist; ++c)
        ranked[c] = {simd_squared_l2(query,
            config.centers.data() + c * kDimension, kDimension), c};
    std::sort(ranked.begin(), ranked.end());

    std::array<Hit, kTopK> best{};
    std::size_t best_size = 0;
    std::size_t scanned = 0;
    for (const auto& item : ranked) {
        const auto& ids = config.lists[item.second].ids;
        const std::size_t take = std::min(ids.size(), kVectorScanBudget - scanned);
        for (std::size_t i = 0; i < take; ++i) {
            const std::uint32_t id = ids[i];
            const float distance = simd_squared_l2(
                base.data() + static_cast<std::size_t>(id) * kDimension,
                query, kDimension);
            insert_hit(best, best_size, Hit{distance, id});
        }
        scanned += take;
        if (scanned == kVectorScanBudget) break;
    }
    if (scanned != kVectorScanBudget || best_size != kTopK)
        throw std::runtime_error("query did not consume the configured vector budget");
    return best;
}

std::vector<double> measure_query_blocks(const Config& config,
                                         const std::vector<float>& base,
                                         const std::vector<float>& queries,
                                         std::size_t trial) {
    std::vector<std::uint32_t> order(kQueryCount);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(kSeed + static_cast<std::uint32_t>(trial * 991 + config.nlist));
    std::shuffle(order.begin(), order.end(), rng);

    volatile std::uint32_t warmup_sink = 0;
    for (std::size_t i = 0; i < 256; ++i) {
        const auto result = search_one(config, base,
            queries.data() + static_cast<std::size_t>(order[i]) * kDimension);
        warmup_sink += result[i % kTopK].id;
    }
    (void)warmup_sink;

    constexpr std::size_t kQueriesPerBlock = 100;
    std::vector<double> block_us;
    block_us.reserve(kQueryCount / kQueriesPerBlock);
    for (std::size_t begin = 0; begin < kQueryCount; begin += kQueriesPerBlock) {
        const auto start = std::chrono::steady_clock::now();
        volatile std::uint32_t result_sink = 0;
        for (std::size_t i = begin; i < begin + kQueriesPerBlock; ++i) {
            const auto query_id = order[i];
            const auto result = search_one(config, base,
                queries.data() + static_cast<std::size_t>(query_id) * kDimension);
            result_sink += result[i % kTopK].id;
        }
        const auto end = std::chrono::steady_clock::now();
        (void)result_sink;
        block_us.push_back(std::chrono::duration<double, std::micro>(end - start).count() /
                           static_cast<double>(kQueriesPerBlock));
        if (((begin / kQueriesPerBlock) + 1) % 20 == 0)
            std::cout << "QUERY_TIMING_PROGRESS nlist=" << config.nlist
                      << " trial=" << (trial + 1) << " queries="
                      << (begin + kQueriesPerBlock) << '/' << kQueryCount << std::endl;
    }
    return block_us;
}

void write_query_csv(const std::filesystem::path& path,
                     const std::vector<Config>& configs, std::size_t workers) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create query timing CSV");
    out << "dataset,nlist,dimension,base_vectors,query_count,top_k,vector_scan_budget,"
           "recall_at_10,query_mean_us,query_median_block_us,query_p95_block_us,"
           "query_stddev_block_us,query_trials,blocks_per_trial,training_rows,"
           "training_iterations,build_workers,query_threads,seed\n";
    out << std::fixed << std::setprecision(8);
    for (const auto& config : configs)
        out << "DEEP10M_2," << config.nlist << ',' << kDimension << ',' << kBaseCount << ','
            << kQueryCount << ',' << kTopK << ',' << kVectorScanBudget << ','
            << config.recall_at_10 << ',' << config.query_mean_us << ','
            << config.query_median_block_us << ',' << config.query_p95_block_us << ','
            << config.query_stddev_block_us << ',' << kInsertionTrials << ','
            << (kQueryCount / 100) << ',' << config.training_rows << ','
            << kTrainingIterations << ',' << workers << ",1," << kSeed << '\n';
    out.close();
    if (!out) throw std::runtime_error("failed writing query timing CSV");
}

void evaluate_queries(Config& config, const std::vector<float>& base,
                      const std::vector<float>& queries,
                      const std::vector<TruthRow>& truth, std::size_t workers) {
    std::vector<std::uint8_t> hit_flags(kQueryCount, 0);
    std::vector<std::uint32_t> scanned_per_query(kQueryCount, 0);
    constexpr std::size_t kRecallBlock = 500;
    for (std::size_t block_begin = 0; block_begin < kQueryCount; block_begin += kRecallBlock) {
        const std::size_t block_count = std::min(kRecallBlock, kQueryCount - block_begin);
        parallel_ranges(block_count, workers, [&](std::size_t begin, std::size_t end,
                                                   std::size_t) {
            std::vector<std::pair<float, std::size_t>> ranked(config.nlist);
            for (std::size_t local = begin; local < end; ++local) {
                const std::size_t q = block_begin + local;
                const float* query = queries.data() + q * kDimension;
                for (std::size_t c = 0; c < config.nlist; ++c)
                    ranked[c] = {simd_squared_l2(query,
                        config.centers.data() + c * kDimension, kDimension), c};
                std::sort(ranked.begin(), ranked.end());

                std::array<Hit, kTopK> best{};
                std::size_t best_size = 0;
                std::size_t scanned = 0;
                for (const auto& item : ranked) {
                    const auto& ids = config.lists[item.second].ids;
                    const std::size_t take = std::min(ids.size(), kVectorScanBudget - scanned);
                    for (std::size_t i = 0; i < take; ++i) {
                        const std::uint32_t id = ids[i];
                        const float distance = simd_squared_l2(
                            base.data() + static_cast<std::size_t>(id) * kDimension,
                            query, kDimension);
                        insert_hit(best, best_size, Hit{distance, id});
                    }
                    scanned += take;
                    if (scanned == kVectorScanBudget) break;
                }
                if (scanned != kVectorScanBudget || best_size != kTopK)
                    throw std::runtime_error("query did not consume the configured vector budget");
                scanned_per_query[q] = static_cast<std::uint32_t>(scanned);
                std::uint8_t hits = 0;
                for (std::size_t i = 0; i < best_size; ++i)
                    if (std::find(truth[q].begin(), truth[q].end(),
                                  static_cast<std::int32_t>(best[i].id)) != truth[q].end())
                        ++hits;
                hit_flags[q] = hits;
            }
        });
        std::cout << "RECALL_PROGRESS nlist=" << config.nlist << " queries="
                  << (block_begin + block_count) << '/' << kQueryCount << std::endl;
    }
    const std::uint64_t hits = std::accumulate(hit_flags.begin(), hit_flags.end(),
                                               std::uint64_t{0});
    config.recall_at_10 = static_cast<double>(hits) / (kQueryCount * kTopK);
    config.scanned_vectors = static_cast<std::size_t>(std::accumulate(
        scanned_per_query.begin(), scanned_per_query.end(), std::uint64_t{0}) / kQueryCount);
}

struct PackedList { std::vector<std::uint32_t> ids; };

struct PackedIndex {
    std::vector<PackedList> lists;
    // Model the incoming-vector append separately from the old base storage;
    // reserving its capacity keeps allocator growth out of per-insert timing.
    std::vector<float> incoming_vectors;
};

PackedIndex make_packed_index(const Config& config) {
    PackedIndex index;
    index.lists.resize(config.nlist);
    for (std::size_t c = 0; c < config.nlist; ++c) {
        index.lists[c].ids = config.lists[c].ids;
    }
    index.incoming_vectors.reserve(kQueryCount * kDimension);
    return index;
}

void insert_one(PackedIndex& index, const Config& config, const float* vector,
                std::uint32_t id) {
    const std::size_t cluster = nearest_center(vector, config.centers, config.nlist);
    PackedList& list = index.lists[cluster];
    list.ids.push_back(id);
    const std::size_t old_size = index.incoming_vectors.size();
    index.incoming_vectors.resize(old_size + kDimension);
    simd_copy_floats(index.incoming_vectors.data() + old_size, vector, kDimension);
}

std::vector<double> measure_insertions(const Config& config,
                                       const std::vector<float>& base,
                                       const std::vector<float>& queries,
                                       std::size_t trial) {
    PackedIndex index = make_packed_index(config);
    std::vector<std::uint32_t> order(kQueryCount);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(kSeed + static_cast<std::uint32_t>(trial * 991 + config.nlist));
    std::shuffle(order.begin(), order.end(), rng);

    // Warm the CPU and centroid data without changing posting-list capacities.
    volatile std::size_t warmup_sink = 0;
    for (std::size_t i = 0; i < 512; ++i)
        warmup_sink += nearest_center(queries.data() +
            static_cast<std::size_t>(order[i]) * kDimension, config.centers, config.nlist);
    (void)warmup_sink;

    std::vector<double> block_us;
    block_us.reserve(kQueryCount / kInsertionBlock);
    std::size_t cursor = 0;
    std::uint32_t next_id = static_cast<std::uint32_t>(kBaseCount);
    while (cursor < kQueryCount) {
        const auto start = std::chrono::steady_clock::now();
        for (std::size_t i = 0; i < kInsertionBlock; ++i) {
            const auto query_id = order[cursor + i];
            insert_one(index, config,
                queries.data() + static_cast<std::size_t>(query_id) * kDimension,
                next_id++);
        }
        const auto end = std::chrono::steady_clock::now();
        block_us.push_back(std::chrono::duration<double, std::micro>(end - start).count() /
                           static_cast<double>(kInsertionBlock));
        cursor += kInsertionBlock;
    }
    return block_us;
}

double percentile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());
    const std::size_t index = static_cast<std::size_t>(
        std::ceil(p * values.size())) - 1;
    return values[std::min(index, values.size() - 1)];
}

void write_csv(const std::filesystem::path& path, const std::vector<Config>& configs,
               std::size_t workers) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create output CSV");
    out << "dataset,nlist,dimension,base_vectors,query_count,top_k,vector_scan_budget,"
           "mean_scanned_vectors,recall_at_10,insertion_mean_us,insertion_median_block_us,"
           "insertion_p95_block_us,insertion_stddev_block_us,insertion_trials,blocks_per_trial,"
           "training_rows,training_iterations,threads,seed,mean_sse_per_vector\n";
    out << std::fixed << std::setprecision(8);
    for (const auto& config : configs) {
        out << "DEEP10M_2," << config.nlist << ',' << kDimension << ',' << kBaseCount << ','
            << kQueryCount << ',' << kTopK << ',' << kVectorScanBudget << ','
            << config.scanned_vectors << ',' << config.recall_at_10 << ','
            << config.insertion_mean_us << ',' << config.insertion_median_block_us << ','
            << config.insertion_p95_block_us << ',' << config.insertion_stddev_block_us << ','
            << kInsertionTrials << ',' << (kQueryCount / kInsertionBlock) << ','
            << config.training_rows << ',' << kTrainingIterations << ',' << workers << ','
            << kSeed << ',' << config.total_sse / kBaseCount << '\n';
    }
    out.close();
    if (!out) throw std::runtime_error("failed writing output CSV");
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 5 || argc > 7)
            throw std::runtime_error(
                "usage: runner BASE.fbin QUERIES.fbin GROUNDTRUTH.ibin OUTPUT_DIR [THREADS [all|query]]");
        const std::size_t workers = argc >= 6
            ? static_cast<std::size_t>(std::stoul(argv[5]))
            : std::min<std::size_t>(12, std::max(1u, std::thread::hardware_concurrency()));
        const std::string mode = argc == 7 ? argv[6] : "all";
        if (workers == 0) throw std::runtime_error("THREADS must be positive");
        if (mode != "all" && mode != "query")
            throw std::runtime_error("MODE must be 'all' or 'query'");

        const std::filesystem::path output_dir(argv[4]);
        std::filesystem::create_directories(output_dir);
        const auto base = read_base(argv[1]);
        const auto queries = read_fbin(argv[2], kQueryCount);
        const auto truth = read_truth(argv[3]);
        std::vector<std::size_t> shuffled_rows(kBaseCount);
        std::iota(shuffled_rows.begin(), shuffled_rows.end(), 0);
        std::mt19937 sample_rng(kSeed);
        std::shuffle(shuffled_rows.begin(), shuffled_rows.end(), sample_rng);

        std::cout << "DATA_READY base=" << kBaseCount << " queries=" << kQueryCount
                  << " dimension=" << kDimension << " workers=" << workers << std::endl;
        std::vector<Config> configs;
        configs.reserve(kNlists.size());
        for (const auto nlist : kNlists) {
            configs.push_back(build_config(nlist, base, shuffled_rows, workers));
            auto& config = configs.back();
            evaluate_queries(config, base, queries, truth, workers);
            std::cout << "QUERY_END nlist=" << nlist
                      << " recall_at_10=" << config.recall_at_10 << std::endl;
        }

        {
            constexpr std::size_t kQueriesPerBlock = 100;
            std::ofstream raw_blocks(output_dir / "query_timing_blocks.csv", std::ios::trunc);
            if (!raw_blocks) throw std::runtime_error("cannot create raw query timing CSV");
            raw_blocks << "nlist,trial,block,mean_query_us\n" << std::fixed << std::setprecision(8);
            std::vector<std::vector<double>> all_query_blocks(configs.size());
            std::vector<std::size_t> query_order(configs.size());
            std::iota(query_order.begin(), query_order.end(), 0);
            std::mt19937 query_order_rng(kSeed + 997);
            for (std::size_t trial = 0; trial < kInsertionTrials; ++trial) {
                std::shuffle(query_order.begin(), query_order.end(), query_order_rng);
                for (const auto ci : query_order) {
                    const auto blocks = measure_query_blocks(configs[ci], base, queries, trial);
                    auto& accumulated = all_query_blocks[ci];
                    accumulated.insert(accumulated.end(), blocks.begin(), blocks.end());
                    for (std::size_t block = 0; block < blocks.size(); ++block)
                        raw_blocks << configs[ci].nlist << ',' << (trial + 1) << ','
                                   << block << ',' << blocks[block] << '\n';
                    std::cout << "QUERY_TIMING nlist=" << configs[ci].nlist
                              << " trial=" << (trial + 1) << '/' << kInsertionTrials
                              << " median_block_us=" << percentile(blocks, 0.5) << std::endl;
                }
            }
            raw_blocks.close();
            if (!raw_blocks) throw std::runtime_error("failed writing raw query timing CSV");
            for (std::size_t i = 0; i < configs.size(); ++i) {
                auto blocks = all_query_blocks[i];
                const double mean = std::accumulate(blocks.begin(), blocks.end(), 0.0) /
                                    static_cast<double>(blocks.size());
                double variance = 0.0;
                for (double value : blocks) variance += (value - mean) * (value - mean);
                variance /= static_cast<double>(blocks.size());
                configs[i].query_mean_us = mean;
                configs[i].query_median_block_us = percentile(blocks, 0.5);
                configs[i].query_p95_block_us = percentile(blocks, 0.95);
                configs[i].query_stddev_block_us = std::sqrt(variance);
                std::cout << "QUERY_RESULT nlist=" << configs[i].nlist
                          << " query_mean_us=" << configs[i].query_mean_us
                          << " query_median_block_us=" << configs[i].query_median_block_us
                          << " recall_at_10=" << configs[i].recall_at_10 << std::endl;
            }
            write_query_csv(output_dir / "deep10m_query_cost_recall.csv", configs, workers);
            std::ofstream manifest(output_dir / "query_experiment_manifest.txt", std::ios::trunc);
            manifest << "dataset=Deep10M_2 standalone 10M base\nbase_vectors=10000000\ndimension=96\n"
                     << "queries=10000\nground_truth=first 10 IDs from local exact top-100 ibin\n"
                     << "nlist=100,500,1000,2000,3000,4000,5000,8000,10000\n"
                     << "query_budget=100000 scanned base vectors per query; center routing is excluded from this vector budget and reported separately\n"
                     << "query_latency=single-threaded complete search; includes center distances and ranking,"
                     << " vector scans, and top-10 maintenance; excludes ground-truth comparison\n"
                     << "query_trials=3; each trial runs all 10000 queries after 256 warmup queries\n"
                     << "query_blocks=100 queries per timing block; total 300 block means per nlist\n"
                     << "training_sample=max(20000,8*nlist)\ntraining_iterations="
                     << kTrainingIterations << "\nbuild_workers=" << workers
                     << "\nquery_threads=1\nseed=" << kSeed << "\n";
            if (!manifest) throw std::runtime_error("failed writing query experiment manifest");
            std::cout << "QUERY_EXPERIMENT_DONE output="
                      << (output_dir / "deep10m_query_cost_recall.csv").string() << std::endl;
        }

        std::vector<std::size_t> trial_order(configs.size());
        std::iota(trial_order.begin(), trial_order.end(), 0);
        std::mt19937 order_rng(kSeed + 77);
        std::shuffle(trial_order.begin(), trial_order.end(), order_rng);
        std::vector<std::vector<double>> all_blocks(configs.size());
        for (std::size_t trial = 0; trial < kInsertionTrials; ++trial) {
            std::shuffle(trial_order.begin(), trial_order.end(), order_rng);
            for (const auto ci : trial_order) {
                auto blocks = measure_insertions(configs[ci], base, queries, trial);
                auto& accumulated = all_blocks[ci];
                accumulated.insert(accumulated.end(), blocks.begin(), blocks.end());
                std::cout << "INSERT_TRIAL nlist=" << configs[ci].nlist
                          << " trial=" << (trial + 1) << '/' << kInsertionTrials
                          << " mean_us="
                          << std::accumulate(blocks.begin(), blocks.end(), 0.0) / blocks.size()
                          << std::endl;
            }
        }

        for (std::size_t i = 0; i < configs.size(); ++i) {
            auto blocks = all_blocks[i];
            const double mean = std::accumulate(blocks.begin(), blocks.end(), 0.0) /
                                static_cast<double>(blocks.size());
            double variance = 0.0;
            for (double value : blocks) variance += (value - mean) * (value - mean);
            variance /= static_cast<double>(blocks.size());
            configs[i].insertion_mean_us = mean;
            configs[i].insertion_median_block_us = percentile(blocks, 0.5);
            configs[i].insertion_p95_block_us = percentile(blocks, 0.95);
            configs[i].insertion_stddev_block_us = std::sqrt(variance);
            std::cout << "RESULT nlist=" << configs[i].nlist
                      << " insert_mean_us=" << configs[i].insertion_mean_us
                      << " insert_median_block_us=" << configs[i].insertion_median_block_us
                      << " recall_at_10=" << configs[i].recall_at_10 << std::endl;
        }
        write_csv(output_dir / "deep10m_insert_cost_recall.csv", configs, workers);
        std::ofstream manifest(output_dir / "experiment_manifest.txt", std::ios::trunc);
        manifest << "dataset=Deep10M_2 standalone 10M base\nbase_vectors=10000000\ndimension=96\n"
                 << "queries=10000\nground_truth=first 10 IDs from local exact top-100 ibin\n"
                 << "nlist=100,500,1000,2000,3000,4000,5000,8000,10000\n"
                 << "query_budget=100000 scanned base vectors per query; center routing is excluded from this vector budget and reported separately\n"
                 << "training_sample=max(20000,8*nlist)\ntraining_iterations="
                 << kTrainingIterations << "\ninsert_measurement=three fresh in-memory index replicas per nlist;"
                 << " each trial adds all 10000 query vectors one at a time (nearest-center search, posting-ID append, and payload copy to a preallocated incoming-vector array)\n"
                 << "reported_insert_mean=mean of 30 block means, each block has 1000 single-vector calls\n"
                 << "reported_insert_median_block=median of the 30 block means\nthreads=" << workers
                 << "\nseed=" << kSeed << "\n";
        if (!manifest) throw std::runtime_error("failed writing experiment manifest");
        std::cout << "EXPERIMENT_DONE output="
                  << (output_dir / "deep10m_insert_cost_recall.csv").string() << std::endl;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
