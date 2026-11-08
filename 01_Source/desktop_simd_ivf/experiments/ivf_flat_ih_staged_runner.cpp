#include "../adaptive_hierarchical_ivf.h"
#include "../ivf_flat.h"
#include "../ivf_flat_u8.h"

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
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::uint32_t kRows = 10'000'000;
constexpr std::uint32_t kInitialRows = 1'000'000;
constexpr std::uint32_t kBatchRows = 10'000;
constexpr std::uint32_t kInsertBatches = 900;
constexpr std::uint32_t kQueryEveryBatches = 9;
constexpr std::uint32_t kStages = 100;
constexpr std::uint32_t kQueries = 10'000;
constexpr std::uint32_t kTopK = 100;
constexpr std::uint32_t kNlist = 1'000;
constexpr std::uint32_t kThreads = 16;

struct DatasetSpec {
    std::string name;
    std::filesystem::path data;
    std::filesystem::path queries;
    std::filesystem::path truth;
    std::filesystem::path initial_index;
    std::uint32_t dimension;
    std::uint32_t budget_divisor;
    bool uint8;
};

struct Header {
    std::uint32_t rows = 0;
    std::uint32_t dimension = 0;
};

Header read_vector_header(std::ifstream& input, const std::filesystem::path& path,
                          std::size_t element_size) {
    Header header;
    input.read(reinterpret_cast<char*>(&header.rows), sizeof(header.rows));
    input.read(reinterpret_cast<char*>(&header.dimension), sizeof(header.dimension));
    if (!input || header.rows == 0 || header.dimension == 0)
        throw std::runtime_error("invalid vector-file header: " + path.string());
    const std::uintmax_t expected = 8ULL +
        static_cast<std::uintmax_t>(header.rows) * header.dimension * element_size;
    if (std::filesystem::file_size(path) != expected)
        throw std::runtime_error("vector-file size does not match header: " + path.string());
    return header;
}

void read_exact(std::ifstream& input, char* output, std::size_t bytes,
                const std::string& what) {
    input.read(output, static_cast<std::streamsize>(bytes));
    if (!input || static_cast<std::size_t>(input.gcount()) != bytes)
        throw std::runtime_error("truncated input while reading " + what);
}

std::vector<float> load_initial_centroids(const DatasetSpec& spec) {
    auto initial = AdaptiveHierarchicalIvf::load(spec.initial_index.string());
    const auto& cfg = initial.config();
    if (initial.size() != kInitialRows || initial.leaf_nlist() != kNlist ||
        cfg.dimension != spec.dimension || cfg.uint8_storage != spec.uint8 ||
        cfg.lvq8_storage || cfg.lvq4_storage) {
        throw std::runtime_error("saved initial index metadata mismatch for " + spec.name);
    }
    auto centroids = initial.leaf_centroids();
    if (centroids.size() != static_cast<std::size_t>(kNlist) * spec.dimension)
        throw std::runtime_error("saved initial centroid count mismatch for " + spec.name);
    return centroids;
}

void validate_truth(const DatasetSpec& spec) {
    std::ifstream input(spec.truth, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open staged truth: " + spec.truth.string());
    std::array<char, 8> magic{};
    std::array<std::uint32_t, 3> shape{};
    read_exact(input, magic.data(), magic.size(), "truth magic");
    read_exact(input, reinterpret_cast<char*>(shape.data()), sizeof(shape), "truth shape");
    if (std::string(magic.data(), magic.size()) != "MSTGTH01" ||
        shape[0] != kStages || shape[1] != kQueries || shape[2] != kTopK)
        throw std::runtime_error("unexpected staged truth layout for " + spec.name);
    const std::uintmax_t expected = 20ULL +
        static_cast<std::uintmax_t>(kStages) * kQueries * kTopK * sizeof(std::uint32_t);
    if (std::filesystem::file_size(spec.truth) != expected)
        throw std::runtime_error("staged truth file size mismatch for " + spec.name);
}

struct QueryBatchResult {
    double wall_seconds = 0.0;
    double mean_individual_ms = 0.0;
    double p95_individual_ms = 0.0;
    double mean_scanned_vectors = 0.0;
    double mean_scanned_lists = 0.0;
    double recall100 = 0.0;
};

template <typename Search>
QueryBatchResult run_queries(const DatasetSpec& spec, const std::vector<float>& queries,
                             const std::vector<std::uint32_t>& truth, std::size_t budget,
                             Search&& search, bool warmup) {
    const std::size_t qcount = kQueries;
    std::vector<std::int64_t> found(qcount * kTopK, -1);
    std::vector<double> per_query_ms(qcount, 0.0);
    std::vector<std::uint64_t> scanned_vectors(kThreads, 0), scanned_lists(kThreads, 0);
    const std::size_t worker_count = std::min<std::size_t>(kThreads, qcount);

    if (warmup) {
        const std::size_t warm_count = std::min<std::size_t>(64, qcount);
        std::vector<std::thread> warmers;
        warmers.reserve(worker_count);
        for (std::size_t worker = 0; worker < worker_count; ++worker) {
            const std::size_t begin = warm_count * worker / worker_count;
            const std::size_t end = warm_count * (worker + 1) / worker_count;
            warmers.emplace_back([&, begin, end] {
                for (std::size_t qi = begin; qi < end; ++qi)
                    (void)search(queries.data() + qi * spec.dimension, budget, nullptr);
            });
        }
        for (auto& thread : warmers) thread.join();
    }

    const auto wall_start = Clock::now();
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
        const std::size_t begin = qcount * worker / worker_count;
        const std::size_t end = qcount * (worker + 1) / worker_count;
        workers.emplace_back([&, worker, begin, end] {
            for (std::size_t qi = begin; qi < end; ++qi) {
                IvfBudgetSearchStats stats;
                const auto start = Clock::now();
                const auto results = search(queries.data() + qi * spec.dimension, budget, &stats);
                per_query_ms[qi] = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                scanned_vectors[worker] += stats.scanned_vectors;
                scanned_lists[worker] += stats.scanned_lists;
                const std::size_t offset = qi * kTopK;
                const std::size_t count = std::min<std::size_t>(kTopK, results.size());
                for (std::size_t rank = 0; rank < count; ++rank)
                    found[offset + rank] = results[rank].id;
            }
        });
    }
    for (auto& thread : workers) thread.join();
    const auto wall_end = Clock::now();

    std::uint64_t hits = 0;
    std::array<std::uint32_t, kTopK> sorted_ids{};
    for (std::size_t qi = 0; qi < qcount; ++qi) {
        std::size_t count = 0;
        for (std::size_t rank = 0; rank < kTopK; ++rank) {
            const auto id = found[qi * kTopK + rank];
            if (id >= 0) sorted_ids[count++] = static_cast<std::uint32_t>(id);
        }
        std::sort(sorted_ids.begin(), sorted_ids.begin() + count);
        for (std::size_t rank = 0; rank < kTopK; ++rank) {
            if (std::binary_search(sorted_ids.begin(), sorted_ids.begin() + count,
                                   truth[qi * kTopK + rank])) ++hits;
        }
    }
    std::sort(per_query_ms.begin(), per_query_ms.end());
    const std::size_t p95_index = static_cast<std::size_t>(std::ceil(0.95 * qcount)) - 1;
    std::uint64_t vectors_sum = 0, lists_sum = 0;
    for (std::size_t i = 0; i < worker_count; ++i) {
        vectors_sum += scanned_vectors[i];
        lists_sum += scanned_lists[i];
    }
    QueryBatchResult result;
    result.wall_seconds = std::chrono::duration<double>(wall_end - wall_start).count();
    result.mean_individual_ms = std::accumulate(per_query_ms.begin(), per_query_ms.end(), 0.0) / qcount;
    result.p95_individual_ms = per_query_ms[p95_index];
    result.mean_scanned_vectors = static_cast<double>(vectors_sum) / qcount;
    result.mean_scanned_lists = static_cast<double>(lists_sum) / qcount;
    result.recall100 = static_cast<double>(hits) / (qcount * kTopK);
    return result;
}

void write_manifest(const std::filesystem::path& path, const DatasetSpec& spec,
                    std::string status, double initial_import_seconds = 0.0,
                    double update_seconds = 0.0, double query_seconds = 0.0) {
    std::ofstream out(path, std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write manifest: " + path.string());
    out << "experiment=project-owned IVF-Flat insert-heavy full staged-query run\n"
        << "status=" << status << "\n"
        << "dataset=" << spec.name << "\nbase=" << spec.data.string()
        << "\nqueries=" << spec.queries.string() << "\ntruth=" << spec.truth.string()
        << "\ninitial_snapshot=" << spec.initial_index.string()
        << "\nsource_rows=" << kRows << ";dimension=" << spec.dimension
        << ";payload=" << (spec.uint8 ? "uint8" : "float32") << ";metric=squared_L2\n"
        << "initial_vectors=" << kInitialRows << ";initial_nlist=" << kNlist
        << ";insert_batches=" << kInsertBatches << ";batch_size=" << kBatchRows << "\n"
        << "query_every_insert_batches=" << kQueryEveryBatches << ";query_stages=" << kStages
        << ";queries_per_stage=" << kQueries << ";topk=" << kTopK << "\n"
        << "query_budget=floor(active_vectors/" << spec.budget_divisor
        << "), includes centroid and vector distances; query_threads=" << kThreads << "\n"
        << "index=custom IvfFlat" << (spec.uint8 ? "U8" : "")
        << ";initial codebook reused from saved PRD snapshot; raw initial vectors routed into IVF-Flat lists"
        << ";no retraining;no maintenance;no LVQ;no APS;no early stop\n"
        << std::setprecision(10)
        << "initial_import_seconds=" << initial_import_seconds << "\n"
        << "cumulative_incremental_add_seconds=" << update_seconds << "\n"
        << "cumulative_staged_query_wall_seconds=" << query_seconds << "\n"
        << "final_index_saved=false\n";
}

double add_vectors(IvfFlat& index, const float* data, std::size_t count) {
    const auto start = Clock::now();
    index.add(data, count);
    return std::chrono::duration<double>(Clock::now() - start).count();
}
double add_vectors(IvfFlatU8& index, const std::uint8_t* data, std::size_t count) {
    const auto start = Clock::now();
    index.add(data, count);
    return std::chrono::duration<double>(Clock::now() - start).count();
}
std::vector<SearchResult> search_budget(const IvfFlat& index, const float* query,
                                        std::size_t budget, IvfBudgetSearchStats* stats) {
    return index.search_with_distance_budget(query, kTopK, budget, stats);
}
std::vector<SearchResult> search_budget(const IvfFlatU8& index, const float* query,
                                        std::size_t budget, IvfBudgetSearchStats* stats) {
    return index.search_with_distance_budget(query, kTopK, budget, stats);
}

template <typename Index, typename Scalar>
double import_rows(Index& index, std::ifstream& data, std::size_t dimension,
                   std::size_t rows, std::vector<Scalar>& buffer) {
    buffer.resize(kBatchRows * dimension);
    double seconds = 0.0;
    std::size_t remaining = rows;
    while (remaining > 0) {
        const std::size_t count = std::min<std::size_t>(kBatchRows, remaining);
        read_exact(data, reinterpret_cast<char*>(buffer.data()), count * dimension * sizeof(Scalar),
                   "initial dataset rows");
        seconds += add_vectors(index, buffer.data(), count);
        remaining -= count;
    }
    return seconds;
}

double read_stage_truth(std::ifstream& truth_file, std::size_t stage,
                        std::vector<std::uint32_t>& truth) {
    const std::uint64_t offset = 20ULL +
        static_cast<std::uint64_t>(stage - 1) * kQueries * kTopK * sizeof(std::uint32_t);
    truth_file.clear();
    truth_file.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!truth_file) throw std::runtime_error("cannot seek staged truth");
    read_exact(truth_file, reinterpret_cast<char*>(truth.data()),
               truth.size() * sizeof(std::uint32_t), "stage truth");
    return 0.0;
}

template <typename Index, typename Scalar>
void run_dataset(const DatasetSpec& spec, const std::filesystem::path& output_root) {
    const auto output = output_root / spec.name;
    std::filesystem::create_directories(output);
    write_manifest(output / "manifest.txt", spec, "running");

    std::ifstream data(spec.data, std::ios::binary);
    if (!data) throw std::runtime_error("cannot open dataset: " + spec.data.string());
    const Header data_header = read_vector_header(data, spec.data, sizeof(Scalar));
    if (data_header.rows != kRows || data_header.dimension != spec.dimension)
        throw std::runtime_error("unexpected base shape for " + spec.name);

    std::ifstream query_file(spec.queries, std::ios::binary);
    if (!query_file) throw std::runtime_error("cannot open queries: " + spec.queries.string());
    const Header query_header = read_vector_header(query_file, spec.queries, sizeof(Scalar));
    if (query_header.rows < kQueries || query_header.dimension != spec.dimension)
        throw std::runtime_error("unexpected query shape for " + spec.name);
    std::vector<float> queries(static_cast<std::size_t>(kQueries) * spec.dimension);
    if constexpr (std::is_same_v<Scalar, float>) {
        read_exact(query_file, reinterpret_cast<char*>(queries.data()),
                   queries.size() * sizeof(float), "query vectors");
    } else {
        std::vector<std::uint8_t> raw_queries(queries.size());
        read_exact(query_file, reinterpret_cast<char*>(raw_queries.data()), raw_queries.size(),
                   "query vectors");
        std::transform(raw_queries.begin(), raw_queries.end(), queries.begin(),
                       [](std::uint8_t value) { return static_cast<float>(value); });
    }

    std::ifstream truth_file(spec.truth, std::ios::binary);
    if (!truth_file) throw std::runtime_error("cannot open truth: " + spec.truth.string());
    std::vector<std::uint32_t> truth(static_cast<std::size_t>(kQueries) * kTopK);
    std::vector<float> centroids = load_initial_centroids(spec);
    Index index(spec.dimension, kNlist, 1234);
    index.train_with_centroids(centroids.data());
    centroids.clear();
    centroids.shrink_to_fit();

    const auto initial_start = Clock::now();
    std::vector<Scalar> buffer;
    const double initial_import_seconds = import_rows(index, data, spec.dimension, kInitialRows, buffer);
    const double initial_wall = std::chrono::duration<double>(Clock::now() - initial_start).count();
    if (index.size() != kInitialRows) throw std::runtime_error("initial import count mismatch");
    std::cout << spec.name << ": initial IVF-Flat ready; reused saved centroids; import="
              << std::fixed << std::setprecision(1) << initial_import_seconds << "s\n" << std::flush;

    std::ofstream insert_csv(output / "insert_batches.csv", std::ios::trunc);
    std::ofstream stage_csv(output / "stage_summary.csv", std::ios::trunc);
    if (!insert_csv || !stage_csv) throw std::runtime_error("cannot create result CSVs");
    insert_csv << "batch,active_vectors,source_read_ms,ivf_add_ms,cumulative_ivf_add_seconds\n";
    stage_csv << "stage,insert_batches_completed,active_vectors,query_count,budget_divisor,distance_budget_per_query,nlist,query_threads,recall_at_100,query_batch_seconds,wall_mean_query_ms,mean_individual_query_ms,p95_individual_query_ms,mean_scanned_vectors,mean_scanned_lists,cumulative_incremental_add_seconds\n";
    insert_csv.flush();
    stage_csv.flush();

    std::vector<Scalar> batch(kBatchRows * spec.dimension);
    double cumulative_add_seconds = 0.0;
    double cumulative_query_seconds = 0.0;
    std::size_t completed_stages = 0;
    for (std::size_t batch_index = 0; batch_index < kInsertBatches; ++batch_index) {
        const auto read_start = Clock::now();
        read_exact(data, reinterpret_cast<char*>(batch.data()), batch.size() * sizeof(Scalar),
                   "incremental dataset batch");
        const double read_seconds = std::chrono::duration<double>(Clock::now() - read_start).count();
        const double add_seconds = add_vectors(index, batch.data(), kBatchRows);
        cumulative_add_seconds += add_seconds;
        const std::size_t active = kInitialRows + (batch_index + 1) * kBatchRows;
        insert_csv << (batch_index + 1) << ',' << active << ',' << std::setprecision(8)
                   << read_seconds * 1000.0 << ',' << add_seconds * 1000.0 << ','
                   << cumulative_add_seconds << '\n';
        insert_csv.flush();

        const std::size_t batches_done = batch_index + 1;
        if (batches_done % kQueryEveryBatches != 0) continue;
        const std::size_t stage = batches_done / kQueryEveryBatches;
        read_stage_truth(truth_file, stage, truth);
        const std::size_t budget = active / spec.budget_divisor;
        if (budget <= kNlist)
            throw std::runtime_error("query budget must exceed centroid count");
        const auto start = Clock::now();
        QueryBatchResult result;
        if (stage == 1) {
            result = run_queries(spec, queries, truth, budget,
                [&index](const float* q, std::size_t b, IvfBudgetSearchStats* s) {
                    return search_budget(index, q, b, s);
                }, true);
        } else {
            result = run_queries(spec, queries, truth, budget,
                [&index](const float* q, std::size_t b, IvfBudgetSearchStats* s) {
                    return search_budget(index, q, b, s);
                }, false);
        }
        // The batch timer is already captured inside run_queries; use its exact wall value.
        cumulative_query_seconds += result.wall_seconds;
        stage_csv << stage << ',' << batches_done << ',' << active << ',' << kQueries << ','
                  << spec.budget_divisor << ',' << budget << ',' << kNlist << ',' << kThreads << ','
                  << std::setprecision(10) << result.recall100 << ',' << result.wall_seconds << ','
                  << result.wall_seconds * 1000.0 / kQueries << ',' << result.mean_individual_ms << ','
                  << result.p95_individual_ms << ',' << result.mean_scanned_vectors << ','
                  << result.mean_scanned_lists << ',' << cumulative_add_seconds << '\n';
        stage_csv.flush();
        ++completed_stages;
        if (stage % 10 == 0) {
            std::cout << spec.name << ": stage " << stage << '/' << kStages
                      << " active=" << active << " R@100=" << std::fixed << std::setprecision(5)
                      << result.recall100 << " query=" << std::setprecision(1)
                      << result.wall_seconds << "s mean=" << std::setprecision(3)
                      << result.wall_seconds * 1000.0 / kQueries << "ms/q, add_total="
                      << cumulative_add_seconds << "s\n" << std::flush;
        }
    }
    insert_csv.close();
    stage_csv.close();
    if (index.size() != kRows || completed_stages != kStages)
        throw std::runtime_error("run finished with incomplete vectors or query stages");

    std::ifstream stages(output / "stage_summary.csv");
    std::string line;
    std::getline(stages, line);
    double recall_sum = 0.0, final_recall = 0.0;
    std::size_t rows = 0;
    while (std::getline(stages, line)) {
        if (line.empty()) continue;
        const auto last_comma = line.find_last_of(',');
        (void)last_comma;
        // Recall@100 is field 9; CSV has no string fields and no embedded commas.
        std::array<std::string, 16> fields{};
        std::size_t field = 0, begin = 0;
        for (std::size_t i = 0; i <= line.size() && field < fields.size(); ++i) {
            if (i == line.size() || line[i] == ',') {
                fields[field++] = line.substr(begin, i - begin);
                begin = i + 1;
            }
        }
        if (field != 16) throw std::runtime_error("malformed stage summary row");
        final_recall = std::stod(fields[8]);
        recall_sum += final_recall;
        ++rows;
    }
    if (rows != kStages) throw std::runtime_error("stage summary row count mismatch");
    const double mean_recall = recall_sum / rows;
    std::ofstream summary(output / "summary.csv", std::ios::trunc);
    summary << "dataset,method,nlist,initial_vectors,final_vectors,insert_batches,query_stages,queries_per_stage,query_budget_divisor,query_threads,mean_recall_at_100,final_recall_at_100,initial_import_add_seconds,incremental_add_seconds,maintenance_seconds,total_staged_query_seconds\n"
            << spec.name << ",project IVF-Flat," << kNlist << ',' << kInitialRows << ',' << kRows
            << ',' << kInsertBatches << ',' << kStages << ',' << kQueries << ',' << spec.budget_divisor
            << ',' << kThreads << ',' << std::setprecision(10) << mean_recall << ',' << final_recall
            << ',' << initial_import_seconds << ',' << cumulative_add_seconds << ",0," << cumulative_query_seconds << '\n';
    summary.close();
    write_manifest(output / "manifest.txt", spec, "complete", initial_import_seconds,
                   cumulative_add_seconds, cumulative_query_seconds);
    std::ofstream complete(output / "COMPLETE.txt", std::ios::trunc);
    complete << "status=complete\nstages=100\n";
    std::cout << "COMPLETE " << spec.name << ": mean/final R@100="
              << std::fixed << std::setprecision(5) << mean_recall << '/' << final_recall
              << ", insert=" << std::setprecision(1) << cumulative_add_seconds
              << "s, query=" << cumulative_query_seconds << "s, maintenance=0s\n" << std::flush;
    (void)initial_wall;
}

std::vector<DatasetSpec> specs() {
    return {
        {"MSTuring10M",
         R"(E:\MSTuring10M\shuffled_input_seed1234_20261025\msturing-10M-shuffled-seed1234.fbin)",
         R"(C:\Users\29188\Downloads\testQuery10K.fbin)",
         R"(E:\MSTuring10M\experiment_results\msturing10m_triad_quake_lire_onepass_full_ih_seed1234_20261026\groundtruth_stage_top100.bin)",
         R"(E:\MSTuring10M\experiment_results\triad_ablation_final_aps09_seed1234_20261026\initial_1m_nlist1000_seed1234_triand.index.bin)",
         100, 20, false},
        {"YFCC10M",
         R"(C:\Users\29188\Downloads\YFCC10m\base.10M.u8bin)",
         R"(D:\ivf_code_bundle_portable\build\experiment_data\yfcc10m_first10k_official_queries_staged_truth_20261030\query.first10k.official.u8bin)",
         R"(D:\ivf_code_bundle_portable\build\experiment_data\yfcc10m_first10k_official_queries_staged_truth_20261030\groundtruth_stage_top100.bin)",
         R"(D:\ivf_code_bundle_portable\build\experiment_data\yfcc10m_triad_quake_lire_first10k_n100_seed1234_20261030\shared_initial_1m_nlist1000.index.bin)",
         192, 100, true},
        {"Deep10M",
         R"(C:\Users\29188\Downloads\Deep10M_2\base.10M.fbin)",
         R"(C:\Users\29188\Downloads\Deep10M_2\query.public.10K.fbin)",
         R"(E:\Deep10M\deep10m_triad_quake_lire_100stages_n20_lire1000_seed1234_20261028\groundtruth_stage_top100.bin)",
         R"(D:\ivf_code_bundle_portable\build\experiment_data\deep10m_triad_quake_lire_final_n20_20261028\initial_1m_nlist1000.index.bin)",
         96, 100, false},
    };
}

template <typename Index, typename Scalar>
void run_insert_only(const DatasetSpec& spec, const std::filesystem::path& output) {
    if (spec.uint8 != std::is_same_v<Scalar, std::uint8_t>)
        throw std::runtime_error("dataset payload type does not match insert-only index type");
    if (std::filesystem::exists(output / "COMPLETE.txt"))
        throw std::runtime_error("refusing to overwrite a completed insert-only result");
    std::filesystem::create_directories(output);
    double initial_add_seconds = 0.0;
    double initial_elapsed_seconds = 0.0;
    double cumulative_add_seconds = 0.0;
    double cumulative_read_seconds = 0.0;
    auto save_manifest = [&](const std::string& status) {
        std::ofstream manifest(output / "manifest.txt", std::ios::trunc);
        if (!manifest) throw std::runtime_error("cannot write YFCC insert-only manifest");
        manifest << "experiment=project-owned IVF-Flat insert-only run\n"
                 << "status=" << status << "\ndataset=" << spec.name << "\nbase=" << spec.data.string()
                 << "\ninitial_snapshot=" << spec.initial_index.string()
                 << "\nrows=10000000\ndimension=" << spec.dimension << "\nmetric=squared_L2\n"
                 << "storage=" << (spec.uint8 ? "source and indexed vectors remain uint8" : "source and indexed vectors remain float32")
                 << "; centroids are float32\n"
                 << "initial_vectors=1000000\ninitial_nlist=1000\nseed=1234\n"
                 << "reused_initial_centroids=true; no retraining\n"
                 << "incremental_batches=900\nbatch_size=10000\nqueries_executed=0\nmaintenance=none\n"
                 << "insertion=one nearest-centroid assignment per vector, then append to its list\n"
                 << std::setprecision(10)
                 << "initial_import_add_seconds=" << initial_add_seconds << "\n"
                 << "initial_import_elapsed_seconds=" << initial_elapsed_seconds << "\n"
                 << "incremental_add_seconds=" << cumulative_add_seconds << "\n"
                 << "incremental_source_read_seconds=" << cumulative_read_seconds << "\n"
                 << "incremental_read_plus_add_seconds="
                 << cumulative_add_seconds + cumulative_read_seconds << "\n";
    };
    save_manifest("running");

    std::ifstream data(spec.data, std::ios::binary);
    if (!data) throw std::runtime_error("cannot open dataset: " + spec.data.string());
    const Header header = read_vector_header(data, spec.data, sizeof(Scalar));
    if (header.rows != kRows || header.dimension != spec.dimension)
        throw std::runtime_error("unexpected base shape for insert-only dataset");

    auto centroids = load_initial_centroids(spec);
    Index index(spec.dimension, kNlist, 1234);
    index.train_with_centroids(centroids.data());
    centroids.clear();
    centroids.shrink_to_fit();

    std::vector<Scalar> buffer;
    const auto initial_start = Clock::now();
    initial_add_seconds = import_rows(index, data, spec.dimension, kInitialRows, buffer);
    initial_elapsed_seconds =
        std::chrono::duration<double>(Clock::now() - initial_start).count();
    if (index.size() != kInitialRows)
        throw std::runtime_error("initial import count mismatch");

    std::ofstream batches(output / "insert_batches.csv", std::ios::trunc);
    if (!batches) throw std::runtime_error("cannot create YFCC insert-batch CSV");
    batches << "batch,active_vectors,source_read_ms,ivf_add_ms,cumulative_ivf_add_seconds,cumulative_read_plus_add_seconds\n";
    batches.flush();
    buffer.resize(static_cast<std::size_t>(kBatchRows) * spec.dimension);
    for (std::size_t batch_index = 0; batch_index < kInsertBatches; ++batch_index) {
        const auto read_start = Clock::now();
        read_exact(data, reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(Scalar),
                   "incremental dataset batch");
        const double read_seconds =
            std::chrono::duration<double>(Clock::now() - read_start).count();
        const double add_seconds = add_vectors(index, buffer.data(), kBatchRows);
        cumulative_read_seconds += read_seconds;
        cumulative_add_seconds += add_seconds;
        const std::size_t active = kInitialRows + (batch_index + 1) * kBatchRows;
        batches << (batch_index + 1) << ',' << active << ',' << std::setprecision(9)
                << read_seconds * 1000.0 << ',' << add_seconds * 1000.0 << ','
                << cumulative_add_seconds << ','
                << cumulative_read_seconds + cumulative_add_seconds << '\n';
        batches.flush();
        if ((batch_index + 1) % 100 == 0) {
            std::cout << spec.name << ": insert batch " << (batch_index + 1) << '/'
                      << kInsertBatches << " active=" << active
                      << " cumulative_add=" << std::fixed << std::setprecision(1)
                      << cumulative_add_seconds << "s\n" << std::flush;
        }
    }
    batches.close();
    if (index.size() != kRows)
        throw std::runtime_error("insert-only run ended with wrong vector count");

    std::ofstream summary(output / "summary.csv", std::ios::trunc);
    summary << "dataset,method,nlist,initial_vectors,final_vectors,insert_batches,batch_size,initial_import_add_seconds,initial_import_elapsed_seconds,incremental_add_seconds,incremental_source_read_seconds,incremental_read_plus_add_seconds,maintenance_seconds,queries_executed\n"
            << spec.name << ",project IVF-Flat " << (spec.uint8 ? "U8" : "FP32") << ','
            << kNlist << ',' << kInitialRows << ','
            << kRows << ',' << kInsertBatches << ',' << kBatchRows << ','
            << std::setprecision(10) << initial_add_seconds << ',' << initial_elapsed_seconds
            << ',' << cumulative_add_seconds << ',' << cumulative_read_seconds << ','
            << cumulative_add_seconds + cumulative_read_seconds << ",0,0\n";
    summary.close();
    save_manifest("complete");
    std::ofstream complete(output / "COMPLETE.txt", std::ios::trunc);
    complete << "status=complete\nmode=insert-only\nbatches=900\nqueries_executed=0\n";
    std::cout << "COMPLETE " << spec.name << " insert-only: 900 batches, add="
              << std::fixed << std::setprecision(1) << cumulative_add_seconds
              << "s; source_read=" << cumulative_read_seconds
              << "s; maintenance=0s; no queries\n" << std::flush;
}

void write_root_summary(const std::filesystem::path& root) {
    std::ofstream out(root / "summary.csv", std::ios::trunc);
    out << "dataset,method,nlist,initial_vectors,final_vectors,insert_batches,query_stages,queries_per_stage,query_budget_divisor,query_threads,mean_recall_at_100,final_recall_at_100,initial_import_add_seconds,incremental_add_seconds,maintenance_seconds,total_staged_query_seconds\n";
    for (const auto& spec : specs()) {
        const auto path = root / spec.name / "summary.csv";
        std::ifstream in(path);
        if (!in) throw std::runtime_error("completed dataset summary missing: " + path.string());
        std::string header, row;
        std::getline(in, header);
        std::getline(in, row);
        if (row.empty()) throw std::runtime_error("empty dataset summary: " + path.string());
        out << row << '\n';
    }
    std::ofstream complete(root / "COMPLETE.txt", std::ios::trunc);
    complete << "status=complete\ndatasets=MSTuring10M,YFCC10M,Deep10M\nmethod=project-owned IVF-Flat\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 4 && std::string(argv[1]) == "--insert-only") {
            const std::string dataset_name(argv[2]);
            const auto all = specs();
            const auto found = std::find_if(all.begin(), all.end(), [&](const DatasetSpec& spec) {
                return spec.name == dataset_name;
            });
            if (found == all.end()) throw std::runtime_error("unknown dataset: " + dataset_name);
            if (found->uint8) run_insert_only<IvfFlatU8, std::uint8_t>(*found, argv[3]);
            else run_insert_only<IvfFlat, float>(*found, argv[3]);
            return 0;
        }
        if (argc != 2) {
            std::cerr << "Usage: ivf_flat_ih_staged_runner OUTPUT_DIRECTORY\n"
                      << "   or: ivf_flat_ih_staged_runner --insert-only DATASET OUTPUT_DIRECTORY\n";
            return 2;
        }
        const std::filesystem::path output_root(argv[1]);
        const auto all = specs();
        // Check every input and saved initial codebook before starting any long run.
        for (const auto& spec : all) {
            if (!std::filesystem::is_regular_file(spec.data) ||
                !std::filesystem::is_regular_file(spec.queries) ||
                !std::filesystem::is_regular_file(spec.initial_index))
                throw std::runtime_error("missing data/query/initial-index input for " + spec.name);
            validate_truth(spec);
            std::ifstream input(spec.data, std::ios::binary);
            const auto header = read_vector_header(input, spec.data, spec.uint8 ? 1 : sizeof(float));
            if (header.rows != kRows || header.dimension != spec.dimension)
                throw std::runtime_error("base vector header mismatch for " + spec.name);
            const auto centers = load_initial_centroids(spec);
            if (centers.size() != static_cast<std::size_t>(kNlist) * spec.dimension)
                throw std::runtime_error("invalid initial codebook for " + spec.name);
            std::cout << "READY " << spec.name << " initial=1M nlist=1000, queries=10k x 100, budget=N/"
                      << spec.budget_divisor << ", threads=16\n" << std::flush;
        }
        std::filesystem::create_directories(output_root);
        for (const auto& spec : all) {
            const auto done = output_root / spec.name / "COMPLETE.txt";
            if (std::filesystem::exists(done)) {
                std::cout << "SKIP completed " << spec.name << '\n' << std::flush;
                continue;
            }
            if (spec.uint8) run_dataset<IvfFlatU8, std::uint8_t>(spec, output_root);
            else run_dataset<IvfFlat, float>(spec, output_root);
        }
        write_root_summary(output_root);
        std::cout << "ALL_COMPLETE " << output_root.string() << '\n' << std::flush;
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
