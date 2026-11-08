#include "../adaptive_hierarchical_ivf.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {
constexpr std::size_t kDimension = 96;
constexpr std::size_t kRows = 1'000'000;
constexpr std::size_t kInitialRows = 500'000;
constexpr std::size_t kBatchRows = 5'000;
constexpr std::size_t kStages = 100;
constexpr std::size_t kQueries = 1'000;
constexpr std::size_t kTopK = 10;
constexpr std::size_t kInitialNlist = 512;
constexpr std::size_t kMaxPartitionSize = 1'000;
constexpr std::size_t kMinPartitionSize = 32;
constexpr std::size_t kRefinementRadius = 50;
constexpr std::size_t kRefinementIterations = 1;
constexpr std::uint32_t kSeed = 1234;
constexpr std::array<const char*, 9> kCases{"G", "A1", "A2", "B1", "B2", "B3", "B4", "C1", "C2"};
using Clock = std::chrono::steady_clock;
using TruthRow = std::array<std::uint32_t, kTopK>;

class MappedFbin {
public:
    explicit MappedFbin(const std::filesystem::path& path) {
#ifdef _WIN32
        file_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open fbin: " + path.string());
        LARGE_INTEGER length{};
        if (!GetFileSizeEx(file_, &length) || length.QuadPart < 8) {
            close(); throw std::runtime_error("cannot get fbin size: " + path.string());
        }
        bytes_ = static_cast<std::uint64_t>(length.QuadPart);
        mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_) { close(); throw std::runtime_error("cannot map fbin: " + path.string()); }
        mapped_ = static_cast<const std::uint8_t*>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
        if (!mapped_) { close(); throw std::runtime_error("cannot view fbin: " + path.string()); }
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open fbin: " + path.string());
        struct stat info {};
        if (::fstat(fd_, &info) != 0 || info.st_size < 8) {
            close(); throw std::runtime_error("cannot get fbin size: " + path.string());
        }
        bytes_ = static_cast<std::uint64_t>(info.st_size);
        mapped_ = static_cast<const std::uint8_t*>(::mmap(nullptr, static_cast<std::size_t>(bytes_),
                            PROT_READ, MAP_SHARED, fd_, 0));
        if (mapped_ == MAP_FAILED) { mapped_ = nullptr; close(); throw std::runtime_error("cannot view fbin: " + path.string()); }
#endif
        std::uint32_t header[2]{};
        std::memcpy(header, mapped_, sizeof(header));
        rows_ = header[0];
        dimension_ = header[1];
        if (dimension_ != kDimension || bytes_ != 8ULL + rows_ * dimension_ * sizeof(float)) {
            close(); throw std::runtime_error("fbin shape/size mismatch: " + path.string());
        }
    }
    ~MappedFbin() { close(); }
    MappedFbin(const MappedFbin&) = delete;
    MappedFbin& operator=(const MappedFbin&) = delete;
    const float* row(std::size_t id) const {
        if (id >= rows_) throw std::out_of_range("fbin row out of range");
        return reinterpret_cast<const float*>(mapped_ + 8) + id * dimension_;
    }
    const float* data() const { return reinterpret_cast<const float*>(mapped_ + 8); }
    std::size_t rows() const noexcept { return rows_; }
private:
    void close() noexcept {
#ifdef _WIN32
        if (mapped_) UnmapViewOfFile(mapped_);
        mapped_ = nullptr;
        if (mapping_) CloseHandle(mapping_);
        mapping_ = nullptr;
        if (file_ != INVALID_HANDLE_VALUE) CloseHandle(file_);
        file_ = INVALID_HANDLE_VALUE;
#else
        if (mapped_) ::munmap(const_cast<std::uint8_t*>(mapped_), static_cast<std::size_t>(bytes_));
        mapped_ = nullptr;
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
#endif
    }
    const std::uint8_t* mapped_ = nullptr;
    std::uint64_t bytes_ = 0;
    std::size_t rows_ = 0;
    std::size_t dimension_ = 0;
#ifdef _WIN32
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

std::vector<TruthRow> read_truth(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read staged truth: " + path.string());
    char magic[8]{};
    std::uint32_t shape[3]{};
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char*>(shape), sizeof(shape));
    if (!in || std::string(magic, sizeof(magic)) != "DLQTRTH1" ||
        shape[0] != kStages + 1 || shape[1] != kQueries || shape[2] != kTopK)
        throw std::runtime_error("staged truth header mismatch: " + path.string());
    std::vector<TruthRow> rows((kStages + 1) * kQueries);
    in.read(reinterpret_cast<char*>(rows.data()), static_cast<std::streamsize>(rows.size() * sizeof(TruthRow)));
    if (!in || in.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("staged truth payload mismatch: " + path.string());
    return rows;
}

const TruthRow& truth_at(const std::vector<TruthRow>& truth, std::size_t stage, std::size_t query) {
    return truth.at(stage * kQueries + query);
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto rank = static_cast<std::size_t>(std::ceil(p * values.size()));
    return values[std::min(values.size() - 1, std::max<std::size_t>(1, rank) - 1)];
}

struct SizeStats {
    std::size_t minimum = 0, median = 0, maximum = 0, empty = 0;
    double mean = 0.0;
};

SizeStats size_stats(const std::vector<std::size_t>& sizes) {
    SizeStats result;
    if (sizes.empty()) return result;
    std::vector<std::size_t> nonempty;
    nonempty.reserve(sizes.size());
    std::size_t total = 0;
    for (const auto value : sizes) {
        if (value == 0) ++result.empty;
        else { nonempty.push_back(value); total += value; }
    }
    if (nonempty.empty()) return result;
    std::sort(nonempty.begin(), nonempty.end());
    result.minimum = nonempty.front();
    result.median = nonempty[nonempty.size() / 2];
    result.maximum = nonempty.back();
    result.mean = static_cast<double>(total) / nonempty.size();
    return result;
}

struct QuerySummary {
    std::uint64_t hits = 0, top1_hits = 0, centers = 0, vectors = 0, total = 0, partitions = 0;
    double mean_ms = 0.0, p50_ms = 0.0, p95_ms = 0.0;
};

QuerySummary run_queries(const AdaptiveHierarchicalIvf& index, const MappedFbin& queries,
                         const std::vector<TruthRow>& truth, std::size_t stage,
                         std::size_t budget, std::size_t threads,
                         std::ofstream& per_query, const std::string& case_name) {
    std::vector<std::uint64_t> hits(kQueries, 0), top1_hits(kQueries, 0), centers(kQueries, 0), vectors(kQueries, 0),
                                total(kQueries, 0), partitions(kQueries, 0);
    std::vector<double> latency(kQueries, 0.0);
    std::exception_ptr failure;
    std::mutex failure_mutex;
    std::atomic<std::size_t> next{0};
    const std::size_t worker_count = std::min(threads, kQueries);
    auto worker = [&]() {
        for (;;) {
            const std::size_t qi = next.fetch_add(1, std::memory_order_relaxed);
            if (qi >= kQueries) break;
            try {
                QuerySearchOptions options;
                options.enable_early_stop = false;
                options.max_total_distance_computations = budget;
                QuerySearchStats stats;
                const auto start = Clock::now();
                const auto found = index.search_with_options(queries.row(qi), kTopK, options, &stats);
                latency[qi] = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                if (stats.early_stopped || stats.total_distance_computations != budget || found.size() != kTopK)
                    throw std::runtime_error("fixed N/100 query budget was not honored");
                std::uint64_t count = 0;
                for (const auto& candidate : found) {
                    for (const auto expected : truth_at(truth, stage, qi)) {
                        if (candidate.id == expected) { ++count; break; }
                    }
                }
                hits[qi] = count;
                top1_hits[qi] = !found.empty() && found.front().id == truth_at(truth, stage, qi)[0];
                centers[qi] = stats.center_distance_computations;
                vectors[qi] = stats.scanned_vectors;
                total[qi] = stats.total_distance_computations;
                partitions[qi] = stats.scanned_partitions;
            } catch (...) {
                std::lock_guard<std::mutex> lock(failure_mutex);
                if (!failure) failure = std::current_exception();
                break;
            }
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(worker_count);
    for (std::size_t i = 0; i < worker_count; ++i) pool.emplace_back(worker);
    for (auto& thread : pool) thread.join();
    if (failure) std::rethrow_exception(failure);

    QuerySummary result;
    std::vector<double> times = latency;
    for (std::size_t qi = 0; qi < kQueries; ++qi) {
        result.hits += hits[qi];
        result.top1_hits += top1_hits[qi];
        result.centers += centers[qi];
        result.vectors += vectors[qi];
        result.total += total[qi];
        result.partitions += partitions[qi];
        per_query << case_name << ',' << stage << ',' << qi << ',' << budget << ',' << centers[qi]
                  << ',' << vectors[qi] << ',' << total[qi] << ',' << partitions[qi] << ','
                  << hits[qi] << ',' << std::setprecision(7) << latency[qi] << '\n';
    }
    result.mean_ms = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    result.p50_ms = percentile(times, 0.50);
    result.p95_ms = percentile(times, 0.95);
    return result;
}

void run_case(const std::filesystem::path& data_root,
              const std::filesystem::path& truth_root,
              const std::filesystem::path& output_root,
              const char* case_name,
              const AdaptiveHierarchicalIvf& common_initial,
              double build_ms, double initial_maintenance_ms,
              const MaintenanceReport& initial_maintenance,
              std::ofstream& all_summary, std::size_t query_threads,
              std::size_t stage_limit) {
    const std::string name(case_name);
    const auto case_dir = data_root / name;
    MappedFbin data(case_dir / "base.fbin");
    if (data.rows() != kRows) throw std::runtime_error("unexpected Deep-1M row count in " + name);
    const auto truth = read_truth(truth_root / "_prepared_truth" / (name + ".u32bin"));
    const auto case_output = output_root / name;
    std::filesystem::create_directories(case_output);
    std::ofstream stages(case_output / "stage_summary.csv");
    std::ofstream queries_out(case_output / "per_query.csv");
    std::ofstream manifest(case_output / "manifest.txt");
    if (!stages || !queries_out || !manifest) throw std::runtime_error("cannot create LIRE result files");
    stages << "case,stage,active_vectors,inserted_vectors,nlist_before,nlist_after,min_list_size,median_list_size,mean_list_size,max_list_size,empty_partitions,build_ms,initial_maintenance_ms,insert_ms,maintenance_ms,update_ms,checked_partitions,marked_partitions,splits,split_failures,small_partitions_removed,small_vectors_routed,empty_partitions_removed,local_refinement_partitions,local_refinement_vectors,local_refinement_moved,recall_at_10,top1_hit_rate,query_mean_ms,query_p50_ms,query_p95_ms,mean_center_distance_computations,mean_vector_distance_computations,mean_total_distance_computations,mean_partitions_scanned,scan_budget,total_sse,mean_sse_per_vector\n";
    queries_out << "case,stage,query_id,scan_budget,center_distances,vector_distances,total_distances,partitions_scanned,recall_hits_at_10,query_ms\n";
    manifest << "method=Current LIRE policy in AdaptiveHierarchicalIvf::maintain_quake_lire_policy\n"
             << "dataset=DEEP-1M FreewayML nine scenarios; 96-D FP32 normalized vectors; squared L2\n"
             << "case=" << name << "\ninitial_vectors=500000; initial_nlist=512; seed=1234; kmeans_iterations=10\n"
             << "updates=100 x 5000 inserts; query at stage 0 and after every insertion batch\n"
             << "LIRE=max_partition_size=" << kMaxPartitionSize << "; min_partition_size=32; local_refinement_radius=50 (includes seeds); iterations=1\n"
             << "query_count=1000; query_threads=" << query_threads << "; budget=floor(current_active_vectors/100), including centroid distances\n"
             << "LVQ=false; early_stop=false; query stages use matching prepared exact top-10 truth\n"
             << "initial_index=one shared in-memory 500k/512 index copied into each scenario; no scenario index snapshots saved\n";

    AdaptiveHierarchicalIvf index = common_initial;
    if (index.size() != kInitialRows || index.leaf_nlist() != initial_maintenance.leaf_nlist_after)
        throw std::runtime_error("common initial LIRE index metadata mismatch");
    const auto write_stage = [&](std::size_t stage, std::size_t active, std::size_t inserted,
                                 const MaintenanceReport& maintenance, double insert_ms,
                                 double maintenance_ms, std::size_t nlist_before,
                                 const QuerySummary& query, const MappedFbin& query_data) {
        const auto sizes = size_stats(index.leaf_sizes());
        const std::size_t budget = active / 100;
        const double nq = static_cast<double>(kQueries);
        stages << name << ',' << stage << ',' << active << ',' << inserted << ',' << nlist_before << ','
               << index.leaf_nlist() << ',' << sizes.minimum << ',' << sizes.median << ','
               << std::fixed << std::setprecision(4) << sizes.mean << ',' << sizes.maximum << ',' << sizes.empty << ','
               << std::setprecision(3) << build_ms << ',' << initial_maintenance_ms << ',' << insert_ms << ','
               << maintenance_ms << ',' << insert_ms + maintenance_ms << ','
               << maintenance.checked_partitions << ',' << maintenance.marked_partitions << ','
               << maintenance.split_partitions << ',' << maintenance.recentered_partitions << ','
               << maintenance.deleted_small_partitions << ',' << maintenance.reassigned_small_partition_vectors << ','
               << maintenance.removed_empty_partitions << ',' << maintenance.local_refinement_partitions << ','
               << maintenance.local_refinement_vectors << ',' << maintenance.local_refinement_reassigned_vectors << ','
               << std::setprecision(7) << static_cast<double>(query.hits) / (nq * kTopK) << ','
               << static_cast<double>(query.top1_hits) / nq << ',' << query.mean_ms << ',' << query.p50_ms << ','
               << query.p95_ms << ',' << static_cast<double>(query.centers) / nq << ','
               << static_cast<double>(query.vectors) / nq << ',' << static_cast<double>(query.total) / nq << ','
               << static_cast<double>(query.partitions) / nq << ',' << budget << ','
               << index.total_sse() << ',' << index.mean_sse_per_vector() << '\n';
        stages.flush();
        (void)query_data;
    };

    double total_insert_ms = 0.0;
    double total_maintenance_ms = 0.0;
    std::uint64_t total_splits = initial_maintenance.split_partitions;
    std::uint64_t total_small_partitions_removed = initial_maintenance.deleted_small_partitions;
    std::uint64_t total_small_vectors_routed = initial_maintenance.reassigned_small_partition_vectors;
    double final_recall = 0.0;
    double final_query_mean_ms = 0.0;
    for (std::size_t stage = 0; stage <= stage_limit; ++stage) {
        std::size_t inserted = 0;
        double insert_ms = 0.0;
        double maintenance_ms = 0.0;
        const std::size_t nlist_before = stage == 0 ? initial_maintenance.leaf_nlist_before : index.leaf_nlist();
        MaintenanceReport maintenance;
        if (stage == 0) {
            maintenance = initial_maintenance;
        } else {
            const auto insert_start = Clock::now();
            index.begin_update_batch();
            index.add(data.row(kInitialRows + (stage - 1) * kBatchRows), kBatchRows);
            const auto end_report = index.end_update_batch();
            insert_ms = std::chrono::duration<double, std::milli>(Clock::now() - insert_start).count();
            if (end_report.vector_count != kInitialRows + stage * kBatchRows)
                throw std::runtime_error("LIRE insertion batch changed the wrong number of active vectors");
            inserted = kBatchRows;
            const auto maintenance_start = Clock::now();
            maintenance = index.maintain_quake_lire_policy(kMaxPartitionSize, kMinPartitionSize,
                                                            kRefinementRadius, kRefinementIterations);
            maintenance_ms = std::chrono::duration<double, std::milli>(Clock::now() - maintenance_start).count();
        }
        total_insert_ms += insert_ms;
        total_maintenance_ms += maintenance_ms;
        total_splits += stage == 0 ? 0 : maintenance.split_partitions;
        total_small_partitions_removed += stage == 0 ? 0 : maintenance.deleted_small_partitions;
        total_small_vectors_routed += stage == 0 ? 0 : maintenance.reassigned_small_partition_vectors;
        const std::size_t active = kInitialRows + stage * kBatchRows;
        if (index.size() != active) throw std::runtime_error("LIRE active population mismatch in " + name);
        std::ostringstream query_name;
        query_name << "query_" << std::setw(3) << std::setfill('0') << stage << ".fbin";
        MappedFbin query_data(case_dir / query_name.str());
        if (query_data.rows() != kQueries) throw std::runtime_error("query row count mismatch: " + query_name.str());
        const std::size_t budget = active / 100;
        const QuerySummary query = run_queries(index, query_data, truth, stage, budget,
                                                query_threads, queries_out, name);
        if (stage == stage_limit) {
            final_recall = static_cast<double>(query.hits) / (kQueries * kTopK);
            final_query_mean_ms = query.mean_ms;
        }
        write_stage(stage, active, inserted, maintenance, insert_ms, maintenance_ms,
                    nlist_before, query, query_data);
        std::cout << "LIRE case=" << name << " stage=" << std::setw(3) << std::setfill('0') << stage
                  << '/' << kStages << " nlist=" << index.leaf_nlist()
                  << " split=" << maintenance.split_partitions
                  << " short_merge=" << maintenance.deleted_small_partitions
                  << " reassigned=" << maintenance.reassigned_small_partition_vectors
                  << " recall@10=" << std::fixed << std::setprecision(4)
                  << static_cast<double>(query.hits) / (kQueries * kTopK)
                  << " query_ms=" << std::setprecision(2) << query.mean_ms
                  << " update_ms=" << insert_ms + maintenance_ms << '\n' << std::flush;
    }
    const auto final_sizes = size_stats(index.leaf_sizes());
    all_summary << name << ',' << index.leaf_nlist() << ',' << final_sizes.minimum << ','
                << final_sizes.median << ',' << std::fixed << std::setprecision(2) << final_sizes.mean << ','
                << final_sizes.maximum << ',' << std::setprecision(7) << final_recall << ','
                << std::setprecision(4) << final_query_mean_ms << ',' << total_insert_ms << ','
                << total_maintenance_ms << ',' << total_insert_ms + total_maintenance_ms << ','
                << total_splits << ',' << total_small_partitions_removed << ',' << total_small_vectors_routed << ','
                << std::setprecision(3) << build_ms << ',' << initial_maintenance_ms << ','
                << index.size() << ',' << index.total_sse() << ',' << index.mean_sse_per_vector() << '\n';
    all_summary.flush();
    manifest << "final_vectors=" << index.size() << "\nfinal_nlist=" << index.leaf_nlist()
             << "\nfinal_sse=" << index.total_sse() << "\nfinal_mean_sse_per_vector="
             << index.mean_sse_per_vector() << "\nindex_snapshots_saved=false\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 5) {
            std::cerr << "Usage: deep1m_current_lire_freewayml_runner DATA_ROOT TRUTH_ROOT OUTPUT_DIR [--smoke]\n";
            return 2;
        }
        const std::filesystem::path data_root(argv[1]), truth_root(argv[2]), output_root(argv[3]);
        const bool smoke = argc == 5 && std::string(argv[4]) == "--smoke";
        if (argc == 5 && !smoke) throw std::invalid_argument("optional mode must be --smoke");
        const std::size_t stage_limit = smoke ? 2 : kStages;
        if (!std::filesystem::is_directory(data_root) || !std::filesystem::is_directory(truth_root / "_prepared_truth"))
            throw std::runtime_error("Deep-1M scenario data or prepared truth directory is missing");
        if (std::filesystem::exists(output_root) &&
            (!std::filesystem::is_directory(output_root) ||
             std::filesystem::directory_iterator(output_root) != std::filesystem::directory_iterator()))
            throw std::runtime_error("output directory must be absent or empty: " + output_root.string());
        std::filesystem::create_directories(output_root);

        MappedFbin initial_base(data_root / "G" / "base.fbin");
        if (initial_base.rows() != kRows) throw std::runtime_error("G base must contain 1M rows");
        for (const char* case_name : kCases) {
            MappedFbin data(data_root / case_name / "base.fbin");
            if (data.rows() != kRows || std::memcmp(initial_base.data(), data.data(),
                    kInitialRows * kDimension * sizeof(float)) != 0)
                throw std::runtime_error(std::string("scenario initial 500k differs from shared starting data: ") + case_name);
            (void)read_truth(truth_root / "_prepared_truth" / (std::string(case_name) + ".u32bin"));
        }

        AdaptiveHierarchicalConfig config;
        config.dimension = kDimension;
        config.initial_leaf_nlist = kInitialNlist;
        config.kmeans_iterations = 10;
        config.seed = kSeed;
        config.automatic_maintenance = false;
        config.prd_threshold = 0.075f;
        config.drift_criterion = DriftCriterion::GlobalFitOnly;
        config.split_local_refinement_radius = 0;
        config.residual_tail_reassign_fraction = 0.0f;

        std::ofstream manifest(output_root / "manifest.txt");
        std::ofstream all_summary(output_root / "all_cases_summary.csv");
        std::ofstream progress(output_root / "progress.txt");
        if (!manifest || !all_summary || !progress) throw std::runtime_error("cannot create experiment summary files");
        manifest << "experiment=Current LIRE policy on Deep-1M FreewayML nine scenarios\n"
                 << "data_root=" << data_root.string() << "\ntruth_root=" << truth_root.string() << '\n'
                 << "cases=G,A1,A2,B1,B2,B3,B4,C1,C2\ninitial_vectors=500000; initial_nlist=512; seed=1234; kmeans_iterations=10\n"
                 << "updates=100 x 5000; query at initial stage and after each update\n"
                 << "LIRE=max_partition_size=" << kMaxPartitionSize << "; min_partition_size=32; local_refinement_radius=50; iterations=1\n"
                 << "query_count=1000 at every stage; budget=floor(current N/100), including center distances; query_threads=16\n"
                 << "query_metric=squared L2; source vectors are individually normalized FP32; LVQ=false; early_stop=false\n"
                 << "common_initial=trained once on G's first 500k, verified identical across all scenario prefixes; reused by value copy\n"
                 << "initial_LIRE_size_repair=applied once to shared initial index before scenario copies\n"
                 << "staged_truth=DLQTRTH1, 101 stages x 1000 queries x top10; all nine files verified before run\n"
                 << "run_mode=" << (smoke ? "smoke (G, stage 0-2)" : "full (all nine, stage 0-100)") << '\n'
                 << "index_snapshots_saved=false\n";
        all_summary << "case,final_nlist,min_list_size,median_list_size,mean_list_size,max_list_size,final_recall_at_10,final_query_mean_ms,total_insert_ms,total_maintenance_ms,total_update_ms,total_splits,total_small_partitions_removed,total_small_vectors_routed,shared_build_ms,initial_maintenance_ms,final_vectors,final_sse,final_mean_sse_per_vector\n";
        progress << "status=running\n";
        const auto build_start = Clock::now();
        AdaptiveHierarchicalIvf common_initial(config);
        common_initial.train(initial_base.data(), kInitialRows);
        const double build_ms = std::chrono::duration<double, std::milli>(Clock::now() - build_start).count();
        std::cout << "COMMON_INITIAL_READY vectors=" << common_initial.size()
                  << " nlist=" << common_initial.leaf_nlist() << " build_ms=" << std::fixed
                  << std::setprecision(1) << build_ms << '\n' << std::flush;
        const auto initial_maintenance_start = Clock::now();
        const MaintenanceReport initial_maintenance = common_initial.maintain_quake_lire_policy(
            kMaxPartitionSize, kMinPartitionSize, kRefinementRadius, kRefinementIterations);
        const double initial_maintenance_ms = std::chrono::duration<double, std::milli>(
            Clock::now() - initial_maintenance_start).count();
        if (common_initial.size() != kInitialRows) throw std::runtime_error("initial LIRE maintenance changed population");
        manifest << "shared_build_ms=" << build_ms << "\ninitial_maintenance_ms=" << initial_maintenance_ms
                 << "\ninitial_nlist_after_lire=" << common_initial.leaf_nlist()
                 << "\ninitial_splits=" << initial_maintenance.split_partitions
                 << "\ninitial_small_partitions_removed=" << initial_maintenance.deleted_small_partitions
                 << "\ninitial_small_vectors_routed=" << initial_maintenance.reassigned_small_partition_vectors << '\n';
        const std::size_t case_count = smoke ? 1 : kCases.size();
        for (std::size_t case_index = 0; case_index < case_count; ++case_index) {
            const char* case_name = kCases[case_index];
            progress << "begin=" << case_name << '\n'; progress.flush();
            std::cout << "BEGIN LIRE case=" << case_name << '\n' << std::flush;
            run_case(data_root, truth_root, output_root, case_name, common_initial,
                     build_ms, initial_maintenance_ms, initial_maintenance,
                     all_summary, 16, stage_limit);
            progress << "complete=" << case_name << '\n'; progress.flush();
        }
        progress << "status=complete\ncompleted_cases=" << case_count
                 << "\nstages_per_case=" << stage_limit + 1 << "\nqueries_per_stage=1000\nquery_threads=16\n";
        std::cout << (smoke ? "LIRE_SMOKE_COMPLETE output=" : "ALL_NINE_LIRE_CASES_COMPLETE output=")
                  << output_root.string() << '\n' << std::flush;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
