#include "../adaptive_hierarchical_ivf.h"
#include "../ivf_quantized_common.h"
#include "../simd_l2.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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
std::filesystem::path g_quake_latency_profile_path;
#ifdef MSTURING_IH_SMOKE
constexpr std::size_t kInitialRows = 20'000;
constexpr std::size_t kInitialNlist = 32;
constexpr std::size_t kInsertBatchRows = 1'000;
constexpr std::size_t kInsertOpsPerQuery = 1;
constexpr std::size_t kInsertOps = 2;
constexpr std::size_t kQueryOps = 2;
constexpr std::size_t kQueries = 1'000;
#elif defined(DEEP10M_100STAGE_IH)
constexpr std::size_t kInitialRows = 1'000'000;
constexpr std::size_t kInitialNlist = 1'000;
constexpr std::size_t kInsertBatchRows = 10'000;
constexpr std::size_t kInsertOpsPerQuery = 9;
constexpr std::size_t kInsertOps = 900;
constexpr std::size_t kQueryOps = 100;
constexpr std::size_t kQueries = 10'000;
#elif defined(DEEP1M_FREEWAYML_ABLATION)
constexpr std::size_t kInitialRows = 500'000;
constexpr std::size_t kInitialNlist = 512;
constexpr std::size_t kInsertBatchRows = 5'000;
constexpr std::size_t kInsertOpsPerQuery = 100;
constexpr std::size_t kInsertOps = 100;
constexpr std::size_t kQueryOps = 1;
constexpr std::size_t kQueries = 1'000;
#elif defined(YFCC10M_IH) && defined(YFCC10M_IH_10K_QUERIES)
constexpr std::size_t kInitialRows = 1'000'000;
constexpr std::size_t kInitialNlist = 1'000;
constexpr std::size_t kInsertBatchRows = 10'000;
constexpr std::size_t kInsertOpsPerQuery = 9;
constexpr std::size_t kInsertOps = 900;
constexpr std::size_t kQueryOps = 100;
constexpr std::size_t kQueries = 10'000;
#elif defined(YFCC10M_IH) || defined(YANDEX10M_IH)
constexpr std::size_t kInitialRows = 1'000'000;
constexpr std::size_t kInitialNlist = 1'000;
constexpr std::size_t kInsertBatchRows = 10'000;
constexpr std::size_t kInsertOpsPerQuery = 9;
constexpr std::size_t kInsertOps = 900;
constexpr std::size_t kQueryOps = 100;
constexpr std::size_t kQueries = 1'000;
#elif defined(MSTURING_IH_QUERY_1K)
constexpr std::size_t kInitialRows = 1'000'000;
constexpr std::size_t kInitialNlist = 1'000;
constexpr std::size_t kInsertBatchRows = 10'000;
constexpr std::size_t kInsertOpsPerQuery = 9;
constexpr std::size_t kInsertOps = 900;
constexpr std::size_t kQueryOps = 100;
constexpr std::size_t kQueries = 1'000;
#else
constexpr std::size_t kInitialRows = 1'000'000;
constexpr std::size_t kInitialNlist = 1'000;
constexpr std::size_t kInsertBatchRows = 10'000;
constexpr std::size_t kInsertOpsPerQuery = 9;
constexpr std::size_t kInsertOps = 900;
constexpr std::size_t kQueryOps = 100;
constexpr std::size_t kQueries = 10'000;
#endif
#ifdef YFCC10M_IH
constexpr std::size_t kDimension = 192;
#elif defined(YANDEX10M_IH)
constexpr std::size_t kDimension = 200;
#elif defined(DEEP1M_FREEWAYML_ABLATION) || defined(DEEP10M_100STAGE_IH)
constexpr std::size_t kDimension = 96;
#else
constexpr std::size_t kDimension = 100;
#endif
#ifdef YANDEX10M_IH
constexpr bool kMaximumInnerProduct = true;
constexpr const char* kMetricName = "maximum inner product";
#else
constexpr bool kMaximumInnerProduct = false;
constexpr const char* kMetricName = "squared L2";
#endif
#if defined(YFCC10M_IH)
constexpr const char* kQuakeProfileDataset = "YFCC10M";
constexpr const char* kQuakeProfileInput = "uint8";
#elif defined(DEEP10M_100STAGE_IH)
constexpr const char* kQuakeProfileDataset = "Deep10M";
constexpr const char* kQuakeProfileInput = "FP32";
#elif defined(MSTURING_QUAKE_PROFILE_DEFAULT)
constexpr const char* kQuakeProfileDataset = "MSTuring-10M";
constexpr const char* kQuakeProfileInput = "FP32";
#else
constexpr const char* kQuakeProfileDataset = "unspecified";
constexpr const char* kQuakeProfileInput = "FP32";
#endif
#ifdef DEEP1M_FREEWAYML_FIXED50K
constexpr std::size_t kTopK = 10;
#else
constexpr std::size_t kTopK = 100;
#endif
constexpr std::size_t kQueryWindow = 1'000;
constexpr std::size_t kKmeansIterations = 10;
constexpr float kApsCandidateFraction = 0.10f;
constexpr float kApsRecallTarget = 0.90f;
constexpr float kApsRecomputeThreshold = 0.001f;
constexpr float kApsPrdThreshold = 0.05f;
constexpr float kApsCapacityGrowthSplitFactor = 2.50f;
constexpr float kApsCapacitySqrtNSplitMultiplier = 1.2f;
constexpr float kApsCapacitySqrtNHighClusterSplitMultiplier = 2.0f;
constexpr std::size_t kLirMaxPosting = 1'000;
constexpr std::size_t kLirNeighborRange = 64;
constexpr double kQuakeAlpha = 0.9;
// Default Quake threshold used by the paper-style runner; selected experiments
// can override split and delete thresholds independently.
constexpr double kQuakeSplitThresholdNs = 250.0;
constexpr double kQuakeDeleteThresholdNs = 250.0;
constexpr std::size_t kQuakeMinPartitionSize = 32;
constexpr std::size_t kQuakeRefinementRadius = 50; // Includes the seed partition.
constexpr std::size_t kQuakeRefinementIterations = 1;
constexpr std::size_t kQuakeLatencyTopK = 10;
constexpr std::uint32_t kSeed = 1234;

class MappedFbin {
public:
    explicit MappedFbin(const std::filesystem::path& path) {
#ifdef _WIN32
        file_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open fbin: " + path.string());
        LARGE_INTEGER size{};
        if (!GetFileSizeEx(file_, &size) || size.QuadPart < 8) { close(); throw std::runtime_error("cannot get fbin size"); }
        bytes_ = static_cast<std::uint64_t>(size.QuadPart);
        mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_) { close(); throw std::runtime_error("cannot map fbin"); }
        mapped_ = static_cast<const std::uint8_t*>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
        if (!mapped_) { close(); throw std::runtime_error("cannot view fbin mapping"); }
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open fbin: " + path.string());
        struct stat st {};
        if (::fstat(fd_, &st) != 0 || st.st_size < 8) { close(); throw std::runtime_error("cannot get fbin size"); }
        bytes_ = static_cast<std::uint64_t>(st.st_size);
        mapped_ = static_cast<const std::uint8_t*>(::mmap(nullptr, static_cast<std::size_t>(bytes_), PROT_READ, MAP_SHARED, fd_, 0));
        if (mapped_ == MAP_FAILED) { mapped_ = nullptr; close(); throw std::runtime_error("cannot map fbin"); }
#endif
        std::uint32_t header[2]{};
        std::memcpy(header, mapped_, sizeof(header));
        rows_ = header[0];
        dimension_ = header[1];
        if (dimension_ != kDimension || bytes_ != 8ull + rows_ * dimension_ * sizeof(float)) {
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
    std::size_t rows_ = 0, dimension_ = 0;
#ifdef _WIN32
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

// Read the source YFCC vectors in-place. The mapped payload remains unsigned
// bytes on disk; no 10M-row FP32 side copy is created.
class MappedU8bin {
public:
    explicit MappedU8bin(const std::filesystem::path& path) {
#ifdef _WIN32
        file_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (file_ == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open u8bin: " + path.string());
        LARGE_INTEGER size{};
        if (!GetFileSizeEx(file_, &size) || size.QuadPart < 8) { close(); throw std::runtime_error("cannot get u8bin size"); }
        bytes_ = static_cast<std::uint64_t>(size.QuadPart);
        mapping_ = CreateFileMappingW(file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!mapping_) { close(); throw std::runtime_error("cannot map u8bin"); }
        mapped_ = static_cast<const std::uint8_t*>(MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, 0));
        if (!mapped_) { close(); throw std::runtime_error("cannot view u8bin mapping"); }
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open u8bin: " + path.string());
        struct stat st {};
        if (::fstat(fd_, &st) != 0 || st.st_size < 8) { close(); throw std::runtime_error("cannot get u8bin size"); }
        bytes_ = static_cast<std::uint64_t>(st.st_size);
        mapped_ = static_cast<const std::uint8_t*>(::mmap(nullptr, static_cast<std::size_t>(bytes_), PROT_READ, MAP_SHARED, fd_, 0));
        if (mapped_ == MAP_FAILED) { mapped_ = nullptr; close(); throw std::runtime_error("cannot map u8bin"); }
#endif
        std::uint32_t header[2]{};
        std::memcpy(header, mapped_, sizeof(header));
        rows_ = header[0]; dimension_ = header[1];
        if (dimension_ != kDimension || bytes_ != 8ull + rows_ * dimension_) {
            close(); throw std::runtime_error("u8bin shape/size mismatch: " + path.string());
        }
    }
    ~MappedU8bin() { close(); }
    MappedU8bin(const MappedU8bin&) = delete;
    MappedU8bin& operator=(const MappedU8bin&) = delete;
    const std::uint8_t* row(std::size_t id) const {
        if (id >= rows_) throw std::out_of_range("u8bin row out of range");
        return mapped_ + 8 + id * dimension_;
    }
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
    std::size_t rows_ = 0, dimension_ = 0;
#ifdef _WIN32
    HANDLE file_ = INVALID_HANDLE_VALUE;
    HANDLE mapping_ = nullptr;
#else
    int fd_ = -1;
#endif
};

class FloatQueryRows {
public:
    explicit FloatQueryRows(const MappedU8bin& source) : values_(source.rows() * kDimension) {
        for (std::size_t q = 0; q < source.rows(); ++q)
            for (std::size_t d = 0; d < kDimension; ++d)
                values_[q * kDimension + d] = static_cast<float>(source.row(q)[d]);
    }
    const float* row(std::size_t id) const { return values_.data() + id * kDimension; }
    std::size_t rows() const noexcept { return values_.size() / kDimension; }
private:
    std::vector<float> values_;
};

template <typename Data>
constexpr bool uses_u8_input_v = std::is_same_v<std::remove_cv_t<std::remove_pointer_t<
    decltype(std::declval<const Data&>().row(std::size_t{}))>>, std::uint8_t>;

using Ids = std::array<std::uint32_t, kTopK>;
struct QueryStats {
    std::uint64_t center = 0, vectors = 0, total = 0, partitions = 0;
    std::vector<double> ms;
};
struct StageStats {
    std::size_t active = 0, budget = 0, nlist = 0, maintenance_splits = 0;
    double mean_ms = 0.0, p50_ms = 0.0, p95_ms = 0.0;
    std::uint64_t center = 0, vectors = 0, total = 0, partitions = 0;
};

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const auto i = std::min(values.size() - 1,
        static_cast<std::size_t>(std::ceil(p * values.size()) - 1.0));
    return values[i];
}

struct OutputFiles {
    std::filesystem::path dir;
    std::ofstream stages, updates, queries, results, quake_decisions;
    explicit OutputFiles(const std::filesystem::path& root, const std::string& method,
                         std::size_t query_stages = kQueryOps,
                         std::size_t insert_ops_per_query = kInsertOpsPerQuery,
                         std::size_t fixed_query_budget = 0,
                         std::size_t query_budget_divisor = 20,
                         bool record_per_query = true) {
        dir = root / method;
        std::filesystem::create_directories(dir);
        stages.open(dir / "stage_summary.csv");
        updates.open(dir / "insert_operations.csv");
        if (record_per_query) queries.open(dir / "per_query.csv");
        results.open(dir / "results.bin", std::ios::binary);
        if (method == "quake") quake_decisions.open(dir / "maintenance_decisions.csv");
        std::ofstream manifest(dir / "manifest.txt");
        if (!stages || !updates || (record_per_query && !queries) || !results || !manifest ||
            (method == "quake" && !quake_decisions))
            throw std::runtime_error("cannot create method outputs");
        stages << "method,query_stage,dataset_prefix_vectors,index_stored_vectors,insert_ops_since_query,query_count,scan_budget_per_query,nlist,maintenance_splits,mean_query_ms,p50_query_ms,p95_query_ms,mean_center_distances,mean_vector_distances,mean_total_distances,mean_partitions_scanned,recall_at_100\n";
        updates << "method,insert_op,query_stage,dataset_prefix_vectors,index_stored_vectors,inserted_vectors,insert_ms,maintenance_ms,nlist_before,nlist_after,load_splits,capacity_splits,prd_splits,merges,empty_removed,local_refinement_partitions,local_refinement_vectors,local_refinement_reassigned_vectors,quake_cost_splits,lire_size_splits,lire_candidates_or_refined_vectors,lire_moved,maintenance_deleted_partitions,maintenance_deleted_vectors,maintenance_reassigned_small_partition_vectors,delete_rejections,maintenance_empty_removed,global_center_refresh_ms,global_centers_moved,global_vectors_covered\n";
        queries << "query_stage,query_id,dataset_prefix_vectors,index_stored_vectors,scan_budget,center_distances,scanned_vectors,total_distances,partitions_scanned,query_ms\n";
        if (method == "quake")
            quake_decisions << "insert_op,query_stage,leaf_before,size,window_hits,hit_rate,split_delta_ns,delete_delta_ns,reassign_delete_delta_ns,decision\n";
        const char magic[8] = {'M','S','T','I','H','R','0','1'};
        const std::uint32_t shape[3] = {static_cast<std::uint32_t>(query_stages), static_cast<std::uint32_t>(kQueries), static_cast<std::uint32_t>(kTopK)};
        results.write(magic, sizeof(magic));
        results.write(reinterpret_cast<const char*>(shape), sizeof(shape));
        manifest << "workload=Quake Table 3 insert-heavy (IH), deterministic insert operations followed by query operation\n"
                 << "dataset_rows=10000000\ndimension=" << kDimension << "\nquery_metric=" << kMetricName
                 << "\nmaintenance_metric=squared L2\ninitial_vectors=" << kInitialRows << "\ninitial_nlist=" << kInitialNlist
                 << "\ninsert_batch=" << kInsertBatchRows << "\ninsert_operations=" << kInsertOps << "\nquery_operations=" << query_stages << "\ninsert_ops_per_query=" << insert_ops_per_query << "\nqueries_per_query_operation=" << kQueries << '\n'
                 << "top_k=" << kTopK << '\n';
        if (fixed_query_budget > 0)
            manifest << "per-query total distance budget=" << fixed_query_budget
                     << " fixed, including centroid and vector comparisons\n";
        else
            manifest << "per-query total distance budget=floor(indexed_vectors/" << query_budget_divisor
                     << "), including centroid and vector comparisons\n";
        manifest << "query order=the configured query sample is reused for each query operation\n"
                 << "LVQ=false\nseed=1234\n";
        if (method == "prd_ivf_no_reassignment") manifest << "method=PRD-IVF\nprd_threshold=0.05; load_split_fraction=2.2; capacity_split=1.05*sqrt(N) only while nlist<=sqrt(N)\nlocal_refinement_radius=0 (disabled)\nglobal_center_refresh=after each maintenance batch; stored cluster means; no vector reassignment; historical PRD reference retained\n";
        else if (method == "prd_ivf_localref_r25_i1") manifest << "method=PRD-IVF local-refinement ablation\nprd_threshold=0.05; load_split_fraction=2.2; capacity_split=1.05*sqrt(N) only while nlist<=sqrt(N)\nlocal_refinement_radius=25; iterations=1\nglobal_center_refresh=after each maintenance batch; stored cluster means; no vector reassignment; historical PRD reference retained\n";
        else if (method == "frozen_ivf") manifest << "method=Frozen-IVF\nautomatic maintenance=disabled\n";
        else if (method == "quake") manifest << "method=Windows Quake source-logic port (single configured leaf-routing level; APS query disabled)\n"
            << "maintenance=window=1000; alpha=0.9; paper_cost_threshold_tau=250ns; split_threshold=250ns; delete_threshold=250ns; min_partition_size=32\n"
            << "parameter_provenance=paper cost threshold tau=250ns applied to both split and delete decisions; source defaults for window/alpha/min-size; refinement radius 50, one iteration\n"
            << "cost_model=k=10; bilinear L(n,k) profile over source grids; see build.txt for latency_profile_source\n"
            << "deletion_rejection=enabled; accepted deletes reassign vectors to nearest surviving centroids\n"
            << "split=2-means with 5 iterations; local_refinement=top-50 centroids including seed; iterations=1\n"
            << "source_behavior=enable_split_rejection exists in params but is not consulted by upstream perform_maintenance\n"
            << "scan_fraction_denominator=initial ntotal (upstream HitCountTracker is not updated after inserts)\n";
        else if (method == "quake_lire") manifest << "method=Quake source size-threshold LIRE policy port\n"
            << "maintenance details and the explicit max-partition-size argument are in the run manifest\n";
        else if (method == "spfresh_lire") manifest << "method=SPFresh-LIRE local port\nmax_posting=1000; neighbor_reassignment_range=64\n"
            << "raw_vector_storage=inline contiguous FP32 vectors in each posting; query scans stored vectors directly\n";
    }
};

class ScanLatencyProfile {
public:
    explicit ScanLatencyProfile(const std::filesystem::path& output_dir,
                                const std::filesystem::path& fixed_profile_path = {}) {
        constexpr std::array<int, 10> n_grid{1,2,4,16,64,256,1024,4096,16384,65536};
        constexpr std::array<int, 6> k_grid{1,4,10,16,64,256};
        values_.assign(n_grid.size(), std::vector<float>(k_grid.size(), 0.0f));
        if (!fixed_profile_path.empty()) {
            load_fixed_profile(fixed_profile_path, n_grid, k_grid);
            profile_source_ = std::filesystem::absolute(fixed_profile_path).string();
        } else {
            constexpr int trials = 5;
            std::mt19937 rng(kSeed ^ 0x514b4345u);
            std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
            std::vector<float> vectors(static_cast<std::size_t>(n_grid.back()) * kDimension);
            std::vector<float> query(kDimension);
            for (float& value : vectors) value = uniform(rng);
            for (float& value : query) value = uniform(rng);
            volatile float sink = 0.0f;
            for (std::size_t ni = 0; ni < n_grid.size(); ++ni) {
                for (std::size_t ki = 0; ki < k_grid.size(); ++ki) {
                    double total_ns = 0.0;
                    for (int trial = 0; trial < trials; ++trial) {
                        std::priority_queue<float> top;
                        const auto start = std::chrono::steady_clock::now();
                        for (int i = 0; i < n_grid[ni]; ++i) {
                            const float distance = simd_squared_l2(
                                query.data(), vectors.data() + static_cast<std::size_t>(i) * kDimension,
                                kDimension);
                            if (top.size() < static_cast<std::size_t>(k_grid[ki])) top.push(distance);
                            else if (distance < top.top()) { top.pop(); top.push(distance); }
                        }
                        total_ns += std::chrono::duration<double, std::nano>(
                            std::chrono::steady_clock::now() - start).count();
                        if (!top.empty()) sink += top.top();
                    }
                    values_[ni][ki] = static_cast<float>(total_ns / trials);
                }
            }
            (void)sink;
            profile_source_ = "runtime_calibration_5_trials_per_grid_point";
        }
        if (!output_dir.empty()) {
            std::ofstream profile(output_dir / "latency_profile.csv");
            if (!profile) throw std::runtime_error("cannot write effective Quake latency profile");
            profile << "n,k,mean_scan_ns\n";
            for (std::size_t ni = 0; ni < n_grid.size(); ++ni)
                for (std::size_t ki = 0; ki < k_grid.size(); ++ki)
                    profile << n_grid[ni] << ',' << k_grid[ki] << ',' << std::setprecision(10)
                            << values_[ni][ki] << '\n';
        }
    }
    const std::string& profile_source() const { return profile_source_; }
    double operator()(std::size_t n, std::size_t k) const {
        if (n == 0 || k == 0) return 0.0;
        constexpr std::array<int, 10> n_grid{1,2,4,16,64,256,1024,4096,16384,65536};
        constexpr std::array<int, 6> k_grid{1,4,10,16,64,256};
        if (n < static_cast<std::size_t>(n_grid.front()) || k < static_cast<std::size_t>(k_grid.front()))
            throw std::out_of_range("Quake latency model queried below its source grid");
        const auto bracket = [](const auto& grid, int value, std::size_t* lo,
                                std::size_t* hi, float* t) {
            if (value <= grid.back()) {
                const auto upper = std::upper_bound(grid.begin(), grid.end(), value);
                if (upper == grid.end()) {
                    *lo = grid.size() - 2; *hi = grid.size() - 1; *t = 1.0f;
                } else {
                    *hi = static_cast<std::size_t>(upper - grid.begin()); *lo = *hi - 1;
                    *t = static_cast<float>(value - grid[*lo]) /
                         static_cast<float>(grid[*hi] - grid[*lo]);
                }
            } else {
                *lo = grid.size() - 2; *hi = grid.size() - 1;
                *t = static_cast<float>(value - grid[*hi]) /
                     static_cast<float>(grid[*hi] - grid[*lo]);
            }
        };
        std::size_t nl, nh, kl, kh; float nt, kt;
        bracket(n_grid, static_cast<int>(n), &nl, &nh, &nt);
        bracket(k_grid, static_cast<int>(k), &kl, &kh, &kt);
        const float f11 = values_[nl][kl], f12 = values_[nl][kh];
        const float f21 = values_[nh][kl], f22 = values_[nh][kh];
        if (n <= static_cast<std::size_t>(n_grid.back()) && k <= static_cast<std::size_t>(k_grid.back()))
            return (1.0f - nt) * (1.0f - kt) * f11 + nt * (1.0f - kt) * f21 +
                   (1.0f - nt) * kt * f12 + nt * kt * f22;
        const float nlo = f11 + nt * (f21 - f11);
        const float nhi = f12 + nt * (f22 - f12);
        return nlo + kt * (nhi - nlo);
    }
private:
    void load_fixed_profile(const std::filesystem::path& path,
                            const std::array<int, 10>& n_grid,
                            const std::array<int, 6>& k_grid) {
        if (std::string(kMetricName) != "squared L2")
            throw std::runtime_error("fixed Quake latency profiles currently require squared-L2 search");
        std::ifstream input(path);
        if (!input) throw std::runtime_error("cannot open Quake latency profile: " + path.string());
        std::array<std::array<bool, 6>, 10> seen{};
        std::string line;
        bool saw_header = false;
        bool saw_metadata = false;
        while (std::getline(input, line)) {
            if (line.empty()) continue;
            if (line.rfind("# dataset=", 0) == 0) {
                const std::string expected_dataset = std::string("dataset=") + kQuakeProfileDataset;
                const std::string expected_input = std::string("input=") + kQuakeProfileInput;
                const std::string expected_dimension = std::string("dimension=") + std::to_string(kDimension);
                if (line.find(expected_dataset) == std::string::npos ||
                    line.find(expected_input) == std::string::npos ||
                    line.find(expected_dimension) == std::string::npos ||
                    line.find("metric=squared_L2") == std::string::npos)
                    throw std::runtime_error("Quake latency profile metadata does not match this dataset build: " + line);
                saw_metadata = true;
                continue;
            }
            if (line[0] == '#') continue;
            if (!saw_header) {
                if (line.rfind("n,k,mean_scan_ns", 0) != 0)
                    throw std::runtime_error("Quake latency profile must start with n,k,mean_scan_ns columns");
                saw_header = true;
                continue;
            }
            std::stringstream row(line);
            std::string n_text, k_text, value_text;
            if (!std::getline(row, n_text, ',') || !std::getline(row, k_text, ',') ||
                !std::getline(row, value_text, ','))
                throw std::runtime_error("malformed Quake latency profile row: " + line);
            const int n = std::stoi(n_text), k = std::stoi(k_text);
            const double value = std::stod(value_text);
            const auto ni = std::find(n_grid.begin(), n_grid.end(), n);
            const auto ki = std::find(k_grid.begin(), k_grid.end(), k);
            if (ni == n_grid.end() || ki == k_grid.end() || !std::isfinite(value) || value <= 0.0)
                throw std::runtime_error("invalid Quake latency profile cell: " + line);
            const std::size_t n_index = static_cast<std::size_t>(ni - n_grid.begin());
            const std::size_t k_index = static_cast<std::size_t>(ki - k_grid.begin());
            if (seen[n_index][k_index]) throw std::runtime_error("duplicate Quake latency profile cell: " + line);
            seen[n_index][k_index] = true;
            values_[n_index][k_index] = static_cast<float>(value);
        }
        if (!saw_header) throw std::runtime_error("empty Quake latency profile: " + path.string());
        if (!saw_metadata) throw std::runtime_error("Quake latency profile is missing dataset/type/dimension metadata");
        for (std::size_t ni = 0; ni < n_grid.size(); ++ni)
            for (std::size_t ki = 0; ki < k_grid.size(); ++ki)
                if (!seen[ni][ki])
                    throw std::runtime_error("Quake latency profile is missing a required n/k grid cell");
    }
    std::vector<std::vector<float>> values_;
    std::string profile_source_;
};

struct QuakeWindowQuery { std::vector<std::size_t> leaves; std::size_t scanned_size_sum = 0; };
enum class QuakeAction { None, Split, Delete, DeleteRejected };
struct QuakeCandidate {
    std::size_t leaf = 0, size = 0, hits = 0;
    double rate = 0.0, split_delta = 0.0, delete_delta = 0.0, reassign_delete_delta = 0.0;
    QuakeAction action = QuakeAction::None;
};
struct QuakeMaintenanceDecision {
    std::vector<QuakeCandidate> candidates;
    double scan_fraction = 0.0;
};

void remap_quake_window(std::deque<QuakeWindowQuery>* window,
                        const std::vector<std::size_t>& old_to_new) {
    const std::size_t removed = std::numeric_limits<std::size_t>::max();
    for (QuakeWindowQuery& query : *window) {
        std::vector<std::size_t> remapped;
        remapped.reserve(query.leaves.size());
        for (const std::size_t old_id : query.leaves) {
            if (old_id < old_to_new.size() && old_to_new[old_id] != removed)
                remapped.push_back(old_to_new[old_id]);
        }
        query.leaves = std::move(remapped);
    }
}

float l2_to_centroid(const float* point, const float* centroid) {
    return simd_squared_l2(point, centroid, kDimension);
}

float l2_to_centroid(const std::uint8_t* point, const float* centroid) {
    return simd_squared_l2_u8_f32(point, centroid, kDimension);
}

template <typename Data>
double quake_reassign_delete_delta(const AdaptiveHierarchicalIvf& index,
        const Data& data, const std::vector<std::size_t>& sizes,
        const std::vector<std::size_t>& hits, std::size_t leaf,
        double hit_rate, const ScanLatencyProfile& latency) {
    const std::size_t total = sizes.size(), partition_size = sizes[leaf];
    if (total <= 1 || partition_size == 0) return 0.0;
    const auto partition_ids = index.leaf_vector_ids(leaf);
    const auto centers = index.leaf_centroids();
    std::vector<std::size_t> reassigned_counts(total, 0);
    for (const std::uint32_t id : partition_ids) {
        const auto* x = data.row(id);
        std::size_t best = total; float best_distance = std::numeric_limits<float>::max();
        for (std::size_t candidate = 0; candidate < total; ++candidate) {
            if (candidate == leaf) continue;
            const float d = l2_to_centroid(x, centers.data() + candidate * kDimension);
            if (d < best_distance) { best_distance = d; best = candidate; }
        }
        if (best != total) ++reassigned_counts[best];
    }
    const double delta_overhead = latency(total - 1, kQuakeLatencyTopK) -
                            latency(total, kQuakeLatencyTopK);
    const double removal_delta = hit_rate * latency(partition_size, kQuakeLatencyTopK);
    double reassign_delta = 0.0;
    for (std::size_t destination = 0; destination < total; ++destination) {
        const std::size_t count = reassigned_counts[destination];
        if (count == 0) continue;
        const double dest_rate = static_cast<double>(hits[destination]) / kQueryWindow;
        const double old_cost = dest_rate * latency(sizes[destination], kQuakeLatencyTopK);
        const double new_cost = (dest_rate + hit_rate) *
            latency(sizes[destination] + partition_size, kQuakeLatencyTopK);
        reassign_delta += new_cost - old_cost;
    }
    return delta_overhead + removal_delta + reassign_delta;
}

template <typename Data>
QuakeMaintenanceDecision choose_quake_actions(const AdaptiveHierarchicalIvf& index,
        const Data& data, const std::deque<QuakeWindowQuery>& window,
        const ScanLatencyProfile& latency,
        double split_threshold_ns, double delete_threshold_ns) {
    const auto sizes = index.leaf_sizes();
    std::vector<std::size_t> hits(sizes.size(), 0);
    std::size_t scanned_sizes = 0;
    for (const auto& query : window) {
        scanned_sizes += query.scanned_size_sum;
        for (const auto id : query.leaves) if (id < hits.size()) ++hits[id];
    }
    QuakeMaintenanceDecision decision;
    if (window.size() < kQueryWindow) return decision;
    decision.scan_fraction = static_cast<double>(scanned_sizes) /
        (static_cast<double>(window.size()) * static_cast<double>(kInitialRows));
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] == 0) continue;
        QuakeCandidate candidate;
        candidate.leaf = i; candidate.size = sizes[i]; candidate.hits = hits[i];
        candidate.rate = static_cast<double>(hits[i]) / kQueryWindow;
        const std::size_t count = sizes.size();
        const std::size_t avg_size = index.size() / count;
        const double t = latency(count, kQuakeLatencyTopK);
        const double t_minus = count > 1 ? latency(count - 1, kQuakeLatencyTopK) : t;
        const double merged_size = static_cast<double>(avg_size) +
            static_cast<double>(sizes[i]) / static_cast<double>(std::max<std::size_t>(1, count - 1));
        const double merged_rate = decision.scan_fraction +
            candidate.rate / static_cast<double>(std::max<std::size_t>(1, count - 1));
        const double old_cost = static_cast<double>(count - 1) * decision.scan_fraction *
            latency(avg_size, kQuakeLatencyTopK) + candidate.rate * latency(sizes[i], kQuakeLatencyTopK);
        double new_cost = 0.0;
        if (sizes[i] < count) {
            new_cost = static_cast<double>(sizes[i]) * merged_rate *
                    latency(avg_size + 1, kQuakeLatencyTopK) +
                static_cast<double>(count - sizes[i] - 1) * merged_rate *
                    latency(avg_size, kQuakeLatencyTopK);
        } else {
            new_cost = static_cast<double>(count - 1) * merged_rate *
                latency(static_cast<std::size_t>(std::ceil(merged_size)), kQuakeLatencyTopK);
        }
        candidate.delete_delta = (t_minus - t) + new_cost - old_cost;
        if (candidate.delete_delta < -delete_threshold_ns) {
            // The no-reassignment estimate is a cheap candidate prefilter only.
            // Final deletion acceptance always uses the complete delete+reassign
            // cost, because quake_delete_partitions moves every source vector.
            candidate.reassign_delete_delta = quake_reassign_delete_delta(
                index, data, sizes, hits, i, candidate.rate, latency);
            candidate.action = candidate.reassign_delete_delta < -delete_threshold_ns
                ? QuakeAction::Delete : QuakeAction::DeleteRejected;
        } else if (sizes[i] > kQuakeMinPartitionSize) {
            candidate.split_delta = latency(count + 1, kQuakeLatencyTopK) - t +
                2.0 * kQuakeAlpha * candidate.rate *
                    latency(sizes[i] / 2, kQuakeLatencyTopK) -
                candidate.rate * latency(sizes[i], kQuakeLatencyTopK);
            if (candidate.split_delta < -split_threshold_ns)
                candidate.action = QuakeAction::Split;
        }
        decision.candidates.push_back(candidate);
    }
    return decision;
}

struct HeapHit { float distance; std::uint32_t id; };
bool hit_better(const HeapHit& a, const HeapHit& b) {
    return a.distance < b.distance || (a.distance == b.distance && a.id < b.id);
}
// Keep each posting's raw vectors inline with its IDs. This makes the LIRE
// query scan use the same contiguous vector access pattern as the other IVF
// implementations instead of dereferencing every candidate ID into data_.
template <typename Scalar>
struct LirePartition {
    std::vector<std::uint32_t> ids;
    std::vector<Scalar> vectors;
    std::vector<float> center;
};
struct LireStats { std::size_t splits = 0, candidates = 0, moved = 0; };
struct LireResult {
    std::vector<std::uint32_t> ids;
    std::size_t centers = 0, vectors = 0, partitions = 0;
    float estimated_recall = 0.0f;
    std::size_t candidate_partitions = 0;
    std::vector<std::size_t> probed_partitions;
};

double runner_aps_incomplete_beta(double a, double b, double x) {
    constexpr double stop = 1.0e-8, tiny = 1.0e-30;
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    if (x > (a + 1.0) / (a + b + 2.0))
        return 1.0 - runner_aps_incomplete_beta(b, a, 1.0 - x);
    const double lbeta = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
    const double front = std::exp(std::log(x) * a + std::log(1.0 - x) * b - lbeta) / a;
    double f = 1.0, c = 1.0, d = 0.0;
    for (int i = 0; i <= 200; ++i) {
        const int m = i / 2;
        double numerator = 1.0;
        if (i != 0 && i % 2 == 0)
            numerator = (m * (b - m) * x) / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        else if (i % 2 != 0)
            numerator = -((a + m) * (a + b + m) * x) / ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        d = 1.0 + numerator * d;
        if (std::fabs(d) < tiny) d = tiny;
        d = 1.0 / d;
        c = 1.0 + numerator / c;
        if (std::fabs(c) < tiny) c = tiny;
        const double cd = c * d;
        f *= cd;
        if (std::fabs(1.0 - cd) < stop) return front * (f - 1.0);
    }
    return std::numeric_limits<double>::quiet_NaN();
}

double runner_aps_beta_lookup(double x, std::size_t dimension) {
    constexpr std::size_t table_size = 1001;
    struct Table { std::size_t dimension = 0; std::array<double, table_size> values{}; };
    static Table table;
    if (table.dimension != dimension) {
        const double a = (static_cast<double>(dimension) + 1.0) / 2.0;
        for (std::size_t i = 0; i < table_size; ++i)
            table.values[i] = runner_aps_incomplete_beta(a, 0.5, static_cast<double>(i) / (table_size - 1));
        table.dimension = dimension;
    }
    x = std::clamp(x, 0.0, 1.0);
    const double scaled = x * (table_size - 1);
    const std::size_t lo = std::min<std::size_t>(static_cast<std::size_t>(scaled), table_size - 2);
    const double x0 = static_cast<double>(lo) / (table_size - 1);
    return table.values[lo] + (x - x0) * (table.values[lo + 1] - table.values[lo]) * (table_size - 1);
}

std::vector<float> runner_aps_recall_profile(const std::vector<float>& boundaries,
                                             float radius, std::size_t dimension) {
    const std::size_t n = boundaries.size();
    if (n == 0) return {};
    std::vector<float> probabilities(n, 0.0f);
    if (n == 1 || radius <= 1e-9f) { probabilities[0] = 1.0f; return probabilities; }
    std::vector<double> raw(n, 0.0);
    double sum = 0.0;
    for (std::size_t i = 1; i < n; ++i) {
        if (boundaries[i] >= radius) continue;
        const double ratio = static_cast<double>(boundaries[i]) / radius;
        const double x = std::clamp(std::sqrt(std::max(0.0, 1.0 - ratio * ratio)), 0.0, 1.0);
        raw[i] = std::clamp(0.5 * runner_aps_beta_lookup(x, dimension), 0.0, 0.5);
        sum += raw[i];
    }
    if (sum <= 1e-9) { probabilities[0] = 1.0f; return probabilities; }
    double survival = 1.0;
    for (std::size_t i = 1; i < n; ++i) survival *= 1.0 - raw[i] / sum;
    probabilities[0] = static_cast<float>(std::clamp(survival, 0.0, 1.0));
    const double target = 1.0 - probabilities[0];
    for (std::size_t i = 1; i < n; ++i)
        probabilities[i] = static_cast<float>(target * raw[i] / sum);
    return probabilities;
}

struct AdaptiveMethodConfig {
    std::string name;
    bool prd = false;
    bool quake = false;
    std::size_t initial_leaf_nlist = kInitialNlist;
    std::size_t split_local_refinement_radius = 0;
    std::size_t split_local_refinement_iterations = 1;
    double quake_split_threshold_ns = kQuakeSplitThresholdNs;
    double quake_delete_threshold_ns = kQuakeDeleteThresholdNs;
    bool quake_lire = false;
    std::size_t lire_max_partition_size = 0;
    std::size_t query_threads = 1;
    bool final_query_only = false;
    bool insertion_only = false;
    bool skip_final_aps_diagnostic = false;
    bool include_initial_query_stage = false;
    bool aps_queries = false;
    bool triad_policy = false;
    bool triad_enable_prd_splits = true;
    bool triad_enable_load_splits = true;
    bool triad_enable_capacity_splits = true;
    bool triad_enable_load_merges = true;
    bool triad_global_center_refresh = true;
    std::size_t query_stages_override = 0;
    std::size_t insert_ops_per_query_override = 0;
    std::size_t fixed_query_budget_override = 0;
    std::size_t query_budget_divisor_override = 0;
    bool record_per_query = true;
    std::vector<std::filesystem::path> query_stage_files;
    bool override_split_local_refinement = false;
    float prd_threshold_override = -1.0f;
    float short_list_sqrt_n_fraction_override = -1.0f;
    float short_list_n_fraction_override = -1.0f;
    float capacity_n_divisor_override = -1.0f;
    std::size_t capacity_cluster_count_gate_override = 0;
};
void write_method_stage(OutputFiles& out, const std::string& method, std::size_t stage,
                        std::size_t active, std::size_t stored_vectors,
                        std::size_t budget, std::size_t nlist,
                        QueryStats& qs, std::size_t maintenance_splits,
                        const std::vector<Ids>& result_rows,
                        std::size_t insert_ops_since_query = kInsertOpsPerQuery);

template <typename Data>
class LireIndex {
public:
    using Scalar = std::remove_cv_t<std::remove_pointer_t<
        decltype(std::declval<const Data&>().row(std::size_t{}))>>;
    using Partition = LirePartition<Scalar>;

    explicit LireIndex(const Data& data) : data_(data) {}
    float center_distance(const float* a, const float* b) const {
        return simd_squared_l2(a, b, kDimension);
    }
    float vector_center_distance(const Scalar* a, const float* b) const {
        if constexpr (std::is_same_v<Scalar, std::uint8_t>)
            return simd_squared_l2_u8_f32(a, b, kDimension);
        else
            return simd_squared_l2(a, b, kDimension);
    }
    float vector_vector_distance(const Scalar* a, const Scalar* b) const {
        if constexpr (std::is_same_v<Scalar, std::uint8_t>) {
            float sum = 0.0f;
            for (std::size_t d = 0; d < kDimension; ++d) {
                const int delta = static_cast<int>(a[d]) - static_cast<int>(b[d]);
                sum += static_cast<float>(delta * delta);
            }
            return sum;
        } else {
            return simd_squared_l2(a, b, kDimension);
        }
    }
    float query_vector_distance(const float* query, const Scalar* vector) const {
        if constexpr (std::is_same_v<Scalar, std::uint8_t>) {
            if constexpr (kMaximumInnerProduct)
                return simd_negative_dot_f32_u8(query, vector, kDimension);
            else
                return simd_squared_l2_f32_u8(query, vector, kDimension);
        } else {
            if constexpr (kMaximumInnerProduct)
                return simd_negative_dot(query, vector, kDimension);
            else
                return simd_squared_l2(query, vector, kDimension);
        }
    }
    const Scalar* vector(std::uint32_t id) const { return data_.row(id); }
    void initialize(const std::vector<float>& centers, std::size_t count) {
        parts_.resize(centers.size() / kDimension);
        for (std::size_t c = 0; c < parts_.size(); ++c)
            parts_[c].center.assign(centers.data() + c * kDimension, centers.data() + (c + 1) * kDimension);
        for (std::size_t id = 0; id < count; ++id)
            append_vector(parts_[nearest(vector(static_cast<std::uint32_t>(id)))],
                          static_cast<std::uint32_t>(id));
        active_ = count;
    }
    void add(std::size_t begin, std::size_t count) {
        for (std::size_t id = begin; id < begin + count; ++id)
            append_vector(parts_[nearest(vector(static_cast<std::uint32_t>(id)))],
                          static_cast<std::uint32_t>(id));
        active_ += count;
    }
    LireStats maintain() {
        LireStats s;
        std::vector<std::size_t> pending;
        std::vector<std::uint8_t> queued(parts_.size(), 0);
        auto enqueue = [&]() {
            if (queued.size() < parts_.size()) queued.resize(parts_.size(), 0);
            for (std::size_t i = 0; i < parts_.size(); ++i)
                if (parts_[i].ids.size() > kLirMaxPosting && !queued[i]) { pending.push_back(i); queued[i] = 1; }
        };
        enqueue();
        while (!pending.empty()) {
            const auto id = pending.back(); pending.pop_back(); queued[id] = 0;
            if (parts_[id].ids.size() <= kLirMaxPosting) continue;
            split_and_reassign(id, s); enqueue();
        }
        return s;
    }
    std::size_t nlist() const { return parts_.size(); }
    std::size_t size() const { return active_; }
    LireResult search(const float* q, std::size_t budget) const {
        LireResult out;
        std::vector<std::pair<float, std::size_t>> ranked;
        ranked.reserve(parts_.size());
        for (std::size_t i = 0; i < parts_.size(); ++i)
            ranked.emplace_back(kMaximumInnerProduct
                ? simd_negative_dot(q, parts_[i].center.data(), kDimension)
                : center_distance(q, parts_[i].center.data()), i);
        out.centers = ranked.size();
        if (budget <= out.centers) throw std::runtime_error("LIRE budget is smaller than nlist");
        std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.first == b.first ? a.second < b.second : a.first < b.first; });
        using Pair = std::pair<float, std::uint32_t>;
        std::priority_queue<Pair> best;
        const std::size_t target = budget - out.centers;
        for (const auto& item : ranked) {
            const auto& part = parts_[item.second];
            for (std::size_t pos = 0; pos < part.ids.size(); ++pos) {
                const auto id = part.ids[pos];
                const Pair hit{query_vector_distance(q, part.vectors.data() + pos * kDimension), id};
                if (best.size() < kTopK) best.push(hit);
                else if (hit < best.top()) { best.pop(); best.push(hit); }
                if (++out.vectors == target) { ++out.partitions; break; }
            }
            if (out.vectors == target) break;
            if (!parts_[item.second].ids.empty()) ++out.partitions;
        }
        if (out.vectors != target || best.size() != kTopK) throw std::runtime_error("LIRE failed exact fixed-budget scan");
        std::vector<Pair> sorted;
        while (!best.empty()) { sorted.push_back(best.top()); best.pop(); }
        std::sort(sorted.begin(), sorted.end());
        for (const auto& h : sorted) out.ids.push_back(h.second);
        return out;
    }
    LireResult search_aps(const float* q, std::size_t k, float candidate_fraction,
                          float recall_target, float recompute_threshold) const {
        if (kMaximumInnerProduct)
            throw std::invalid_argument("Quake APS port currently supports Euclidean L2 search only");
        LireResult out;
        if (q == nullptr || k == 0 || parts_.empty()) return out;
        candidate_fraction = std::clamp(candidate_fraction, 0.0f, 1.0f);
        recall_target = std::clamp(recall_target, 0.0f, 1.0f);
        const std::size_t candidate_count = std::min(parts_.size(), std::max<std::size_t>(
            1, static_cast<std::size_t>(static_cast<double>(parts_.size()) * candidate_fraction)));
        std::vector<std::pair<float, std::size_t>> coarse;
        coarse.reserve(parts_.size());
        for (std::size_t id = 0; id < parts_.size(); ++id)
            coarse.emplace_back(center_distance(q, parts_[id].center.data()), id);
        std::partial_sort(coarse.begin(), coarse.begin() + candidate_count, coarse.end(),
            [](const auto& a, const auto& b) { return a.first == b.first ? a.second < b.second : a.first < b.first; });
        coarse.resize(candidate_count);

        const float* c0 = parts_[coarse[0].second].center.data();
        const double norm0 = std::inner_product(c0, c0 + kDimension, c0, 0.0);
        std::vector<float> boundaries(candidate_count, 0.0f);
        std::vector<float> diff(kDimension);
        for (std::size_t p = 1; p < candidate_count; ++p) {
            const float* cj = parts_[coarse[p].second].center.data();
            double normj = 0.0, dot_qv = 0.0, normv = 0.0;
            for (std::size_t d = 0; d < kDimension; ++d) {
                diff[d] = cj[d] - c0[d];
                normj += static_cast<double>(cj[d]) * cj[d];
                normv += static_cast<double>(diff[d]) * diff[d];
                dot_qv += static_cast<double>(q[d]) * diff[d];
            }
            const double b = 0.5 * (normj - norm0);
            boundaries[p] = normv > 0.0
                ? static_cast<float>(std::fabs(dot_qv - b) / std::sqrt(normv))
                : std::numeric_limits<float>::infinity();
        }

        using Pair = std::pair<float, std::uint32_t>;
        std::priority_queue<Pair> best;
        std::vector<float> probabilities;
        float query_radius = std::numeric_limits<float>::max();
        for (std::size_t p = 0; p < candidate_count; ++p) {
            const std::size_t part_id = coarse[p].second;
            const Partition& part = parts_[part_id];
            for (std::size_t pos = 0; pos < part.ids.size(); ++pos) {
                const Pair hit{query_vector_distance(q, part.vectors.data() + pos * kDimension), part.ids[pos]};
                if (best.size() < k) best.push(hit);
                else if (hit < best.top()) { best.pop(); best.push(hit); }
            }
            ++out.partitions;
            out.vectors += part.ids.size();
            out.probed_partitions.push_back(part_id);
            const float kth_sq = best.size() < k ? std::numeric_limits<float>::infinity() : best.top().first;
            const float radius = std::sqrt(std::max(0.0f, kth_sq));
            const float percent_change = std::fabs(radius - query_radius) /
                std::max(radius, std::numeric_limits<float>::min());
            if (radius != 0.0f && (p == 0 || percent_change > recompute_threshold)) {
                query_radius = radius;
                probabilities = runner_aps_recall_profile(boundaries, radius, kDimension);
            }
            out.estimated_recall = 0.0f;
            for (std::size_t i = 0; i <= p && i < probabilities.size(); ++i)
                out.estimated_recall += probabilities[i];
            if (out.estimated_recall >= recall_target) break;
        }
        out.centers = parts_.size();
        out.candidate_partitions = candidate_count;
        std::vector<Pair> sorted;
        while (!best.empty()) { sorted.push_back(best.top()); best.pop(); }
        std::sort(sorted.begin(), sorted.end());
        for (const auto& h : sorted) out.ids.push_back(h.second);
        return out;
    }
private:
    void append_vector(Partition& part, std::uint32_t id) {
        const Scalar* x = vector(id);
        part.ids.push_back(id);
        part.vectors.insert(part.vectors.end(), x, x + kDimension);
    }
    std::vector<Scalar> copy_vectors(const std::vector<std::uint32_t>& ids) const {
        std::vector<Scalar> copy;
        copy.reserve(ids.size() * kDimension);
        for (const auto id : ids) {
            const Scalar* x = vector(id);
            copy.insert(copy.end(), x, x + kDimension);
        }
        return copy;
    }
    std::size_t nearest(const Scalar* x) const {
        std::size_t best = 0; float d = vector_center_distance(x, parts_[0].center.data());
        for (std::size_t c = 1; c < parts_.size(); ++c) {
            const float next = vector_center_distance(x, parts_[c].center.data());
            if (next < d) { best = c; d = next; }
        }
        return best;
    }
    struct Split { std::array<std::vector<std::uint32_t>, 2> ids; std::array<std::vector<float>, 2> centers; };
    Split balanced_split(const std::vector<std::uint32_t>& ids, std::uint32_t seed) const {
        if (ids.size() < 2) throw std::runtime_error("cannot split tiny LIRE posting");
        std::mt19937 rng(seed); const std::size_t first = static_cast<std::size_t>(rng()) % ids.size();
        const Scalar* p0 = vector(ids[first]); std::size_t second = first; float farthest = -1.0f;
        for (std::size_t i = 0; i < ids.size(); ++i) { const float d = vector_vector_distance(p0, vector(ids[i])); if (d > farthest) { farthest = d; second = i; } }
        Split s; s.centers[0].assign(p0, p0 + kDimension);
        const Scalar* p1 = vector(ids[second]); s.centers[1].assign(p1, p1 + kDimension);
        std::vector<std::size_t> assignment(ids.size(), 0);
        for (std::size_t it = 0; it < 5; ++it) {
            std::vector<std::pair<float, float>> ds(ids.size());
            std::vector<std::pair<float, std::size_t>> regret(ids.size());
            for (std::size_t i = 0; i < ids.size(); ++i) {
                ds[i] = {vector_center_distance(vector(ids[i]), s.centers[0].data()), vector_center_distance(vector(ids[i]), s.centers[1].data())};
                regret[i] = {std::abs(ds[i].first - ds[i].second), i};
            }
            std::sort(regret.begin(), regret.end(), [](const auto& a, const auto& b) { return a.first == b.first ? a.second < b.second : a.first > b.first; });
            s.ids[0].clear(); s.ids[1].clear(); const std::array<std::size_t, 2> cap{(ids.size() + 1) / 2, ids.size() / 2};
            std::array<std::size_t, 2> n{}; std::array<std::vector<double>, 2> sums{std::vector<double>(kDimension, 0.0), std::vector<double>(kDimension, 0.0)};
            for (const auto& item : regret) {
                const std::size_t i = item.second, preferred = ds[i].first <= ds[i].second ? 0 : 1;
                const std::size_t c = n[preferred] < cap[preferred] ? preferred : 1 - preferred;
                assignment[i] = c; ++n[c]; const Scalar* x = vector(ids[i]);
                for (std::size_t d = 0; d < kDimension; ++d) sums[c][d] += static_cast<double>(x[d]);
            }
            for (std::size_t c = 0; c < 2; ++c) {
                s.centers[c].resize(kDimension);
                for (std::size_t d = 0; d < kDimension; ++d) s.centers[c][d] = static_cast<float>(sums[c][d] / n[c]);
            }
        }
        for (std::size_t i = 0; i < ids.size(); ++i) s.ids[assignment[i]].push_back(ids[i]);
        if (s.ids[0].empty() || s.ids[1].empty()) throw std::runtime_error("LIRE split produced empty child");
        return s;
    }
    void split_and_reassign(std::size_t leaf, LireStats& stats) {
        const auto old_ids = parts_[leaf].ids; const auto old_center = parts_[leaf].center;
        std::vector<std::pair<float, std::size_t>> neighbor_rank;
        for (std::size_t i = 0; i < parts_.size(); ++i)
            if (i != leaf && !parts_[i].ids.empty()) neighbor_rank.emplace_back(center_distance(old_center.data(), parts_[i].center.data()), i);
        const std::size_t nn = std::min(kLirNeighborRange, neighbor_rank.size());
        std::partial_sort(neighbor_rank.begin(), neighbor_rank.begin() + nn, neighbor_rank.end(),
            [](const auto& a, const auto& b) { return a.first == b.first ? a.second < b.second : a.first < b.first; });
        std::vector<std::size_t> neighbors; for (std::size_t i = 0; i < nn; ++i) neighbors.push_back(neighbor_rank[i].second);
        const std::uint32_t seed = kSeed + static_cast<std::uint32_t>(active_) * 17u + static_cast<std::uint32_t>(leaf * 31u + stats.splits);
        auto split = balanced_split(old_ids, seed);
        auto left_vectors = copy_vectors(split.ids[0]);
        auto right_vectors = copy_vectors(split.ids[1]);
        parts_[leaf].ids = std::move(split.ids[0]);
        parts_[leaf].vectors = std::move(left_vectors);
        parts_[leaf].center = std::move(split.centers[0]);
        const std::size_t child = parts_.size();
        Partition child_partition;
        child_partition.ids = std::move(split.ids[1]);
        child_partition.vectors = std::move(right_vectors);
        child_partition.center = std::move(split.centers[1]);
        parts_.push_back(std::move(child_partition));
        std::unordered_set<std::uint32_t> left(parts_[leaf].ids.begin(), parts_[leaf].ids.end());
        std::vector<std::pair<std::size_t, std::uint32_t>> candidates;
        for (const auto id : old_ids) {
            const Scalar* x = vector(id); const float d0 = vector_center_distance(x, old_center.data());
            const float d1 = vector_center_distance(x, parts_[leaf].center.data()), d2 = vector_center_distance(x, parts_[child].center.data());
            if (d0 <= d1 && d0 <= d2) candidates.emplace_back(left.count(id) ? leaf : child, id);
        }
        for (const auto neighbor : neighbors) {
            const auto ids = parts_[neighbor].ids;
            for (const auto id : ids) {
                const Scalar* x = vector(id); const float d0 = vector_center_distance(x, old_center.data());
                const float d1 = vector_center_distance(x, parts_[leaf].center.data()), d2 = vector_center_distance(x, parts_[child].center.data());
                if (std::min(d1, d2) <= d0) candidates.emplace_back(neighbor, id);
            }
        }
        stats.candidates += candidates.size(); std::vector<std::pair<std::size_t, std::uint32_t>> moves;
        moves.reserve(candidates.size());
        for (const auto& item : candidates) {
            const std::size_t from = item.first; const auto id = item.second; const Scalar* x = vector(id);
            const float current = vector_center_distance(x, parts_[from].center.data()); std::size_t best = from; float bd = current;
            for (std::size_t p = 0; p < parts_.size(); ++p) { const float d = vector_center_distance(x, parts_[p].center.data()); if (d < bd) { bd = d; best = p; } }
            if (best != from) moves.emplace_back(from, id);
        }
        std::unordered_map<std::size_t, std::unordered_set<std::uint32_t>> moved;
        for (const auto& m : moves) moved[m.first].insert(m.second);
        for (const auto& item : moved) {
            auto& part = parts_[item.first];
            std::vector<std::uint32_t> kept_ids;
            std::vector<Scalar> kept_vectors;
            kept_ids.reserve(part.ids.size() - item.second.size());
            kept_vectors.reserve((part.ids.size() - item.second.size()) * kDimension);
            for (std::size_t pos = 0; pos < part.ids.size(); ++pos) {
                if (item.second.count(part.ids[pos]) != 0) continue;
                kept_ids.push_back(part.ids[pos]);
                const Scalar* x = part.vectors.data() + pos * kDimension;
                kept_vectors.insert(kept_vectors.end(), x, x + kDimension);
            }
            part.ids = std::move(kept_ids);
            part.vectors = std::move(kept_vectors);
        }
        // Candidates that are moved are rerouted to their best existing partition.
        for (const auto& item : candidates) {
            const std::uint32_t id = item.second;
            if (!moved[item.first].count(id)) continue;
            append_vector(parts_[nearest(vector(id))], id);
        }
        stats.moved += moves.size(); ++stats.splits;
    }
    const Data& data_;
    std::vector<Partition> parts_;
    std::size_t active_ = 0;
};

template <typename Data>
std::vector<float> train_lire_centers(const Data& data) {
    if constexpr (!uses_u8_input_v<Data>) {
        std::vector<float> initial(kInitialRows * kDimension);
        std::copy_n(data.row(0), initial.size(), initial.data());
        return ivf_quantized_detail::train_kmeans(
            initial, kInitialRows, kDimension, kInitialNlist, kKmeansIterations, kSeed);
    } else {
        std::vector<std::size_t> order(kInitialRows);
        std::iota(order.begin(), order.end(), 0);
        std::mt19937 rng(kSeed);
        std::shuffle(order.begin(), order.end(), rng);
        std::vector<float> centers(kInitialNlist * kDimension);
        for (std::size_t c = 0; c < kInitialNlist; ++c) {
            const auto* point = data.row(order[c]);
            for (std::size_t d = 0; d < kDimension; ++d)
                centers[c * kDimension + d] = static_cast<float>(point[d]);
        }
        std::vector<float> sums(kInitialNlist * kDimension);
        std::vector<std::size_t> sizes(kInitialNlist);
        for (std::size_t iteration = 0; iteration < kKmeansIterations; ++iteration) {
            std::fill(sums.begin(), sums.end(), 0.0f);
            std::fill(sizes.begin(), sizes.end(), 0);
            for (std::size_t row = 0; row < kInitialRows; ++row) {
                const auto* point = data.row(row);
                std::size_t best = 0;
                float best_distance = simd_squared_l2_u8_f32(
                    point, centers.data(), kDimension);
                for (std::size_t c = 1; c < kInitialNlist; ++c) {
                    const float distance = simd_squared_l2_u8_f32(
                        point, centers.data() + c * kDimension, kDimension);
                    if (distance < best_distance) { best = c; best_distance = distance; }
                }
                ++sizes[best];
                float* sum = sums.data() + best * kDimension;
                for (std::size_t d = 0; d < kDimension; ++d)
                    sum[d] += static_cast<float>(point[d]);
            }
            for (std::size_t c = 0; c < kInitialNlist; ++c) {
                float* center = centers.data() + c * kDimension;
                if (sizes[c] == 0) {
                    const auto* replacement = data.row(order[rng() % kInitialRows]);
                    for (std::size_t d = 0; d < kDimension; ++d)
                        center[d] = static_cast<float>(replacement[d]);
                } else {
                    const float inverse = 1.0f / static_cast<float>(sizes[c]);
                    const float* sum = sums.data() + c * kDimension;
                    for (std::size_t d = 0; d < kDimension; ++d)
                        center[d] = sum[d] * inverse;
                }
            }
        }
        return centers;
    }
}

template <typename Data, typename QueryRows>
void run_lire(const std::filesystem::path& root, const Data& data, const QueryRows& queries,
              bool aps_queries = false) {
    OutputFiles out(root, "spfresh_lire");
    std::ofstream aps_per_query, aps_stage_summary;
    if (aps_queries) {
        aps_per_query.open(out.dir / "aps09_per_query.csv", std::ios::trunc);
        aps_stage_summary.open(out.dir / "aps09_stage_summary.csv", std::ios::trunc);
        if (!aps_per_query || !aps_stage_summary)
            throw std::runtime_error("cannot create LIRE APS=0.9 outputs");
        aps_per_query << "query_stage,query_id,estimated_recall,candidate_partitions,scanned_partitions,scanned_vectors,query_ms\n";
        aps_stage_summary << "query_stage,active_vectors,nlist,aps_target,candidate_fraction,mean_estimated_recall,mean_candidate_partitions,mean_scanned_partitions,mean_scanned_vectors,mean_query_ms,p95_query_ms,total_query_seconds\n";
    }
    {
        std::ofstream manifest(out.dir / "manifest.txt", std::ios::trunc);
        if (!manifest) throw std::runtime_error("cannot write LIRE method manifest");
        manifest << "method=SPFresh-LIRE local port\nmax_posting=" << kLirMaxPosting
                 << "\nneighbor_reassignment_range=" << kLirNeighborRange
                 << "\nraw_vector_storage=inline contiguous exact "
                 << (uses_u8_input_v<Data> ? "uint8 bytes" : "FP32 values")
                 << "\nmetric=" << kMetricName << "\nseed=" << kSeed
                 << "\nquery_policy=" << (aps_queries
                        ? "APS target=0.9; candidate_fraction=0.1; recompute_threshold=0.001; warmup=100 before first measured stage"
                        : "fixed budget=floor(current active vectors/20), includes centroid distances")
                 << "\nquery_schedule=one query stage after each 9 insert batches\n";
    }
    const auto build_start = std::chrono::steady_clock::now();
    const auto centers = train_lire_centers(data);
    LireIndex<Data> index(data); index.initialize(centers, kInitialRows);
    const LireStats initial_stats = index.maintain();
    const double build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - build_start).count();
    { std::ofstream manifest(out.dir / "build.txt"); manifest << "initial_build_ms=" << std::setprecision(10) << build_ms << "\ninitial_capacity_splits=" << initial_stats.splits << '\n'; }
    std::cout << "spfresh_lire initial_build_done_ms=" << build_ms << " initial_splits=" << initial_stats.splits << '\n' << std::flush;
    std::size_t insert_op = 0;
    bool aps_warmed = false;
    for (std::size_t stage = 0; stage < kQueryOps; ++stage) {
        std::size_t batch_splits = 0, batch_candidates = 0, batch_moved = 0;
        for (std::size_t within = 0; within < kInsertOpsPerQuery; ++within) {
            const std::size_t begin = kInitialRows + insert_op * kInsertBatchRows;
            const std::size_t active = begin + kInsertBatchRows, nlist_before = index.nlist();
            const auto a = std::chrono::steady_clock::now(); index.add(begin, kInsertBatchRows);
            const double insert_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
            const auto b = std::chrono::steady_clock::now(); const auto m = index.maintain();
            const double maintain_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - b).count();
            batch_splits += m.splits; batch_candidates += m.candidates; batch_moved += m.moved;
            out.updates << "spfresh_lire," << insert_op + 1 << ',' << stage << ',' << active << ',' << index.size() << ',' << kInsertBatchRows << ','
                << std::setprecision(8) << insert_ms << ',' << maintain_ms << ',' << nlist_before << ',' << index.nlist()
                << ",0,0,0,0,0,0,0,0,0," << m.splits << ',' << m.candidates << ',' << m.moved << ",0,0,0,0\n";
            ++insert_op;
            if (insert_op % 100 == 0 || insert_op == kInsertOps) { out.updates.flush(); std::cout << "spfresh_lire insert_progress=" << insert_op << '/' << kInsertOps << " active=" << active << " nlist=" << index.nlist() << '\n' << std::flush; }
        }
        const std::size_t active = kInitialRows + insert_op * kInsertBatchRows;
        const std::size_t budget = aps_queries ? 0 : active / 20;
        if (aps_queries && !aps_warmed) {
            for (std::size_t qi = 0; qi < 100; ++qi)
                (void)index.search_aps(queries.row(qi), kTopK, kApsCandidateFraction,
                    kApsRecallTarget, kApsRecomputeThreshold);
            aps_warmed = true;
            std::cout << "spfresh_lire aps09_warmup=100 target=" << kApsRecallTarget << '\n' << std::flush;
        }
        QueryStats qs; std::vector<Ids> result_rows(kQueries);
        std::vector<double> aps_estimates, aps_query_ms;
        std::uint64_t aps_candidates = 0, aps_partitions = 0, aps_vectors = 0;
        for (std::size_t qi = 0; qi < kQueries; ++qi) {
            const auto start = std::chrono::steady_clock::now();
            LireResult found = aps_queries
                ? index.search_aps(queries.row(qi), kTopK, kApsCandidateFraction,
                    kApsRecallTarget, kApsRecomputeThreshold)
                : index.search(queries.row(qi), budget);
            const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            if (found.ids.size() != kTopK) throw std::runtime_error("LIRE failed to return top-100");
            qs.center += found.centers; qs.vectors += found.vectors; qs.total += found.centers + found.vectors; qs.partitions += found.partitions; qs.ms.push_back(elapsed);
            std::copy(found.ids.begin(), found.ids.end(), result_rows[qi].begin());
            out.queries << stage << ',' << qi << ',' << active << ',' << index.size() << ',' << budget << ',' << found.centers << ',' << found.vectors << ',' << found.centers + found.vectors << ',' << found.partitions << ',' << std::setprecision(8) << elapsed << '\n';
            if (aps_queries) {
                aps_estimates.push_back(found.estimated_recall);
                aps_query_ms.push_back(elapsed);
                aps_candidates += found.candidate_partitions;
                aps_partitions += found.partitions;
                aps_vectors += found.vectors;
                aps_per_query << stage << ',' << qi << ',' << found.estimated_recall << ','
                    << found.candidate_partitions << ',' << found.partitions << ',' << found.vectors
                    << ',' << std::setprecision(8) << elapsed << '\n';
            }
            if ((qi + 1) % 1000 == 0) { out.queries.flush(); if (aps_queries) aps_per_query.flush(); }
        }
        write_method_stage(out, "spfresh_lire", stage, active, index.size(), budget, index.nlist(), qs, batch_splits, result_rows);
        const double mean_ms = std::accumulate(qs.ms.begin(), qs.ms.end(), 0.0) / qs.ms.size();
        if (aps_queries) {
            const double mean_estimated = std::accumulate(aps_estimates.begin(), aps_estimates.end(), 0.0) / kQueries;
            aps_stage_summary << stage + 1 << ',' << active << ',' << index.nlist() << ',' << kApsRecallTarget << ','
                << kApsCandidateFraction << ',' << mean_estimated << ','
                << static_cast<double>(aps_candidates) / kQueries << ','
                << static_cast<double>(aps_partitions) / kQueries << ','
                << static_cast<double>(aps_vectors) / kQueries << ',' << mean_ms << ','
                << percentile(qs.ms, 0.95) << ',' << std::accumulate(aps_query_ms.begin(), aps_query_ms.end(), 0.0) / 1000.0 << '\n';
            aps_stage_summary.flush(); aps_per_query.flush();
        }
        if ((stage + 1) % 10 == 0 || stage + 1 == kQueryOps)
            std::cout << "spfresh_lire query_checkpoint=" << stage + 1 << '/' << kQueryOps << " active=" << active << " nlist=" << index.nlist()
                      << (aps_queries ? " aps_target=0.9 mean_estimated_recall=" + std::to_string(aps_queries ? std::accumulate(aps_estimates.begin(), aps_estimates.end(), 0.0) / kQueries : 0.0)
                                      : " budget=" + std::to_string(budget))
                      << " mean_query_ms=" << mean_ms << " p95_ms=" << percentile(qs.ms, .95) << '\n' << std::flush;
    }
}

void write_method_stage(OutputFiles& out, const std::string& method, std::size_t stage,
                        std::size_t active, std::size_t stored_vectors,
                        std::size_t budget, std::size_t nlist,
                        QueryStats& qs, std::size_t maintenance_splits,
                        const std::vector<Ids>& result_rows,
                        std::size_t insert_ops_since_query) {
    const double qcount = static_cast<double>(result_rows.size());
    const double mean_ms = std::accumulate(qs.ms.begin(), qs.ms.end(), 0.0) / qcount;
    out.stages << method << ',' << stage << ',' << active << ',' << stored_vectors << ',' << insert_ops_since_query << ',' << kQueries << ',' << budget << ',' << nlist << ',' << maintenance_splits << ','
               << std::setprecision(8) << mean_ms << ',' << percentile(qs.ms, .50) << ',' << percentile(qs.ms, .95) << ','
               << static_cast<double>(qs.center) / qcount << ',' << static_cast<double>(qs.vectors) / qcount << ','
               << static_cast<double>(qs.total) / qcount << ',' << static_cast<double>(qs.partitions) / qcount << ",\n";
    out.results.write(reinterpret_cast<const char*>(result_rows.data()),
                      static_cast<std::streamsize>(result_rows.size() * sizeof(Ids)));
    if (!out.results) throw std::runtime_error("failed writing result IDs");
    out.stages.flush();
    if (out.queries.is_open()) out.queries.flush();
    out.updates.flush();
}

template <typename Data, typename Queries>
void run_adaptive_clean(const std::filesystem::path& root, const Data& data,
                        const Queries& queries, const AdaptiveMethodConfig& method,
                        std::filesystem::path reusable_initial_index = {},
                        std::filesystem::path final_index_output = {}) {
    const std::size_t query_stages = method.insertion_only ? 1 : (method.final_query_only ? 1 :
        (method.query_stages_override ? method.query_stages_override : kQueryOps));
    const std::size_t insert_ops_per_query = method.insertion_only || method.final_query_only ? kInsertOps :
        (method.insert_ops_per_query_override ? method.insert_ops_per_query_override : kInsertOpsPerQuery);
    const std::size_t update_query_stages = query_stages -
        ((method.include_initial_query_stage && query_stages > 0) ? 1 : 0);
    if (!method.insertion_only && update_query_stages * insert_ops_per_query != kInsertOps)
        throw std::runtime_error("query checkpoint schedule must cover exactly all insert batches");
    if (!method.query_stage_files.empty() && method.query_stage_files.size() != query_stages)
        throw std::runtime_error("query-stage file count must match query checkpoints");
    std::vector<std::unique_ptr<MappedFbin>> query_stage_mmaps;
    query_stage_mmaps.reserve(method.query_stage_files.size());
    for (const auto& path : method.query_stage_files) {
        auto stage_queries = std::make_unique<MappedFbin>(path);
        if (stage_queries->rows() != kQueries)
            throw std::runtime_error("query-stage file does not contain the expected query count: " + path.string());
        query_stage_mmaps.push_back(std::move(stage_queries));
    }
    OutputFiles out(root, method.name, query_stages, insert_ops_per_query,
                    method.fixed_query_budget_override,
                    method.query_budget_divisor_override > 0 ? method.query_budget_divisor_override : 20,
                    method.insertion_only ? false : method.record_per_query);
    std::ofstream lire_maintenance;
    std::ofstream aps_per_query, aps_stage_summary;
    if (method.aps_queries) {
        aps_per_query.open(out.dir / "aps09_per_query.csv", std::ios::trunc);
        aps_stage_summary.open(out.dir / "aps09_stage_summary.csv", std::ios::trunc);
        if (!aps_per_query || !aps_stage_summary)
            throw std::runtime_error("cannot create per-stage APS=0.9 outputs");
        aps_per_query << "query_stage,query_id,estimated_recall,candidate_partitions,scanned_partitions,scanned_vectors,query_ms\n";
        aps_stage_summary << "query_stage,active_vectors,nlist,aps_target,candidate_fraction,mean_estimated_recall,mean_candidate_partitions,mean_scanned_partitions,mean_scanned_vectors,mean_query_ms,p95_query_ms,total_query_seconds\n";
    }
    if (method.quake_lire) {
        lire_maintenance.open(out.dir / "lire_size_maintenance.csv", std::ios::trunc);
        if (!lire_maintenance) throw std::runtime_error("cannot create Quake-LIRE maintenance log");
        lire_maintenance << "operation_type,operation_number,query_stage,dataset_prefix_vectors,index_stored_vectors,nlist_before,nlist_after,maintenance_ms,retired_small_partitions,deleted_vectors,reassigned_small_partition_vectors,splits,refinement_partitions,refinement_vectors,reassigned_vectors\n";
    }
    AdaptiveHierarchicalConfig cfg;
    cfg.dimension = kDimension; cfg.initial_leaf_nlist = method.initial_leaf_nlist; cfg.kmeans_iterations = kKmeansIterations;
    cfg.seed = kSeed; cfg.lvq8_storage = false; cfg.lvq4_storage = false; cfg.angular_metric = false;
    cfg.uint8_storage = uses_u8_input_v<Data>;
    cfg.prd_threshold = method.prd_threshold_override >= 0.0f
        ? method.prd_threshold_override : 0.05f;
    cfg.drift_criterion = DriftCriterion::GlobalFitOnly;
    cfg.load_split_fraction = 2.20f;
    cfg.short_list_sqrt_n_fraction = method.short_list_sqrt_n_fraction_override >= 0.0f
        ? method.short_list_sqrt_n_fraction_override : 0.0f;
    cfg.capacity_sqrt_n_split_multiplier = 1.05f;
    cfg.capacity_n_divisor = method.capacity_n_divisor_override >= 0.0f
        ? method.capacity_n_divisor_override : 0.0f;
    cfg.capacity_cluster_count_gate = method.capacity_cluster_count_gate_override;
    cfg.short_list_n_fraction = method.short_list_n_fraction_override >= 0.0f
        ? method.short_list_n_fraction_override : 0.0f;
    cfg.automatic_maintenance = method.prd && !method.triad_policy;
    cfg.split_local_refinement_radius = method.split_local_refinement_radius;
    cfg.split_local_refinement_iterations = method.split_local_refinement_iterations;
    AdaptiveHierarchicalIvf index(cfg);
    double initial_build_ms = 0.0;
    double initial_index_load_ms = 0.0;
    if (reusable_initial_index.empty()) {
        const auto build_start = std::chrono::steady_clock::now();
        if constexpr (uses_u8_input_v<Data>) index.train_u8(data.row(0), kInitialRows);
        else index.train(data.row(0), kInitialRows);
        initial_build_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - build_start).count();
    } else {
        const auto load_start = std::chrono::steady_clock::now();
        index = AdaptiveHierarchicalIvf::load(reusable_initial_index.string());
        initial_index_load_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - load_start).count();
        if (index.size() != kInitialRows || index.leaf_nlist() != method.initial_leaf_nlist ||
            index.config().dimension != kDimension || index.config().lvq8_storage ||
            index.config().lvq4_storage || index.config().angular_metric ||
            index.config().uint8_storage != uses_u8_input_v<Data>)
            throw std::runtime_error("reused initial index metadata does not match this experiment");
        index.set_automatic_maintenance(method.prd && !method.triad_policy);
    }
    if (method.prd_threshold_override >= 0.0f)
        index.set_prd_threshold(method.prd_threshold_override);
    if (method.short_list_sqrt_n_fraction_override >= 0.0f)
        index.set_short_list_sqrt_n_fraction(method.short_list_sqrt_n_fraction_override);
    if (method.short_list_n_fraction_override >= 0.0f)
        index.set_short_list_n_fraction(method.short_list_n_fraction_override);
    if (method.capacity_n_divisor_override >= 0.0f)
        index.set_capacity_n_divisor(method.capacity_n_divisor_override);
    if (method.capacity_cluster_count_gate_override > 0)
        index.set_capacity_cluster_count_gate(method.capacity_cluster_count_gate_override);
    if (method.override_split_local_refinement)
        index.set_split_local_refinement(method.split_local_refinement_radius,
                                         method.split_local_refinement_iterations);
    std::unique_ptr<ScanLatencyProfile> latency;
    double profile_ms = 0.0;
    std::deque<QuakeWindowQuery> hit_window;
    if (method.quake) {
        const auto profile_start = std::chrono::steady_clock::now();
        latency = std::make_unique<ScanLatencyProfile>(out.dir, g_quake_latency_profile_path);
        profile_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - profile_start).count();
        std::ofstream manifest(out.dir / "manifest.txt", std::ios::trunc);
        manifest << "method=Quake local port\nsplit_threshold_ns=" << method.quake_split_threshold_ns
                 << "\ndelete_threshold_ns=" << method.quake_delete_threshold_ns
                 << "\nalpha=" << kQuakeAlpha << "\nquery_hit_window=" << kQueryWindow
                 << "\nlocal_refinement_radius=" << kQuakeRefinementRadius
                 << "\nlocal_refinement_iterations=" << kQuakeRefinementIterations
                 << "\nquery_metric=" << kMetricName << "\nmaintenance_metric=squared L2\nseed=" << kSeed
                 << "\nquery_policy=" << (method.aps_queries
                        ? "APS target=0.9; candidate_fraction=0.1; recompute_threshold=0.001; warmup=100 before first measured stage"
                        : "fixed total-distance budget=floor(active_vectors/" +
                            std::to_string(method.query_budget_divisor_override > 0
                                ? method.query_budget_divisor_override : 20) + ")") << '\n';
    }
    if (method.quake_lire) {
        std::ofstream manifest(out.dir / "lire_policy.txt", std::ios::trunc);
        if (!manifest) throw std::runtime_error("cannot create Quake-LIRE policy manifest");
        manifest << "method=Quake source size-threshold LIRE branch\n"
                 << "max_partition_size=" << method.lire_max_partition_size << " (provided by command line)\n"
                 << "query_threads=" << method.query_threads << " (independent read-only queries)\n"
                 << "min_partition_size=32\nrefinement_radius=50 total partitions including each seed\n"
                 << "refinement_iterations=0 in Quake API, which performs one assignment pass\n"
                 << "short partition objects are retired only after their vectors are routed to nearest surviving centroids\n"
                 << "indexed vector count is checked after every update batch\n"
                 << "splits are collected before deletion; deletions happen first; all splits then run; one local refinement pass follows\n"
                 << "maintenance is invoked after each insert batch; query batches do not trigger maintenance\n"
                 << "source vectors=uncompressed exact values\n"
                 << "query_policy=" << (method.aps_queries
                        ? "APS target=0.9; candidate_fraction=0.1; recompute_threshold=0.001; warmup=100 before first measured stage"
                        : "fixed total-distance budget=floor(active_vectors/" +
                            std::to_string(method.query_budget_divisor_override > 0
                                ? method.query_budget_divisor_override : 20) + ")")
                 << "; early stopping disabled\n";
    }
    { std::ofstream manifest(out.dir / "build.txt"); manifest << "initial_build_ms=" << std::setprecision(10) << initial_build_ms << '\n'
        << "initial_index_load_ms=" << initial_index_load_ms << '\n'
        << "initial_index_reused=" << (!reusable_initial_index.empty()) << '\n'
        << "latency_profile_ms=" << profile_ms << '\n';
        if (method.quake)
            manifest << "latency_profile_source=" << latency->profile_source() << '\n'; }
    std::cout << method.name << " initial_build_ms=" << initial_build_ms
              << " initial_index_load_ms=" << initial_index_load_ms
              << " latency_profile_ms=" << profile_ms << '\n' << std::flush;
    if (method.insertion_only) {
        std::ofstream manifest(out.dir / "manifest.txt", std::ios::trunc);
        if (!manifest) throw std::runtime_error("cannot write insert-only manifest");
        manifest << "workload=900-batch insertion-only maintenance-cost run\n"
                 << "query_operations=0\ninitial_vectors=" << kInitialRows
                 << "\ninitial_nlist=" << method.initial_leaf_nlist
                 << "\ninsert_batch=" << kInsertBatchRows << "\ninsert_operations=" << kInsertOps
                 << "\nseed=" << kSeed << "\nLVQ=false\n"
                 << "method=" << method.name << '\n';
        if (method.triad_policy)
            manifest << "Triad=PRD=" << method.prd_threshold_override
                     << "; load split=2.2T; merge<0.2*sqrt(N); capacity=1.05*sqrt(N) while K<=sqrt(N)"
                     << "; split-local reassignment radius=" << method.split_local_refinement_radius
                     << "; iterations=" << method.split_local_refinement_iterations
                     << "; global center refresh after every insert batch\n";
        if (method.quake_lire)
            manifest << "LIRE=max_partition_size=" << method.lire_max_partition_size
                     << "; min_partition_size=32; radius=50; one assignment pass after splits"
                     << "; maintenance after every insert batch\n";
        manifest.close();
    }
    std::size_t insert_op = 0;
    double total_insert_ms = 0.0, total_maintenance_ms = 0.0;
    double total_global_center_refresh_ms = 0.0;
    std::size_t total_load_splits = 0, total_capacity_splits = 0, total_prd_splits = 0;
    std::size_t total_merges = 0, total_removed_empty = 0, total_lire_splits = 0;
    std::size_t total_lire_deleted_partitions = 0, total_lire_deleted_vectors = 0;
    std::size_t total_lire_reassigned_small_vectors = 0;
    std::size_t total_refinement_partitions = 0, total_refinement_vectors = 0;
    std::size_t total_refinement_reassigned_vectors = 0;
    bool aps_warmed = false;
    const auto all_inserts_start = std::chrono::steady_clock::now();
    for (std::size_t stage = 0; stage < query_stages; ++stage) {
        std::size_t stage_splits = 0;
        const bool initial_query_only_stage = method.include_initial_query_stage && stage == 0;
        for (std::size_t within = 0; !initial_query_only_stage && within < insert_ops_per_query; ++within) {
            const std::size_t begin = kInitialRows + insert_op * kInsertBatchRows;
            const std::size_t active = begin + kInsertBatchRows, nlist_before = index.leaf_nlist();
            const auto ins = std::chrono::steady_clock::now();
            if (method.prd || method.triad_policy) index.begin_update_batch();
            if constexpr (uses_u8_input_v<Data>) index.add_u8(data.row(begin), kInsertBatchRows);
            else index.add(data.row(begin), kInsertBatchRows);
            const double ins_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ins).count();
            const auto mt = std::chrono::steady_clock::now(); MaintenanceReport report;
            GlobalCenterRefreshReport global_center_refresh{};
            double global_center_refresh_ms = 0.0;
            std::size_t qsplit = 0, qdelete = 0, qdropped = 0, qdelete_rejected = 0, qempty = 0;
            if (method.triad_policy) {
                (void)index.end_update_batch();
                if (index.maintenance_due())
                    report = index.maintain_with_policy(
                        method.triad_enable_prd_splits,
                        method.triad_enable_load_splits,
                        method.triad_enable_capacity_splits,
                        method.triad_enable_load_merges);
                if (method.triad_global_center_refresh) {
                    const auto refresh_start = std::chrono::steady_clock::now();
                    global_center_refresh = index.refresh_all_centroids_to_means();
                    global_center_refresh_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - refresh_start).count();
                    if (global_center_refresh.vectors_covered != index.size())
                        throw std::runtime_error("global center refresh did not cover all active vectors");
                }
            }
            else if (method.prd) {
                report = index.end_update_batch();
                const auto refresh_start = std::chrono::steady_clock::now();
                global_center_refresh = index.refresh_all_centroids_to_means();
                global_center_refresh_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - refresh_start).count();
                if (global_center_refresh.vectors_covered != index.size())
                    throw std::runtime_error("global center refresh did not cover all active vectors");
            }
            else if (method.quake && !hit_window.empty()) {
                index.quake_refresh_centroids();
                const auto decision = choose_quake_actions(index, data, hit_window, *latency,
                    method.quake_split_threshold_ns, method.quake_delete_threshold_ns);
                std::vector<std::size_t> selected_deletes, selected_splits;
                for (const auto& candidate : decision.candidates) {
                    const char* action = "none";
                    if (candidate.action == QuakeAction::Delete) { selected_deletes.push_back(candidate.leaf); action = "delete"; }
                    else if (candidate.action == QuakeAction::Split) { selected_splits.push_back(candidate.leaf); action = "split"; }
                    else if (candidate.action == QuakeAction::DeleteRejected) { ++qdelete_rejected; action = "delete_rejected"; }
                    out.quake_decisions << insert_op + 1 << ',' << stage << ',' << candidate.leaf << ','
                        << candidate.size << ',' << candidate.hits << ',' << std::setprecision(9)
                        << candidate.rate << ',' << candidate.split_delta << ',' << candidate.delete_delta << ','
                        << candidate.reassign_delete_delta << ',' << action << '\n';
                }
                out.quake_decisions.flush();
                if (!selected_deletes.empty()) {
                    const std::size_t old_size = index.size();
                    const auto old_to_new = index.quake_delete_partitions(selected_deletes);
                    qdropped = old_size - index.size();
                    qdelete = selected_deletes.size();
                    remap_quake_window(&hit_window, old_to_new);
                    std::vector<std::size_t> remapped_splits;
                    for (const std::size_t old_id : selected_splits)
                        if (old_id < old_to_new.size() &&
                            old_to_new[old_id] != std::numeric_limits<std::size_t>::max())
                            remapped_splits.push_back(old_to_new[old_id]);
                    selected_splits = std::move(remapped_splits);
                }
                if (!selected_splits.empty()) {
                    report = index.maintain_selected(selected_splits,
                        SelectedMaintenanceAction::Split,
                        kQuakeRefinementRadius - 1, kQuakeRefinementIterations);
                    qsplit = report.split_partitions; stage_splits += qsplit;
                    if (!report.source_split_leaf_ids.empty()) {
                        const std::unordered_set<std::size_t> retired(
                            report.source_split_leaf_ids.begin(), report.source_split_leaf_ids.end());
                        for (QuakeWindowQuery& query : hit_window)
                            query.leaves.erase(std::remove_if(query.leaves.begin(), query.leaves.end(),
                                [&](std::size_t id) { return retired.count(id) != 0; }),
                                query.leaves.end());
                    }
                }
                const std::size_t nlist_before_cleanup = index.leaf_nlist();
                const auto cleanup_map = index.quake_delete_partitions({});
                qempty = nlist_before_cleanup - index.leaf_nlist();
                if (qempty != 0) remap_quake_window(&hit_window, cleanup_map);
            }
            else if (method.quake_lire) {
                report = index.maintain_quake_lire_policy(
                    method.lire_max_partition_size, kQuakeMinPartitionSize,
                    kQuakeRefinementRadius, 0);
                qsplit = report.split_partitions;
                stage_splits += qsplit;
                qdelete = report.deleted_small_partitions;
                qdropped = report.deleted_small_vectors;
                qempty = report.removed_empty_partitions;
            }
            if (method.prd) stage_splits += report.total_split_partitions;
            total_load_splits += report.load_split_partitions;
            total_capacity_splits += report.capacity_split_partitions;
            total_prd_splits += report.prd_split_partitions;
            total_merges += report.merged_short_partitions;
            total_removed_empty += report.removed_empty_partitions;
            total_refinement_partitions += report.local_refinement_partitions;
            total_refinement_vectors += report.local_refinement_vectors;
            total_refinement_reassigned_vectors += report.local_refinement_reassigned_vectors;
            total_global_center_refresh_ms += global_center_refresh_ms;
            if (method.quake_lire) {
                total_lire_splits += report.split_partitions;
                total_lire_deleted_partitions += report.deleted_small_partitions;
                total_lire_deleted_vectors += report.deleted_small_vectors;
                total_lire_reassigned_small_vectors += report.reassigned_small_partition_vectors;
            }
            if (method.quake_lire && index.size() != active)
                throw std::logic_error("Quake LIRE changed indexed vector count during partition maintenance");
            const double mt_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mt).count();
            total_insert_ms += ins_ms;
            total_maintenance_ms += mt_ms;
            out.updates << method.name << ',' << insert_op + 1 << ',' << stage << ',' << active << ',' << index.size() << ',' << kInsertBatchRows << ',' << std::setprecision(8)
                << ins_ms << ',' << mt_ms << ',' << nlist_before << ',' << index.leaf_nlist() << ',' << report.load_split_partitions << ','
                << report.capacity_split_partitions << ',' << report.prd_split_partitions << ',' << report.merged_short_partitions << ','
                << report.removed_empty_partitions << ',' << report.local_refinement_partitions << ','
                << report.local_refinement_vectors << ',' << report.local_refinement_reassigned_vectors << ','
                << (method.quake ? qsplit : 0) << ','
                << (method.quake_lire ? qsplit : 0) << ','
                << (method.quake_lire ? report.local_refinement_vectors : 0) << ','
                << (method.quake_lire ? report.local_refinement_reassigned_vectors : 0) << ',' << qdelete << ','
                << qdropped << ',' << report.reassigned_small_partition_vectors << ','
                << qdelete_rejected << ',' << qempty << ','
                << global_center_refresh_ms << ',' << global_center_refresh.centers_updated << ','
                << global_center_refresh.vectors_covered << '\n';
            if (method.quake_lire) {
                lire_maintenance << "insert," << insert_op + 1 << ',' << stage << ',' << active << ','
                    << index.size() << ',' << nlist_before << ',' << index.leaf_nlist() << ','
                    << std::setprecision(8) << mt_ms << ',' << report.deleted_small_partitions << ','
                    << report.deleted_small_vectors << ',' << report.reassigned_small_partition_vectors << ','
                    << report.split_partitions << ','
                    << report.local_refinement_partitions << ',' << report.local_refinement_vectors << ','
                    << report.local_refinement_reassigned_vectors << '\n';
                lire_maintenance.flush();
            }
            ++insert_op;
            if (insert_op % 100 == 0 || insert_op == kInsertOps) {
                out.updates.flush();
                const double elapsed_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - all_inserts_start).count();
                std::cout << method.name << " insert_progress=" << insert_op << '/' << kInsertOps
                          << " active=" << active << " nlist=" << index.leaf_nlist()
                          << " insert_total_s=" << total_insert_ms / 1000.0
                          << " maintenance_total_s=" << total_maintenance_ms / 1000.0
                          << " insert_plus_maintenance_elapsed_s=" << elapsed_s << '\n' << std::flush;
            }
        }
        if (method.insertion_only) {
            out.updates.flush();
            if (lire_maintenance.is_open()) lire_maintenance.flush();
            const double elapsed_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - all_inserts_start).count();
            std::ofstream summary(out.dir / "insert_only_summary.csv", std::ios::trunc);
            if (!summary) throw std::runtime_error("cannot write insert-only summary");
            summary << "method,insert_batches,final_vectors,final_clusters,total_insert_s,total_maintenance_s,global_center_refresh_s,insert_plus_maintenance_elapsed_s,load_splits,capacity_splits,prd_splits,merges,empty_removed,lire_splits,lire_retired_small_partitions,lire_deleted_vectors,lire_reassigned_small_partition_vectors,refinement_partitions,refinement_vectors,reassigned_vectors,initial_build_ms,initial_index_load_ms\n"
                    << method.name << ',' << insert_op << ',' << index.size() << ',' << index.leaf_nlist() << ','
                    << std::setprecision(10) << total_insert_ms / 1000.0 << ','
                    << total_maintenance_ms / 1000.0 << ','
                    << total_global_center_refresh_ms / 1000.0 << ',' << elapsed_s << ','
                    << total_load_splits << ',' << total_capacity_splits << ',' << total_prd_splits << ','
                    << total_merges << ',' << total_removed_empty << ',' << total_lire_splits << ','
                    << total_lire_deleted_partitions << ',' << total_lire_deleted_vectors << ','
                    << total_lire_reassigned_small_vectors << ','
                    << total_refinement_partitions << ',' << total_refinement_vectors << ','
                    << total_refinement_reassigned_vectors << ',' << initial_build_ms << ','
                    << initial_index_load_ms << '\n';
            summary.close();
            // LIRE batch-level cardinality checks above ensure retiring a short
            // partition never drops its indexed vectors.
            if (insert_op != kInsertOps)
                throw std::runtime_error("insert-only run ended before all input batches were processed");
            std::ofstream complete(out.dir / "RUN_COMPLETE.txt", std::ios::trunc);
            if (!complete) throw std::runtime_error("cannot write insert-only completion marker");
            complete << "status=complete\nmethod=" << method.name << "\ninsert_batches=" << insert_op
                     << "\nqueries_executed=0\ninput_vectors_processed="
                     << (insert_op * kInsertBatchRows) << "\nfinal_vectors=" << index.size()
                     << "\nfinal_clusters=" << index.leaf_nlist() << '\n';
            complete.close();
            std::cout << method.name << " INSERT_ONLY_COMPLETE batches=" << insert_op
                      << " active=" << index.size() << " nlist=" << index.leaf_nlist()
                      << " insert_total_s=" << total_insert_ms / 1000.0
                      << " maintenance_total_s=" << total_maintenance_ms / 1000.0
                      << " elapsed_s=" << elapsed_s << '\n' << std::flush;
            break;
        }
        const std::size_t active = kInitialRows + insert_op * kInsertBatchRows;
        const auto query_row = [&](std::size_t query_id) -> const float* {
            return query_stage_mmaps.empty() ? queries.row(query_id) : query_stage_mmaps[stage]->row(query_id);
        };
        const std::size_t budget = method.aps_queries ? 0 :
            (method.fixed_query_budget_override > 0 ? method.fixed_query_budget_override :
             active / (method.query_budget_divisor_override > 0 ? method.query_budget_divisor_override : 20));
        QuerySearchOptions options; options.enable_early_stop = false; options.max_total_distance_computations = budget;
        options.maximum_inner_product = kMaximumInnerProduct;
        QueryStats qs; std::vector<Ids> result_rows(kQueries);
        std::vector<double> aps_estimated_recalls, aps_query_times;
        std::uint64_t aps_candidate_partitions = 0, aps_scanned_partitions = 0, aps_scanned_vectors = 0;
        const auto leaf_sizes = (method.aps_queries || method.quake)
            ? index.leaf_sizes() : std::vector<std::size_t>{};
        if (method.aps_queries && !aps_warmed) {
            for (std::size_t qi = 0; qi < 100; ++qi)
                (void)index.search_aps(query_row(qi), kTopK, kApsCandidateFraction,
                    kApsRecallTarget, kApsRecomputeThreshold, nullptr, nullptr);
            aps_warmed = true;
            std::cout << method.name << " aps09_warmup=100 target=" << kApsRecallTarget
                      << " candidate_fraction=" << kApsCandidateFraction << '\n' << std::flush;
        }
        std::vector<QuerySearchStats> parallel_stats;
        std::vector<double> parallel_ms;
        std::vector<std::vector<std::size_t>> parallel_probed;
        if (!method.aps_queries && method.query_threads > 1) {
            // Fixed-budget searches are read-only and independent for both
            // LIRE and Triad. CSV and result output remain serialized.
            parallel_stats.resize(kQueries);
            parallel_ms.assign(kQueries, 0.0);
            if (method.quake) parallel_probed.resize(kQueries);
            std::atomic<std::size_t> next_query{0};
            std::exception_ptr worker_error;
            std::mutex worker_error_mutex;
            const auto worker = [&]() {
                for (;;) {
                    const std::size_t qi = next_query.fetch_add(1);
                    if (qi >= kQueries) break;
                    try {
                        QuerySearchStats stats;
                        const auto t = std::chrono::steady_clock::now();
                        const auto found = index.search_with_options(query_row(qi), kTopK,
                            options, &stats, method.quake ? &parallel_probed[qi] : nullptr);
                        parallel_ms[qi] = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - t).count();
                        if (stats.total_distance_computations > budget ||
                            stats.center_distance_computations != index.leaf_nlist() ||
                            found.size() != kTopK)
                            throw std::runtime_error("parallel query failed to consume expected budget or return top-100");
                        for (std::size_t k = 0; k < kTopK; ++k)
                            result_rows[qi][k] = static_cast<std::uint32_t>(found[k].id);
                        parallel_stats[qi] = stats;
                    } catch (...) {
                        std::lock_guard<std::mutex> lock(worker_error_mutex);
                        if (!worker_error) worker_error = std::current_exception();
                    }
                }
            };
            const std::size_t thread_count = std::min(method.query_threads, kQueries);
            std::vector<std::thread> workers;
            workers.reserve(thread_count);
            for (std::size_t i = 0; i < thread_count; ++i) workers.emplace_back(worker);
            for (std::thread& thread : workers) thread.join();
            if (worker_error) std::rethrow_exception(worker_error);
            if (!method.quake && !method.quake_lire) {
                for (const QuerySearchStats& stats : parallel_stats)
                    if (stats.total_distance_computations != budget)
                        throw std::runtime_error("parallel fixed-budget query did not consume the exact budget");
            }
        }
        for (std::size_t qi = 0; qi < kQueries; ++qi) {
            if (!parallel_stats.empty()) {
                const QuerySearchStats& stats = parallel_stats[qi];
                const double elapsed = parallel_ms[qi];
                qs.center += stats.center_distance_computations;
                qs.vectors += stats.scanned_vectors;
                qs.total += stats.total_distance_computations;
                qs.partitions += stats.scanned_partitions;
                qs.ms.push_back(elapsed);
                if (out.queries.is_open()) {
                    out.queries << stage << ',' << qi << ',' << active << ',' << index.size() << ',' << budget << ','
                        << stats.center_distance_computations << ',' << stats.scanned_vectors << ','
                        << stats.total_distance_computations << ',' << stats.scanned_partitions << ','
                        << std::setprecision(8) << elapsed << '\n';
                    if ((qi + 1) % 1000 == 0) out.queries.flush();
                }
                if (method.quake) {
                    QuakeWindowQuery window_query;
                    window_query.leaves = std::move(parallel_probed[qi]);
                    for (const std::size_t id : window_query.leaves)
                        if (id < leaf_sizes.size()) window_query.scanned_size_sum += leaf_sizes[id];
                    hit_window.push_back(std::move(window_query));
                    while (hit_window.size() > kQueryWindow) hit_window.pop_front();
                }
                continue;
            }
            QuerySearchStats stats; std::vector<std::size_t> probed;
            const auto t = std::chrono::steady_clock::now();
            std::vector<SearchResult> found;
            ApsSearchStats aps_stats;
            if (method.aps_queries) {
                found = index.search_aps(query_row(qi), kTopK, kApsCandidateFraction,
                    kApsRecallTarget, kApsRecomputeThreshold, &probed, &aps_stats);
                stats.center_distance_computations = index.leaf_nlist();
                stats.scanned_partitions = aps_stats.scanned_partitions;
                for (const std::size_t id : probed)
                    if (id < leaf_sizes.size()) stats.scanned_vectors += leaf_sizes[id];
                stats.total_distance_computations = stats.center_distance_computations + stats.scanned_vectors;
            } else {
                found = index.search_with_options(query_row(qi), kTopK, options, &stats, method.quake ? &probed : nullptr);
            }
            const double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
            if ((!method.aps_queries && !method.quake && !method.quake_lire && stats.total_distance_computations != budget) ||
                (!method.aps_queries && (method.quake || method.quake_lire) && stats.total_distance_computations > budget) ||
                stats.center_distance_computations != index.leaf_nlist() || found.size() != kTopK)
                throw std::runtime_error(method.name + " query failed to consume expected budget or return top-100");
            qs.center += stats.center_distance_computations; qs.vectors += stats.scanned_vectors; qs.total += stats.total_distance_computations; qs.partitions += stats.scanned_partitions; qs.ms.push_back(elapsed);
            for (std::size_t k = 0; k < kTopK; ++k) result_rows[qi][k] = static_cast<std::uint32_t>(found[k].id);
            if (out.queries.is_open())
                out.queries << stage << ',' << qi << ',' << active << ',' << index.size() << ',' << budget << ',' << stats.center_distance_computations << ',' << stats.scanned_vectors << ',' << stats.total_distance_computations << ',' << stats.scanned_partitions << ',' << std::setprecision(8) << elapsed << '\n';
            if (method.aps_queries) {
                aps_estimated_recalls.push_back(aps_stats.estimated_recall);
                aps_query_times.push_back(elapsed);
                aps_candidate_partitions += aps_stats.candidate_partitions;
                aps_scanned_partitions += aps_stats.scanned_partitions;
                aps_scanned_vectors += stats.scanned_vectors;
                aps_per_query << stage << ',' << qi << ',' << aps_stats.estimated_recall << ','
                    << aps_stats.candidate_partitions << ',' << aps_stats.scanned_partitions << ','
                    << stats.scanned_vectors << ',' << std::setprecision(8) << elapsed << '\n';
            }
            if (method.quake) {
                QuakeWindowQuery window_query; window_query.leaves = std::move(probed);
                for (const std::size_t id : window_query.leaves)
                    if (id < leaf_sizes.size()) window_query.scanned_size_sum += leaf_sizes[id];
                hit_window.push_back(std::move(window_query));
                while (hit_window.size() > kQueryWindow) hit_window.pop_front();
            }
            if (out.queries.is_open() && (qi + 1) % 1000 == 0) out.queries.flush();
            if (method.aps_queries && (qi + 1) % 1000 == 0) aps_per_query.flush();
        }
        write_method_stage(out, method.name, stage, active, index.size(), budget, index.leaf_nlist(), qs,
                           stage_splits, result_rows,
                           initial_query_only_stage ? 0 : insert_ops_per_query);
        const double mean_ms = std::accumulate(qs.ms.begin(), qs.ms.end(), 0.0) / qs.ms.size();
        if (method.aps_queries) {
            const double aps_mean_estimated_recall = std::accumulate(aps_estimated_recalls.begin(), aps_estimated_recalls.end(), 0.0) / kQueries;
            aps_stage_summary << stage << ',' << active << ',' << index.leaf_nlist() << ',' << kApsRecallTarget << ','
                << kApsCandidateFraction << ',' << aps_mean_estimated_recall << ','
                << static_cast<double>(aps_candidate_partitions) / kQueries << ','
                << static_cast<double>(aps_scanned_partitions) / kQueries << ','
                << static_cast<double>(aps_scanned_vectors) / kQueries << ',' << mean_ms << ','
                << percentile(aps_query_times, .95) << ','
                << std::accumulate(aps_query_times.begin(), aps_query_times.end(), 0.0) / 1000.0 << '\n';
            aps_stage_summary.flush(); aps_per_query.flush();
        }
        if (method.final_query_only || (stage + 1) % 10 == 0 || stage + 1 == kQueryOps) {
            std::cout << method.name << " query_checkpoint=" << stage + 1 << '/' << query_stages
                      << " active=" << active << " nlist=" << index.leaf_nlist();
            if (method.aps_queries)
                std::cout << " aps_target=" << kApsRecallTarget
                          << " mean_estimated_recall="
                          << std::accumulate(aps_estimated_recalls.begin(), aps_estimated_recalls.end(), 0.0) / kQueries;
            else
                std::cout << " budget=" << budget;
            std::cout << " mean_query_ms=" << mean_ms << " p95_ms=" << percentile(qs.ms, .95)
                      << '\n' << std::flush;
        }
        if (method.final_query_only && !method.aps_queries && !method.skip_final_aps_diagnostic) {
            std::ofstream aps_per_query(out.dir / "aps09_per_query.csv", std::ios::trunc);
            std::ofstream aps_results(out.dir / "aps09_results.bin", std::ios::binary | std::ios::trunc);
            std::ofstream aps_summary(out.dir / "aps09_summary.csv", std::ios::trunc);
            if (!aps_per_query || !aps_results || !aps_summary)
                throw std::runtime_error("cannot create final APS=0.9 outputs");
            aps_per_query << "query_id,estimated_recall,candidate_partitions,scanned_partitions,scanned_vectors,query_ms\n";
            aps_summary << "method,aps_target,aps_candidate_fraction,queries,mean_estimated_recall,mean_candidate_partitions,mean_scanned_partitions,mean_scanned_vectors,mean_query_ms,p95_query_ms,total_query_seconds\n";
            const char aps_magic[8] = {'M','S','T','I','H','R','0','1'};
            const std::uint32_t aps_shape[3] = {1, static_cast<std::uint32_t>(kQueries), static_cast<std::uint32_t>(kTopK)};
            aps_results.write(aps_magic, sizeof(aps_magic));
            aps_results.write(reinterpret_cast<const char*>(aps_shape), sizeof(aps_shape));
            std::cout << method.name << " aps09_warmup=100 target=" << kApsRecallTarget
                      << " candidate_fraction=" << kApsCandidateFraction << '\n' << std::flush;
            for (std::size_t q = 0; q < 100; ++q)
                (void)index.search_aps(query_row(q), kTopK, kApsCandidateFraction,
                    kApsRecallTarget, kApsRecomputeThreshold, nullptr, nullptr);
            const auto leaf_sizes = index.leaf_sizes();
            std::vector<double> aps_query_ms;
            aps_query_ms.reserve(kQueries);
            std::uint64_t aps_candidates = 0, aps_scanned_partitions = 0, aps_scanned_vectors = 0;
            double aps_estimated_recall = 0.0;
            for (std::size_t qi = 0; qi < kQueries; ++qi) {
                ApsSearchStats aps_stats;
                std::vector<std::size_t> probed;
                const auto aps_start = std::chrono::steady_clock::now();
                const auto found = index.search_aps(query_row(qi), kTopK, kApsCandidateFraction,
                    kApsRecallTarget, kApsRecomputeThreshold, &probed, &aps_stats);
                const double aps_ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - aps_start).count();
                if (found.size() != kTopK) throw std::runtime_error("APS=0.9 did not return top-100");
                Ids ids{};
                for (std::size_t k = 0; k < kTopK; ++k) {
                    if (found[k].id < 0 || static_cast<std::size_t>(found[k].id) >= data.rows())
                        throw std::runtime_error("APS returned an invalid shuffled row ID");
                    ids[k] = static_cast<std::uint32_t>(found[k].id);
                }
                aps_results.write(reinterpret_cast<const char*>(ids.data()), static_cast<std::streamsize>(sizeof(ids)));
                std::uint64_t query_scanned_vectors = 0;
                for (const std::size_t leaf : probed) {
                    if (leaf >= leaf_sizes.size()) throw std::runtime_error("APS returned an invalid leaf ID");
                    query_scanned_vectors += leaf_sizes[leaf];
                }
                aps_query_ms.push_back(aps_ms);
                aps_candidates += aps_stats.candidate_partitions;
                aps_scanned_partitions += aps_stats.scanned_partitions;
                aps_scanned_vectors += query_scanned_vectors;
                aps_estimated_recall += aps_stats.estimated_recall;
                aps_per_query << qi << ',' << aps_stats.estimated_recall << ',' << aps_stats.candidate_partitions
                              << ',' << aps_stats.scanned_partitions << ',' << query_scanned_vectors << ','
                              << std::setprecision(8) << aps_ms << '\n';
                if ((qi + 1) % 1000 == 0) {
                    aps_per_query.flush(); aps_results.flush();
                    std::cout << method.name << " aps09_progress=" << qi + 1 << '/' << kQueries << '\n' << std::flush;
                }
            }
            const double aps_mean_ms = std::accumulate(aps_query_ms.begin(), aps_query_ms.end(), 0.0) / kQueries;
            aps_summary << method.name << ',' << kApsRecallTarget << ',' << kApsCandidateFraction << ',' << kQueries << ','
                        << std::setprecision(10) << aps_estimated_recall / kQueries << ','
                        << static_cast<double>(aps_candidates) / kQueries << ','
                        << static_cast<double>(aps_scanned_partitions) / kQueries << ','
                        << static_cast<double>(aps_scanned_vectors) / kQueries << ',' << aps_mean_ms << ','
                        << percentile(aps_query_ms, .95) << ','
                        << std::accumulate(aps_query_ms.begin(), aps_query_ms.end(), 0.0) / 1000.0 << '\n';
            aps_summary.flush(); aps_per_query.flush(); aps_results.flush();
            std::cout << method.name << " aps09_query_checkpoint=" << kQueries
                      << " mean_query_ms=" << aps_mean_ms << " p95_query_ms=" << percentile(aps_query_ms, .95)
                      << " mean_scanned_vectors=" << static_cast<double>(aps_scanned_vectors) / kQueries
                      << " mean_estimated_recall=" << aps_estimated_recall / kQueries << '\n' << std::flush;
        }
    }
    if (!final_index_output.empty()) {
        index.save(final_index_output.string());
        std::ofstream snapshot_note(root / "final_index_snapshot.txt", std::ios::trunc);
        if (!snapshot_note) throw std::runtime_error("cannot write final index snapshot note");
        snapshot_note << "final_index=" << final_index_output.string()
                     << "\nindexed_vectors=" << index.size()
                     << "\nfinal_nlist=" << index.leaf_nlist()
                     << "\nquery_stages=" << query_stages
                     << "\nquery_feedback_or_maintenance_after_snapshot=false\n";
        std::cout << method.name << " final_index_saved=" << final_index_output.string()
                  << " bytes=" << std::filesystem::file_size(final_index_output) << '\n' << std::flush;
    }
}

std::vector<Ids> calculate_truth(const MappedFbin& data, const MappedFbin& queries) {
    // Each worker handles a small query block and scans the dataset once. This
    // computes all 100 prefix checkpoints in one pass per query block.
    const std::size_t truth_rows = kQueryOps * kQueries;
    std::vector<Ids> truth(truth_rows);
    const std::size_t qblock = 8;
    std::atomic<std::size_t> next{0};
    const unsigned hc = std::thread::hardware_concurrency();
    const std::size_t workers = std::max<std::size_t>(1, std::min<std::size_t>(8, hc ? hc : 4));
    auto work = [&]() {
        for (;;) {
            const std::size_t q0 = next.fetch_add(qblock);
            if (q0 >= kQueries) break;
            const std::size_t qn = std::min(qblock, kQueries - q0);
            std::vector<std::priority_queue<std::pair<float, std::uint32_t>>> heaps(qn);
            std::size_t checkpoint = 0;
            for (std::size_t row = 0; row < data.rows(); ++row) {
                const float* x = data.row(row);
                for (std::size_t j = 0; j < qn; ++j) {
                    auto& heap = heaps[j];
                    const auto d = simd_squared_l2(queries.row(q0 + j), x, kDimension);
                    const std::pair<float, std::uint32_t> candidate{d, static_cast<std::uint32_t>(row)};
                    if (heap.size() < kTopK) heap.push(candidate);
                    else if (candidate < heap.top()) { heap.pop(); heap.push(candidate); }
                }
                const std::size_t active = row + 1;
                if (checkpoint < kQueryOps && active == kInitialRows + (checkpoint + 1) * kInsertOpsPerQuery * kInsertBatchRows) {
                    for (std::size_t j = 0; j < qn; ++j) {
                        std::vector<std::pair<float, std::uint32_t>> sorted;
                        auto heap = heaps[j];
                        while (!heap.empty()) { sorted.push_back(heap.top()); heap.pop(); }
                        std::sort(sorted.begin(), sorted.end());
                        auto& target = truth[checkpoint * kQueries + q0 + j];
                        for (std::size_t k = 0; k < kTopK; ++k) target[k] = sorted[k].second;
                    }
                    ++checkpoint;
                    if (checkpoint == kQueryOps) break;
                }
            }
            if (checkpoint != kQueryOps) throw std::runtime_error("truth pass did not reach all checkpoints");
            if (q0 % (qblock * 16) == 0 || q0 + qn >= kQueries)
                std::cout << "truth query_block=" << q0 << '-' << (q0 + qn - 1)
                          << '/' << kQueries << '\n' << std::flush;
        }
    };
    std::vector<std::thread> pool;
    for (std::size_t i = 0; i < workers; ++i) pool.emplace_back(work);
    for (auto& t : pool) t.join();
    return truth;
}

template <typename Queries>
std::vector<Ids> calculate_truth_u8(const MappedU8bin& data, const Queries& queries) {
    const std::size_t query_count = queries.rows();
    const std::size_t truth_rows = kQueryOps * query_count;
    std::vector<Ids> truth(truth_rows);
    constexpr std::size_t qblock = 8;
    std::atomic<std::size_t> next{0};
    const unsigned hc = std::thread::hardware_concurrency();
    const std::size_t workers = std::max<std::size_t>(1, std::min<std::size_t>(8, hc ? hc : 4));
    auto work = [&]() {
        for (;;) {
            const std::size_t q0 = next.fetch_add(qblock);
            if (q0 >= query_count) break;
            const std::size_t qn = std::min(qblock, query_count - q0);
            std::vector<float> query_values(qn * kDimension);
            for (std::size_t j = 0; j < qn; ++j) {
                const auto* source = queries.row(q0 + j);
                float* target = query_values.data() + j * kDimension;
                if constexpr (uses_u8_input_v<Queries>) {
                    for (std::size_t d = 0; d < kDimension; ++d)
                        target[d] = static_cast<float>(source[d]);
                } else {
                    std::copy_n(source, kDimension, target);
                }
            }
            std::vector<std::priority_queue<std::pair<float, std::uint32_t>>> heaps(qn);
            std::size_t checkpoint = 0;
            for (std::size_t row = 0; row < data.rows(); ++row) {
                const std::uint8_t* x = data.row(row);
                for (std::size_t j = 0; j < qn; ++j) {
                    auto& heap = heaps[j];
                    const float d = simd_squared_l2_u8_f32(
                        x, query_values.data() + j * kDimension, kDimension);
                    const std::pair<float, std::uint32_t> candidate{
                        d, static_cast<std::uint32_t>(row)};
                    if (heap.size() < kTopK) heap.push(candidate);
                    else if (candidate < heap.top()) { heap.pop(); heap.push(candidate); }
                }
                const std::size_t active = row + 1;
                if (checkpoint < kQueryOps && active == kInitialRows +
                    (checkpoint + 1) * kInsertOpsPerQuery * kInsertBatchRows) {
                    for (std::size_t j = 0; j < qn; ++j) {
                        std::vector<std::pair<float, std::uint32_t>> sorted;
                        auto heap = heaps[j];
                        while (!heap.empty()) { sorted.push_back(heap.top()); heap.pop(); }
                        std::sort(sorted.begin(), sorted.end());
                        auto& target = truth[checkpoint * query_count + q0 + j];
                        for (std::size_t k = 0; k < kTopK; ++k) target[k] = sorted[k].second;
                    }
                    ++checkpoint;
                    if (checkpoint == kQueryOps) break;
                }
            }
            if (checkpoint != kQueryOps) throw std::runtime_error("uint8 truth pass missed a checkpoint");
            if (q0 % (qblock * 16) == 0 || q0 + qn >= query_count)
                std::cout << "truth_u8 query_block=" << q0 << '-' << (q0 + qn - 1)
                          << '/' << query_count << '\n' << std::flush;
        }
    };
    std::vector<std::thread> pool;
    for (std::size_t i = 0; i < workers; ++i) pool.emplace_back(work);
    for (auto& t : pool) t.join();
    return truth;
}

std::vector<Ids> calculate_truth_prefix(const MappedFbin& data,
                                        const MappedFbin& queries,
                                        std::size_t rows) {
    if (rows == 0 || rows > data.rows())
        throw std::runtime_error("invalid row count for prefix groundtruth");
    std::vector<Ids> truth(kQueries);
    constexpr std::size_t qblock = 8;
    std::atomic<std::size_t> next{0};
    const unsigned hc = std::thread::hardware_concurrency();
    const std::size_t workers = std::max<std::size_t>(1, std::min<std::size_t>(8, hc ? hc : 4));
    auto work = [&]() {
        for (;;) {
            const std::size_t q0 = next.fetch_add(qblock);
            if (q0 >= kQueries) break;
            const std::size_t qn = std::min(qblock, kQueries - q0);
            std::vector<std::priority_queue<std::pair<float, std::uint32_t>>> heaps(qn);
            for (std::size_t row = 0; row < rows; ++row) {
                const float* x = data.row(row);
                for (std::size_t j = 0; j < qn; ++j) {
                    auto& heap = heaps[j];
                    const float d = simd_squared_l2(queries.row(q0 + j), x, kDimension);
                    const std::pair<float, std::uint32_t> candidate{
                        d, static_cast<std::uint32_t>(row)};
                    if (heap.size() < kTopK) heap.push(candidate);
                    else if (candidate < heap.top()) { heap.pop(); heap.push(candidate); }
                }
            }
            for (std::size_t j = 0; j < qn; ++j) {
                std::vector<std::pair<float, std::uint32_t>> sorted;
                auto heap = heaps[j];
                while (!heap.empty()) { sorted.push_back(heap.top()); heap.pop(); }
                std::sort(sorted.begin(), sorted.end());
                for (std::size_t k = 0; k < kTopK; ++k)
                    truth[q0 + j][k] = sorted[k].second;
            }
            if (q0 % (qblock * 16) == 0 || q0 + qn >= kQueries)
                std::cout << "initial_truth query_block=" << q0 << '-'
                          << q0 + qn - 1 << '/' << kQueries << '\n' << std::flush;
        }
    };
    std::vector<std::thread> pool;
    for (std::size_t i = 0; i < workers; ++i) pool.emplace_back(work);
    for (auto& t : pool) t.join();
    return truth;
}

void ensure_empty_or_create(const std::filesystem::path& path) {
    if (std::filesystem::exists(path)) {
        if (!std::filesystem::is_directory(path) || std::filesystem::directory_iterator(path) != std::filesystem::directory_iterator())
            throw std::runtime_error("output directory must be new or empty: " + path.string());
    } else std::filesystem::create_directories(path);
}

template <typename Data, typename QueryRows>
void run_prd_final_ih(const std::filesystem::path& output_root,
                      const std::filesystem::path& data_path,
                      const std::filesystem::path& query_path,
                      const std::filesystem::path& truth_path,
                      const Data& data, const QueryRows& queries,
                      float load_split_fraction,
                      float capacity_sqrt_n_split_multiplier,
                      bool query_every_stage = false,
                      const std::filesystem::path& reusable_initial_index = {}) {
    ensure_empty_or_create(output_root);
    if (data.rows() != 10'000'000 || queries.rows() < kQueries)
        throw std::runtime_error("expected 10M base vectors and at least the configured query sample");
    if (!std::isfinite(load_split_fraction) || load_split_fraction <= 0.0f ||
        !std::isfinite(capacity_sqrt_n_split_multiplier) || capacity_sqrt_n_split_multiplier <= 0.0f)
        throw std::runtime_error("invalid load/capacity split threshold");

    std::ifstream truth_file(truth_path, std::ios::binary);
    char truth_magic[8]{};
    std::uint32_t truth_shape[3]{};
    truth_file.read(truth_magic, sizeof(truth_magic));
    truth_file.read(reinterpret_cast<char*>(truth_shape), sizeof(truth_shape));
    constexpr std::uint64_t truth_header_bytes = 8 + 3 * sizeof(std::uint32_t);
    constexpr std::uint64_t truth_stage_bytes =
        static_cast<std::uint64_t>(kQueries) * sizeof(Ids);
    if (!truth_file || std::string(truth_magic, sizeof(truth_magic)) != "MSTGTH01" ||
        truth_shape[0] != kQueryOps || truth_shape[1] != kQueries ||
        truth_shape[2] != kTopK || std::filesystem::file_size(truth_path) !=
        truth_header_bytes + kQueryOps * truth_stage_bytes)
        throw std::runtime_error("expected exact MSTuring stage truth [100,10000,100]");
    truth_file.seekg(static_cast<std::streamoff>(truth_header_bytes +
        (query_every_stage ? 0 : (kQueryOps - 1) * truth_stage_bytes)));
    std::vector<Ids> final_truth(kQueries);
    if (!query_every_stage) {
        truth_file.read(reinterpret_cast<char*>(final_truth.data()),
                        static_cast<std::streamsize>(truth_stage_bytes));
        if (!truth_file) throw std::runtime_error("cannot read final 10M exact ground truth");
    }

    const auto final_index_path = output_root / "final_prd_ivf.index.bin";
    const auto local_initial_index_path = output_root / "initial_prd_ivf.index.bin";
    const auto initial_index_path = reusable_initial_index.empty()
        ? local_initial_index_path : reusable_initial_index;
    std::ofstream manifest(output_root / "manifest.txt");
    std::ofstream operations(output_root / "insert_operations.csv");
    std::ofstream per_query(output_root / "per_query.csv");
    std::ofstream summary(output_root / "final_summary.csv");
    std::ofstream stage_summary(output_root / "stage_summary.csv");
    std::ofstream query_totals(output_root / "query_total_summary.csv");
    std::ofstream global_center_move(output_root / "global_center_move.csv");
    if (!manifest || !operations || !per_query || !summary || !stage_summary ||
        !query_totals || !global_center_move)
        throw std::runtime_error("cannot create final-only PRD experiment outputs");
    manifest << "experiment=" << (uses_u8_input_v<Data> ? "YFCC-10M" : "MSTuring-10M")
             << " IH, PRD-IVF local reassignment"
             << (query_every_stage ? ", 100 query stages" : ", final query only") << "\n"
             << "dataset=" << data_path.string() << "\nqueries=" << query_path.string()
             << "\nexact_groundtruth=" << truth_path.string()
             << "\ninitial_vectors=" << kInitialRows << "\ninitial_nlist=" << kInitialNlist
             << "\ninsert_batch_size=" << kInsertBatchRows << "\ninsert_batches=" << kInsertOps
             << "\nquery_schedule=" << (query_every_stage
                    ? "query 10K vectors after each group of 9 insert+maintenance batches (100 stages)"
                    : "insert and maintain all 900 batches, then query once")
             << "\nmerge_rule=route each source vector of a short leaf to its nearest other live centroid; recenter each touched receiver; later split passes handle oversized receivers"
             << "\nprd_rule=every PRD-marked leaf directly attempts a split; no gain-based recenter; degenerate PRD splits are counted without recentering"
              << "\nqueries_per_stage=" << kQueries << "\nmetric=squared L2"
              << "\nquery_budget=floor(current active vectors/20), including centroid and vector distances"
             << "\nprd_threshold=0.05\ndrift_criterion=GlobalFitOnly (maintenance marking uses mean SSE only)\nload_split_fraction=" << load_split_fraction
             << "\ncapacity_sqrt_n_split_multiplier=" << capacity_sqrt_n_split_multiplier
             << "\ncapacity_split_rule=when live nlist <= sqrt(active vectors), split a leaf only if size > multiplier*sqrt(active vectors); disabled above sqrt(active vectors)"
             << "\ncapacity_growth_split_factor=3.0 (inactive while sqrt-N threshold is enabled)\nshort_list_fraction=0.20"
             << "\nmerge_rule=route each source vector of a short leaf to its nearest other live centroid; recenter each touched receiver; later split passes handle oversized receivers"
             << "\nprd_rule=every PRD-marked leaf directly attempts a split; no gain-based recenter; degenerate PRD splits are counted without recentering"
             << "\nlocal_refinement_radius=50"
             << "\nlocal_refinement_iterations=1"
             << "\nfinal_global_center_action=move every non-empty center to its already-maintained per-cluster mean statistic; no vector reassignment"
             << "\nref_mean_refresh=false (historical baseline retained for cumulative drift)"
             << "\nref_p95_refresh=false (unused by GlobalFitOnly maintenance marking)"
              << "\nLVQ=false\nearly_stop=false\nseed=" << kSeed
             << "\nwarmup_queries=" << (query_every_stage ? "100 against initial index (excluded from measured query results)" : "none; only the final query pass is measured")
             << "\ninitial_index_snapshot=" << initial_index_path.string()
             << (reusable_initial_index.empty()
                    ? " (saved immediately after training and before any inserts)"
                    : " (reused from the previously saved initial index)")
             << "\nfinal_index=" << final_index_path.string() << '\n';
    operations << "insert_batch,active_vectors,nlist_before,nlist_after,load_target_T,insert_ms,maintenance_ms,"
                   "prd_maintenance_ms,global_center_move_ms,"
                   "checked_partitions,marked_partitions,recentered_partitions,load_splits,capacity_splits,prd_splits,"
                   "total_splits,merges,merge_recipient_partitions,merge_routed_vectors,empty_removed,prd_split_failures,local_refinement_partitions,local_refinement_vectors,"
                   "local_refinement_reassigned_vectors\n";
    per_query << "query_stage,active_vectors,nlist,query_id,budget,center_distances,scanned_vectors,total_distances,partitions_scanned,"
                 "hits_at_10,recall_at_10,hits_at_100,recall_at_100,query_ms\n";
    stage_summary << "query_stage,active_vectors,nlist,budget,recall_at_10,recall_at_100,"
                     "mean_query_ms,p95_query_ms,total_query_seconds,mean_center_distances,"
                     "mean_scanned_vectors,mean_total_distances,insert_batches_since_query,"
                     "load_splits,capacity_splits,prd_splits,merges,local_refinement_reassigned_vectors\n";
    query_totals << "query_stages,queries_per_stage,total_queries,total_query_seconds,"
                     "mean_query_ms,mean_center_distances,mean_scanned_vectors,mean_total_distances,"
                     "overall_recall_at_10,overall_recall_at_100\n";
    summary << "indexed_vectors,initial_nlist,final_nlist,insert_batches,insert_batch_size,load_split_fraction,capacity_sqrt_n_split_multiplier,"
                "local_refinement_radius,local_refinement_iterations,build_ms,total_insert_ms,total_maintenance_ms,"
                "total_prd_maintenance_ms,total_global_center_move_ms,"
                "query_count,budget_per_query,recall_at_10,recall_at_100,mean_query_ms,p95_query_ms,total_query_seconds,"
                "mean_center_distances,mean_scanned_vectors,mean_total_distances,total_load_splits,total_capacity_splits,"
                "total_prd_splits,total_splits,total_merges,total_merge_recipient_partitions,total_merge_routed_vectors,total_empty_removed,total_prd_split_failures,total_local_refinement_partitions,"
                "total_local_refinement_vectors,total_local_refinement_reassigned_vectors,global_move_ms,"
                "global_centers_moved,global_vectors_covered,index_bytes\n";
    global_center_move << "insert_batch,active_vectors,nlist,action,centers_moved,vectors_covered,elapsed_ms,total_sse_before,total_sse_after,"
                          "mean_sse_before,mean_sse_after,vector_reassignments\n";

    AdaptiveHierarchicalConfig config;
    config.dimension = kDimension;
    config.initial_leaf_nlist = kInitialNlist;
    config.kmeans_iterations = kKmeansIterations;
    config.seed = kSeed;
    config.prd_threshold = 0.05f;
    config.drift_criterion = DriftCriterion::GlobalFitOnly;
    config.lvq8_storage = false;
    config.lvq4_storage = false;
    config.uint8_storage = uses_u8_input_v<Data>;
    config.angular_metric = false;
    config.automatic_maintenance = true;
    config.load_split_fraction = load_split_fraction;
    config.capacity_sqrt_n_split_multiplier = capacity_sqrt_n_split_multiplier;
    config.split_local_refinement_radius = 50;
    config.split_local_refinement_iterations = 1;
    double build_ms = 0.0, initial_index_load_ms = 0.0;
    AdaptiveHierarchicalIvf index = [&]() {
        if (!reusable_initial_index.empty()) {
            const auto load_start = std::chrono::steady_clock::now();
            auto loaded = AdaptiveHierarchicalIvf::load(reusable_initial_index.string());
            initial_index_load_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - load_start).count();
            const auto& loaded_config = loaded.config();
            if (loaded.size() != kInitialRows || loaded.leaf_nlist() != kInitialNlist ||
                loaded_config.dimension != kDimension || !loaded_config.uint8_storage ||
                loaded_config.lvq8_storage || loaded_config.lvq4_storage || loaded_config.angular_metric ||
                std::fabs(loaded_config.prd_threshold - 0.05f) > 1e-6f ||
                std::fabs(loaded_config.load_split_fraction - load_split_fraction) > 1e-6f ||
                std::fabs(loaded_config.capacity_sqrt_n_split_multiplier - capacity_sqrt_n_split_multiplier) > 1e-6f)
                throw std::runtime_error("saved initial PRD index metadata does not match this YFCC workload");
            loaded.set_split_local_refinement(50, 1);
            return loaded;
        }
        AdaptiveHierarchicalIvf built(config);
        const auto build_start = std::chrono::steady_clock::now();
        if constexpr (uses_u8_input_v<Data>) built.train_u8(data.row(0), kInitialRows);
        else built.train(data.row(0), kInitialRows);
        build_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - build_start).count();
        return built;
    }();
    std::ofstream build_file(output_root / "build.txt");
    build_file << "initial_build_ms=" << std::setprecision(10) << build_ms
               << "\ninitial_index_load_ms=" << initial_index_load_ms
               << "\ninitial_vectors=" << kInitialRows << "\ninitial_nlist="
               << index.leaf_nlist() << "\ninitial_index_source="
               << initial_index_path.string() << '\n';
    build_file.close();
    if (reusable_initial_index.empty()) {
        const auto initial_save_start = std::chrono::steady_clock::now();
        index.save(initial_index_path.string());
        const double initial_save_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - initial_save_start).count();
        std::ofstream initial_save_file(output_root / "initial_index_save.txt");
        initial_save_file << "initial_index=" << initial_index_path.string()
                          << "\nsave_ms=" << std::setprecision(10) << initial_save_ms
                          << "\nindex_bytes=" << std::filesystem::file_size(initial_index_path)
                          << "\ncontains_vectors=" << kInitialRows
                          << "\nnlist=" << index.leaf_nlist() << '\n';
    }
    std::cout << "prd_final_ih initial_index_ready build_ms=" << std::fixed
              << std::setprecision(3) << build_ms << " nlist=" << index.leaf_nlist()
              << " load_split_fraction=" << load_split_fraction
              << " capacity_sqrt_n_split_multiplier=" << capacity_sqrt_n_split_multiplier
              << " local_refinement_radius=50 local_refinement_iterations=1"
              << " initial_index=" << initial_index_path.string()
              << " initial_index_load_ms=" << initial_index_load_ms << '\n' << std::flush;

    double total_insert_ms = 0.0, total_maintenance_ms = 0.0;
    double total_prd_maintenance_ms = 0.0, total_global_center_move_ms = 0.0;
    std::uint64_t total_global_centers_moved = 0, total_global_vectors_covered = 0;
    std::uint64_t total_checked = 0, total_marked = 0, total_recentered = 0;
    std::uint64_t total_load_splits = 0, total_capacity_splits = 0, total_prd_splits = 0;
    std::uint64_t total_merges = 0, total_merge_recipient_partitions = 0;
    std::uint64_t total_merge_routed_vectors = 0, total_empty_removed = 0;
    std::uint64_t total_prd_split_failures = 0;
    std::uint64_t total_refinement_partitions = 0, total_refinement_vectors = 0;
    std::uint64_t total_refinement_reassigned = 0;
    struct QueryStageResult {
        QueryStats stats;
        std::uint64_t hits10 = 0, hits100 = 0;
        std::size_t active = 0, budget = 0, nlist = 0;
        double elapsed_ms = 0.0;
    };
    std::vector<std::uint32_t> truth_marks10(data.rows(), 0);
    std::vector<std::uint32_t> truth_marks100(data.rows(), 0);
    std::uint64_t all_query_hits10 = 0, all_query_hits100 = 0;
    std::uint64_t all_query_centers = 0, all_query_vectors = 0, all_query_distances = 0;
    double all_query_ms = 0.0;
    std::size_t all_query_count = 0;
    std::size_t stage_load_splits = 0, stage_capacity_splits = 0, stage_prd_splits = 0;
    std::size_t stage_merges = 0, stage_refinement_reassigned = 0;
    QueryStageResult last_query;
    auto run_query_stage = [&](std::size_t stage, std::size_t active,
                               const std::vector<Ids>& stage_truth) {
        QueryStageResult result;
        result.active = active;
        result.budget = active / 20;
        result.nlist = index.leaf_nlist();
        QuerySearchOptions options;
        options.enable_early_stop = false;
        options.max_total_distance_computations = result.budget;
        for (std::size_t qi = 0; qi < kQueries; ++qi) {
            const std::uint32_t mark = static_cast<std::uint32_t>(stage * kQueries + qi + 1);
            for (std::size_t rank = 0; rank < kTopK; ++rank) {
                const std::uint32_t id = stage_truth[qi][rank];
                if (id >= active) throw std::runtime_error("stage truth contains an ID outside the active prefix");
                truth_marks100[id] = mark;
                if (rank < 10) truth_marks10[id] = mark;
            }
            QuerySearchStats stats;
            const auto query_start = std::chrono::steady_clock::now();
            const auto found = index.search_with_options(queries.row(qi), kTopK, options, &stats);
            const double elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - query_start).count();
            if (found.size() != kTopK || stats.total_distance_computations != result.budget ||
                stats.center_distance_computations != index.leaf_nlist())
                throw std::runtime_error("PRD stage query violated fixed active-N/20 budget or top-100 output");
            std::uint64_t query_hits10 = 0, query_hits100 = 0;
            for (std::size_t rank = 0; rank < kTopK; ++rank) {
                if (found[rank].id < 0 || static_cast<std::size_t>(found[rank].id) >= active)
                    throw std::runtime_error("PRD stage query returned an ID outside the active prefix");
                const auto id = static_cast<std::size_t>(found[rank].id);
                if (truth_marks100[id] == mark) ++query_hits100;
                if (rank < 10 && truth_marks10[id] == mark) ++query_hits10;
            }
            result.hits10 += query_hits10;
            result.hits100 += query_hits100;
            result.stats.center += stats.center_distance_computations;
            result.stats.vectors += stats.scanned_vectors;
            result.stats.total += stats.total_distance_computations;
            result.stats.partitions += stats.scanned_partitions;
            result.stats.ms.push_back(elapsed);
            all_query_hits10 += query_hits10;
            all_query_hits100 += query_hits100;
            all_query_centers += stats.center_distance_computations;
            all_query_vectors += stats.scanned_vectors;
            all_query_distances += stats.total_distance_computations;
            all_query_ms += elapsed;
            ++all_query_count;
            per_query << stage + 1 << ',' << active << ',' << index.leaf_nlist() << ',' << qi << ','
                      << result.budget << ',' << stats.center_distance_computations << ','
                      << stats.scanned_vectors << ',' << stats.total_distance_computations << ','
                      << stats.scanned_partitions << ',' << query_hits10 << ','
                      << std::setprecision(8) << static_cast<double>(query_hits10) / 10.0 << ','
                      << query_hits100 << ',' << static_cast<double>(query_hits100) / 100.0 << ','
                      << elapsed << '\n';
            if ((qi + 1) % 1000 == 0) {
                per_query.flush();
                if ((stage + 1) % 10 == 0 || stage + 1 == kQueryOps)
                    std::cout << "prd_full_ih query_stage=" << stage + 1 << '/' << kQueryOps
                              << " queries=" << qi + 1 << '/' << kQueries
                              << " active=" << active << '\n' << std::flush;
            }
        }
        result.elapsed_ms = std::accumulate(result.stats.ms.begin(), result.stats.ms.end(), 0.0);
        stage_summary << stage + 1 << ',' << active << ',' << result.nlist << ',' << result.budget << ','
                      << std::setprecision(10)
                      << static_cast<double>(result.hits10) / (kQueries * 10.0) << ','
                      << static_cast<double>(result.hits100) / (kQueries * 100.0) << ','
                      << result.elapsed_ms / kQueries << ',' << percentile(result.stats.ms, 0.95) << ','
                      << result.elapsed_ms / 1000.0 << ','
                      << static_cast<double>(result.stats.center) / kQueries << ','
                      << static_cast<double>(result.stats.vectors) / kQueries << ','
                      << static_cast<double>(result.stats.total) / kQueries << ','
                      << kInsertOpsPerQuery << ',' << stage_load_splits << ',' << stage_capacity_splits << ','
                      << stage_prd_splits << ',' << stage_merges << ',' << stage_refinement_reassigned << '\n';
        stage_summary.flush(); per_query.flush();
        stage_load_splits = stage_capacity_splits = stage_prd_splits = 0;
        stage_merges = stage_refinement_reassigned = 0;
        const double mean_ms = result.elapsed_ms / kQueries;
        if ((stage + 1) % 10 == 0 || stage + 1 == kQueryOps)
            std::cout << "prd_full_ih query_checkpoint=" << stage + 1 << '/' << kQueryOps
                      << " active=" << active << " nlist=" << result.nlist << " budget=" << result.budget
                      << " recall@10=" << static_cast<double>(result.hits10) / (kQueries * 10.0)
                      << " recall@100=" << static_cast<double>(result.hits100) / (kQueries * 100.0)
                      << " mean_query_ms=" << mean_ms << " p95_ms=" << percentile(result.stats.ms, 0.95)
                      << " stage_query_seconds=" << result.elapsed_ms / 1000.0 << '\n' << std::flush;
        return result;
    };
    if (query_every_stage) {
        QuerySearchOptions warmup_options;
        warmup_options.enable_early_stop = false;
        warmup_options.max_total_distance_computations = kInitialRows / 20;
        for (std::size_t qi = 0; qi < 100; ++qi)
            (void)index.search_with_options(queries.row(qi), kTopK, warmup_options, nullptr);
        std::cout << "prd_full_ih warmup_done queries=100 budget="
                  << warmup_options.max_total_distance_computations << '\n' << std::flush;
    }
    if (query_every_stage) {
        // Full stage truth is read in sequence alongside the 100 query checkpoints.
    }
    for (std::size_t batch = 0; batch < kInsertOps; ++batch) {
        const std::size_t begin = kInitialRows + batch * kInsertBatchRows;
        const std::size_t nlist_before = index.leaf_nlist();
        index.begin_update_batch();
        const auto insert_start = std::chrono::steady_clock::now();
        if constexpr (uses_u8_input_v<Data>) index.add_u8(data.row(begin), kInsertBatchRows);
        else index.add(data.row(begin), kInsertBatchRows);
        const double insert_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - insert_start).count();
        const auto maintenance_start = std::chrono::steady_clock::now();
        const MaintenanceReport report = index.end_update_batch();
        const double maintenance_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - maintenance_start).count();
        const std::size_t active = begin + kInsertBatchRows;
        const auto global_move_start = std::chrono::steady_clock::now();
        const GlobalCenterRefreshReport global_move = index.refresh_all_centroids_to_means();
        const double global_move_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - global_move_start).count();
        if (global_move.vectors_covered != active)
            throw std::runtime_error("per-batch global center action did not cover every active vector");
        const double batch_total_maintenance_ms = maintenance_ms + global_move_ms;
        total_insert_ms += insert_ms;
        total_prd_maintenance_ms += maintenance_ms;
        total_global_center_move_ms += global_move_ms;
        total_maintenance_ms += batch_total_maintenance_ms;
        total_global_centers_moved += global_move.centers_updated;
        total_global_vectors_covered += global_move.vectors_covered;
        total_checked += report.checked_partitions;
        total_marked += report.marked_partitions;
        total_recentered += report.recentered_partitions;
        total_load_splits += report.load_split_partitions;
        total_capacity_splits += report.capacity_split_partitions;
        total_prd_splits += report.prd_split_partitions;
        total_prd_split_failures += report.prd_split_failures;
        total_merges += report.merged_short_partitions;
        total_merge_recipient_partitions += report.merge_recipient_partitions;
        total_merge_routed_vectors += report.merge_routed_vectors;
        total_empty_removed += report.removed_empty_partitions;
        total_refinement_partitions += report.local_refinement_partitions;
        total_refinement_vectors += report.local_refinement_vectors;
        total_refinement_reassigned += report.local_refinement_reassigned_vectors;
        stage_load_splits += report.load_split_partitions;
        stage_capacity_splits += report.capacity_split_partitions;
        stage_prd_splits += report.prd_split_partitions;
        stage_merges += report.merged_short_partitions;
        stage_refinement_reassigned += report.local_refinement_reassigned_vectors;
        operations << batch + 1 << ',' << active << ',' << nlist_before << ','
                   << index.leaf_nlist() << ',' << std::setprecision(8)
                   << report.load_target_T << ',' << insert_ms << ',' << batch_total_maintenance_ms << ','
                   << maintenance_ms << ',' << global_move_ms << ','
                   << report.checked_partitions << ',' << report.marked_partitions << ','
                   << report.recentered_partitions << ',' << report.load_split_partitions << ','
                   << report.capacity_split_partitions << ',' << report.prd_split_partitions << ','
                   << report.total_split_partitions << ',' << report.merged_short_partitions << ','
                   << report.merge_recipient_partitions << ',' << report.merge_routed_vectors << ','
                   << report.removed_empty_partitions << ',' << report.prd_split_failures << ','
                   << report.local_refinement_partitions << ','
                   << report.local_refinement_vectors << ','
                   << report.local_refinement_reassigned_vectors << '\n';
        if ((batch + 1) % 10 == 0) operations.flush();
        global_center_move << batch + 1 << ',' << active << ',' << index.leaf_nlist()
                           << ",move_centers_to_stored_cluster_means," << global_move.centers_updated << ','
                           << global_move.vectors_covered << ',' << std::setprecision(10) << global_move_ms << ','
                           << global_move.total_sse_before << ',' << global_move.total_sse_after << ','
                           << global_move.total_sse_before / active << ','
                           << global_move.total_sse_after / active << ",0\n";
        if ((batch + 1) % 10 == 0) global_center_move.flush();
        if ((batch + 1) % 100 == 0) {
            std::cout << "prd_final_ih insert_progress=" << batch + 1 << '/' << kInsertOps
                      << " active=" << active << " nlist=" << index.leaf_nlist()
                      << " insert_ms_total=" << std::setprecision(1) << total_insert_ms
                      << " maintenance_ms_total=" << total_maintenance_ms
                      << " load_splits=" << total_load_splits
                      << " capacity_splits=" << total_capacity_splits
                      << " prd_splits=" << total_prd_splits
                      << " refinement_moved=" << total_refinement_reassigned << '\n' << std::flush;
        }
        if (query_every_stage && (batch + 1) % kInsertOpsPerQuery == 0) {
            const std::size_t stage = (batch + 1) / kInsertOpsPerQuery - 1;
            std::vector<Ids> stage_truth(kQueries);
            truth_file.read(reinterpret_cast<char*>(stage_truth.data()),
                            static_cast<std::streamsize>(truth_stage_bytes));
            if (!truth_file) throw std::runtime_error("cannot read exact truth for PRD query stage");
            last_query = run_query_stage(stage, active, stage_truth);
        }
    }
    operations.flush();
    if (index.size() != 10'000'000)
        throw std::runtime_error("final PRD index does not contain exactly 10M vectors");
    global_center_move.flush();
    manifest << "global_center_action_schedule=after each of the 900 insert-and-PRD-maintenance batches"
             << "\ntotal_global_center_move_ms=" << total_global_center_move_ms
             << "\ntotal_global_centers_moved=" << total_global_centers_moved
             << "\ntotal_global_vectors_covered=" << total_global_vectors_covered
             << "\nglobal_vector_reassignments=0\n";
    std::cout << "prd_final_ih per_batch_global_center_moves_done actions=" << kInsertOps
              << " centers_moved_total=" << total_global_centers_moved
              << " vectors_covered_total=" << total_global_vectors_covered
              << " elapsed_ms_total=" << std::fixed << std::setprecision(3)
              << total_global_center_move_ms << '\n' << std::flush;
    const auto save_start = std::chrono::steady_clock::now();
    index.save(final_index_path.string());
    const double save_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - save_start).count();

    if (!query_every_stage) {
        last_query = run_query_stage(kQueryOps - 1, index.size(), final_truth);
    }
    const std::size_t budget = last_query.budget;
    const double recall10 = static_cast<double>(last_query.hits10) / (kQueries * 10.0);
    const double recall100 = static_cast<double>(last_query.hits100) / (kQueries * 100.0);
    const double mean_query_ms = last_query.elapsed_ms / kQueries;
    const double total_query_seconds = last_query.elapsed_ms / 1000.0;
    const double overall_recall10 = static_cast<double>(all_query_hits10) / (all_query_count * 10.0);
    const double overall_recall100 = static_cast<double>(all_query_hits100) / (all_query_count * 100.0);
    query_totals << (query_every_stage ? kQueryOps : 1) << ',' << kQueries << ',' << all_query_count << ','
                 << std::setprecision(10) << all_query_ms / 1000.0 << ','
                 << all_query_ms / std::max<std::size_t>(1, all_query_count) << ','
                 << static_cast<double>(all_query_centers) / std::max<std::size_t>(1, all_query_count) << ','
                 << static_cast<double>(all_query_vectors) / std::max<std::size_t>(1, all_query_count) << ','
                 << static_cast<double>(all_query_distances) / std::max<std::size_t>(1, all_query_count) << ','
                 << overall_recall10 << ',' << overall_recall100 << '\n';
    query_totals.flush();
    const auto index_bytes = std::filesystem::file_size(final_index_path);
    summary << index.size() << ',' << kInitialNlist << ',' << index.leaf_nlist() << ','
            << kInsertOps << ',' << kInsertBatchRows << ',' << load_split_fraction << ','
            << capacity_sqrt_n_split_multiplier << ','
            << 50 << ',' << 1 << ',' << std::setprecision(10)
            << build_ms << ',' << total_insert_ms << ',' << total_maintenance_ms << ','
            << total_prd_maintenance_ms << ',' << total_global_center_move_ms << ','
            << kQueries << ',' << budget << ',' << recall10 << ',' << recall100 << ','
             << mean_query_ms << ',' << percentile(last_query.stats.ms, 0.95) << ',' << total_query_seconds << ','
             << static_cast<double>(last_query.stats.center) / kQueries << ','
             << static_cast<double>(last_query.stats.vectors) / kQueries << ','
             << static_cast<double>(last_query.stats.total) / kQueries << ','
            << total_load_splits << ',' << total_capacity_splits << ',' << total_prd_splits << ','
            << (total_load_splits + total_capacity_splits + total_prd_splits) << ','
            << total_merges << ',' << total_merge_recipient_partitions << ','
            << total_merge_routed_vectors << ',' << total_empty_removed << ','
            << total_prd_split_failures << ',' << total_refinement_partitions << ','
            << total_refinement_vectors << ',' << total_refinement_reassigned << ','
            << total_global_center_move_ms << ',' << total_global_centers_moved << ','
            << total_global_vectors_covered << ',' << index_bytes << '\n';
    std::ofstream save_file(output_root / "index_save.txt");
    save_file << "final_index=" << final_index_path.string() << "\nsave_ms=" << save_ms
              << "\nindex_bytes=" << index_bytes << '\n';
    std::ofstream complete(output_root / "COMPLETE.txt");
    complete << "status=complete\nindexed_vectors=" << index.size()
             << "\nfinal_nlist=" << index.leaf_nlist()
             << "\ninsert_batches=" << kInsertOps << "\nquery_stages=" << (query_every_stage ? kQueryOps : 1)
             << "\nqueries_per_stage=" << kQueries << "\ntotal_queries=" << all_query_count
             << "\nbudget_per_query=" << budget << "\nrecall_at_10=" << recall10
             << "\nrecall_at_100=" << recall100
             << "\noverall_recall_at_10=" << overall_recall10
             << "\noverall_recall_at_100=" << overall_recall100
             << "\ntotal_query_seconds=" << all_query_ms / 1000.0
             << "\ninitial_index=" << initial_index_path.string()
             << "\ntotal_merges=" << total_merges
             << "\ntotal_merge_recipient_partitions=" << total_merge_recipient_partitions
             << "\ntotal_merge_routed_vectors=" << total_merge_routed_vectors
             << "\nprd_split_failures=" << total_prd_split_failures
             << "\nfinal_index="
             << final_index_path.string() << '\n';
    std::cout << (query_every_stage ? "PRD_FULL_IH_COMPLETE" : "PRD_FINAL_IH_COMPLETE")
              << " active=" << index.size()
              << " nlist=" << index.leaf_nlist()
              << " recall@10=" << std::setprecision(6) << recall10
              << " recall@100=" << recall100 << " budget=" << budget
              << " mean_query_ms=" << mean_query_ms << " p95_query_ms="
              << percentile(last_query.stats.ms, 0.95) << " total_query_seconds="
              << all_query_ms / 1000.0 << " overall_recall@10=" << overall_recall10
              << " overall_recall@100=" << overall_recall100 << " total_insert_ms=" << total_insert_ms
              << " total_maintenance_ms=" << total_maintenance_ms
              << " load_splits=" << total_load_splits
              << " capacity_splits=" << total_capacity_splits
              << " prd_splits=" << total_prd_splits
              << " prd_split_failures=" << total_prd_split_failures
              << " merges=" << total_merges
              << " merge_recipient_partitions=" << total_merge_recipient_partitions
              << " merge_routed_vectors=" << total_merge_routed_vectors
              << " refinement_moved=" << total_refinement_reassigned
              << " output=" << output_root.string() << '\n' << std::flush;
}

void recompute_saved_prd_final_query(const std::filesystem::path& index_path,
                                    const std::filesystem::path& data_path,
                                    const std::filesystem::path& query_path,
                                    const std::filesystem::path& truth_path,
                                    const std::filesystem::path& output_root,
                                    std::size_t budget,
                                    const MappedFbin& queries,
                                    const MappedFbin& data) {
    if (!std::filesystem::is_directory(output_root) || queries.rows() < kQueries ||
        data.rows() != 10'000'000)
        throw std::runtime_error("saved PRD query inputs/output directory are invalid");
    std::cout << "prd_final_requery loading_index=" << index_path.string() << '\n' << std::flush;
    AdaptiveHierarchicalIvf index = AdaptiveHierarchicalIvf::load(index_path.string());
    if (index.size() != 10'000'000 || index.config().dimension != kDimension ||
        budget <= index.leaf_nlist())
        throw std::runtime_error("saved PRD index does not match final fixed-budget query");

    std::ifstream truth_file(truth_path, std::ios::binary);
    char truth_magic[8]{};
    std::uint32_t truth_shape[3]{};
    truth_file.read(truth_magic, sizeof(truth_magic));
    truth_file.read(reinterpret_cast<char*>(truth_shape), sizeof(truth_shape));
    constexpr std::uint64_t truth_header_bytes = 8 + 3 * sizeof(std::uint32_t);
    constexpr std::uint64_t truth_stage_bytes =
        static_cast<std::uint64_t>(kQueries) * sizeof(Ids);
    if (!truth_file || std::string(truth_magic, sizeof(truth_magic)) != "MSTGTH01" ||
        truth_shape[0] != kQueryOps || truth_shape[1] != kQueries ||
        truth_shape[2] != kTopK || std::filesystem::file_size(truth_path) !=
        truth_header_bytes + kQueryOps * truth_stage_bytes)
        throw std::runtime_error("expected exact MSTuring stage truth [100,10000,100]");
    truth_file.seekg(static_cast<std::streamoff>(
        truth_header_bytes + (kQueryOps - 1) * truth_stage_bytes));
    std::vector<Ids> exact(kQueries);
    truth_file.read(reinterpret_cast<char*>(exact.data()),
                    static_cast<std::streamsize>(truth_stage_bytes));
    if (!truth_file) throw std::runtime_error("cannot read final exact top-100 truth");

    QuerySearchOptions options;
    options.enable_early_stop = false;
    options.max_total_distance_computations = budget;
    for (std::size_t qi = 0; qi < 100; ++qi)
        (void)index.search_with_options(queries.row(qi), kTopK, options, nullptr);

    struct QueryRecord {
        std::size_t query_id = 0;
        std::uint64_t hits10 = 0, hits100 = 0;
        QuerySearchStats stats;
        double elapsed_ms = 0.0;
    };
    std::vector<QueryRecord> records;
    records.reserve(kQueries);
    std::uint64_t hits10_total = 0, hits100_total = 0;
    std::uint64_t center_total = 0, vector_total = 0, distance_total = 0;
    std::vector<double> query_ms;
    query_ms.reserve(kQueries);
    for (std::size_t qi = 0; qi < kQueries; ++qi) {
        std::unordered_set<std::uint32_t> exact10, exact100;
        exact10.reserve(20);
        exact100.reserve(200);
        for (std::size_t rank = 0; rank < 10; ++rank) exact10.insert(exact[qi][rank]);
        for (std::size_t rank = 0; rank < kTopK; ++rank) exact100.insert(exact[qi][rank]);
        QueryRecord record;
        record.query_id = qi;
        const auto start = std::chrono::steady_clock::now();
        const auto found = index.search_with_options(queries.row(qi), kTopK, options,
                                                     &record.stats);
        record.elapsed_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        if (found.size() != kTopK || record.stats.total_distance_computations != budget ||
            record.stats.center_distance_computations != index.leaf_nlist())
            throw std::runtime_error("saved PRD query violated fixed distance budget");
        for (std::size_t rank = 0; rank < kTopK; ++rank) {
            if (found[rank].id < 0 || static_cast<std::size_t>(found[rank].id) >= data.rows())
                throw std::runtime_error("saved PRD query returned invalid row ID");
            const auto id = static_cast<std::uint32_t>(found[rank].id);
            if (rank < 10 && exact10.count(id)) ++record.hits10;
            if (exact100.count(id)) ++record.hits100;
        }
        hits10_total += record.hits10;
        hits100_total += record.hits100;
        center_total += record.stats.center_distance_computations;
        vector_total += record.stats.scanned_vectors;
        distance_total += record.stats.total_distance_computations;
        query_ms.push_back(record.elapsed_ms);
        records.push_back(record);
        if ((qi + 1) % 1000 == 0)
            std::cout << "prd_final_requery progress=" << qi + 1 << '/' << kQueries << '\n'
                      << std::flush;
    }

    const double recall10 = static_cast<double>(hits10_total) / (kQueries * 10.0);
    const double recall100 = static_cast<double>(hits100_total) / (kQueries * 100.0);
    const double mean_ms = std::accumulate(query_ms.begin(), query_ms.end(), 0.0) / kQueries;
    const double total_seconds = std::accumulate(query_ms.begin(), query_ms.end(), 0.0) / 1000.0;
    {
        std::ofstream per_query(output_root / "per_query.csv", std::ios::trunc);
        if (!per_query) throw std::runtime_error("cannot rewrite corrected per-query metrics");
        per_query << "query_id,budget,center_distances,scanned_vectors,total_distances,partitions_scanned,"
                     "hits_at_10,recall_at_10,hits_at_100,recall_at_100,query_ms\n";
        for (const auto& record : records)
            per_query << record.query_id << ',' << budget << ','
                      << record.stats.center_distance_computations << ','
                      << record.stats.scanned_vectors << ','
                      << record.stats.total_distance_computations << ','
                      << record.stats.scanned_partitions << ',' << record.hits10 << ','
                      << std::setprecision(8) << static_cast<double>(record.hits10) / 10.0 << ','
                      << record.hits100 << ',' << static_cast<double>(record.hits100) / 100.0 << ','
                      << record.elapsed_ms << '\n';
    }

    const auto summary_path = output_root / "final_summary.csv";
    std::ifstream old_summary(summary_path);
    std::string header, row;
    std::getline(old_summary, header);
    std::getline(old_summary, row);
    if (!old_summary && row.empty())
        throw std::runtime_error("cannot read previous final summary for correction");
    auto split_csv = [](const std::string& line) {
        std::vector<std::string> fields;
        std::stringstream stream(line);
        std::string field;
        while (std::getline(stream, field, ',')) fields.push_back(field);
        return fields;
    };
    auto fields = split_csv(row);
    if (fields.size() != 31) throw std::runtime_error("unexpected final summary column count");
    fields[13] = std::to_string(recall10);
    fields[14] = std::to_string(recall100);
    fields[15] = std::to_string(mean_ms);
    fields[16] = std::to_string(percentile(query_ms, 0.95));
    fields[17] = std::to_string(total_seconds);
    fields[18] = std::to_string(static_cast<double>(center_total) / kQueries);
    fields[19] = std::to_string(static_cast<double>(vector_total) / kQueries);
    fields[20] = std::to_string(static_cast<double>(distance_total) / kQueries);
    old_summary.close();
    {
        std::ofstream corrected(summary_path, std::ios::trunc);
        if (!corrected) throw std::runtime_error("cannot write corrected final summary");
        corrected << header << '\n';
        for (std::size_t i = 0; i < fields.size(); ++i) {
            if (i) corrected << ',';
            corrected << fields[i];
        }
        corrected << '\n';
    }
    {
        std::ofstream note(output_root / "RECALL_CORRECTION.txt");
        note << "Recall@10 was corrected by rerunning the final query on the saved index.\n"
             << "The initial report mistakenly compared returned top-10 IDs to the exact top-100 set.\n"
             << "Correct metric: returned top-10 IDs are compared only with exact top-10 truth IDs.\n"
             << "Recall@10=" << std::setprecision(10) << recall10
             << "\nRecall@100=" << recall100 << '\n';
    }
    {
        std::ofstream complete(output_root / "COMPLETE.txt", std::ios::trunc);
        complete << "status=complete\nindexed_vectors=" << index.size()
                 << "\nfinal_nlist=" << index.leaf_nlist()
                 << "\ninsert_batches=" << kInsertOps << "\nquery_count=" << kQueries
                 << "\nbudget_per_query=" << budget << "\nrecall_at_10=" << recall10
                 << "\nrecall_at_100=" << recall100
                 << "\nrecall_at_10_recomputed_against_exact_top10=true\n";
    }
    std::cout << "PRD_FINAL_IH_REQUERY_CORRECTED recall@10=" << std::setprecision(8)
              << recall10 << " recall@100=" << recall100 << " budget=" << budget
              << " mean_query_ms=" << mean_ms << " p95_query_ms="
              << percentile(query_ms, 0.95) << '\n' << std::flush;
}

void run_prd_aps_initial(const std::filesystem::path& output_root,
                         const std::filesystem::path& data_path,
                         const std::filesystem::path& query_path,
                         const MappedFbin& data, const MappedFbin& queries) {
    ensure_empty_or_create(output_root);
    if (data.rows() < kInitialRows || queries.rows() < kQueries)
        throw std::runtime_error("expected at least 1M base vectors and 10K queries");

    std::ofstream manifest(output_root / "manifest.txt");
    std::ofstream summary(output_root / "summary.csv");
    std::ofstream per_query(output_root / "per_query.csv");
    if (!manifest || !summary || !per_query)
        throw std::runtime_error("cannot create initial PRD APS outputs");
    manifest << "experiment=MSTuring10M initial 1M prefix, PRD-IVF, APS query\n"
             << "dataset=" << data_path.string() << "\nqueries=" << query_path.string()
             << "\nindexed_vectors=" << kInitialRows << "\nnlist=" << kInitialNlist
             << "\nmetric=squared L2\nseed=" << kSeed << "\nLVQ=false"
             << "\naps_candidate_fraction=" << kApsCandidateFraction
             << "\naps_recall_target=" << kApsRecallTarget
             << "\naps_recompute_threshold=" << kApsRecomputeThreshold
             << "\nqueries=" << kQueries << "\ntop_k=" << kTopK
             << "\nwarmup_queries=100\n";
    summary << "method,indexed_vectors,nlist,build_ms,truth_compute_ms,query_count,"
               "aps_target,aps_estimated_recall,recall_at_100,mean_candidate_partitions,"
               "mean_scanned_partitions,mean_scanned_vectors,mean_query_ms,p95_query_ms,"
               "total_query_seconds\n";
    per_query << "query_id,recall_at_100,aps_estimated_recall,candidate_partitions,"
                  "scanned_partitions,scanned_vectors,query_ms\n";

    std::cout << "initial_truth_start rows=" << kInitialRows
              << " queries=" << kQueries << " topk=" << kTopK << '\n' << std::flush;
    const auto truth_start = std::chrono::steady_clock::now();
    const auto truth = calculate_truth_prefix(data, queries, kInitialRows);
    const double truth_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - truth_start).count();
    {
        std::ofstream truth_out(output_root / "groundtruth_top100_1m.bin", std::ios::binary);
        const char magic[8] = {'M','S','T','G','T','H','0','1'};
        const std::uint32_t shape[3] = {1, static_cast<std::uint32_t>(kQueries),
            static_cast<std::uint32_t>(kTopK)};
        truth_out.write(magic, sizeof(magic));
        truth_out.write(reinterpret_cast<const char*>(shape), sizeof(shape));
        truth_out.write(reinterpret_cast<const char*>(truth.data()),
            static_cast<std::streamsize>(truth.size() * sizeof(Ids)));
        if (!truth_out) throw std::runtime_error("failed writing initial 1M exact truth");
    }

    AdaptiveHierarchicalConfig cfg;
    cfg.dimension = kDimension;
    cfg.initial_leaf_nlist = kInitialNlist;
    cfg.kmeans_iterations = kKmeansIterations;
    cfg.seed = kSeed;
    cfg.lvq8_storage = false;
    cfg.lvq4_storage = false;
    cfg.angular_metric = false;
    cfg.automatic_maintenance = false;
    AdaptiveHierarchicalIvf index(cfg);
    const auto build_start = std::chrono::steady_clock::now();
    index.train(data.row(0), kInitialRows);
    const double build_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - build_start).count();
    const auto leaf_sizes = index.leaf_sizes();
    std::cout << "initial_prd_build_done_ms=" << std::fixed << std::setprecision(3)
              << build_ms << " nlist=" << index.leaf_nlist()
              << " warmup_queries=100 APS_target=0.9\n" << std::flush;
    for (std::size_t q = 0; q < 100; ++q)
        (void)index.search_aps(queries.row(q), kTopK, kApsCandidateFraction,
            kApsRecallTarget, kApsRecomputeThreshold, nullptr, nullptr);

    std::vector<std::uint32_t> marks(kInitialRows, 0);
    std::vector<double> query_ms;
    query_ms.reserve(kQueries);
    std::uint64_t hits = 0, scanned_vectors = 0, scanned_partitions = 0;
    std::uint64_t candidate_partitions = 0;
    double estimated_recall = 0.0;
    for (std::size_t qi = 0; qi < kQueries; ++qi) {
        const std::uint32_t mark = static_cast<std::uint32_t>(qi + 1);
        for (const std::uint32_t id : truth[qi]) marks[id] = mark;
        ApsSearchStats aps_stats;
        std::vector<std::size_t> probed;
        const auto query_start = std::chrono::steady_clock::now();
        const auto found = index.search_aps(queries.row(qi), kTopK,
            kApsCandidateFraction, kApsRecallTarget, kApsRecomputeThreshold,
            &probed, &aps_stats);
        const double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - query_start).count();
        if (found.size() != kTopK) throw std::runtime_error("APS did not return top-100");
        query_ms.push_back(elapsed);
        candidate_partitions += aps_stats.candidate_partitions;
        scanned_partitions += aps_stats.scanned_partitions;
        estimated_recall += aps_stats.estimated_recall;
        for (const std::size_t leaf : probed)
            if (leaf < leaf_sizes.size()) scanned_vectors += leaf_sizes[leaf];
        std::uint64_t query_hits = 0;
        for (const SearchResult& result : found) {
            if (result.id >= 0 && static_cast<std::size_t>(result.id) < kInitialRows &&
                marks[static_cast<std::size_t>(result.id)] == mark)
                ++query_hits;
        }
        hits += query_hits;
        per_query << qi << ',' << std::setprecision(8)
                  << static_cast<double>(query_hits) / kTopK << ','
                  << aps_stats.estimated_recall << ','
                  << aps_stats.candidate_partitions << ','
                  << aps_stats.scanned_partitions << ',';
        std::uint64_t query_scanned_vectors = 0;
        for (const std::size_t leaf : probed)
            if (leaf < leaf_sizes.size()) query_scanned_vectors += leaf_sizes[leaf];
        per_query << query_scanned_vectors << ',' << elapsed << '\n';
    }
    const double recall = static_cast<double>(hits) / (kQueries * kTopK);
    const double mean_estimated = estimated_recall / kQueries;
    const double mean_ms = std::accumulate(query_ms.begin(), query_ms.end(), 0.0) / kQueries;
    const double query_seconds = std::accumulate(query_ms.begin(), query_ms.end(), 0.0) / 1000.0;
    summary << "PRD-IVF," << kInitialRows << ',' << index.leaf_nlist() << ','
            << std::setprecision(8) << build_ms << ',' << truth_ms << ',' << kQueries << ','
            << kApsRecallTarget << ',' << mean_estimated << ',' << recall << ','
            << static_cast<double>(candidate_partitions) / kQueries << ','
            << static_cast<double>(scanned_partitions) / kQueries << ','
            << static_cast<double>(scanned_vectors) / kQueries << ',' << mean_ms << ','
            << percentile(query_ms, 0.95) << ',' << query_seconds << '\n';
    summary.flush();
    per_query.flush();
    std::ofstream complete(output_root / "COMPLETE.txt");
    complete << "status=complete\nmethod=PRD-IVF only\nquery=APS\n"
             << "indexed_vectors=1000000\nqueries=10000\ntopk=100\n";
    std::cout << "INITIAL_APS_COMPLETE recall@100=" << std::setprecision(6) << recall
              << " aps_estimated_recall=" << mean_estimated
              << " mean_query_ms=" << mean_ms << " p95_query_ms="
              << percentile(query_ms, 0.95) << " scanned_vectors_per_query="
              << static_cast<double>(scanned_vectors) / kQueries
              << " output=" << output_root.string() << '\n' << std::flush;
}

void run_prd_aps_full_build(const std::filesystem::path& output_root,
                            const std::filesystem::path& data_path,
                            const std::filesystem::path& query_path,
                            const std::filesystem::path& truth_path,
                            const MappedFbin& data, const MappedFbin& queries,
                            std::size_t nlist, bool run_aps_query = true,
                            double training_fraction = 0.15,
                            std::uint32_t training_seed = kSeed) {
    ensure_empty_or_create(output_root);
    if (data.rows() != 10'000'000 || queries.rows() < kQueries ||
        nlist == 0 || nlist > data.rows() || training_fraction <= 0.0 || training_fraction > 1.0)
        throw std::runtime_error("expected 10M vectors, at least 10K queries, and valid nlist");

    std::ifstream truth_file(truth_path, std::ios::binary);
    char truth_magic[8]{};
    std::uint32_t truth_shape[3]{};
    truth_file.read(truth_magic, sizeof(truth_magic));
    truth_file.read(reinterpret_cast<char*>(truth_shape), sizeof(truth_shape));
    if (!truth_file || std::string(truth_magic, sizeof(truth_magic)) != "MSTGTH01" ||
        truth_shape[0] != kQueryOps || truth_shape[1] != kQueries ||
        truth_shape[2] != kTopK)
        throw std::runtime_error("expected exact stage truth MSTGTH01 [100,10000,100]");
    constexpr std::uint64_t truth_header_bytes = 8 + 3 * sizeof(std::uint32_t);
    constexpr std::uint64_t truth_stage_bytes =
        static_cast<std::uint64_t>(kQueries) * sizeof(Ids);
    if (std::filesystem::file_size(truth_path) !=
        truth_header_bytes + kQueryOps * truth_stage_bytes)
        throw std::runtime_error("stage groundtruth file size does not match its header");
    truth_file.seekg(static_cast<std::streamoff>(
        truth_header_bytes + (kQueryOps - 1) * truth_stage_bytes));
    std::vector<Ids> truth(kQueries);
    truth_file.read(reinterpret_cast<char*>(truth.data()),
                    static_cast<std::streamsize>(truth_stage_bytes));
    if (!truth_file) throw std::runtime_error("cannot read final 10M exact groundtruth");

    std::ofstream manifest(output_root / "manifest.txt");
    std::ofstream summary(output_root / "aps_metrics.csv");
    std::ofstream per_query(output_root / "per_query.csv");
    if (!manifest || !summary || !per_query)
        throw std::runtime_error("cannot create full-build APS outputs");
    const std::size_t training_count = static_cast<std::size_t>(
        std::floor(static_cast<double>(data.rows()) * training_fraction));
    const bool full_training = training_count == data.rows();
    const unsigned detected_workers = std::thread::hardware_concurrency();
    const std::size_t build_worker_threads =
        std::max<std::size_t>(1, detected_workers ? detected_workers : 1);
    manifest << "experiment=MSTuring10M PRD-IVF build and APS query\n"
             << "dataset=" << data_path.string() << "\nqueries=" << query_path.string()
             << "\nexact_groundtruth=" << truth_path.string()
             << "\nindexed_vectors=" << data.rows() << "\ntraining_sample_vectors=" << training_count
             << "\ntraining_sample_fraction=" << training_fraction
             << "\ntraining_sample=" << (full_training ? "all vectors in source order" : "random without replacement")
             << "\nnlist=" << nlist
             << "\nkmeans_iterations=" << kKmeansIterations << "\nseed=" << training_seed
             << "\nbuild_worker_threads=" << build_worker_threads
             << "\nmetric=squared L2\nLVQ=false\nautomatic_maintenance=false"
             << "\nqueries=" << kQueries << "\ntop_k=" << kTopK
             << "\naps_candidate_fraction=" << kApsCandidateFraction
             << "\naps_recall_target=" << kApsRecallTarget
             << "\naps_recompute_threshold=" << kApsRecomputeThreshold
             << "\nwarmup_queries=100\nquery_stage=final 10M stage\n"
             << "index_internal_id_mapping=internal_to_original_row_ids.u32bin\n";
    summary << "indexed_vectors,nlist,kmeans_iterations,seed,build_ms,index_save_ms,"
               "query_count,aps_target,aps_estimated_recall,recall_at_100,"
               "mean_candidate_partitions,mean_scanned_partitions,mean_scanned_vectors,"
               "mean_query_ms,p95_query_ms,total_query_seconds\n";
    per_query << "query_id,recall_at_100,aps_estimated_recall,candidate_partitions,"
                  "scanned_partitions,scanned_vectors,query_ms\n";

    AdaptiveHierarchicalConfig cfg;
    cfg.dimension = kDimension;
    cfg.initial_leaf_nlist = nlist;
    cfg.kmeans_iterations = kKmeansIterations;
    cfg.seed = training_seed;
    cfg.worker_threads = build_worker_threads;
    cfg.lvq8_storage = false;
    cfg.lvq4_storage = false;
    cfg.angular_metric = false;
    cfg.automatic_maintenance = false;
    AdaptiveHierarchicalIvf index(cfg);
    std::cout << "full_aps build_started vectors=" << data.rows()
              << " training_rows=" << training_count << " nlist=" << nlist
              << " kmeans_iterations=" << kKmeansIterations
              << " seed=" << training_seed
              << " build_worker_threads=" << cfg.worker_threads;
    if (!full_training) std::cout << " sample_seed=" << training_seed;
    std::cout << '\n' << std::flush;
    const auto build_start = std::chrono::steady_clock::now();
    std::vector<std::uint8_t> is_training_row(data.rows(), 0);
    std::vector<std::uint32_t> original_to_internal(data.rows(),
        std::numeric_limits<std::uint32_t>::max());
    std::vector<std::uint32_t> internal_to_original;
    internal_to_original.reserve(data.rows());
    if (full_training) {
        for (std::uint32_t original_id = 0; original_id < data.rows(); ++original_id) {
            original_to_internal[original_id] = original_id;
            internal_to_original.push_back(original_id);
        }
        std::cout << "full_aps training_data_ready vectors=" << training_count
                  << "\nfull_aps kmeans_started\n" << std::flush;
        index.train(data.row(0), training_count);
    } else {
        std::vector<std::uint32_t> shuffled_rows(data.rows());
        std::iota(shuffled_rows.begin(), shuffled_rows.end(), 0u);
        std::mt19937 sample_rng(training_seed);
        std::shuffle(shuffled_rows.begin(), shuffled_rows.end(), sample_rng);
        shuffled_rows.resize(training_count);
        std::vector<float> training_vectors(training_count * kDimension);
        for (std::size_t i = 0; i < training_count; ++i) {
            const std::uint32_t original_id = shuffled_rows[i];
            is_training_row[original_id] = 1;
            original_to_internal[original_id] = static_cast<std::uint32_t>(i);
            internal_to_original.push_back(original_id);
            std::copy_n(data.row(original_id), kDimension,
                        training_vectors.data() + i * kDimension);
        }
        std::vector<std::uint32_t>().swap(shuffled_rows);
        std::cout << "full_aps training_sample_ready vectors=" << training_count
                  << "\nfull_aps kmeans_started\n" << std::flush;
        index.train(training_vectors.data(), training_count);
    }
    if (index.size() != training_count)
        throw std::runtime_error("training did not index the expected number of vectors");
    std::cout << "full_aps kmeans_done; adding_remaining_vectors="
              << (data.rows() - training_count) << '\n' << std::flush;
    constexpr std::size_t add_batch_size = 50'000;
    std::vector<float> add_buffer;
    add_buffer.reserve(add_batch_size * kDimension);
    std::vector<std::uint32_t> add_original_ids;
    add_original_ids.reserve(add_batch_size);
    std::size_t added_count = 0;
    const auto flush_add_batch = [&]() {
        if (add_original_ids.empty()) return;
        index.add(add_buffer.data(), add_original_ids.size());
        const std::size_t first_internal_id = internal_to_original.size();
        for (std::size_t i = 0; i < add_original_ids.size(); ++i) {
            const std::uint32_t original_id = add_original_ids[i];
            original_to_internal[original_id] = static_cast<std::uint32_t>(first_internal_id + i);
            internal_to_original.push_back(original_id);
        }
        added_count += add_original_ids.size();
        add_buffer.clear();
        add_original_ids.clear();
        if (added_count % 1'000'000 == 0 || added_count == data.rows() - training_count)
            std::cout << "full_aps add_progress=" << added_count << '/'
                      << (data.rows() - training_count) << '\n' << std::flush;
    };
    for (std::uint32_t original_id = 0; original_id < data.rows(); ++original_id) {
        if (full_training || is_training_row[original_id]) continue;
        const float* row = data.row(original_id);
        add_buffer.insert(add_buffer.end(), row, row + kDimension);
        add_original_ids.push_back(original_id);
        if (add_original_ids.size() == add_batch_size) flush_add_batch();
    }
    flush_add_batch();
    if (index.size() != data.rows() || internal_to_original.size() != data.rows())
        throw std::runtime_error("full-build insertion did not index every source vector exactly once");
    const double build_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - build_start).count();
    std::cout << "full_aps build_done_ms=" << std::fixed << std::setprecision(3)
              << build_ms << " nlist=" << index.leaf_nlist() << '\n' << std::flush;

    const auto index_path = output_root /
        ("prd_ivf_10m_nlist" + std::to_string(nlist) + ".index.bin");
    const auto save_start = std::chrono::steady_clock::now();
    index.save(index_path.string());
    const double index_save_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - save_start).count();
    const auto id_map_path = output_root / "internal_to_original_row_ids.u32bin";
    {
        std::ofstream id_map(id_map_path, std::ios::binary);
        const char map_magic[8] = {'M','S','T','I','D','M','0','1'};
        const std::uint64_t map_count = internal_to_original.size();
        id_map.write(map_magic, sizeof(map_magic));
        id_map.write(reinterpret_cast<const char*>(&map_count), sizeof(map_count));
        id_map.write(reinterpret_cast<const char*>(internal_to_original.data()),
                     static_cast<std::streamsize>(internal_to_original.size() * sizeof(std::uint32_t)));
        if (!id_map) throw std::runtime_error("failed writing index ID mapping sidecar");
    }
    manifest << "index_file=" << index_path.string() << "\n";
    manifest << "index_id_mapping_file=" << id_map_path.string()
             << "\nindex_id_mapping_format=MSTIDM01, uint64 count, then uint32 original row IDs\n";
    if (!run_aps_query) {
        manifest << "post_build_query_evaluation=skipped; index build only\n";
        summary.flush();
        per_query.flush();
        std::ofstream complete(output_root / "COMPLETE.txt");
        complete << "status=complete\noperation=index build only\nmethod=PRD-IVF\ntraining_fraction="
                 << training_fraction << "\ntraining_vectors=" << training_count << "\n"
                 << "indexed_vectors=" << data.rows() << "\nnlist=" << index.leaf_nlist()
                 << "\nseed=" << training_seed
                 << "\ntraining_sample_vectors=" << training_count << '\n';
        std::cout << "FULL_INDEX_BUILD_COMPLETE active=" << data.rows()
                  << " nlist=" << index.leaf_nlist() << " index=" << index_path.string()
                  << " mapping=" << id_map_path.string() << '\n' << std::flush;
        return;
    }
    const auto leaf_sizes = index.leaf_sizes();
    std::cout << "full_aps index_saved bytes=" << std::filesystem::file_size(index_path)
              << " save_ms=" << index_save_ms << " warmup_queries=100 APS_target=0.9\n"
              << std::flush;
    for (std::size_t q = 0; q < 100; ++q)
        (void)index.search_aps(queries.row(q), kTopK, kApsCandidateFraction,
            kApsRecallTarget, kApsRecomputeThreshold, nullptr, nullptr);

    std::vector<std::uint32_t> marks(data.rows(), 0);
    std::vector<double> query_ms;
    query_ms.reserve(kQueries);
    std::uint64_t hits = 0, scanned_vectors = 0, scanned_partitions = 0;
    std::uint64_t candidate_partitions = 0;
    double estimated_recall = 0.0;
    for (std::size_t qi = 0; qi < kQueries; ++qi) {
        const std::uint32_t mark = static_cast<std::uint32_t>(qi + 1);
        for (const std::uint32_t original_id : truth[qi]) {
            if (original_id >= data.rows() ||
                original_to_internal[original_id] == std::numeric_limits<std::uint32_t>::max())
                throw std::runtime_error("truth ID is outside the 10M index mapping");
            marks[original_to_internal[original_id]] = mark;
        }
        ApsSearchStats aps_stats;
        std::vector<std::size_t> probed;
        const auto query_start = std::chrono::steady_clock::now();
        const auto found = index.search_aps(queries.row(qi), kTopK,
            kApsCandidateFraction, kApsRecallTarget, kApsRecomputeThreshold,
            &probed, &aps_stats);
        const double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - query_start).count();
        if (found.size() != kTopK) throw std::runtime_error("APS did not return top-100");
        query_ms.push_back(elapsed);
        candidate_partitions += aps_stats.candidate_partitions;
        scanned_partitions += aps_stats.scanned_partitions;
        estimated_recall += aps_stats.estimated_recall;
        std::uint64_t query_scanned_vectors = 0;
        for (const std::size_t leaf : probed)
            if (leaf < leaf_sizes.size()) query_scanned_vectors += leaf_sizes[leaf];
        scanned_vectors += query_scanned_vectors;
        std::uint64_t query_hits = 0;
        for (const SearchResult& result : found) {
            if (result.id >= 0 && static_cast<std::size_t>(result.id) < data.rows() &&
                marks[static_cast<std::size_t>(result.id)] == mark)
                ++query_hits;
        }
        hits += query_hits;
        per_query << qi << ',' << std::setprecision(8)
                  << static_cast<double>(query_hits) / kTopK << ','
                  << aps_stats.estimated_recall << ',' << aps_stats.candidate_partitions
                  << ',' << aps_stats.scanned_partitions << ',' << query_scanned_vectors
                  << ',' << elapsed << '\n';
        if ((qi + 1) % 1000 == 0)
            std::cout << "full_aps query_progress=" << qi + 1 << '/' << kQueries << '\n'
                      << std::flush;
    }
    const double recall = static_cast<double>(hits) / (kQueries * kTopK);
    const double mean_estimated = estimated_recall / kQueries;
    const double mean_ms = std::accumulate(query_ms.begin(), query_ms.end(), 0.0) / kQueries;
    const double query_seconds = std::accumulate(query_ms.begin(), query_ms.end(), 0.0) / 1000.0;
    summary << data.rows() << ',' << index.leaf_nlist() << ',' << kKmeansIterations << ','
            << kSeed << ',' << std::setprecision(10) << build_ms << ',' << index_save_ms
            << ',' << kQueries << ',' << kApsRecallTarget << ',' << mean_estimated << ','
            << recall << ',' << static_cast<double>(candidate_partitions) / kQueries << ','
            << static_cast<double>(scanned_partitions) / kQueries << ','
            << static_cast<double>(scanned_vectors) / kQueries << ',' << mean_ms << ','
            << percentile(query_ms, 0.95) << ',' << query_seconds << '\n';
    summary.flush();
    per_query.flush();
    std::ofstream complete(output_root / "COMPLETE.txt");
    complete << "status=complete\nmethod=PRD-IVF trained on random 15% sample; all 10M indexed\nquery=APS\n"
             << "indexed_vectors=" << data.rows() << "\nnlist=" << index.leaf_nlist()
             << "\nqueries=" << kQueries << "\ntopk=" << kTopK << '\n';
    std::cout << "FULL_APS_COMPLETE active=" << data.rows()
              << " nlist=" << index.leaf_nlist() << " recall@100=" << std::setprecision(6)
              << recall << " estimated_recall=" << mean_estimated
              << " mean_query_ms=" << mean_ms << " p95_query_ms="
              << percentile(query_ms, 0.95) << " output=" << output_root.string()
              << '\n' << std::flush;
}

void run_fixed_nprobe_query(const std::filesystem::path& output_root,
                            const std::filesystem::path& index_path,
                            const std::filesystem::path& mapping_path,
                            const std::filesystem::path& data_path,
                            const std::filesystem::path& query_path,
                            const std::filesystem::path& truth_path,
                            std::size_t nprobe,
                            const MappedFbin& queries) {
    ensure_empty_or_create(output_root);
    if (queries.rows() < kQueries || nprobe == 0)
        throw std::runtime_error("expected at least 10K queries and positive nprobe");

    std::cout << "fixed_nprobe loading_index=" << index_path.string() << '\n' << std::flush;
    const auto load_start = std::chrono::steady_clock::now();
    AdaptiveHierarchicalIvf index = AdaptiveHierarchicalIvf::load(index_path.string());
    const double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - load_start).count();
    if (index.size() != 10'000'000 || index.config().dimension != kDimension ||
        nprobe > index.leaf_nlist())
        throw std::runtime_error("loaded index shape does not match fixed-nprobe experiment");

    std::ifstream mapping_file(mapping_path, std::ios::binary);
    char mapping_magic[8]{};
    std::uint64_t mapping_count = 0;
    mapping_file.read(mapping_magic, sizeof(mapping_magic));
    mapping_file.read(reinterpret_cast<char*>(&mapping_count), sizeof(mapping_count));
    if (!mapping_file || std::string(mapping_magic, sizeof(mapping_magic)) != "MSTIDM01" ||
        mapping_count != index.size() || std::filesystem::file_size(mapping_path) !=
            sizeof(mapping_magic) + sizeof(mapping_count) + mapping_count * sizeof(std::uint32_t))
        throw std::runtime_error("index ID mapping header/size mismatch");
    std::vector<std::uint32_t> internal_to_original(static_cast<std::size_t>(mapping_count));
    mapping_file.read(reinterpret_cast<char*>(internal_to_original.data()),
                      static_cast<std::streamsize>(internal_to_original.size() * sizeof(std::uint32_t)));
    if (!mapping_file) throw std::runtime_error("cannot read full index ID mapping");

    std::ifstream truth_file(truth_path, std::ios::binary);
    char truth_magic[8]{};
    std::uint32_t truth_shape[3]{};
    truth_file.read(truth_magic, sizeof(truth_magic));
    truth_file.read(reinterpret_cast<char*>(truth_shape), sizeof(truth_shape));
    if (!truth_file || std::string(truth_magic, sizeof(truth_magic)) != "MSTGTH01" ||
        truth_shape[0] != kQueryOps || truth_shape[1] != kQueries ||
        truth_shape[2] != kTopK)
        throw std::runtime_error("expected exact stage truth MSTGTH01 [100,10000,100]");
    constexpr std::uint64_t truth_header_bytes = 8 + 3 * sizeof(std::uint32_t);
    constexpr std::uint64_t truth_stage_bytes =
        static_cast<std::uint64_t>(kQueries) * sizeof(Ids);
    if (std::filesystem::file_size(truth_path) !=
        truth_header_bytes + kQueryOps * truth_stage_bytes)
        throw std::runtime_error("stage truth file size does not match header");
    truth_file.seekg(static_cast<std::streamoff>(
        truth_header_bytes + (kQueryOps - 1) * truth_stage_bytes));
    std::vector<Ids> truth(kQueries);
    truth_file.read(reinterpret_cast<char*>(truth.data()),
                    static_cast<std::streamsize>(truth_stage_bytes));
    if (!truth_file) throw std::runtime_error("cannot read final 10M exact truth");

    std::ofstream manifest(output_root / "manifest.txt");
    std::ofstream summary(output_root / "fixed_nprobe_metrics.csv");
    std::ofstream per_query(output_root / "per_query.csv");
    if (!manifest || !summary || !per_query)
        throw std::runtime_error("cannot create fixed-nprobe outputs");
    manifest << "experiment=MSTuring10M PRD-IVF fixed-nprobe query\n"
             << "dataset=" << data_path.string() << "\nqueries=" << query_path.string()
             << "\nexact_groundtruth=" << truth_path.string()
             << "\nindex=" << index_path.string() << "\nid_mapping=" << mapping_path.string()
             << "\nindexed_vectors=" << index.size() << "\nnlist=" << index.leaf_nlist()
             << "\nnprobe=" << nprobe << "\nmetric=squared L2\nLVQ=false"
             << "\nquery_count=" << kQueries << "\ntop_k=" << kTopK
             << "\nwarmup_queries=100\nquery_stage=final 10M stage\n";
    summary << "indexed_vectors,nlist,nprobe,query_count,recall_at_100,mean_scanned_partitions,"
               "mean_scanned_vectors,mean_query_ms,p95_query_ms,total_query_seconds,index_load_ms\n";
    per_query << "query_id,recall_at_100,scanned_partitions,scanned_vectors,query_ms\n";

    for (std::size_t q = 0; q < 100; ++q)
        (void)index.search(queries.row(q), kTopK, nprobe, nullptr);
    const auto leaf_sizes = index.leaf_sizes();
    std::vector<std::uint32_t> marks(index.size(), 0);
    std::vector<double> query_ms;
    query_ms.reserve(kQueries);
    std::uint64_t total_hits = 0, scanned_partitions = 0, scanned_vectors = 0;
    for (std::size_t qi = 0; qi < kQueries; ++qi) {
        const std::uint32_t mark = static_cast<std::uint32_t>(qi + 1);
        for (const std::uint32_t original_id : truth[qi]) {
            if (original_id >= index.size())
                throw std::runtime_error("truth ID exceeds original row range");
            marks[original_id] = mark;
        }
        std::vector<std::size_t> probed;
        const auto query_start = std::chrono::steady_clock::now();
        const auto found = index.search(queries.row(qi), kTopK, nprobe, &probed);
        const double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - query_start).count();
        if (found.size() != kTopK || probed.size() != nprobe)
            throw std::runtime_error("fixed-nprobe search returned unexpected result/probe count");
        query_ms.push_back(elapsed);
        scanned_partitions += probed.size();
        std::uint64_t query_scanned_vectors = 0;
        for (const std::size_t leaf : probed) {
            if (leaf >= leaf_sizes.size()) throw std::runtime_error("invalid probed leaf ID");
            query_scanned_vectors += leaf_sizes[leaf];
        }
        scanned_vectors += query_scanned_vectors;
        std::uint64_t hits = 0;
        for (const SearchResult& result : found) {
            if (result.id < 0 || static_cast<std::size_t>(result.id) >= internal_to_original.size())
                throw std::runtime_error("search result ID outside internal-to-original mapping");
            if (marks[internal_to_original[static_cast<std::size_t>(result.id)]] == mark) ++hits;
        }
        total_hits += hits;
        per_query << qi << ',' << std::setprecision(8)
                  << static_cast<double>(hits) / kTopK << ',' << probed.size() << ','
                  << query_scanned_vectors << ',' << elapsed << '\n';
        if ((qi + 1) % 1000 == 0)
            std::cout << "fixed_nprobe query_progress=" << qi + 1 << '/' << kQueries << '\n'
                      << std::flush;
    }
    const double recall = static_cast<double>(total_hits) / (kQueries * kTopK);
    const double mean_ms = std::accumulate(query_ms.begin(), query_ms.end(), 0.0) / kQueries;
    const double total_seconds = std::accumulate(query_ms.begin(), query_ms.end(), 0.0) / 1000.0;
    summary << index.size() << ',' << index.leaf_nlist() << ',' << nprobe << ',' << kQueries
            << ',' << std::setprecision(10) << recall << ','
            << static_cast<double>(scanned_partitions) / kQueries << ','
            << static_cast<double>(scanned_vectors) / kQueries << ',' << mean_ms << ','
            << percentile(query_ms, 0.95) << ',' << total_seconds << ',' << load_ms << '\n';
    summary.flush();
    per_query.flush();
    std::ofstream complete(output_root / "COMPLETE.txt");
    complete << "status=complete\nquery=standard IVF fixed nprobe\nindexed_vectors="
             << index.size() << "\nnlist=" << index.leaf_nlist() << "\nnprobe=" << nprobe
             << "\nqueries=" << kQueries << "\ntopk=" << kTopK << '\n';
    std::cout << "FIXED_NPROBE_COMPLETE recall@100=" << std::setprecision(6) << recall
              << " nprobe=" << nprobe << " mean_query_ms=" << mean_ms
              << " p95_query_ms=" << percentile(query_ms, 0.95)
              << " mean_scanned_vectors=" << static_cast<double>(scanned_vectors) / kQueries
              << " output=" << output_root.string() << '\n' << std::flush;
}

#ifdef YFCC10M_IH
void run_yfcc_static_sample20(const std::filesystem::path& output_root,
                              const MappedU8bin& data,
                              const MappedU8bin& query_bytes,
                              const std::filesystem::path& truth_path,
                              const std::filesystem::path& permutation_path,
                              std::size_t nlist,
                              std::uint32_t seed) {
    if (data.rows() != 10'000'000 || query_bytes.rows() < kQueries ||
        nlist == 0 || nlist >= data.rows())
        throw std::runtime_error("YFCC static run requires 10M vectors, queries, and a valid nlist");
    ensure_empty_or_create(output_root);

    std::ifstream permutation_file(permutation_path, std::ios::binary);
    char permutation_magic[8]{};
    std::uint64_t permutation_count = 0;
    permutation_file.read(permutation_magic, sizeof(permutation_magic));
    permutation_file.read(reinterpret_cast<char*>(&permutation_count), sizeof(permutation_count));
    if (!permutation_file || std::string(permutation_magic, sizeof(permutation_magic)) != "YFCPRM01" ||
        permutation_count != data.rows() || std::filesystem::file_size(permutation_path) !=
            sizeof(permutation_magic) + sizeof(permutation_count) + permutation_count * sizeof(std::uint32_t))
        throw std::runtime_error("YFCC shuffle permutation header/size mismatch");
    std::vector<std::uint32_t> shuffled_to_source(data.rows());
    permutation_file.read(reinterpret_cast<char*>(shuffled_to_source.data()),
                          static_cast<std::streamsize>(shuffled_to_source.size() * sizeof(std::uint32_t)));
    if (!permutation_file) throw std::runtime_error("cannot read YFCC shuffle permutation");
    std::vector<std::uint32_t>().swap(shuffled_to_source);

    std::ifstream truth_file(truth_path, std::ios::binary);
    char truth_magic[8]{};
    std::uint32_t truth_shape[3]{};
    truth_file.read(truth_magic, sizeof(truth_magic));
    truth_file.read(reinterpret_cast<char*>(truth_shape), sizeof(truth_shape));
    if (!truth_file || std::string(truth_magic, sizeof(truth_magic)) != "MSTGTH01" ||
        truth_shape[0] != 1 || truth_shape[1] != kQueries || truth_shape[2] != kTopK ||
        std::filesystem::file_size(truth_path) != 8 + 3 * sizeof(std::uint32_t) +
            static_cast<std::uint64_t>(kQueries) * kTopK * sizeof(std::uint32_t))
        throw std::runtime_error("expected single-stage exact YFCC final top-100 truth");
    std::vector<Ids> truth(kQueries);
    truth_file.read(reinterpret_cast<char*>(truth.data()), static_cast<std::streamsize>(truth.size() * sizeof(Ids)));
    if (!truth_file) throw std::runtime_error("cannot read YFCC final exact truth");

    constexpr double training_fraction = 0.20;
    const std::size_t training_count = static_cast<std::size_t>(data.rows() * training_fraction);
    const std::size_t budget = data.rows() / 20;
    AdaptiveHierarchicalConfig cfg;
    cfg.dimension = kDimension;
    cfg.initial_leaf_nlist = nlist;
    cfg.kmeans_iterations = kKmeansIterations;
    cfg.seed = seed;
    const unsigned detected_workers = std::thread::hardware_concurrency();
    cfg.worker_threads = std::max<std::size_t>(1, detected_workers ? detected_workers : 1);
    cfg.lvq8_storage = false;
    cfg.lvq4_storage = false;
    cfg.angular_metric = false;
    cfg.uint8_storage = true;
    cfg.automatic_maintenance = false;
    AdaptiveHierarchicalIvf index(cfg);

    std::ofstream manifest(output_root / "manifest.txt", std::ios::trunc);
    std::ofstream summary(output_root / "fixed_budget_metrics.csv", std::ios::trunc);
    std::ofstream per_query(output_root / "per_query.csv", std::ios::trunc);
    if (!manifest || !summary || !per_query)
        throw std::runtime_error("cannot create YFCC static-run outputs");
    manifest << "experiment=YFCC-10M static IVF fixed-budget comparison\n"
             << "dataset=" << "YFCC 10M u8bin with first 1M rows preserved and insertion tail shuffled" << '\n'
             << "shuffle_permutation=" << permutation_path.string() << "\nshuffle_seed=1234\n"
             << "query_source=" << "seed-1234 sample of 1000 YFCC public queries" << '\n'
             << "exact_groundtruth=" << truth_path.string() << " (final 10M set; IDs remapped to source rows)\n"
             << "rows=10000000; dimension=192; metric=squared L2; source and stored vectors=uint8 exact\n"
             << "training_sample_fraction=" << training_fraction << "; training_rows=" << training_count
             << "; without replacement; sample_seed=" << seed << "\nnlist=" << nlist
             << "\nkmeans_iterations=" << kKmeansIterations << "\nmaintenance=disabled\n"
             << "build_worker_threads=" << cfg.worker_threads << "\n"
             << "query_count=" << kQueries << "; top_k=100; budget_per_query=" << budget
             << " total distance computations including centroid distances\nwarmup_queries=100\n"
             << "truth IDs and internal_to_shuffled_row mapping use the shuffled YFCC row order\n"
             << "temporary_index_retained=false\n";
    summary << "indexed_vectors,nlist,training_rows,training_fraction,query_count,budget_per_query,"
               "recall_at_10,recall_at_100,mean_center_distances,mean_vector_distances,"
               "mean_total_distances,mean_query_ms,p95_query_ms,total_query_seconds,build_ms\n";
    per_query << "query_id,hits_at_10,recall_at_10,hits_at_100,recall_at_100,center_distances,"
                  "scanned_vectors,total_distances,query_ms\n";

    std::vector<std::uint32_t> sample_ids(data.rows());
    std::iota(sample_ids.begin(), sample_ids.end(), 0u);
    std::mt19937 sample_rng(seed);
    std::shuffle(sample_ids.begin(), sample_ids.end(), sample_rng);
    sample_ids.resize(training_count);
    std::vector<std::uint8_t> is_training(data.rows(), 0);
    std::vector<std::uint8_t> training_vectors(training_count * kDimension);
    std::vector<std::uint32_t> internal_to_source;
    internal_to_source.reserve(data.rows());
    for (std::size_t i = 0; i < training_count; ++i) {
        const std::uint32_t shuffled_id = sample_ids[i];
        is_training[shuffled_id] = 1;
        internal_to_source.push_back(shuffled_id);
        std::copy_n(data.row(shuffled_id), kDimension,
                    training_vectors.data() + i * kDimension);
    }
    std::vector<std::uint32_t>().swap(sample_ids);

    std::cout << "YFCC_STATIC_START nlist=" << nlist << " training_rows=" << training_count
              << " seed=" << seed << "\nYFCC_STATIC_KMEANS_START\n" << std::flush;
    const auto build_start = std::chrono::steady_clock::now();
    index.train_u8(training_vectors.data(), training_count);
    std::vector<std::uint8_t>().swap(training_vectors);
    std::cout << "YFCC_STATIC_KMEANS_DONE; adding_remaining=" << (data.rows() - training_count)
              << "\n" << std::flush;
    constexpr std::size_t add_batch_size = 50'000;
    std::vector<std::uint8_t> add_buffer;
    add_buffer.reserve(add_batch_size * kDimension);
    std::size_t added = 0;
    for (std::uint32_t shuffled_id = 0; shuffled_id < data.rows(); ++shuffled_id) {
        if (is_training[shuffled_id]) continue;
        const auto* row = data.row(shuffled_id);
        add_buffer.insert(add_buffer.end(), row, row + kDimension);
        internal_to_source.push_back(shuffled_id);
        if (add_buffer.size() == add_batch_size * kDimension) {
            index.add_u8(add_buffer.data(), add_batch_size);
            added += add_batch_size;
            add_buffer.clear();
            if (added % 1'000'000 == 0 || added == data.rows() - training_count)
                std::cout << "YFCC_STATIC_ADD_PROGRESS nlist=" << nlist << " added=" << added
                          << '/' << (data.rows() - training_count) << '\n' << std::flush;
        }
    }
    if (!add_buffer.empty()) {
        const std::size_t count = add_buffer.size() / kDimension;
        index.add_u8(add_buffer.data(), count);
        added += count;
    }
    if (index.size() != data.rows() || internal_to_source.size() != data.rows())
        throw std::runtime_error("YFCC static build failed to add every vector exactly once");
    const double build_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - build_start).count();
    std::cout << "YFCC_STATIC_BUILD_DONE nlist=" << nlist << " build_ms=" << build_ms
              << "\n" << std::flush;

    FloatQueryRows queries(query_bytes);
    QuerySearchOptions options;
    options.enable_early_stop = false;
    options.max_total_distance_computations = budget;
    for (std::size_t q = 0; q < std::min<std::size_t>(100, queries.rows()); ++q)
        (void)index.search_with_options(queries.row(q), kTopK, options, nullptr);

    std::vector<std::uint32_t> marks10(data.rows(), 0), marks100(data.rows(), 0);
    std::vector<double> query_ms;
    query_ms.reserve(kQueries);
    std::uint64_t hits10_total = 0, hits100_total = 0;
    std::uint64_t centers_total = 0, vectors_total = 0, distances_total = 0;
    for (std::size_t qi = 0; qi < kQueries; ++qi) {
        const auto mark = static_cast<std::uint32_t>(qi + 1);
        for (std::size_t rank = 0; rank < 10; ++rank) {
            const auto source_id = truth[qi][rank];
            if (source_id >= data.rows()) throw std::runtime_error("YFCC truth ID out of source range");
            marks10[source_id] = mark;
            marks100[source_id] = mark;
        }
        for (std::size_t rank = 10; rank < kTopK; ++rank) {
            const auto source_id = truth[qi][rank];
            if (source_id >= data.rows()) throw std::runtime_error("YFCC truth ID out of source range");
            marks100[source_id] = mark;
        }
        QuerySearchStats stats;
        const auto start = std::chrono::steady_clock::now();
        const auto found = index.search_with_options(queries.row(qi), kTopK, options, &stats);
        const double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        if (found.size() != kTopK || stats.total_distance_computations != budget ||
            stats.center_distance_computations != nlist)
            throw std::runtime_error("YFCC fixed-budget query failed to honor its exact budget");
        std::uint64_t hits10 = 0, hits100 = 0;
        for (std::size_t rank = 0; rank < found.size(); ++rank) {
            const auto internal = found[rank].id;
            if (internal < 0 || static_cast<std::size_t>(internal) >= internal_to_source.size())
                throw std::runtime_error("YFCC result ID outside internal-to-source mapping");
            const auto source_id = internal_to_source[static_cast<std::size_t>(internal)];
            if (rank < 10 && marks10[source_id] == mark) ++hits10;
            if (marks100[source_id] == mark) ++hits100;
        }
        query_ms.push_back(elapsed);
        hits10_total += hits10;
        hits100_total += hits100;
        centers_total += stats.center_distance_computations;
        vectors_total += stats.scanned_vectors;
        distances_total += stats.total_distance_computations;
        per_query << qi << ',' << hits10 << ',' << std::setprecision(8)
                  << static_cast<double>(hits10) / 10.0 << ',' << hits100 << ','
                  << static_cast<double>(hits100) / 100.0 << ','
                  << stats.center_distance_computations << ',' << stats.scanned_vectors << ','
                  << stats.total_distance_computations << ',' << elapsed << '\n';
        if ((qi + 1) % 100 == 0)
            std::cout << "YFCC_STATIC_QUERY_PROGRESS nlist=" << nlist << " queries="
                      << qi + 1 << '/' << kQueries << '\n' << std::flush;
    }
    const double recall10 = static_cast<double>(hits10_total) / (kQueries * 10);
    const double recall100 = static_cast<double>(hits100_total) / (kQueries * kTopK);
    const double mean_ms = std::accumulate(query_ms.begin(), query_ms.end(), 0.0) / kQueries;
    summary << data.rows() << ',' << nlist << ',' << training_count << ',' << training_fraction << ','
            << kQueries << ',' << budget << ',' << std::setprecision(10) << recall10 << ','
            << recall100 << ',' << static_cast<double>(centers_total) / kQueries << ','
            << static_cast<double>(vectors_total) / kQueries << ','
            << static_cast<double>(distances_total) / kQueries << ',' << mean_ms << ','
            << percentile(query_ms, 0.95) << ','
            << std::accumulate(query_ms.begin(), query_ms.end(), 0.0) / 1000.0 << ','
            << build_ms << '\n';
    std::ofstream complete(output_root / "COMPLETE.txt", std::ios::trunc);
    complete << "status=complete\nindexed_vectors=" << index.size() << "\nnlist=" << nlist
             << "\ntraining_rows=" << training_count << "\nseed=" << seed
             << "\nrecall_at_10=" << recall10 << "\nrecall_at_100=" << recall100
             << "\nper_query_budget=" << budget << "\ntemporary_index_retained=false\n";
    std::cout << "YFCC_STATIC_COMPLETE nlist=" << nlist << " recall@10=" << recall10
              << " recall@100=" << recall100 << " mean_query_ms=" << mean_ms
              << " p95_query_ms=" << percentile(query_ms, 0.95)
              << " build_ms=" << build_ms << "\n" << std::flush;
}
#endif

void run_fixed_budget_query(const std::filesystem::path& output_root,
                            const std::filesystem::path& index_path,
                            const std::filesystem::path& mapping_path,
                            const std::filesystem::path& data_path,
                            const std::filesystem::path& query_path,
                            const std::filesystem::path& truth_path,
                            std::size_t budget,
                            const MappedFbin& queries,
                            std::size_t query_threads = 1) {
    ensure_empty_or_create(output_root);
    if (queries.rows() < kQueries || budget == 0 || query_threads == 0)
        throw std::runtime_error("expected at least 10K queries and a positive fixed budget");

    std::cout << "fixed_budget loading_index=" << index_path.string() << '\n' << std::flush;
    const auto load_start = std::chrono::steady_clock::now();
    AdaptiveHierarchicalIvf index = AdaptiveHierarchicalIvf::load(index_path.string());
    const double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - load_start).count();
    if (index.size() != 10'000'000 || index.config().dimension != kDimension ||
        budget <= index.leaf_nlist())
        throw std::runtime_error("loaded index shape/budget does not match fixed-budget experiment");

    std::ifstream mapping_file(mapping_path, std::ios::binary);
    char mapping_magic[8]{};
    std::uint64_t mapping_count = 0;
    mapping_file.read(mapping_magic, sizeof(mapping_magic));
    mapping_file.read(reinterpret_cast<char*>(&mapping_count), sizeof(mapping_count));
    if (!mapping_file || std::string(mapping_magic, sizeof(mapping_magic)) != "MSTIDM01" ||
        mapping_count != index.size() || std::filesystem::file_size(mapping_path) !=
            sizeof(mapping_magic) + sizeof(mapping_count) + mapping_count * sizeof(std::uint32_t))
        throw std::runtime_error("index ID mapping header/size mismatch");
    std::vector<std::uint32_t> internal_to_original(static_cast<std::size_t>(mapping_count));
    mapping_file.read(reinterpret_cast<char*>(internal_to_original.data()),
                      static_cast<std::streamsize>(internal_to_original.size() * sizeof(std::uint32_t)));
    if (!mapping_file) throw std::runtime_error("cannot read full index ID mapping");

    std::ifstream truth_file(truth_path, std::ios::binary);
    char truth_magic[8]{};
    std::uint32_t truth_shape[3]{};
    truth_file.read(truth_magic, sizeof(truth_magic));
    truth_file.read(reinterpret_cast<char*>(truth_shape), sizeof(truth_shape));
    if (!truth_file || std::string(truth_magic, sizeof(truth_magic)) != "MSTGTH01" ||
        truth_shape[0] != kQueryOps || truth_shape[1] != kQueries || truth_shape[2] != kTopK)
        throw std::runtime_error("expected exact stage truth MSTGTH01 [100,10000,100]");
    constexpr std::uint64_t truth_header_bytes = 8 + 3 * sizeof(std::uint32_t);
    constexpr std::uint64_t truth_stage_bytes = static_cast<std::uint64_t>(kQueries) * sizeof(Ids);
    if (std::filesystem::file_size(truth_path) != truth_header_bytes + kQueryOps * truth_stage_bytes)
        throw std::runtime_error("stage truth file size does not match header");
    truth_file.seekg(static_cast<std::streamoff>(truth_header_bytes + (kQueryOps - 1) * truth_stage_bytes));
    std::vector<Ids> truth(kQueries);
    truth_file.read(reinterpret_cast<char*>(truth.data()), static_cast<std::streamsize>(truth_stage_bytes));
    if (!truth_file) throw std::runtime_error("cannot read final 10M exact truth");

    std::ofstream manifest(output_root / "manifest.txt");
    std::ofstream summary(output_root / "fixed_budget_metrics.csv");
    std::ofstream per_query(output_root / "per_query.csv");
    if (!manifest || !summary || !per_query)
        throw std::runtime_error("cannot create fixed-budget outputs");
    manifest << "experiment=MSTuring10M full-index fixed-distance-budget query\n"
             << "dataset=" << data_path.string() << "\nqueries=" << query_path.string()
             << "\nexact_groundtruth=" << truth_path.string()
             << "\nindex=" << index_path.string() << "\nid_mapping=" << mapping_path.string()
             << "\nindexed_vectors=" << index.size() << "\nnlist=" << index.leaf_nlist()
             << "\nper_query_total_distance_budget=" << budget
             << " including centroid and vector distances\nmetric=squared L2\nquery_count=" << kQueries
              << "\ntop_k=" << kTopK << "\nwarmup_queries=100\nquery_threads=" << query_threads
              << "\nquery_stage=final 10M stage\n";
    summary << "indexed_vectors,nlist,query_count,budget_per_query,recall_at_10,recall_at_100,"
               "mean_center_distances,mean_scanned_vectors,mean_total_distances,mean_query_ms,p95_query_ms,total_query_seconds,index_load_ms\n";
    per_query << "query_id,hits_at_10,recall_at_10,hits_at_100,recall_at_100,center_distances,"
                  "scanned_vectors,total_distances,query_ms\n";

    QuerySearchOptions options;
    options.enable_early_stop = false;
    options.max_total_distance_computations = budget;
    for (std::size_t q = 0; q < 100; ++q)
        (void)index.search_with_options(queries.row(q), kTopK, options, nullptr);

    struct QueryOutcome {
        Ids ids{};
        QuerySearchStats stats{};
        double query_ms = 0.0;
    };
    std::vector<QueryOutcome> outcomes(kQueries);
    std::atomic<std::size_t> next_query{0};
    std::exception_ptr worker_error;
    std::mutex worker_error_mutex;
    const auto worker = [&]() {
        for (;;) {
            const std::size_t qi = next_query.fetch_add(1);
            if (qi >= kQueries) break;
            try {
                QuerySearchStats stats;
                const auto start = std::chrono::steady_clock::now();
                const auto found = index.search_with_options(queries.row(qi), kTopK, options, &stats);
                const double elapsed = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                if (found.size() != kTopK || stats.total_distance_computations != budget ||
                    stats.center_distance_computations != index.leaf_nlist())
                    throw std::runtime_error("query failed to honor fixed total-distance budget");
                outcomes[qi].stats = stats;
                outcomes[qi].query_ms = elapsed;
                for (std::size_t rank = 0; rank < kTopK; ++rank) {
                    if (found[rank].id < 0 ||
                        static_cast<std::size_t>(found[rank].id) >= internal_to_original.size())
                        throw std::runtime_error("search result ID outside internal-to-original mapping");
                    outcomes[qi].ids[rank] = static_cast<std::uint32_t>(found[rank].id);
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(worker_error_mutex);
                if (!worker_error) worker_error = std::current_exception();
            }
        }
    };
    const std::size_t thread_count = std::min(query_threads, kQueries);
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (std::size_t i = 0; i < thread_count; ++i) workers.emplace_back(worker);
    for (std::thread& thread : workers) thread.join();
    if (worker_error) std::rethrow_exception(worker_error);

    std::vector<std::uint32_t> marks10(index.size(), 0), marks100(index.size(), 0);
    std::vector<double> query_ms;
    query_ms.reserve(kQueries);
    std::uint64_t hits10_total = 0, hits100_total = 0;
    std::uint64_t center_total = 0, vector_total = 0, distance_total = 0;
    for (std::size_t qi = 0; qi < kQueries; ++qi) {
        const std::uint32_t mark = static_cast<std::uint32_t>(qi + 1);
        for (std::size_t rank = 0; rank < 10; ++rank) {
            const std::uint32_t id = truth[qi][rank];
            if (id >= marks100.size()) throw std::runtime_error("truth ID exceeds original row range");
            marks10[id] = mark;
            marks100[id] = mark;
        }
        for (std::size_t rank = 10; rank < kTopK; ++rank) {
            const std::uint32_t id = truth[qi][rank];
            if (id >= marks100.size()) throw std::runtime_error("truth ID exceeds original row range");
            marks100[id] = mark;
        }

        const QueryOutcome& outcome = outcomes[qi];
        const QuerySearchStats& stats = outcome.stats;
        std::uint64_t hits10 = 0, hits100 = 0;
        for (std::size_t rank = 0; rank < kTopK; ++rank) {
            const std::uint32_t original = internal_to_original[outcome.ids[rank]];
            if (rank < 10 && marks10[original] == mark) ++hits10;
            if (marks100[original] == mark) ++hits100;
        }
        query_ms.push_back(outcome.query_ms);
        hits10_total += hits10;
        hits100_total += hits100;
        center_total += stats.center_distance_computations;
        vector_total += stats.scanned_vectors;
        distance_total += stats.total_distance_computations;
        per_query << qi << ',' << hits10 << ',' << std::setprecision(8)
                  << static_cast<double>(hits10) / 10.0 << ',' << hits100 << ','
                  << static_cast<double>(hits100) / 100.0 << ','
                  << stats.center_distance_computations << ',' << stats.scanned_vectors << ','
                  << stats.total_distance_computations << ',' << outcome.query_ms << '\n';
        if ((qi + 1) % 500 == 0) {
            per_query.flush();
            std::cout << "fixed_budget query_progress=" << qi + 1 << '/' << kQueries << '\n' << std::flush;
        }
    }
    const double recall10 = static_cast<double>(hits10_total) / (kQueries * 10);
    const double recall100 = static_cast<double>(hits100_total) / (kQueries * kTopK);
    const double mean_ms = std::accumulate(query_ms.begin(), query_ms.end(), 0.0) / kQueries;
    const double query_seconds = std::accumulate(query_ms.begin(), query_ms.end(), 0.0) / 1000.0;
    summary << index.size() << ',' << index.leaf_nlist() << ',' << kQueries << ',' << budget << ','
            << std::setprecision(10) << recall10 << ',' << recall100 << ','
            << static_cast<double>(center_total) / kQueries << ','
            << static_cast<double>(vector_total) / kQueries << ','
            << static_cast<double>(distance_total) / kQueries << ',' << mean_ms << ','
            << percentile(query_ms, 0.95) << ',' << query_seconds << ',' << load_ms << '\n';
    std::ofstream complete(output_root / "COMPLETE.txt");
    complete << "status=complete\nquery=fixed total distance budget\nindexed_vectors=" << index.size()
             << "\nnlist=" << index.leaf_nlist() << "\nbudget_per_query=" << budget
             << "\nqueries=" << kQueries << "\nquery_threads=" << query_threads
             << "\nrecall_metrics=exact Recall@10 and Recall@100\n";
    std::cout << "FIXED_BUDGET_COMPLETE recall@10=" << std::setprecision(6) << recall10
              << " recall@100=" << recall100 << " budget=" << budget
              << " mean_query_ms=" << mean_ms << " p95_query_ms=" << percentile(query_ms, 0.95)
              << " output=" << output_root.string() << '\n' << std::flush;
}

void run_fixed_budget_snapshot_eval(const std::filesystem::path& output_root,
                                    const std::filesystem::path& index_path,
                                    const std::filesystem::path& query_path,
                                    const std::filesystem::path& truth_path,
                                    std::size_t budget,
                                    std::size_t query_threads) {
    ensure_empty_or_create(output_root);
    MappedFbin queries(query_path);
    if (queries.rows() == 0 || budget == 0 || query_threads == 0)
        throw std::runtime_error("snapshot evaluation requires queries, a positive fixed budget, and positive thread count");

    std::cout << "snapshot_eval loading_index=" << index_path.string() << '\n' << std::flush;
    const auto load_start = std::chrono::steady_clock::now();
    AdaptiveHierarchicalIvf index = AdaptiveHierarchicalIvf::load(index_path.string());
    const double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - load_start).count();
    if (index.size() != 10'000'000 || index.config().dimension != kDimension ||
        budget <= index.leaf_nlist())
        throw std::runtime_error("snapshot index shape or fixed budget is invalid");

    std::ifstream truth_file(truth_path, std::ios::binary);
    char truth_magic[8]{};
    std::uint32_t truth_shape[3]{};
    truth_file.read(truth_magic, sizeof(truth_magic));
    truth_file.read(reinterpret_cast<char*>(truth_shape), sizeof(truth_shape));
    constexpr std::uint64_t truth_header_bytes = 8 + 3 * sizeof(std::uint32_t);
    if (!truth_file || std::string(truth_magic, sizeof(truth_magic)) != "MSTGTH01" ||
        truth_shape[0] == 0 || truth_shape[1] != queries.rows() || truth_shape[2] != kTopK ||
        std::filesystem::file_size(truth_path) != truth_header_bytes +
            static_cast<std::uint64_t>(truth_shape[0]) * queries.rows() * sizeof(Ids))
        throw std::runtime_error("expected exact truth MSTGTH01 [stages,query_count,100]");
    std::vector<Ids> truth(queries.rows());
    truth_file.seekg(static_cast<std::streamoff>(truth_header_bytes +
        static_cast<std::uint64_t>(truth_shape[0] - 1) * queries.rows() * sizeof(Ids)));
    truth_file.read(reinterpret_cast<char*>(truth.data()),
                    static_cast<std::streamsize>(truth.size() * sizeof(Ids)));
    if (!truth_file) throw std::runtime_error("cannot read final snapshot evaluation truth");

    std::ofstream manifest(output_root / "manifest.txt", std::ios::trunc);
    std::ofstream summary(output_root / "fixed_budget_metrics.csv", std::ios::trunc);
    std::ofstream per_query(output_root / "per_query.csv", std::ios::trunc);
    std::ofstream result_file(output_root / "results_top100.bin", std::ios::binary | std::ios::trunc);
    if (!manifest || !summary || !per_query || !result_file)
        throw std::runtime_error("cannot create frozen-snapshot evaluation outputs");
    manifest << "experiment=frozen final-index snapshot evaluation\n"
             << "index=" << index_path.string() << "\nqueries=" << query_path.string()
             << "\nexact_groundtruth=" << truth_path.string()
             << "\nindexed_vectors=" << index.size() << "\nnlist=" << index.leaf_nlist()
             << "\nper_query_total_distance_budget=" << budget
             << " including centroid and vector distances\nmetric=squared L2\nquery_count=" << queries.rows()
             << "\ntop_k=" << kTopK << "\nquery_threads=" << query_threads
             << "\nwarmup_queries=" << std::min<std::size_t>(100, queries.rows())
             << "\nquery_feedback=false\nmaintenance=false\nindex_mutated_during_evaluation=false\n";
    summary << "indexed_vectors,nlist,query_count,budget_per_query,recall_at_10,recall_at_100,"
               "mean_center_distances,mean_scanned_vectors,mean_total_distances,mean_query_ms,p95_query_ms,total_query_seconds,wall_query_seconds,index_load_ms\n";
    per_query << "query_id,hits_at_10,recall_at_10,hits_at_100,recall_at_100,center_distances,"
                  "scanned_vectors,total_distances,query_ms\n";
    const char result_magic[8] = {'M','S','T','I','H','R','0','1'};
    const std::uint32_t result_shape[3] = {1, static_cast<std::uint32_t>(queries.rows()),
                                           static_cast<std::uint32_t>(kTopK)};
    result_file.write(result_magic, sizeof(result_magic));
    result_file.write(reinterpret_cast<const char*>(result_shape), sizeof(result_shape));

    QuerySearchOptions options;
    options.enable_early_stop = false;
    options.max_total_distance_computations = budget;
    for (std::size_t q = 0; q < std::min<std::size_t>(100, queries.rows()); ++q)
        (void)index.search_with_options(queries.row(q), kTopK, options, nullptr);

    struct SnapshotQueryOutcome {
        Ids ids{};
        QuerySearchStats stats{};
        double query_ms = 0.0;
    };
    std::vector<SnapshotQueryOutcome> outcomes(queries.rows());
    std::atomic<std::size_t> next_query{0}, completed_queries{0};
    std::exception_ptr worker_error;
    std::mutex worker_error_mutex, progress_mutex;
    const auto query_wall_start = std::chrono::steady_clock::now();
    const auto worker = [&]() {
        for (;;) {
            const std::size_t qi = next_query.fetch_add(1);
            if (qi >= queries.rows()) break;
            try {
                QuerySearchStats stats;
                const auto start = std::chrono::steady_clock::now();
                const auto found = index.search_with_options(queries.row(qi), kTopK, options, &stats);
                const double elapsed = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - start).count();
                if (found.size() != kTopK || stats.total_distance_computations != budget ||
                    stats.center_distance_computations != index.leaf_nlist())
                    throw std::runtime_error("frozen snapshot query did not honor the exact distance budget");
                SnapshotQueryOutcome& outcome = outcomes[qi];
                outcome.stats = stats;
                outcome.query_ms = elapsed;
                for (std::size_t rank = 0; rank < kTopK; ++rank) {
                    if (found[rank].id < 0 || static_cast<std::size_t>(found[rank].id) >= index.size())
                        throw std::runtime_error("snapshot query returned an invalid row ID");
                    outcome.ids[rank] = static_cast<std::uint32_t>(found[rank].id);
                }
                const std::size_t done = completed_queries.fetch_add(1) + 1;
                if (done % 500 == 0 || done == queries.rows()) {
                    std::lock_guard<std::mutex> lock(progress_mutex);
                    std::cout << "snapshot_eval query_progress=" << done << '/' << queries.rows() << '\n' << std::flush;
                }
            } catch (...) {
                std::lock_guard<std::mutex> lock(worker_error_mutex);
                if (!worker_error) worker_error = std::current_exception();
            }
        }
    };
    const std::size_t thread_count = std::min(query_threads, queries.rows());
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (std::size_t i = 0; i < thread_count; ++i) workers.emplace_back(worker);
    for (std::thread& thread : workers) thread.join();
    if (worker_error) std::rethrow_exception(worker_error);
    const double wall_query_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - query_wall_start).count();

    std::vector<double> query_ms;
    query_ms.reserve(queries.rows());
    std::uint64_t hits10_total = 0, hits100_total = 0;
    std::uint64_t center_total = 0, vector_total = 0, distance_total = 0;
    for (std::size_t qi = 0; qi < queries.rows(); ++qi) {
        const SnapshotQueryOutcome& outcome = outcomes[qi];
        const QuerySearchStats& stats = outcome.stats;
        std::uint64_t hits10 = 0, hits100 = 0;
        Ids result_ids{};
        for (std::size_t rank = 0; rank < kTopK; ++rank) {
            const auto id = outcome.ids[rank];
            result_ids[rank] = id;
            if (rank < 10 && std::find(truth[qi].begin(), truth[qi].begin() + 10, id) != truth[qi].begin() + 10)
                ++hits10;
            if (std::find(truth[qi].begin(), truth[qi].end(), id) != truth[qi].end()) ++hits100;
        }
        result_file.write(reinterpret_cast<const char*>(result_ids.data()), sizeof(result_ids));
        if (!result_file) throw std::runtime_error("cannot write frozen-snapshot result IDs");
        query_ms.push_back(outcome.query_ms);
        hits10_total += hits10;
        hits100_total += hits100;
        center_total += stats.center_distance_computations;
        vector_total += stats.scanned_vectors;
        distance_total += stats.total_distance_computations;
        per_query << qi << ',' << hits10 << ',' << std::setprecision(8)
                  << static_cast<double>(hits10) / 10.0 << ',' << hits100 << ','
                  << static_cast<double>(hits100) / 100.0 << ','
                  << stats.center_distance_computations << ',' << stats.scanned_vectors << ','
                  << stats.total_distance_computations << ',' << outcome.query_ms << '\n';
        if ((qi + 1) % 500 == 0 || qi + 1 == queries.rows()) {
            per_query.flush(); result_file.flush();
            per_query.flush(); result_file.flush();
        }
    }
    const double recall10 = static_cast<double>(hits10_total) / (queries.rows() * 10);
    const double recall100 = static_cast<double>(hits100_total) / (queries.rows() * kTopK);
    const double mean_ms = std::accumulate(query_ms.begin(), query_ms.end(), 0.0) / queries.rows();
    summary << index.size() << ',' << index.leaf_nlist() << ',' << queries.rows() << ',' << budget << ','
            << std::setprecision(10) << recall10 << ',' << recall100 << ','
            << static_cast<double>(center_total) / queries.rows() << ','
            << static_cast<double>(vector_total) / queries.rows() << ','
            << static_cast<double>(distance_total) / queries.rows() << ',' << mean_ms << ','
            << percentile(query_ms, 0.95) << ','
            << std::accumulate(query_ms.begin(), query_ms.end(), 0.0) / 1000.0 << ','
            << wall_query_seconds << ',' << load_ms << '\n';
    std::ofstream complete(output_root / "COMPLETE.txt", std::ios::trunc);
    complete << "status=complete\nindex=" << index_path.string() << "\nqueries=" << queries.rows()
             << "\nbudget_per_query=" << budget << "\nrecall_at_10=" << recall10
             << "\nrecall_at_100=" << recall100
             << "\nquery_feedback=false\nmaintenance=false\nindex_mutated_during_evaluation=false\n";
    std::cout << "SNAPSHOT_EVAL_COMPLETE queries=" << queries.rows() << " recall@10="
              << std::setprecision(6) << recall10 << " recall@100=" << recall100
              << " mean_query_ms=" << mean_ms << " p95_query_ms=" << percentile(query_ms, 0.95)
              << " output=" << output_root.string() << '\n' << std::flush;
}

void write_cluster_size_summary(const std::filesystem::path& output_root,
                                const std::filesystem::path& index_path) {
    ensure_empty_or_create(output_root);
    std::cout << "cluster_summary loading_index=" << index_path.string() << '\n' << std::flush;
    AdaptiveHierarchicalIvf index = AdaptiveHierarchicalIvf::load(index_path.string());
    const auto sizes = index.leaf_sizes();
    if (sizes.empty()) throw std::runtime_error("loaded index has no leaf partitions");
    std::vector<std::size_t> sorted = sizes;
    std::sort(sorted.begin(), sorted.end());
    const auto nearest_rank = [&](double p) -> std::size_t {
        const auto rank = static_cast<std::size_t>(std::ceil(p * sorted.size()));
        return sorted[std::max<std::size_t>(1, rank) - 1];
    };
    const auto mean_value = std::accumulate(sizes.begin(), sizes.end(), 0.0) / sizes.size();
    double variance = 0.0;
    for (const auto size : sizes) {
        const double delta = static_cast<double>(size) - mean_value;
        variance += delta * delta;
    }
    const double stddev = std::sqrt(variance / sizes.size());
    const auto largest_id = static_cast<std::size_t>(std::max_element(sizes.begin(), sizes.end()) - sizes.begin());
    const auto smallest_id = static_cast<std::size_t>(std::min_element(sizes.begin(), sizes.end()) - sizes.begin());
    std::ofstream summary(output_root / "summary.csv", std::ios::trunc);
    std::ofstream details(output_root / "cluster_sizes.csv", std::ios::trunc);
    std::ofstream histogram(output_root / "histogram_1000_vectors.csv", std::ios::trunc);
    if (!summary || !details || !histogram)
        throw std::runtime_error("cannot create cluster-size summary files");
    summary << "indexed_vectors,cluster_count,min_size,min_cluster_id,p10,p25,median,p75,p90,p95,p99,max_size,max_cluster_id,mean_size,stddev,coefficient_of_variation\n"
            << "" << index.size() << ',' << sizes.size() << ',' << sorted.front() << ',' << smallest_id << ','
            << nearest_rank(.10) << ',' << nearest_rank(.25) << ',' << nearest_rank(.50) << ','
            << nearest_rank(.75) << ',' << nearest_rank(.90) << ',' << nearest_rank(.95) << ','
            << nearest_rank(.99) << ',' << sorted.back() << ',' << largest_id << ','
            << std::setprecision(12) << mean_value << ',' << stddev << ','
            << (mean_value == 0.0 ? 0.0 : stddev / mean_value) << '\n';
    details << "cluster_id,vector_count\n";
    for (std::size_t i = 0; i < sizes.size(); ++i) details << i << ',' << sizes[i] << '\n';
    histogram << "size_range,cluster_count,cluster_fraction\n";
    const std::size_t bin_width = 1000;
    const std::size_t max_bin = sorted.back() / bin_width;
    std::size_t cursor = 0;
    for (std::size_t b = 0; b <= max_bin; ++b) {
        const std::size_t lower = b * bin_width;
        const std::size_t upper = lower + bin_width - 1;
        const auto begin = std::lower_bound(sorted.begin() + static_cast<std::ptrdiff_t>(cursor), sorted.end(), lower);
        const auto end = std::upper_bound(begin, sorted.end(), upper);
        cursor = static_cast<std::size_t>(end - sorted.begin());
        const auto count = static_cast<std::size_t>(end - begin);
        histogram << lower << '-' << upper << ',' << count << ','
                  << std::setprecision(10) << static_cast<double>(count) / sizes.size() << '\n';
    }
    std::ofstream manifest(output_root / "manifest.txt", std::ios::trunc);
    if (!manifest) throw std::runtime_error("cannot create cluster-size manifest");
    manifest << "source_index=" << index_path.string() << "\nindexed_vectors=" << index.size()
             << "\ncluster_count=" << sizes.size()
             << "\npercentiles=nearest-rank\nhistogram_bin_width=1000 vectors\n";
    std::cout << "CLUSTER_SUMMARY_COMPLETE clusters=" << sizes.size() << " min=" << sorted.front()
              << " p50=" << nearest_rank(.50) << " p90=" << nearest_rank(.90)
              << " mean=" << mean_value << " max=" << sorted.back()
              << " output=" << output_root.string() << '\n' << std::flush;
}

void run_fixed_nprobe_sweep_sample(const std::filesystem::path& output_root,
                                   const std::filesystem::path& index_path,
                                   const std::filesystem::path& mapping_path,
                                   const std::filesystem::path& data_path,
                                   const std::filesystem::path& query_path,
                                   const std::filesystem::path& truth_path,
                                   std::size_t sample_count, std::uint32_t sample_seed,
                                   const std::vector<std::size_t>& nprobes,
                                   const MappedFbin& queries) {
    ensure_empty_or_create(output_root);
    if (queries.rows() < kQueries || sample_count == 0 || sample_count > kQueries)
        throw std::runtime_error("expected 1..10000 sampled queries from the 10K query file");
    if (nprobes.empty()) throw std::runtime_error("nprobe sweep cannot be empty");
    std::cout << "nprobe_sweep loading_index=" << index_path.string() << '\n' << std::flush;
    const auto load_start = std::chrono::steady_clock::now();
    AdaptiveHierarchicalIvf index = AdaptiveHierarchicalIvf::load(index_path.string());
    const double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - load_start).count();
    if (index.size() != 10'000'000 || index.config().dimension != kDimension ||
        *std::max_element(nprobes.begin(), nprobes.end()) > index.leaf_nlist())
        throw std::runtime_error("loaded index shape does not match nprobe sweep");

    std::ifstream mapping_file(mapping_path, std::ios::binary);
    char mapping_magic[8]{};
    std::uint64_t mapping_count = 0;
    mapping_file.read(mapping_magic, sizeof(mapping_magic));
    mapping_file.read(reinterpret_cast<char*>(&mapping_count), sizeof(mapping_count));
    if (!mapping_file || std::string(mapping_magic, sizeof(mapping_magic)) != "MSTIDM01" ||
        mapping_count != index.size() || std::filesystem::file_size(mapping_path) !=
            sizeof(mapping_magic) + sizeof(mapping_count) + mapping_count * sizeof(std::uint32_t))
        throw std::runtime_error("index ID mapping header/size mismatch");
    std::vector<std::uint32_t> internal_to_original(static_cast<std::size_t>(mapping_count));
    mapping_file.read(reinterpret_cast<char*>(internal_to_original.data()),
                      static_cast<std::streamsize>(internal_to_original.size() * sizeof(std::uint32_t)));
    if (!mapping_file) throw std::runtime_error("cannot read full index ID mapping");

    std::ifstream truth_file(truth_path, std::ios::binary);
    char truth_magic[8]{};
    std::uint32_t truth_shape[3]{};
    truth_file.read(truth_magic, sizeof(truth_magic));
    truth_file.read(reinterpret_cast<char*>(truth_shape), sizeof(truth_shape));
    if (!truth_file || std::string(truth_magic, sizeof(truth_magic)) != "MSTGTH01" ||
        truth_shape[0] != kQueryOps || truth_shape[1] != kQueries ||
        truth_shape[2] != kTopK)
        throw std::runtime_error("expected exact stage truth MSTGTH01 [100,10000,100]");
    constexpr std::uint64_t truth_header_bytes = 8 + 3 * sizeof(std::uint32_t);
    constexpr std::uint64_t truth_stage_bytes =
        static_cast<std::uint64_t>(kQueries) * sizeof(Ids);
    if (std::filesystem::file_size(truth_path) !=
        truth_header_bytes + kQueryOps * truth_stage_bytes)
        throw std::runtime_error("stage truth file size does not match header");
    truth_file.seekg(static_cast<std::streamoff>(
        truth_header_bytes + (kQueryOps - 1) * truth_stage_bytes));
    std::vector<Ids> truth(kQueries);
    truth_file.read(reinterpret_cast<char*>(truth.data()),
                    static_cast<std::streamsize>(truth_stage_bytes));
    if (!truth_file) throw std::runtime_error("cannot read final 10M exact truth");

    std::vector<std::size_t> sample_ids(kQueries);
    std::iota(sample_ids.begin(), sample_ids.end(), 0);
    std::mt19937 sample_rng(sample_seed);
    std::shuffle(sample_ids.begin(), sample_ids.end(), sample_rng);
    sample_ids.resize(sample_count);
    std::sort(sample_ids.begin(), sample_ids.end());

    std::ofstream manifest(output_root / "manifest.txt");
    std::ofstream summary(output_root / "fixed_nprobe_sweep.csv");
    std::ofstream per_query(output_root / "per_query.csv");
    if (!manifest || !summary || !per_query)
        throw std::runtime_error("cannot create nprobe-sweep outputs");
    manifest << "experiment=MSTuring10M PRD-IVF fixed nprobe sweep\n"
             << "dataset=" << data_path.string() << "\nqueries=" << query_path.string()
             << "\nexact_groundtruth=" << truth_path.string()
             << "\nindex=" << index_path.string() << "\nid_mapping=" << mapping_path.string()
             << "\nindexed_vectors=" << index.size() << "\nnlist=" << index.leaf_nlist()
             << "\nnprobes=";
    for (std::size_t i = 0; i < nprobes.size(); ++i) {
        if (i) manifest << ',';
        manifest << nprobes[i];
    }
    manifest << "\nquery_sample_count=" << sample_count
             << "\nquery_sample_seed=" << sample_seed
             << "\nquery_sample=uniform without replacement from first 10000 rows"
             << "\nmetric=squared L2\nLVQ=false\ntop_k=" << kTopK
             << "\nwarmup_queries=100 per nprobe\nquery_stage=final 10M stage\n";
    summary << "indexed_vectors,nlist,nprobe,query_count,sample_seed,recall_at_100,"
               "mean_scanned_partitions,mean_scanned_vectors,mean_query_ms,p95_query_ms,"
               "total_query_seconds,index_load_ms\n";
    per_query << "nprobe,sample_position,query_id,recall_at_100,scanned_partitions,"
                  "scanned_vectors,query_ms\n";
    const auto leaf_sizes = index.leaf_sizes();
    std::vector<std::uint32_t> marks(index.size(), 0);
    for (const std::size_t nprobe : nprobes) {
        for (std::size_t i = 0; i < std::min<std::size_t>(100, sample_count); ++i)
            (void)index.search(queries.row(sample_ids[i]), kTopK, nprobe, nullptr);
        std::vector<double> query_ms;
        query_ms.reserve(sample_count);
        std::uint64_t total_hits = 0, scanned_partitions = 0, scanned_vectors = 0;
        for (std::size_t si = 0; si < sample_count; ++si) {
            const std::size_t qi = sample_ids[si];
            const std::uint32_t mark = static_cast<std::uint32_t>(si + 1);
            for (const std::uint32_t original_id : truth[qi]) {
                if (original_id >= index.size())
                    throw std::runtime_error("truth ID exceeds original row range");
                marks[original_id] = mark;
            }
            std::vector<std::size_t> probed;
            const auto query_start = std::chrono::steady_clock::now();
            const auto found = index.search(queries.row(qi), kTopK, nprobe, &probed);
            const double elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - query_start).count();
            if (found.size() != kTopK || probed.size() != nprobe)
                throw std::runtime_error("fixed-nprobe search returned unexpected result/probe count");
            query_ms.push_back(elapsed);
            scanned_partitions += probed.size();
            std::uint64_t query_scanned_vectors = 0;
            for (const std::size_t leaf : probed) {
                if (leaf >= leaf_sizes.size()) throw std::runtime_error("invalid probed leaf ID");
                query_scanned_vectors += leaf_sizes[leaf];
            }
            scanned_vectors += query_scanned_vectors;
            std::uint64_t hits = 0;
            for (const SearchResult& result : found) {
                if (result.id < 0 || static_cast<std::size_t>(result.id) >= internal_to_original.size())
                    throw std::runtime_error("search result ID outside internal-to-original mapping");
                if (marks[internal_to_original[static_cast<std::size_t>(result.id)]] == mark) ++hits;
            }
            total_hits += hits;
            per_query << nprobe << ',' << si << ',' << qi << ',' << std::setprecision(8)
                      << static_cast<double>(hits) / kTopK << ',' << probed.size() << ','
                      << query_scanned_vectors << ',' << elapsed << '\n';
        }
        const double recall = static_cast<double>(total_hits) / (sample_count * kTopK);
        const double mean_ms = std::accumulate(query_ms.begin(), query_ms.end(), 0.0) / sample_count;
        const double total_seconds = std::accumulate(query_ms.begin(), query_ms.end(), 0.0) / 1000.0;
        summary << index.size() << ',' << index.leaf_nlist() << ',' << nprobe << ',' << sample_count
                << ',' << sample_seed << ',' << std::setprecision(10) << recall << ','
                << static_cast<double>(scanned_partitions) / sample_count << ','
                << static_cast<double>(scanned_vectors) / sample_count << ',' << mean_ms << ','
                << percentile(query_ms, 0.95) << ',' << total_seconds << ',' << load_ms << '\n';
        summary.flush();
        per_query.flush();
        std::cout << "nprobe_sweep_complete nprobe=" << nprobe << " recall@100="
                  << std::setprecision(6) << recall << " mean_query_ms=" << mean_ms
                  << " p95_query_ms=" << percentile(query_ms, 0.95)
                  << " mean_scanned_vectors=" << static_cast<double>(scanned_vectors) / sample_count
                  << '\n' << std::flush;
    }
    std::ofstream complete(output_root / "COMPLETE.txt");
    complete << "status=complete\nquery=standard IVF fixed nprobe sweep\nnlist="
             << index.leaf_nlist() << "\nnprobes=";
    for (std::size_t i = 0; i < nprobes.size(); ++i) {
        if (i) complete << ',';
        complete << nprobes[i];
    }
    complete << "\nqueries=" << sample_count
             << "\ntopk=" << kTopK << "\nsample_seed=" << sample_seed << '\n';
}

void run_aps_sample_query(const std::filesystem::path& output_root,
                          const std::filesystem::path& index_path,
                          const std::filesystem::path& mapping_path,
                          const std::filesystem::path& data_path,
                          const std::filesystem::path& query_path,
                          const std::filesystem::path& truth_path,
                          std::size_t sample_count, std::uint32_t sample_seed,
                          std::size_t candidate_partition_limit,
                          const MappedFbin& queries) {
    ensure_empty_or_create(output_root);
    if (queries.rows() < kQueries || sample_count == 0 || sample_count > kQueries)
        throw std::runtime_error("expected 1..10000 sampled queries from the 10K query file");

    std::cout << "aps_sample loading_index=" << index_path.string() << '\n' << std::flush;
    const auto load_start = std::chrono::steady_clock::now();
    AdaptiveHierarchicalIvf index = AdaptiveHierarchicalIvf::load(index_path.string());
    const double load_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - load_start).count();
    if (index.size() != 10'000'000 || index.config().dimension != kDimension ||
        candidate_partition_limit == 0 || candidate_partition_limit > index.leaf_nlist())
        throw std::runtime_error("loaded index shape does not match APS query experiment");
    const float candidate_fraction = static_cast<float>(candidate_partition_limit) /
        static_cast<float>(index.leaf_nlist());

    std::ifstream mapping_file(mapping_path, std::ios::binary);
    char mapping_magic[8]{};
    std::uint64_t mapping_count = 0;
    mapping_file.read(mapping_magic, sizeof(mapping_magic));
    mapping_file.read(reinterpret_cast<char*>(&mapping_count), sizeof(mapping_count));
    if (!mapping_file || std::string(mapping_magic, sizeof(mapping_magic)) != "MSTIDM01" ||
        mapping_count != index.size() || std::filesystem::file_size(mapping_path) !=
            sizeof(mapping_magic) + sizeof(mapping_count) + mapping_count * sizeof(std::uint32_t))
        throw std::runtime_error("index ID mapping header/size mismatch");
    std::vector<std::uint32_t> internal_to_original(static_cast<std::size_t>(mapping_count));
    mapping_file.read(reinterpret_cast<char*>(internal_to_original.data()),
                      static_cast<std::streamsize>(internal_to_original.size() * sizeof(std::uint32_t)));
    if (!mapping_file) throw std::runtime_error("cannot read full index ID mapping");

    std::ifstream truth_file(truth_path, std::ios::binary);
    char truth_magic[8]{};
    std::uint32_t truth_shape[3]{};
    truth_file.read(truth_magic, sizeof(truth_magic));
    truth_file.read(reinterpret_cast<char*>(truth_shape), sizeof(truth_shape));
    if (!truth_file || std::string(truth_magic, sizeof(truth_magic)) != "MSTGTH01" ||
        truth_shape[0] != kQueryOps || truth_shape[1] != kQueries ||
        truth_shape[2] != kTopK)
        throw std::runtime_error("expected exact stage truth MSTGTH01 [100,10000,100]");
    constexpr std::uint64_t truth_header_bytes = 8 + 3 * sizeof(std::uint32_t);
    constexpr std::uint64_t truth_stage_bytes =
        static_cast<std::uint64_t>(kQueries) * sizeof(Ids);
    if (std::filesystem::file_size(truth_path) !=
        truth_header_bytes + kQueryOps * truth_stage_bytes)
        throw std::runtime_error("stage truth file size does not match header");
    truth_file.seekg(static_cast<std::streamoff>(
        truth_header_bytes + (kQueryOps - 1) * truth_stage_bytes));
    std::vector<Ids> truth(kQueries);
    truth_file.read(reinterpret_cast<char*>(truth.data()),
                    static_cast<std::streamsize>(truth_stage_bytes));
    if (!truth_file) throw std::runtime_error("cannot read final 10M exact truth");

    std::vector<std::size_t> sample_ids(kQueries);
    std::iota(sample_ids.begin(), sample_ids.end(), 0);
    std::mt19937 sample_rng(sample_seed);
    std::shuffle(sample_ids.begin(), sample_ids.end(), sample_rng);
    sample_ids.resize(sample_count);
    std::sort(sample_ids.begin(), sample_ids.end());

    std::ofstream manifest(output_root / "manifest.txt");
    std::ofstream summary(output_root / "aps_metrics.csv");
    std::ofstream per_query(output_root / "per_query.csv");
    if (!manifest || !summary || !per_query)
        throw std::runtime_error("cannot create APS sample outputs");
    manifest << "experiment=MSTuring10M PRD-IVF APS query sample\n"
             << "dataset=" << data_path.string() << "\nqueries=" << query_path.string()
             << "\nexact_groundtruth=" << truth_path.string()
             << "\nindex=" << index_path.string() << "\nid_mapping=" << mapping_path.string()
             << "\nindexed_vectors=" << index.size() << "\nnlist=" << index.leaf_nlist()
             << "\nquery_sample_count=" << sample_count << "\nquery_sample_seed=" << sample_seed
             << "\nquery_sample=uniform without replacement from first 10000 rows"
             << "\nmetric=squared L2\nLVQ=false\naps_candidate_partition_limit="
             << candidate_partition_limit << "\naps_candidate_fraction="
             << candidate_fraction << "\naps_recall_target=1.0\naps_recompute_threshold="
             << kApsRecomputeThreshold << "\ntop_k=" << kTopK
             << "\nwarmup_queries=100\nquery_stage=final 10M stage\n";
    summary << "indexed_vectors,nlist,query_count,sample_seed,candidate_partition_limit,aps_target,aps_estimated_recall,"
               "recall_at_100,mean_candidate_partitions,mean_scanned_partitions,"
               "mean_scanned_vectors,mean_query_ms,p95_query_ms,total_query_seconds,index_load_ms\n";
    per_query << "sample_position,query_id,recall_at_100,aps_estimated_recall,candidate_partitions,"
                  "scanned_partitions,scanned_vectors,query_ms\n";

    for (std::size_t i = 0; i < std::min<std::size_t>(100, sample_count); ++i)
        (void)index.search_aps(queries.row(sample_ids[i]), kTopK, candidate_fraction,
            1.0f, kApsRecomputeThreshold, nullptr, nullptr);
    const auto leaf_sizes = index.leaf_sizes();
    std::vector<std::uint32_t> marks(index.size(), 0);
    std::vector<double> query_ms;
    query_ms.reserve(sample_count);
    std::uint64_t total_hits = 0, total_candidates = 0;
    std::uint64_t total_scanned_partitions = 0, total_scanned_vectors = 0;
    double total_estimated_recall = 0.0;
    for (std::size_t si = 0; si < sample_count; ++si) {
        const std::size_t qi = sample_ids[si];
        const std::uint32_t mark = static_cast<std::uint32_t>(si + 1);
        for (const std::uint32_t original_id : truth[qi]) {
            if (original_id >= index.size())
                throw std::runtime_error("truth ID exceeds original row range");
            marks[original_id] = mark;
        }
        ApsSearchStats stats;
        std::vector<std::size_t> probed;
        const auto query_start = std::chrono::steady_clock::now();
        const auto found = index.search_aps(queries.row(qi), kTopK,
            candidate_fraction, 1.0f, kApsRecomputeThreshold, &probed, &stats);
        const double elapsed = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - query_start).count();
        if (found.size() != kTopK)
            throw std::runtime_error("APS did not return top-100");
        query_ms.push_back(elapsed);
        total_candidates += stats.candidate_partitions;
        total_scanned_partitions += stats.scanned_partitions;
        total_estimated_recall += stats.estimated_recall;
        std::uint64_t query_scanned_vectors = 0;
        for (const std::size_t leaf : probed) {
            if (leaf >= leaf_sizes.size()) throw std::runtime_error("invalid probed leaf ID");
            query_scanned_vectors += leaf_sizes[leaf];
        }
        total_scanned_vectors += query_scanned_vectors;
        std::uint64_t hits = 0;
        for (const SearchResult& result : found) {
            if (result.id < 0 || static_cast<std::size_t>(result.id) >= internal_to_original.size())
                throw std::runtime_error("search result ID outside internal-to-original mapping");
            if (marks[internal_to_original[static_cast<std::size_t>(result.id)]] == mark) ++hits;
        }
        total_hits += hits;
        per_query << si << ',' << qi << ',' << std::setprecision(8)
                  << static_cast<double>(hits) / kTopK << ',' << stats.estimated_recall << ','
                  << stats.candidate_partitions << ',' << stats.scanned_partitions << ','
                  << query_scanned_vectors << ',' << elapsed << '\n';
        if ((si + 1) % 100 == 0)
            std::cout << "aps_sample query_progress=" << si + 1 << '/' << sample_count << '\n'
                      << std::flush;
    }
    const double recall = static_cast<double>(total_hits) / (sample_count * kTopK);
    const double mean_estimated = total_estimated_recall / sample_count;
    const double mean_ms = std::accumulate(query_ms.begin(), query_ms.end(), 0.0) / sample_count;
    const double total_seconds = std::accumulate(query_ms.begin(), query_ms.end(), 0.0) / 1000.0;
    summary << index.size() << ',' << index.leaf_nlist() << ',' << sample_count << ','
            << sample_seed << ',' << candidate_partition_limit << ",1.0," << std::setprecision(10) << mean_estimated << ','
            << recall << ',' << static_cast<double>(total_candidates) / sample_count << ','
            << static_cast<double>(total_scanned_partitions) / sample_count << ','
            << static_cast<double>(total_scanned_vectors) / sample_count << ',' << mean_ms << ','
            << percentile(query_ms, 0.95) << ',' << total_seconds << ',' << load_ms << '\n';
    summary.flush();
    per_query.flush();
    std::ofstream complete(output_root / "COMPLETE.txt");
    complete << "status=complete\nquery=APS target 1.0\nindexed_vectors=" << index.size()
             << "\nnlist=" << index.leaf_nlist() << "\nqueries=" << sample_count
             << "\ntopk=" << kTopK << "\nsample_seed=" << sample_seed << '\n';
    std::cout << "APS_SAMPLE_COMPLETE recall@100=" << std::setprecision(6) << recall
              << " estimated_recall=" << mean_estimated << " query_count=" << sample_count
              << " mean_scanned_partitions="
              << static_cast<double>(total_scanned_partitions) / sample_count
              << " mean_query_ms=" << mean_ms << " p95_query_ms="
              << percentile(query_ms, 0.95) << " output=" << output_root.string()
              << '\n' << std::flush;
}

void run_prd_aps(const std::filesystem::path& output_root,
                 const std::filesystem::path& data_path,
                 const std::filesystem::path& query_path,
                 const std::filesystem::path& truth_path,
                 const MappedFbin& data, const MappedFbin& queries,
                 bool final_only = false,
                 const std::filesystem::path& initial_index_input = {},
                 std::size_t split_refinement_radius = 0,
                 std::size_t split_refinement_iterations = 1) {
    ensure_empty_or_create(output_root);

    std::ifstream truth_file(truth_path, std::ios::binary);
    char truth_magic[8]{};
    std::uint32_t truth_shape[3]{};
    truth_file.read(truth_magic, sizeof(truth_magic));
    truth_file.read(reinterpret_cast<char*>(truth_shape), sizeof(truth_shape));
    if (!truth_file || std::string(truth_magic, sizeof(truth_magic)) != "MSTGTH01" ||
        truth_shape[0] != kQueryOps || truth_shape[1] != kQueries ||
        truth_shape[2] != kTopK)
        throw std::runtime_error("expected exact stage truth MSTGTH01 [100,10000,100]");
    constexpr std::uint64_t truth_header_bytes = 8 + 3 * sizeof(std::uint32_t);
    constexpr std::uint64_t truth_stage_bytes =
        static_cast<std::uint64_t>(kQueries) * sizeof(Ids);
    if (std::filesystem::file_size(truth_path) !=
        truth_header_bytes + kQueryOps * truth_stage_bytes)
        throw std::runtime_error("stage groundtruth file size does not match its header");
    if (data.rows() != 10'000'000 || queries.rows() < kQueries)
        throw std::runtime_error("expected 10M vectors and at least 10K query vectors");

    std::ofstream stages(output_root / (final_only
        ? "prd_aps_final_metrics.csv" : "prd_aps_stage_metrics.csv"));
    std::ofstream groups;
    if (!final_only) groups.open(output_root / "prd_aps_recall_every5.csv");
    std::ofstream insert_ops(output_root / "insert_operations.csv");
    std::ofstream manifest(output_root / "manifest.txt");
    if (!stages || !insert_ops || !manifest || (!final_only && !groups))
        throw std::runtime_error("cannot create PRD APS outputs");
    stages << "stage,active_vectors,nlist,insert_ms,maintenance_ms,recall_at_100,"
              "mean_aps_estimated_recall,mean_candidate_partitions,mean_scanned_partitions,"
              "mean_scanned_vectors,mean_query_ms,p95_query_ms,total_query_seconds\n";
    if (!final_only) {
        groups << "stage_start,stage_end,active_vectors,recall_at_100_stage1,"
                  "recall_at_100_stage2,recall_at_100_stage3,recall_at_100_stage4,"
                  "recall_at_100_stage5,mean_recall_at_100_5_stages,"
                  "mean_aps_estimated_recall,mean_query_ms\n";
    }
    insert_ops << "insert_op,active_vectors,nlist_before,nlist_after,insert_ms,maintenance_ms,"
                   "load_splits,capacity_splits,prd_splits,merges,empty_removed,"
                   "local_refinement_partitions,local_refinement_vectors,"
                   "local_refinement_reassigned_vectors\n";
    manifest << "experiment=MSTuring10M IH, PRD-IVF only, APS query\n"
             << "dataset=" << data_path.string() << "\nqueries=" << query_path.string()
             << "\nstage_groundtruth=" << truth_path.string()
             << "\ninitial_vectors=" << kInitialRows << "\ninitial_nlist=" << kInitialNlist
             << "\ninsert_batch=" << kInsertBatchRows << "\ninsert_operations=" << kInsertOps
             << "\ninsert_ops_per_query_stage=" << kInsertOpsPerQuery
             << "\nquery_stages=" << (final_only ? 1 : kQueryOps)
             << "\nqueries_per_stage=" << kQueries
             << "\nupdate_checkpoints=" << kQueryOps
             << "\ntop_k=" << kTopK << "\nmetric=squared L2\nLVQ=false\nseed=" << kSeed
             << "\nprd_threshold=" << kApsPrdThreshold
             << "\ncapacity_growth_split_factor=" << kApsCapacityGrowthSplitFactor
             << "\ncapacity_split_trigger=if live_nlist<sqrt(active_vectors), size>"
             << kApsCapacitySqrtNSplitMultiplier << "*sqrt(active_vectors); otherwise size>"
             << kApsCapacitySqrtNHighClusterSplitMultiplier
             << "*sqrt(active_vectors); growth-factor trigger disabled for this run\n"
             << "\nsplit_local_refinement_radius=" << split_refinement_radius
             << "\nsplit_local_refinement_iterations=" << split_refinement_iterations
             << "\nother_maintenance_settings=current core defaults"
             << "\naps_candidate_fraction=" << kApsCandidateFraction
             << "\naps_recall_target=" << kApsRecallTarget
             << "\naps_recompute_threshold=" << kApsRecomputeThreshold
             << "\nfixed_distance_budget=disabled\nwarmup_queries=100\n"
             << "initial_index_source=" << (initial_index_input.empty()
                    ? "built_and_saved_this_run" : initial_index_input.string()) << '\n'
             << (initial_index_input.empty()
                    ? "initial_index_snapshot=initial_1m_prd_ivf.index.bin\n" : "")
             << "query_schedule=" << (final_only
                 ? "only once after all 900 insert batches and maintenance rounds"
                 : "after each 9 insert batches") << '\n';

    AdaptiveHierarchicalConfig cfg;
    cfg.dimension = kDimension;
    cfg.initial_leaf_nlist = kInitialNlist;
    cfg.kmeans_iterations = kKmeansIterations;
    cfg.seed = kSeed;
    cfg.lvq8_storage = false;
    cfg.lvq4_storage = false;
    cfg.angular_metric = false;
    cfg.automatic_maintenance = true;
    cfg.prd_threshold = kApsPrdThreshold;
    cfg.capacity_growth_split_factor = kApsCapacityGrowthSplitFactor;
    cfg.capacity_sqrt_n_split_multiplier = kApsCapacitySqrtNSplitMultiplier;
    cfg.split_local_refinement_radius = split_refinement_radius;
    cfg.split_local_refinement_iterations = split_refinement_iterations;
    double build_ms = 0.0, load_ms = 0.0;
    AdaptiveHierarchicalIvf index = [&]() {
        if (!initial_index_input.empty()) {
            const auto load_start = std::chrono::steady_clock::now();
            AdaptiveHierarchicalIvf loaded = AdaptiveHierarchicalIvf::load(initial_index_input.string());
            load_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - load_start).count();
            const auto& loaded_cfg = loaded.config();
            if (loaded.size() != kInitialRows || loaded.leaf_nlist() != kInitialNlist ||
                loaded_cfg.dimension != kDimension || loaded_cfg.lvq8_storage ||
                loaded_cfg.lvq4_storage || loaded_cfg.angular_metric ||
                std::fabs(loaded_cfg.prd_threshold - kApsPrdThreshold) > 1e-6f ||
                std::fabs(loaded_cfg.capacity_growth_split_factor -
                          kApsCapacityGrowthSplitFactor) > 1e-6f)
                throw std::runtime_error("initial snapshot metadata does not match PRD APS experiment");
            loaded.set_split_local_refinement(split_refinement_radius,
                                              split_refinement_iterations);
            loaded.set_capacity_sqrt_n_split_multiplier(
                kApsCapacitySqrtNSplitMultiplier);
            return loaded;
        }
        AdaptiveHierarchicalIvf built(cfg);
        const auto build_start = std::chrono::steady_clock::now();
        built.train(data.row(0), kInitialRows);
        build_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - build_start).count();
        return built;
    }();
    const auto initial_index_path = output_root / "initial_1m_prd_ivf.index.bin";
    if (initial_index_input.empty()) index.save(initial_index_path.string());
    {
        std::ofstream build_file(output_root / "build.txt");
        build_file << "initial_build_ms=" << std::setprecision(10) << build_ms
                   << "\ninitial_index_load_ms=" << load_ms
                   << "\ninitial_index_source=" << (initial_index_input.empty()
                        ? initial_index_path.string() : initial_index_input.string()) << '\n';
        if (initial_index_input.empty())
            build_file << "initial_index_bytes=" << std::filesystem::file_size(initial_index_path) << '\n';
    }
    std::cout << "prd_aps initial_index_ready build_ms=" << std::fixed << std::setprecision(3)
              << build_ms << " load_ms=" << load_ms << " source="
              << (initial_index_input.empty() ? initial_index_path.string() : initial_index_input.string())
              << " refinement_radius=" << split_refinement_radius
              << " refinement_iterations=" << split_refinement_iterations << " APS_target=0.9"
              << (final_only ? " warmup_deferred_until_final_index\n"
                             : " warmup_queries=100\n") << std::flush;
    if (!final_only) {
        for (std::size_t q = 0; q < 100; ++q)
            (void)index.search_aps(queries.row(q), kTopK, kApsCandidateFraction,
                kApsRecallTarget, kApsRecomputeThreshold, nullptr, nullptr);
    }

    std::vector<std::uint32_t> truth_marks(data.rows(), 0);
    std::vector<double> group_recalls, group_estimated, group_query_ms;
    group_recalls.reserve(5);
    group_estimated.reserve(5);
    group_query_ms.reserve(5);
    std::size_t insert_op = 0;
    double total_insert_ms = 0.0, total_maintenance_ms = 0.0;
    std::uint64_t total_load_splits = 0, total_capacity_splits = 0, total_prd_splits = 0;
    std::uint64_t total_refinement_partitions = 0, total_refinement_vectors = 0;
    std::uint64_t total_refinement_reassigned_vectors = 0;
    for (std::size_t stage = 0; stage < kQueryOps; ++stage) {
        double insert_ms = 0.0, maintenance_ms = 0.0;
        for (std::size_t within = 0; within < kInsertOpsPerQuery; ++within) {
            const std::size_t begin = kInitialRows + insert_op * kInsertBatchRows;
            const std::size_t nlist_before = index.leaf_nlist();
            index.begin_update_batch();
            const auto insert_start = std::chrono::steady_clock::now();
            index.add(data.row(begin), kInsertBatchRows);
            const double one_insert_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - insert_start).count();
            const auto maintenance_start = std::chrono::steady_clock::now();
            const MaintenanceReport report = index.end_update_batch();
            const double one_maintenance_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - maintenance_start).count();
            insert_ms += one_insert_ms;
            maintenance_ms += one_maintenance_ms;
            total_insert_ms += one_insert_ms;
            total_maintenance_ms += one_maintenance_ms;
            total_load_splits += report.load_split_partitions;
            total_capacity_splits += report.capacity_split_partitions;
            total_prd_splits += report.prd_split_partitions;
            total_refinement_partitions += report.local_refinement_partitions;
            total_refinement_vectors += report.local_refinement_vectors;
            total_refinement_reassigned_vectors += report.local_refinement_reassigned_vectors;
            ++insert_op;
            insert_ops << insert_op << ',' << begin + kInsertBatchRows << ','
                       << nlist_before << ',' << index.leaf_nlist() << ','
                       << std::setprecision(8) << one_insert_ms << ','
                       << one_maintenance_ms << ',' << report.load_split_partitions << ','
                       << report.capacity_split_partitions << ','
                       << report.prd_split_partitions << ','
                       << report.merged_short_partitions << ','
                       << report.removed_empty_partitions << ','
                       << report.local_refinement_partitions << ','
                       << report.local_refinement_vectors << ','
                       << report.local_refinement_reassigned_vectors << '\n';
            if (insert_op % 100 == 0) {
                insert_ops.flush();
                std::cout << "prd_aps insert_progress=" << insert_op << '/' << kInsertOps
                          << " active=" << begin + kInsertBatchRows
                          << " nlist=" << index.leaf_nlist()
                          << " insert_ms_total=" << std::setprecision(1) << total_insert_ms
                          << " maintenance_ms_total=" << total_maintenance_ms
                          << " load_splits_total=" << total_load_splits
                          << " capacity_splits_total=" << total_capacity_splits
                          << " prd_splits_total=" << total_prd_splits
                          << " local_refinement_partitions_total=" << total_refinement_partitions
                          << " local_refinement_vectors_total=" << total_refinement_vectors
                          << " local_refinement_reassigned_total="
                          << total_refinement_reassigned_vectors << '\n' << std::flush;
            }
        }

        if (final_only && stage + 1 < kQueryOps) continue;
        if (final_only) {
            truth_file.seekg(static_cast<std::streamoff>(
                truth_header_bytes + (kQueryOps - 1) * truth_stage_bytes));
            for (std::size_t q = 0; q < 100; ++q)
                (void)index.search_aps(queries.row(q), kTopK, kApsCandidateFraction,
                    kApsRecallTarget, kApsRecomputeThreshold, nullptr, nullptr);
        }

        std::vector<Ids> stage_truth(kQueries);
        truth_file.read(reinterpret_cast<char*>(stage_truth.data()),
                        static_cast<std::streamsize>(truth_stage_bytes));
        if (!truth_file) throw std::runtime_error("truncated stage groundtruth");

        const std::size_t active = kInitialRows + insert_op * kInsertBatchRows;
        const auto leaf_sizes = index.leaf_sizes();
        std::vector<double> query_ms;
        query_ms.reserve(kQueries);
        std::uint64_t hits = 0, scanned_vectors = 0, scanned_partitions = 0;
        std::uint64_t candidate_partitions = 0;
        double estimated_recall = 0.0;
        for (std::size_t qi = 0; qi < kQueries; ++qi) {
            const std::uint32_t mark = static_cast<std::uint32_t>(stage * kQueries + qi + 1);
            for (const std::uint32_t id : stage_truth[qi]) {
                if (id >= active)
                    throw std::runtime_error("groundtruth ID is outside the active prefix");
                truth_marks[id] = mark;
            }
            ApsSearchStats aps_stats;
            std::vector<std::size_t> probed;
            const auto query_start = std::chrono::steady_clock::now();
            const auto found = index.search_aps(queries.row(qi), kTopK,
                kApsCandidateFraction, kApsRecallTarget,
                kApsRecomputeThreshold, &probed, &aps_stats);
            const double elapsed = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - query_start).count();
            if (found.size() != kTopK)
                throw std::runtime_error("APS did not return top-100 results");
            query_ms.push_back(elapsed);
            candidate_partitions += aps_stats.candidate_partitions;
            scanned_partitions += aps_stats.scanned_partitions;
            estimated_recall += aps_stats.estimated_recall;
            for (const std::size_t leaf : probed)
                if (leaf < leaf_sizes.size()) scanned_vectors += leaf_sizes[leaf];
            for (const SearchResult& result : found) {
                if (result.id < 0 || static_cast<std::size_t>(result.id) >= active)
                    throw std::runtime_error("APS returned an invalid vector ID");
                if (truth_marks[static_cast<std::size_t>(result.id)] == mark)
                    ++hits;
            }
        }
        const double recall = static_cast<double>(hits) /
            static_cast<double>(kQueries * kTopK);
        const double mean_estimated = estimated_recall / kQueries;
        const double mean_ms = std::accumulate(query_ms.begin(), query_ms.end(), 0.0) /
            static_cast<double>(kQueries);
        const double query_seconds = std::accumulate(query_ms.begin(), query_ms.end(), 0.0) / 1000.0;
        stages << stage + 1 << ',' << active << ',' << index.leaf_nlist() << ','
               << std::setprecision(8)
               << (final_only ? total_insert_ms : insert_ms) << ','
               << (final_only ? total_maintenance_ms : maintenance_ms) << ','
               << recall << ',' << mean_estimated << ','
               << static_cast<double>(candidate_partitions) / kQueries << ','
               << static_cast<double>(scanned_partitions) / kQueries << ','
               << static_cast<double>(scanned_vectors) / kQueries << ','
               << mean_ms << ',' << percentile(query_ms, 0.95) << ',' << query_seconds << '\n';
        stages.flush();

        if (final_only) {
            groups.close();
            std::cout << "FINAL_APS_RECALL100 active=" << active
                      << " nlist=" << index.leaf_nlist()
                      << " recall@100=" << std::setprecision(6) << recall
                      << " estimated_recall=" << mean_estimated
                      << " mean_query_ms=" << mean_ms << " p95_query_ms="
                      << percentile(query_ms, 0.95) << '\n' << std::flush;
        } else {
        group_recalls.push_back(recall);
        group_estimated.push_back(mean_estimated);
        group_query_ms.push_back(mean_ms);
        const double group_recall = std::accumulate(group_recalls.begin(), group_recalls.end(), 0.0) /
            static_cast<double>(group_recalls.size());
        const double group_estimate = std::accumulate(group_estimated.begin(), group_estimated.end(), 0.0) /
            static_cast<double>(group_estimated.size());
        const double group_latency = std::accumulate(group_query_ms.begin(), group_query_ms.end(), 0.0) /
            static_cast<double>(group_query_ms.size());
        std::cout << "prd_aps stage_done=" << stage + 1 << '/' << kQueryOps
                  << " active=" << active << " nlist=" << index.leaf_nlist()
                  << " recall@100=" << std::setprecision(6) << recall
                  << " estimated_recall=" << mean_estimated
                  << " mean_query_ms=" << mean_ms << " p95_query_ms="
                  << percentile(query_ms, 0.95) << "\n" << std::flush;

        if ((stage + 1) % 5 == 0) {
            const std::size_t group_end = stage + 1;
            const std::size_t group_start = group_end - 4;
            groups << group_start << ',' << group_end << ',' << active;
            for (const double item : group_recalls) groups << ',' << std::setprecision(8) << item;
            groups << ',' << group_recall << ',' << group_estimate << ',' << group_latency << '\n';
            groups.flush();
            std::cout << "APS_RECALL100_BATCH stages=" << group_start << '-' << group_end
                      << " by_stage=";
            for (std::size_t i = 0; i < group_recalls.size(); ++i) {
                if (i) std::cout << '|';
                std::cout << std::setprecision(6) << group_recalls[i];
            }
            std::cout << " mean_5_stage=" << group_recall
                      << " mean_query_ms=" << group_latency << '\n' << std::flush;
            group_recalls.clear();
            group_estimated.clear();
            group_query_ms.clear();
        }
        }
    }
    std::ofstream complete(output_root / "COMPLETE.txt");
    complete << "status=complete\nmethod=PRD-IVF only\nquery=APS\n"
             << "insert_batches=" << kInsertOps << "\ninsert_batch_size=" << kInsertBatchRows
             << "\nquery_stages=" << (final_only ? 1 : kQueryOps)
             << "\nqueries_per_stage=" << kQueries
             << "\nrecall=exact Recall@100 against saved stage truth\n";
    std::cout << "COMPLETE output=" << output_root.string() << '\n' << std::flush;
}

template <typename Data, typename Queries>
void score_existing_outputs(const std::filesystem::path& results_root,
                           const std::filesystem::path& output_root,
                           const Data& data, const Queries& queries,
                           const std::filesystem::path& exact_truth_override = {}) {
    if (results_root != output_root) ensure_empty_or_create(output_root);
    const auto truth_path = exact_truth_override.empty()
        ? output_root / "groundtruth_stage_top100.bin" : exact_truth_override;
    std::vector<Ids> truth;
    if (std::filesystem::exists(truth_path)) {
        std::ifstream gt(truth_path, std::ios::binary);
        char magic[8]{};
        std::uint32_t shape[3]{};
        gt.read(magic, sizeof(magic));
        gt.read(reinterpret_cast<char*>(shape), sizeof(shape));
        constexpr std::uint64_t header_bytes = 8 + 3 * sizeof(std::uint32_t);
        const std::uint64_t expected_bytes = header_bytes +
            static_cast<std::uint64_t>(kQueryOps) * kQueries * sizeof(Ids);
        if (!gt || std::string(magic, sizeof(magic)) != "MSTGTH01" ||
            shape[0] != kQueryOps || shape[1] != kQueries || shape[2] != kTopK ||
            std::filesystem::file_size(truth_path) != expected_bytes)
            throw std::runtime_error("cached exact stage truth header/size mismatch");
        truth.resize(kQueryOps * kQueries);
        gt.read(reinterpret_cast<char*>(truth.data()),
                static_cast<std::streamsize>(truth.size() * sizeof(Ids)));
        if (!gt) throw std::runtime_error("short cached exact stage truth");
        std::cout << "posthoc_truth_reused path=" << truth_path.string()
                  << " stages=" << kQueryOps << " queries=" << kQueries << '\n' << std::flush;
    } else {
        std::cout << "posthoc_truth_start stages=" << kQueryOps << " queries=" << kQueries
                  << " topk=" << kTopK << '\n' << std::flush;
        if constexpr (uses_u8_input_v<Data>) truth = calculate_truth_u8(data, queries);
        else truth = calculate_truth(data, queries);
        std::ofstream gt(truth_path, std::ios::binary);
        if (!gt) throw std::runtime_error("cannot create exact stage groundtruth output");
        const char magic[8] = {'M','S','T','G','T','H','0','1'};
        const std::uint32_t shape[3] = {static_cast<std::uint32_t>(kQueryOps),
            static_cast<std::uint32_t>(kQueries), static_cast<std::uint32_t>(kTopK)};
        gt.write(magic, sizeof(magic));
        gt.write(reinterpret_cast<const char*>(shape), sizeof(shape));
        for (const auto& ids : truth)
            gt.write(reinterpret_cast<const char*>(ids.data()), static_cast<std::streamsize>(sizeof(ids)));
        if (!gt) throw std::runtime_error("failed writing exact stage groundtruth");
    }
    std::ofstream all(output_root / "stage_recall_at_100.csv");
    std::ofstream status(output_root / "method_status.csv");
    if (!all || !status) throw std::runtime_error("cannot create post-hoc recall summaries");
    all << "method,stage,indexed_vectors,recall_at_100,queries,topk\n";
    status << "method,available_query_stages,last_stage,last_indexed_vectors,status\n";
    const std::array<std::string, 7> methods{
        "prd_ivf_no_reassignment", "quake", "spfresh_lire", "prd_ivf_localref_r25_i1",
        "triad", "lire", "quake_lire"};
    std::vector<std::uint32_t> truth_marks(data.rows(), 0);
    for (const auto& method : methods) {
        const auto method_dir = results_root / method;
        const auto result_path = method_dir / "results.bin";
        if (!std::filesystem::exists(result_path)) {
            status << method << ",0,,,,not_run\n";
            continue;
        }
        std::ifstream result(result_path, std::ios::binary);
        char result_magic[8]{}; std::uint32_t result_shape[3]{};
        result.read(result_magic, sizeof(result_magic));
        result.read(reinterpret_cast<char*>(result_shape), sizeof(result_shape));
        if (!result || std::string(result_magic, sizeof(result_magic)) != "MSTIHR01" ||
            result_shape[0] != kQueryOps || result_shape[1] != kQueries || result_shape[2] != kTopK)
            throw std::runtime_error("method result header mismatch: " + method);
        constexpr std::uint64_t header_bytes = 8 + 3 * sizeof(std::uint32_t);
        const std::uint64_t stage_bytes = static_cast<std::uint64_t>(kQueries) * sizeof(Ids);
        const std::uint64_t file_bytes = std::filesystem::file_size(result_path);
        if (file_bytes < header_bytes || (file_bytes - header_bytes) % stage_bytes != 0)
            throw std::runtime_error("truncated or malformed result file for " + method);
        const std::size_t available_stages = static_cast<std::size_t>((file_bytes - header_bytes) / stage_bytes);
        std::vector<std::size_t> active_at;
        {
            std::ifstream stages_file(method_dir / "stage_summary.csv");
            std::string line; std::getline(stages_file, line);
            while (std::getline(stages_file, line)) {
                std::stringstream fields(line); std::string field; std::vector<std::string> row;
                while (std::getline(fields, field, ',')) row.push_back(field);
                if (row.size() < 3) continue;
                const std::size_t stage = static_cast<std::size_t>(std::stoull(row[1]));
                if (stage >= active_at.size()) active_at.resize(stage + 1, 0);
                active_at[stage] = static_cast<std::size_t>(std::stoull(row[2]));
            }
        }
        const std::size_t score_stages = std::min({available_stages, active_at.size(), kQueryOps});
        std::ofstream detail(output_root / (method + "_stage_recall_at_100.csv"));
        if (!detail) throw std::runtime_error("cannot create per-method recall file for " + method);
        detail << "method,stage,indexed_vectors,recall_at_100,queries,topk\n";
        for (std::size_t stage = 0; stage < score_stages; ++stage) {
            std::vector<Ids> found(kQueries);
            result.seekg(static_cast<std::streamoff>(header_bytes + stage * stage_bytes));
            result.read(reinterpret_cast<char*>(found.data()), static_cast<std::streamsize>(stage_bytes));
            if (!result) throw std::runtime_error("short result stage for " + method);
            std::uint64_t hits = 0;
            for (std::size_t qi = 0; qi < kQueries; ++qi) {
                const auto& expected = truth[stage * kQueries + qi];
                const std::uint32_t token = static_cast<std::uint32_t>(stage * kQueries + qi + 1);
                for (const auto id : expected) if (id < truth_marks.size()) truth_marks[id] = token;
                for (const auto id : found[qi])
                    if (id < truth_marks.size() && truth_marks[id] == token) ++hits;
            }
            const double recall = static_cast<double>(hits) /
                static_cast<double>(kQueries * kTopK);
            detail << method << ',' << stage << ',' << active_at[stage] << ','
                   << std::setprecision(10) << recall << ',' << kQueries << ',' << kTopK << '\n';
            all << method << ',' << stage << ',' << active_at[stage] << ','
                << std::setprecision(10) << recall << ',' << kQueries << ',' << kTopK << '\n';
        }
        detail.close();
        status << method << ',' << score_stages << ','
               << (score_stages ? std::to_string(score_stages - 1) : std::string()) << ','
               << (score_stages ? std::to_string(active_at[score_stages - 1]) : std::string()) << ','
               << (score_stages == kQueryOps ? "complete" : "partial") << '\n';
        std::cout << "posthoc_recall_done method=" << method << " stages=" << score_stages << '\n' << std::flush;
    }
    std::ofstream complete(output_root / "COMPLETE.txt");
    complete << "exact_truth_stages=" << kQueryOps << "\nqueries_per_stage=" << kQueries
             << "\ntopk=" << kTopK << "\nscored_methods=available outputs only\n";
}

void write_truth_and_recall(const std::filesystem::path& root, const MappedFbin& data,
                            const MappedFbin& queries) {
    score_existing_outputs(root, root, data, queries);
}

void score_previous_final_recall(const std::filesystem::path& root,
                                const std::filesystem::path& truth_path) {
    std::ifstream truth_file(truth_path, std::ios::binary);
    std::uint32_t truth_shape[2]{};
    truth_file.read(reinterpret_cast<char*>(truth_shape), sizeof(truth_shape));
    if (!truth_file || truth_shape[0] != kQueries || truth_shape[1] != kTopK)
        throw std::runtime_error("final groundtruth must be MSTuring gt100: 10000 x 100 int32 IDs followed by float distances");
    const std::size_t truth_count = static_cast<std::size_t>(truth_shape[0]) * truth_shape[1];
    std::vector<std::int32_t> truth(truth_count);
    truth_file.read(reinterpret_cast<char*>(truth.data()),
                    static_cast<std::streamsize>(truth.size() * sizeof(std::int32_t)));
    if (!truth_file) throw std::runtime_error("short final groundtruth file");

    const auto summary_path = root / "previous_experiment_final_recall.csv";
    if (std::filesystem::exists(summary_path))
        throw std::runtime_error("refusing to overwrite existing recall summary: " + summary_path.string());
    std::ofstream summary(summary_path);
    if (!summary) throw std::runtime_error("cannot create previous_experiment_final_recall.csv");
    summary << "method,available_query_stages,last_stage,indexed_vectors,recall_at_100,status\n";
    for (const std::string method : {"prd_ivf", "frozen_ivf", "quake", "spfresh_lire"}) {
        const auto dir = root / method;
        const auto result_path = dir / "results.bin";
        if (!std::filesystem::exists(result_path)) {
            summary << method << ",0,,,,not_run\n";
            continue;
        }
        std::ifstream result(result_path, std::ios::binary);
        char magic[8]{}; std::uint32_t shape[3]{};
        result.read(magic, sizeof(magic));
        result.read(reinterpret_cast<char*>(shape), sizeof(shape));
        if (!result || std::string(magic, sizeof(magic)) != "MSTIHR01" ||
            shape[1] != kQueries || shape[2] != kTopK)
            throw std::runtime_error("unexpected result format for " + method);
        constexpr std::uint64_t header_bytes = 8 + 3 * sizeof(std::uint32_t);
        const std::uint64_t row_bytes = static_cast<std::uint64_t>(shape[1]) * shape[2] * sizeof(std::uint32_t);
        const std::uint64_t file_bytes = std::filesystem::file_size(result_path);
        if (file_bytes < header_bytes || row_bytes == 0 || (file_bytes - header_bytes) % row_bytes != 0)
            throw std::runtime_error("truncated or malformed result file for " + method);
        const std::size_t stages = static_cast<std::size_t>((file_bytes - header_bytes) / row_bytes);
        if (stages == 0) {
            summary << method << ",0,,,,no_query_stages\n";
            continue;
        }
        std::size_t active = 0;
        {
            std::ifstream stages_file(dir / "stage_summary.csv");
            std::string line; std::getline(stages_file, line);
            while (std::getline(stages_file, line)) {
                std::stringstream fields(line); std::string field;
                std::vector<std::string> row;
                while (std::getline(fields, field, ',')) row.push_back(field);
                if (row.size() >= 3 && static_cast<std::size_t>(std::stoull(row[1])) == stages - 1)
                    active = static_cast<std::size_t>(std::stoull(row[2]));
            }
        }
        if (stages - 1 != kQueryOps - 1 || active != 10'000'000) {
            summary << method << ',' << stages << ',' << stages - 1 << ',' << active
                    << ",,partial_no_matching_prefix_groundtruth\n";
            continue;
        }
        result.seekg(static_cast<std::streamoff>(header_bytes + (stages - 1) * row_bytes));
        std::vector<std::uint32_t> found(static_cast<std::size_t>(shape[1]) * shape[2]);
        result.read(reinterpret_cast<char*>(found.data()),
                    static_cast<std::streamsize>(found.size() * sizeof(std::uint32_t)));
        if (!result) throw std::runtime_error("cannot read final query stage for " + method);
        std::uint64_t hits = 0;
        for (std::size_t q = 0; q < shape[1]; ++q) {
            std::unordered_set<std::uint32_t> expected;
            expected.reserve(kTopK * 2);
            for (std::size_t k = 0; k < kTopK; ++k) {
                const std::int32_t id = truth[q * kTopK + k];
                if (id >= 0)
                    expected.insert(static_cast<std::uint32_t>(id));
            }
            for (std::size_t k = 0; k < kTopK; ++k)
                if (expected.count(found[q * kTopK + k]) != 0) ++hits;
        }
        const double recall = static_cast<double>(hits) /
            static_cast<double>(shape[1] * shape[2]);
        summary << method << ',' << stages << ',' << stages - 1 << ',' << active << ','
                << std::setprecision(10) << recall << ",exact_final_prefix\n";
    }
}

void score_shuffled_final_lire_recall(const std::filesystem::path& root,
                                     const std::filesystem::path& truth_path,
                                     const std::filesystem::path& permutation_path,
                                     const std::string& method_dir = "quake_lire") {
    constexpr std::size_t rows = 10'000'000;
    std::ifstream permutation_file(permutation_path, std::ios::binary);
    char permutation_magic[8]{};
    std::uint32_t permutation_shape[2]{};
    permutation_file.read(permutation_magic, sizeof(permutation_magic));
    permutation_file.read(reinterpret_cast<char*>(permutation_shape), sizeof(permutation_shape));
    const std::uint64_t permutation_bytes = 8 + sizeof(permutation_shape) +
        static_cast<std::uint64_t>(rows) * sizeof(std::uint32_t);
    if (!permutation_file || std::string(permutation_magic, sizeof(permutation_magic)) != "MSTPERM1" ||
        permutation_shape[0] != rows || permutation_shape[1] != kSeed ||
        std::filesystem::file_size(permutation_path) != permutation_bytes)
        throw std::runtime_error("invalid seed-1234 MSTuring row permutation");
    std::vector<std::uint32_t> inverse(rows, std::numeric_limits<std::uint32_t>::max());
    for (std::size_t shuffled_id = 0; shuffled_id < rows; ++shuffled_id) {
        std::uint32_t original_id = 0;
        permutation_file.read(reinterpret_cast<char*>(&original_id), sizeof(original_id));
        if (!permutation_file || original_id >= rows || inverse[original_id] != std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("permutation is truncated, out of range, or not bijective");
        inverse[original_id] = static_cast<std::uint32_t>(shuffled_id);
    }

    std::ifstream truth_file(truth_path, std::ios::binary);
    std::uint32_t truth_shape[2]{};
    truth_file.read(reinterpret_cast<char*>(truth_shape), sizeof(truth_shape));
    const std::uint64_t truth_count = static_cast<std::uint64_t>(kQueries) * kTopK;
    const std::uint64_t truth_bytes = 2 * sizeof(std::uint32_t) + 2 * truth_count * sizeof(std::uint32_t);
    if (!truth_file || truth_shape[0] != kQueries || truth_shape[1] != kTopK ||
        std::filesystem::file_size(truth_path) != truth_bytes)
        throw std::runtime_error("final MSTuring truth must be [10000,100] IDs followed by float distances");
    std::vector<std::uint32_t> truth(truth_count);
    truth_file.read(reinterpret_cast<char*>(truth.data()), static_cast<std::streamsize>(truth.size() * sizeof(std::uint32_t)));
    if (!truth_file) throw std::runtime_error("short read from final MSTuring ground truth");
    for (std::uint32_t& id : truth) {
        if (id >= rows || inverse[id] == std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("ground-truth ID is outside the shuffled dataset mapping");
        id = inverse[id];
    }

    const auto read_single_stage = [&](const std::filesystem::path& result_path) {
        std::ifstream result_file(result_path, std::ios::binary);
        char result_magic[8]{};
        std::uint32_t result_shape[3]{};
        result_file.read(result_magic, sizeof(result_magic));
        result_file.read(reinterpret_cast<char*>(result_shape), sizeof(result_shape));
        if (!result_file || std::string(result_magic, sizeof(result_magic)) != "MSTIHR01" ||
            result_shape[0] != 1 || result_shape[1] != kQueries || result_shape[2] != kTopK)
            throw std::runtime_error("final-query result must contain exactly one [10000,100] stage: " + result_path.string());
        std::vector<Ids> values(kQueries);
        result_file.read(reinterpret_cast<char*>(values.data()),
                         static_cast<std::streamsize>(values.size() * sizeof(Ids)));
        if (!result_file) throw std::runtime_error("short read from final query result: " + result_path.string());
        return values;
    };
    const auto fixed_budget_results = read_single_stage(root / method_dir / "results.bin");
    const auto aps_results = read_single_stage(root / method_dir / "aps09_results.bin");

    std::ofstream remapped_truth(root / "groundtruth_final_shuffled_ids.bin", std::ios::binary);
    if (!remapped_truth) throw std::runtime_error("cannot create remapped final ground truth");
    const char truth_magic[8] = {'M','S','T','G','T','H','0','1'};
    const std::uint32_t output_shape[3] = {1, static_cast<std::uint32_t>(kQueries), static_cast<std::uint32_t>(kTopK)};
    remapped_truth.write(truth_magic, sizeof(truth_magic));
    remapped_truth.write(reinterpret_cast<const char*>(output_shape), sizeof(output_shape));
    remapped_truth.write(reinterpret_cast<const char*>(truth.data()), static_cast<std::streamsize>(truth.size() * sizeof(std::uint32_t)));
    if (!remapped_truth) throw std::runtime_error("failed writing remapped final ground truth");

    std::uint64_t fixed_hits10 = 0, fixed_hits100 = 0, aps_hits10 = 0, aps_hits100 = 0;
    std::ofstream per_query(root / "final_recall_per_query.csv");
    std::ofstream summary(root / "final_recall.csv");
    if (!per_query || !summary) throw std::runtime_error("cannot create final recall reports");
    per_query << "query_id,fixed_budget_hits_at_10,fixed_budget_recall_at_10,fixed_budget_hits_at_100,fixed_budget_recall_at_100,aps09_hits_at_10,aps09_recall_at_10,aps09_hits_at_100,aps09_recall_at_100\n";
    for (std::size_t qi = 0; qi < kQueries; ++qi) {
        std::unordered_set<std::uint32_t> truth10, truth100;
        for (std::size_t k = 0; k < 10; ++k) truth10.insert(truth[qi * kTopK + k]);
        for (std::size_t k = 0; k < kTopK; ++k) truth100.insert(truth[qi * kTopK + k]);
        std::size_t fixed_qhits10 = 0, fixed_qhits100 = 0, aps_qhits10 = 0, aps_qhits100 = 0;
        for (std::size_t k = 0; k < 10; ++k) {
            fixed_qhits10 += truth10.count(fixed_budget_results[qi][k]);
            aps_qhits10 += truth10.count(aps_results[qi][k]);
        }
        for (std::size_t k = 0; k < kTopK; ++k) {
            fixed_qhits100 += truth100.count(fixed_budget_results[qi][k]);
            aps_qhits100 += truth100.count(aps_results[qi][k]);
        }
        fixed_hits10 += fixed_qhits10; fixed_hits100 += fixed_qhits100;
        aps_hits10 += aps_qhits10; aps_hits100 += aps_qhits100;
        per_query << qi << ',' << fixed_qhits10 << ',' << static_cast<double>(fixed_qhits10) / 10.0 << ','
                  << fixed_qhits100 << ',' << static_cast<double>(fixed_qhits100) / 100.0 << ','
                  << aps_qhits10 << ',' << static_cast<double>(aps_qhits10) / 10.0 << ','
                  << aps_qhits100 << ',' << static_cast<double>(aps_qhits100) / 100.0 << '\n';
    }
    const double fixed_recall10 = static_cast<double>(fixed_hits10) / (kQueries * 10.0);
    const double fixed_recall100 = static_cast<double>(fixed_hits100) / (kQueries * 100.0);
    const double aps_recall10 = static_cast<double>(aps_hits10) / (kQueries * 10.0);
    const double aps_recall100 = static_cast<double>(aps_hits100) / (kQueries * 100.0);
    summary << "query_mode,queries,recall_at_10,recall_at_100\n"
            << "fixed_budget_500k," << kQueries << ',' << std::setprecision(10) << fixed_recall10 << ',' << fixed_recall100 << '\n'
            << "APS_target_0.9," << kQueries << ',' << aps_recall10 << ',' << aps_recall100 << '\n';
    std::cout << "final_recall fixed_budget_recall_at_10=" << fixed_recall10
              << " fixed_budget_recall_at_100=" << fixed_recall100
              << " aps09_recall_at_10=" << aps_recall10 << " aps09_recall_at_100=" << aps_recall100 << '\n' << std::flush;
}

void score_final_aps_recall(const std::filesystem::path& method_root,
                            const std::filesystem::path& truth_path,
                            const std::filesystem::path& permutation_path,
                            const std::string& method_name) {
    constexpr std::size_t rows = 10'000'000;
    std::ifstream permutation_file(permutation_path, std::ios::binary);
    char permutation_magic[8]{};
    std::uint32_t permutation_shape[2]{};
    permutation_file.read(permutation_magic, sizeof(permutation_magic));
    permutation_file.read(reinterpret_cast<char*>(permutation_shape), sizeof(permutation_shape));
    const std::uint64_t permutation_bytes = 8 + sizeof(permutation_shape) +
        static_cast<std::uint64_t>(rows) * sizeof(std::uint32_t);
    if (!permutation_file || std::string(permutation_magic, sizeof(permutation_magic)) != "MSTPERM1" ||
        permutation_shape[0] != rows || permutation_shape[1] != kSeed ||
        std::filesystem::file_size(permutation_path) != permutation_bytes)
        throw std::runtime_error("invalid seed-1234 MSTuring row permutation");
    std::vector<std::uint32_t> inverse(rows, std::numeric_limits<std::uint32_t>::max());
    for (std::size_t shuffled_id = 0; shuffled_id < rows; ++shuffled_id) {
        std::uint32_t original_id = 0;
        permutation_file.read(reinterpret_cast<char*>(&original_id), sizeof(original_id));
        if (!permutation_file || original_id >= rows ||
            inverse[original_id] != std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("permutation is truncated, out of range, or not bijective");
        inverse[original_id] = static_cast<std::uint32_t>(shuffled_id);
    }

    std::ifstream truth_file(truth_path, std::ios::binary);
    std::uint32_t truth_shape[2]{};
    truth_file.read(reinterpret_cast<char*>(truth_shape), sizeof(truth_shape));
    const std::uint64_t truth_count = static_cast<std::uint64_t>(kQueries) * kTopK;
    const std::uint64_t truth_bytes = 2 * sizeof(std::uint32_t) +
        2 * truth_count * sizeof(std::uint32_t);
    if (!truth_file || truth_shape[0] != kQueries || truth_shape[1] != kTopK ||
        std::filesystem::file_size(truth_path) != truth_bytes)
        throw std::runtime_error("final MSTuring truth must be [10000,100] IDs followed by float distances");
    std::vector<std::uint32_t> truth(truth_count);
    truth_file.read(reinterpret_cast<char*>(truth.data()),
                    static_cast<std::streamsize>(truth.size() * sizeof(std::uint32_t)));
    if (!truth_file) throw std::runtime_error("short read from final MSTuring ground truth");
    for (std::uint32_t& id : truth) {
        if (id >= rows || inverse[id] == std::numeric_limits<std::uint32_t>::max())
            throw std::runtime_error("ground-truth ID is outside the shuffled dataset mapping");
        id = inverse[id];
    }

    const auto result_path = method_root / "results.bin";
    std::ifstream result_file(result_path, std::ios::binary);
    char result_magic[8]{};
    std::uint32_t result_shape[3]{};
    result_file.read(result_magic, sizeof(result_magic));
    result_file.read(reinterpret_cast<char*>(result_shape), sizeof(result_shape));
    const std::uint64_t result_bytes = 8 + 3 * sizeof(std::uint32_t) +
        static_cast<std::uint64_t>(kQueries) * sizeof(Ids);
    if (!result_file || std::string(result_magic, sizeof(result_magic)) != "MSTIHR01" ||
        result_shape[0] != 1 || result_shape[1] != kQueries || result_shape[2] != kTopK ||
        std::filesystem::file_size(result_path) != result_bytes)
        throw std::runtime_error("final APS result must contain exactly one [10000,100] stage");
    std::vector<Ids> found(kQueries);
    result_file.read(reinterpret_cast<char*>(found.data()),
                     static_cast<std::streamsize>(found.size() * sizeof(Ids)));
    if (!result_file) throw std::runtime_error("short read from final APS result");

    std::uint64_t hits10 = 0, hits100 = 0;
    std::ofstream per_query(method_root / "final_aps_recall_per_query.csv");
    std::ofstream summary(method_root / "final_aps_recall.csv");
    if (!per_query || !summary) throw std::runtime_error("cannot create final APS recall reports");
    per_query << "query_id,hits_at_10,recall_at_10,hits_at_100,recall_at_100\n";
    for (std::size_t qi = 0; qi < kQueries; ++qi) {
        std::unordered_set<std::uint32_t> truth10, truth100;
        for (std::size_t k = 0; k < 10; ++k) truth10.insert(truth[qi * kTopK + k]);
        for (std::size_t k = 0; k < kTopK; ++k) truth100.insert(truth[qi * kTopK + k]);
        std::size_t query_hits10 = 0, query_hits100 = 0;
        for (std::size_t k = 0; k < 10; ++k) query_hits10 += truth10.count(found[qi][k]);
        for (std::size_t k = 0; k < kTopK; ++k) query_hits100 += truth100.count(found[qi][k]);
        hits10 += query_hits10;
        hits100 += query_hits100;
        per_query << qi << ',' << query_hits10 << ',' << static_cast<double>(query_hits10) / 10.0
                  << ',' << query_hits100 << ',' << static_cast<double>(query_hits100) / 100.0 << '\n';
    }
    const double recall10 = static_cast<double>(hits10) / (kQueries * 10.0);
    const double recall100 = static_cast<double>(hits100) / (kQueries * 100.0);
    summary << "method,query_mode,queries,recall_at_10,recall_at_100\n"
            << method_name << ",APS_target_0.9_final,10000," << std::setprecision(10)
            << recall10 << ',' << recall100 << '\n';
    std::cout << method_name << " final_aps_recall_at_10=" << recall10
              << " final_aps_recall_at_100=" << recall100 << '\n' << std::flush;
}

void score_existing_recall_at_10(const std::filesystem::path& results_root,
                                 const std::filesystem::path& truth_path,
                                 const std::filesystem::path& output_root) {
    ensure_empty_or_create(output_root);
    std::ifstream truth_file(truth_path, std::ios::binary);
    char truth_magic[8]{};
    std::uint32_t truth_shape[3]{};
    truth_file.read(truth_magic, sizeof(truth_magic));
    truth_file.read(reinterpret_cast<char*>(truth_shape), sizeof(truth_shape));
    if (!truth_file || std::string(truth_magic, sizeof(truth_magic)) != "MSTGTH01" ||
        truth_shape[0] != kQueryOps || truth_shape[1] != kQueries || truth_shape[2] != kTopK)
        throw std::runtime_error("expected stage exact truth MSTGTH01 [100,10000,100]");

    constexpr std::uint64_t header_bytes = 8 + 3 * sizeof(std::uint32_t);
    const std::uint64_t stage_bytes = static_cast<std::uint64_t>(kQueries) * sizeof(Ids);
    const std::uint64_t truth_bytes = std::filesystem::file_size(truth_path);
    if (truth_bytes != header_bytes + kQueryOps * stage_bytes)
        throw std::runtime_error("stage exact truth file size does not match its header");

    std::ofstream detail(output_root / "recall_at_10_by_stage.csv");
    std::ofstream summary(output_root / "recall_at_10_summary.csv");
    if (!detail || !summary) throw std::runtime_error("cannot create Recall@10 output files");
    detail << "method,stage,indexed_vectors,recall_at_10,queries,topk\n";
    summary << "method,available_stages,mean_recall_at_10,stage0_recall_at_10,stage56_recall_at_10,final_recall_at_10,status\n";

    for (const std::string method : {"prd_ivf_no_reassignment", "quake", "spfresh_lire", "prd_ivf_localref_r25_i1", "triad", "lire", "quake_lire"}) {
        const auto method_dir = results_root / method;
        const auto result_path = method_dir / "results.bin";
        if (!std::filesystem::exists(result_path)) {
            summary << method << ",0,,,,,not_run\n";
            continue;
        }
        std::ifstream result(result_path, std::ios::binary);
        char result_magic[8]{};
        std::uint32_t result_shape[3]{};
        result.read(result_magic, sizeof(result_magic));
        result.read(reinterpret_cast<char*>(result_shape), sizeof(result_shape));
        if (!result || std::string(result_magic, sizeof(result_magic)) != "MSTIHR01" ||
            result_shape[0] != kQueryOps || result_shape[1] != kQueries || result_shape[2] != kTopK)
            throw std::runtime_error("method result header mismatch: " + method);
        const std::uint64_t result_bytes = std::filesystem::file_size(result_path);
        if (result_bytes < header_bytes || (result_bytes - header_bytes) % stage_bytes != 0)
            throw std::runtime_error("truncated or malformed result file for " + method);
        const std::size_t available_stages = static_cast<std::size_t>((result_bytes - header_bytes) / stage_bytes);

        std::vector<std::size_t> active_at;
        {
            std::ifstream stages_file(method_dir / "stage_summary.csv");
            std::string line;
            std::getline(stages_file, line);
            while (std::getline(stages_file, line)) {
                std::stringstream fields(line);
                std::string field;
                std::vector<std::string> row;
                while (std::getline(fields, field, ',')) row.push_back(field);
                if (row.size() < 3) continue;
                const std::size_t stage = static_cast<std::size_t>(std::stoull(row[1]));
                if (stage >= active_at.size()) active_at.resize(stage + 1, 0);
                active_at[stage] = static_cast<std::size_t>(std::stoull(row[2]));
            }
        }
        const std::size_t score_stages = std::min({available_stages, active_at.size(), kQueryOps});
        std::vector<Ids> expected(kQueries), found(kQueries);
        std::vector<double> recalls(score_stages, 0.0);
        for (std::size_t stage = 0; stage < score_stages; ++stage) {
            truth_file.seekg(static_cast<std::streamoff>(header_bytes + stage * stage_bytes));
            truth_file.read(reinterpret_cast<char*>(expected.data()), static_cast<std::streamsize>(stage_bytes));
            if (!truth_file) throw std::runtime_error("short exact truth stage " + std::to_string(stage));
            result.seekg(static_cast<std::streamoff>(header_bytes + stage * stage_bytes));
            result.read(reinterpret_cast<char*>(found.data()), static_cast<std::streamsize>(stage_bytes));
            if (!result) throw std::runtime_error("short result stage for " + method);

            std::uint64_t hits = 0;
            for (std::size_t qi = 0; qi < kQueries; ++qi) {
                for (std::size_t rank = 0; rank < 10; ++rank) {
                    const auto id = found[qi][rank];
                    bool match = false;
                    for (std::size_t exact_rank = 0; exact_rank < 10; ++exact_rank)
                        if (id == expected[qi][exact_rank]) { match = true; break; }
                    if (match) ++hits;
                }
            }
            recalls[stage] = static_cast<double>(hits) / static_cast<double>(kQueries * 10);
            detail << method << ',' << stage << ',' << active_at[stage] << ','
                   << std::setprecision(10) << recalls[stage] << ',' << kQueries << ",10\n";
        }

        if (score_stages == 0) {
            summary << method << ",0,,,,,no_query_stages\n";
            continue;
        }
        const double mean = std::accumulate(recalls.begin(), recalls.end(), 0.0) / score_stages;
        summary << method << ',' << score_stages << ',' << std::setprecision(10) << mean << ','
                << recalls.front() << ',';
        if (score_stages > 56) summary << recalls[56];
        summary << ',';
        if (score_stages == kQueryOps) summary << recalls.back();
        summary << ',' << (score_stages == kQueryOps ? "complete" : "partial") << '\n';
        std::cout << "recall_at_10_done method=" << method << " stages=" << score_stages << '\n' << std::flush;
    }
}

template <typename Data, typename Queries>
void run_triad_fixed_n20(const std::filesystem::path& data_path,
                         const std::filesystem::path& query_path,
                         const std::filesystem::path& truth_path,
                         const std::filesystem::path& initial_index_path,
                         const std::filesystem::path& output_root,
                         const Data& data, const Queries& queries) {
    if (data.rows() != 10'000'000 || queries.rows() != kQueries)
        throw std::runtime_error("Triad fixed-N/20 run expects 10M base vectors and the configured query sample");
    if (!std::filesystem::is_regular_file(truth_path) ||
        !std::filesystem::is_regular_file(initial_index_path))
        throw std::runtime_error("cached stage truth or reusable initial index does not exist");

    {
        std::ifstream truth(truth_path, std::ios::binary);
        char magic[8]{};
        std::uint32_t shape[3]{};
        truth.read(magic, sizeof(magic));
        truth.read(reinterpret_cast<char*>(shape), sizeof(shape));
        const std::uint64_t expected_bytes = 8 + 3 * sizeof(std::uint32_t) +
            static_cast<std::uint64_t>(kQueryOps) * kQueries * sizeof(Ids);
        if (!truth || std::string(magic, sizeof(magic)) != "MSTGTH01" ||
            shape[0] != kQueryOps || shape[1] != kQueries || shape[2] != kTopK ||
            std::filesystem::file_size(truth_path) != expected_bytes)
            throw std::runtime_error("cached exact stage truth does not match this 100-stage Triad run");
    }

    ensure_empty_or_create(output_root);
    std::filesystem::copy_file(truth_path, output_root / "groundtruth_stage_top100.bin");
    {
        std::ofstream manifest(output_root / "manifest.txt", std::ios::trunc);
        if (!manifest) throw std::runtime_error("cannot create Triad fixed-N/20 manifest");
        manifest << "experiment=Triad fixed-budget insert-heavy workload (IH)\n"
                 << "dataset=" << data_path.string() << "\nqueries=" << query_path.string()
                 << "\nrows=10000000; dimension=" << kDimension << "; metric=" << kMetricName
                 << "\ninitial_vectors=" << kInitialRows << "; initial_nlist=" << kInitialNlist
                 << "\ninsert_batch=" << kInsertBatchRows << "; insert_ops=" << kInsertOps
                 << "\nquery_schedule=after every " << kInsertOpsPerQuery
                 << " insert batches; query_stages=" << kQueryOps
                 << "; queries_per_stage=" << kQueries
                 << "\ntopk=" << kTopK
                 << "\nquery_budget=floor(current active indexed vectors/20), including centroid and vector distances"
                 << "\nTriad PRD threshold=0.075"
                 << "\nshort-list merge cutoff=0.2*sqrt(current active N)"
                 << "\nload split cutoff=2.2*T; capacity split=1.05*sqrt(N), active only while nlist<=sqrt(N)"
                 << "\nPRD criterion=GlobalFitOnly; PRD action=split directly; load merge+split enabled"
                 << "\nlocal split reassignment radius=50; iterations=1"
                 << "\nglobal center refresh=all centers to stored partition means after each update batch"
                 << "\ninitial_index_reused=true; initial_index=" << initial_index_path.string()
                 << "\nLVQ=false; query early-stop=false; seed=" << kSeed
                 << "\ngroundtruth=reused exact top-100 for each active prefix\n"
                 << "final_index_saved=false\n";
    }

    AdaptiveMethodConfig triad;
    triad.name = "triad";
    triad.prd = true;
    triad.triad_policy = true;
    triad.split_local_refinement_radius = 50;
    triad.split_local_refinement_iterations = 1;
    triad.prd_threshold_override = 0.075f;
    triad.short_list_sqrt_n_fraction_override = 0.20f;
    run_adaptive_clean(output_root, data, queries, triad, initial_index_path);
    score_existing_outputs(output_root, output_root, data, queries);
    score_existing_recall_at_10(output_root, output_root / "groundtruth_stage_top100.bin",
                                output_root / "recall_at_10");

    std::ofstream complete(output_root / "COMPLETE.txt", std::ios::trunc);
    if (!complete) throw std::runtime_error("cannot write Triad fixed-N/20 completion marker");
    complete << "status=complete\nmethod=Triad\ninsert_ops=" << kInsertOps
             << "\nquery_stages=" << kQueryOps << "\nqueries_per_stage=" << kQueries
             << "\nquery_budget=floor(current active indexed vectors/20), including centroid distances"
             << "\nprd_threshold=0.075\nshort_list_merge_cutoff=0.2*sqrt(N)"
             << "\nload_split_cutoff=2.2*T\ncapacity_split=1.05*sqrt(N) while nlist<=sqrt(N)"
             << "\nlocal_split_reassignment=radius 50, one iteration"
             << "\ninitial_index_reused=" << initial_index_path.string()
             << "\nrecall=post-hoc exact Recall@100 and Recall@10\nfinal_index_saved=false\n";
    complete.close();
    std::cout << "TRIAD_N20_COMPLETE output=" << output_root.string() << '\n' << std::flush;
}

void score_triad_final_fixed_budget(const std::filesystem::path& method_root,
                                    const std::filesystem::path& truth_path,
                                    const std::string& method_name) {
    constexpr std::uint64_t header_bytes = 8 + 3 * sizeof(std::uint32_t);
    std::ifstream truth_file(truth_path, std::ios::binary);
    char truth_magic[8]{};
    std::uint32_t truth_shape[3]{};
    truth_file.read(truth_magic, sizeof(truth_magic));
    truth_file.read(reinterpret_cast<char*>(truth_shape), sizeof(truth_shape));
    if (!truth_file || std::string(truth_magic, sizeof(truth_magic)) != "MSTGTH01" ||
        truth_shape[0] != kQueryOps || truth_shape[1] != kQueries || truth_shape[2] != kTopK)
        throw std::runtime_error("final fixed-budget scoring expects exact stage truth [100,10000,100]");

    const std::uint64_t row_bytes = static_cast<std::uint64_t>(kQueries) * sizeof(Ids);
    const std::uint64_t expected_truth_bytes = header_bytes +
        static_cast<std::uint64_t>(kQueryOps) * row_bytes;
    if (std::filesystem::file_size(truth_path) != expected_truth_bytes)
        throw std::runtime_error("final stage truth file size mismatch");
    truth_file.seekg(static_cast<std::streamoff>(header_bytes +
        static_cast<std::uint64_t>(kQueryOps - 1) * row_bytes));
    std::vector<Ids> truth(kQueries);
    truth_file.read(reinterpret_cast<char*>(truth.data()), static_cast<std::streamsize>(row_bytes));
    if (!truth_file) throw std::runtime_error("short read from final exact truth stage");

    const auto result_path = method_root / "results.bin";
    std::ifstream result_file(result_path, std::ios::binary);
    char result_magic[8]{};
    std::uint32_t result_shape[3]{};
    result_file.read(result_magic, sizeof(result_magic));
    result_file.read(reinterpret_cast<char*>(result_shape), sizeof(result_shape));
    if (!result_file || std::string(result_magic, sizeof(result_magic)) != "MSTIHR01" ||
        result_shape[0] != 1 || result_shape[1] != kQueries || result_shape[2] != kTopK ||
        std::filesystem::file_size(result_path) != header_bytes + row_bytes)
        throw std::runtime_error("final fixed-budget result must contain one [10000,100] query stage");
    std::vector<Ids> found(kQueries);
    result_file.read(reinterpret_cast<char*>(found.data()), static_cast<std::streamsize>(row_bytes));
    if (!result_file) throw std::runtime_error("short read from final fixed-budget result");

    std::uint64_t hits10 = 0, hits100 = 0;
    std::ofstream per_query(method_root / "final_recall_per_query.csv", std::ios::trunc);
    std::ofstream summary(method_root / "final_recall.csv", std::ios::trunc);
    if (!per_query || !summary) throw std::runtime_error("cannot create final recall reports");
    per_query << "query_id,hits_at_10,recall_at_10,hits_at_100,recall_at_100\n";
    for (std::size_t qi = 0; qi < kQueries; ++qi) {
        std::unordered_set<std::uint32_t> truth10, truth100;
        truth10.reserve(20);
        truth100.reserve(kTopK * 2);
        for (std::size_t rank = 0; rank < 10; ++rank) truth10.insert(truth[qi][rank]);
        for (std::size_t rank = 0; rank < kTopK; ++rank) truth100.insert(truth[qi][rank]);
        std::size_t query_hits10 = 0, query_hits100 = 0;
        for (std::size_t rank = 0; rank < 10; ++rank)
            query_hits10 += truth10.count(found[qi][rank]);
        for (std::size_t rank = 0; rank < kTopK; ++rank)
            query_hits100 += truth100.count(found[qi][rank]);
        hits10 += query_hits10;
        hits100 += query_hits100;
        per_query << qi << ',' << query_hits10 << ',' << static_cast<double>(query_hits10) / 10.0
                  << ',' << query_hits100 << ',' << static_cast<double>(query_hits100) / 100.0 << '\n';
    }
    const double recall10 = static_cast<double>(hits10) / (kQueries * 10.0);
    const double recall100 = static_cast<double>(hits100) / (kQueries * 100.0);
    summary << "method,active_vectors,queries,topk,recall_at_10,recall_at_100\n"
            << method_name << ",10000000," << kQueries << ',' << kTopK << ','
            << std::setprecision(10) << recall10 << ',' << recall100 << '\n';
    std::cout << "FINAL_RECALL method=" << method_name << " recall@10=" << recall10
              << " recall@100=" << recall100 << '\n' << std::flush;
}

template <typename Data, typename Queries>
void run_triad_final_n20_no_reassignment(const std::filesystem::path& data_path,
                                         const std::filesystem::path& query_path,
                                         const std::filesystem::path& truth_path,
                                         const std::filesystem::path& initial_index_path,
                                         const std::filesystem::path& output_root,
                                         const Data& data, const Queries& queries) {
    if (data.rows() != 10'000'000 || queries.rows() != kQueries)
        throw std::runtime_error("final-only Triad run expects shuffled MSTuring-10M and the configured 10K queries");
    if (!std::filesystem::is_regular_file(truth_path) ||
        !std::filesystem::is_regular_file(initial_index_path))
        throw std::runtime_error("cached exact stage truth and reusable initial index must exist");
    {
        const AdaptiveHierarchicalIvf initial = AdaptiveHierarchicalIvf::load(initial_index_path.string());
        const auto& cfg = initial.config();
        if (initial.size() != kInitialRows || initial.leaf_nlist() != kInitialNlist ||
            cfg.dimension != kDimension || cfg.lvq8_storage || cfg.lvq4_storage ||
            cfg.angular_metric || cfg.uint8_storage ||
            std::abs(cfg.prd_threshold - 0.05f) > 1e-6f ||
            std::abs(cfg.load_split_fraction - 2.20f) > 1e-6f ||
            std::abs(cfg.capacity_sqrt_n_split_multiplier - 1.05f) > 1e-6f)
            throw std::runtime_error("reused initial index does not match the shuffled MSTuring Triad baseline");
    }
    {
        std::ifstream truth(truth_path, std::ios::binary);
        char magic[8]{};
        std::uint32_t shape[3]{};
        truth.read(magic, sizeof(magic));
        truth.read(reinterpret_cast<char*>(shape), sizeof(shape));
        const std::uint64_t expected_bytes = 8 + 3 * sizeof(std::uint32_t) +
            static_cast<std::uint64_t>(kQueryOps) * kQueries * sizeof(Ids);
        if (!truth || std::string(magic, sizeof(magic)) != "MSTGTH01" ||
            shape[0] != kQueryOps || shape[1] != kQueries || shape[2] != kTopK ||
            std::filesystem::file_size(truth_path) != expected_bytes)
            throw std::runtime_error("cached exact stage truth must match [100,10000,100]");
    }

    ensure_empty_or_create(output_root);
    {
        std::ofstream manifest(output_root / "manifest.txt", std::ios::trunc);
        if (!manifest) throw std::runtime_error("cannot create final-only Triad manifest");
        manifest << "experiment=Triad on shuffled MSTuring-10M; insert all batches, then one final fixed-budget query\n"
                 << "dataset=" << data_path.string() << "\nqueries=" << query_path.string()
                 << "\nexact_stage_truth=" << truth_path.string()
                 << "\nrows=10000000; dimension=" << kDimension << "; metric=squared L2"
                 << "\nshuffle_seed=1234; initial_vectors=1000000; initial_nlist=1000"
                 << "\ninsert_batch=10000; insert_ops=900; queries=10000; topk=100"
                 << "\nquery_schedule=one final query stage after all 900 insertions; no intermediate queries"
                 << "\nquery_budget=floor(final active vectors/20)=500000 total distance computations per query"
                 << "\nTriad PRD threshold=0.075; short-list merge cutoff=0.2*sqrt(current active N)"
                 << "\nload split cutoff=2.2*T; capacity split=1.05*sqrt(N), active only while nlist<=sqrt(N)"
                 << "\nPRD criterion=GlobalFitOnly; PRD action=split directly; load merge+split enabled"
                 << "\nlocal split reassignment=disabled (radius 0); required split/merge vector routing retained"
                 << "\nglobal center refresh=all centers to stored partition means after each update batch"
                 << "\ninitial_index_reused=true; initial_index=" << initial_index_path.string()
                 << "\nLVQ=false; query early-stop=false; warmup=none; seed=" << kSeed
                 << "\nfinal_index_saved=false\n";
    }

    AdaptiveMethodConfig triad;
    triad.name = "triad_no_local_reassignment";
    triad.prd = true;
    triad.triad_policy = true;
    triad.query_stages_override = 1;
    triad.insert_ops_per_query_override = kInsertOps;
    triad.split_local_refinement_radius = 0;
    triad.split_local_refinement_iterations = 1;
    triad.override_split_local_refinement = true;
    triad.prd_threshold_override = 0.075f;
    triad.short_list_sqrt_n_fraction_override = 0.20f;
    run_adaptive_clean(output_root, data, queries, triad, initial_index_path);
    score_triad_final_fixed_budget(output_root / triad.name, truth_path, triad.name);
    std::ofstream complete(output_root / "COMPLETE.txt", std::ios::trunc);
    if (!complete) throw std::runtime_error("cannot write final-only Triad completion marker");
    complete << "status=complete\nmethod=Triad; local split reassignment disabled\n"
             << "insert_ops=900\nquery_stages=1\nqueries_per_stage=10000\nquery_budget=500000\n"
             << "initial_index_reused=" << initial_index_path.string()
             << "\nrecall=exact final-stage Recall@10 and Recall@100\nfinal_index_saved=false\n";
    complete.close();
    std::cout << "TRIAD_FINAL_N20_NO_REASSIGNMENT_COMPLETE output="
              << output_root.string() << '\n' << std::flush;
}

template <typename Data, typename Queries>
void run_triad_threshold_robustness_final_n20(
        const std::filesystem::path& data_path,
        const std::filesystem::path& query_path,
        const std::filesystem::path& truth_path,
        const std::filesystem::path& initial_index_path,
        const std::filesystem::path& output_root,
        const Data& data, const Queries& queries,
        std::size_t query_threads, std::size_t n_divisor,
        std::size_t capacity_cluster_count_gate) {
    if (data.rows() != 10'000'000 || queries.rows() < kQueries)
        throw std::runtime_error("Triad threshold robustness run expects shuffled MSTuring-10M and 10K queries");
    if (!std::filesystem::is_regular_file(truth_path) ||
        !std::filesystem::is_regular_file(initial_index_path))
        throw std::runtime_error("cached exact stage truth and reusable initial index must exist");
    if (query_threads == 0 || n_divisor == 0 || capacity_cluster_count_gate == 0)
        throw std::invalid_argument("query thread count, N divisor, and capacity K gate must be positive");

    {
        const AdaptiveHierarchicalIvf initial = AdaptiveHierarchicalIvf::load(initial_index_path.string());
        const auto& cfg = initial.config();
        if (initial.size() != kInitialRows || initial.leaf_nlist() != kInitialNlist ||
            cfg.dimension != kDimension || cfg.lvq8_storage || cfg.lvq4_storage ||
            cfg.angular_metric || cfg.uint8_storage ||
            std::abs(cfg.prd_threshold - 0.05f) > 1e-6f ||
            std::abs(cfg.load_split_fraction - 2.20f) > 1e-6f ||
            std::abs(cfg.capacity_sqrt_n_split_multiplier - 1.05f) > 1e-6f)
            throw std::runtime_error("reused initial index does not match the shuffled MSTuring Triad baseline");
    }
    {
        std::ifstream truth(truth_path, std::ios::binary);
        char magic[8]{};
        std::uint32_t shape[3]{};
        truth.read(magic, sizeof(magic));
        truth.read(reinterpret_cast<char*>(shape), sizeof(shape));
        const std::uint64_t expected_bytes = 8 + 3 * sizeof(std::uint32_t) +
            static_cast<std::uint64_t>(kQueryOps) * kQueries * sizeof(Ids);
        if (!truth || std::string(magic, sizeof(magic)) != "MSTGTH01" ||
            shape[0] != kQueryOps || shape[1] != kQueries || shape[2] != kTopK ||
            std::filesystem::file_size(truth_path) != expected_bytes)
            throw std::runtime_error("cached exact truth must match [100,10000,100]");
    }

    ensure_empty_or_create(output_root);
    {
        std::ofstream manifest(output_root / "manifest.txt", std::ios::trunc);
        if (!manifest) throw std::runtime_error("cannot create threshold robustness manifest");
        manifest << "experiment=Triad capacity/short-list threshold robustness on shuffled MSTuring-10M\n"
                 << "dataset=" << data_path.string() << "\nqueries=" << query_path.string()
                 << "\nexact_stage_truth=" << truth_path.string()
                 << "\nrows=10000000; dimension=" << kDimension << "; metric=squared L2"
                 << "\nshuffle_seed=1234; initial_vectors=1000000; initial_nlist=1000"
                 << "\ninsert_batch=10000; insert_ops=900; queries=10000; topk=100"
                 << "\nquery_schedule=one final query stage after all 900 insertions; no intermediate queries"
                 << "\nquery_budget=floor(final active vectors/20)=500000 total distance computations per query"
                 << "\nquery_threads=" << query_threads << " (parallel read-only query execution)"
                 << "\nPRD=0.075; load split cutoff=2.2*T"
                 << "\nshort-list merge cutoff=0.2*(N/" << n_divisor << "), replacing 0.2*sqrt(N)"
                 << "\ncapacity split cutoff=1.05*(N/" << n_divisor << "), replacing 1.05*sqrt(N)"
                 << "\ncapacity split cluster-count gate=live K<" << capacity_cluster_count_gate
                 << "\nPRD criterion=GlobalFitOnly; PRD action=split directly; load merge+split enabled"
                 << "\nlocal split reassignment=radius 50, one pass"
                 << "\nglobal center refresh=all centers to stored partition means after each update batch"
                 << "\ninitial_index_reused=true; initial_index=" << initial_index_path.string()
                 << "\nLVQ=false; query early-stop=false; APS=false; warmup=none; seed=" << kSeed
                 << "\nfinal_index_saved=false\n";
    }

    AdaptiveMethodConfig triad;
    triad.name = "triad_n" + std::to_string(n_divisor) + "_kgate" +
        std::to_string(capacity_cluster_count_gate) + "_thresholds";
    triad.prd = true;
    triad.triad_policy = true;
    triad.final_query_only = true;
    triad.skip_final_aps_diagnostic = true;
    triad.query_threads = query_threads;
    triad.split_local_refinement_radius = 50;
    triad.split_local_refinement_iterations = 1;
    triad.prd_threshold_override = 0.075f;
    triad.short_list_n_fraction_override = 0.20f / static_cast<float>(n_divisor);
    triad.capacity_n_divisor_override = static_cast<float>(n_divisor);
    triad.capacity_cluster_count_gate_override = capacity_cluster_count_gate;
    run_adaptive_clean(output_root, data, queries, triad, initial_index_path);
    score_triad_final_fixed_budget(output_root / triad.name, truth_path, triad.name);

    std::ofstream complete(output_root / "COMPLETE.txt", std::ios::trunc);
    if (!complete) throw std::runtime_error("cannot write threshold robustness completion marker");
    complete << "status=complete\nmethod=Triad; threshold robustness: sqrt(N) replaced by N/"
             << n_divisor << "\n"
             << "insert_ops=900\nquery_stages=1\nqueries_per_stage=10000\nquery_budget=500000\n"
             << "query_threads=" << query_threads << "\nshort_list_merge_cutoff=0.2*(N/"
             << n_divisor << ")\ncapacity_split_cutoff=1.05*(N/" << n_divisor
             << "); capacity_gate=live K<" << capacity_cluster_count_gate << "\n"
             << "APS=false; one fixed-budget N/20 query stage only\n"
             << "initial_index_reused=" << initial_index_path.string()
             << "\nrecall=exact final-stage Recall@10 and Recall@100\nfinal_index_saved=false\n";
    complete.close();
    std::cout << "TRIAD_THRESHOLD_ROBUSTNESS_COMPLETE output="
              << output_root.string() << '\n' << std::flush;
}
} // namespace

int run_msturing_triad_quake_lire_full(const std::filesystem::path& data_path,
                                       const std::filesystem::path& query_path,
                                       const std::filesystem::path& truth_path,
                                       const std::filesystem::path& initial_index_path,
                                       const std::filesystem::path& output_root,
                                       std::size_t lire_max_partition_size) {
    MappedFbin data(data_path), queries(query_path);
    if (data.rows() != 10'000'000 || queries.rows() < kQueries)
        throw std::runtime_error("expected shuffled 10M MSTuring vectors and at least 10K queries");
    if (!std::filesystem::is_regular_file(truth_path) ||
        !std::filesystem::is_regular_file(initial_index_path))
        throw std::runtime_error("cached stage truth and shared initial index must exist");
    if (lire_max_partition_size == 0)
        throw std::invalid_argument("LIRE max partition size must be positive");

    {
        std::ifstream truth(truth_path, std::ios::binary);
        char magic[8]{};
        std::uint32_t shape[3]{};
        truth.read(magic, sizeof(magic));
        truth.read(reinterpret_cast<char*>(shape), sizeof(shape));
        const std::uint64_t expected_bytes = 8 + 3 * sizeof(std::uint32_t) +
            static_cast<std::uint64_t>(kQueryOps) * kQueries * sizeof(Ids);
        if (!truth || std::string(magic, sizeof(magic)) != "MSTGTH01" ||
            shape[0] != kQueryOps || shape[1] != kQueries || shape[2] != kTopK ||
            std::filesystem::file_size(truth_path) != expected_bytes)
            throw std::runtime_error("stage truth must match the 100x10000x100 shuffled MSTuring workload");
    }
    {
        const AdaptiveHierarchicalIvf initial =
            AdaptiveHierarchicalIvf::load(initial_index_path.string());
        const auto& cfg = initial.config();
        if (initial.size() != kInitialRows || initial.leaf_nlist() != kInitialNlist ||
            cfg.dimension != kDimension || cfg.lvq8_storage || cfg.lvq4_storage ||
            cfg.angular_metric || cfg.uint8_storage ||
            std::abs(cfg.prd_threshold - 0.05f) > 1e-6f ||
            std::abs(cfg.load_split_fraction - 2.20f) > 1e-6f ||
            std::abs(cfg.capacity_sqrt_n_split_multiplier - 1.05f) > 1e-6f)
            throw std::runtime_error("shared initial index does not match current MSTuring settings");
    }
    ensure_empty_or_create(output_root);
    std::filesystem::copy_file(truth_path, output_root / "groundtruth_stage_top100.bin");
    {
        std::ofstream manifest(output_root / "manifest.txt", std::ios::trunc);
        if (!manifest) throw std::runtime_error("cannot create three-method MSTuring manifest");
        manifest << "experiment=Triad, Quake, and one-pass LIRE comparison on shuffled MSTuring-10M\n"
                 << "dataset=" << data_path.string() << "\nqueries=" << query_path.string()
                 << "\nrows=10000000; dimension=" << kDimension << "; metric=squared L2\n"
                 << "shuffle_seed=1234; initial_vectors=" << kInitialRows
                 << "; initial_nlist=" << kInitialNlist
                 << "\ninsert_batch=" << kInsertBatchRows << "; insert_batches=" << kInsertOps
                 << "\nquery_schedule=after every 9 insert batches; total query stages=" << kQueryOps
                 << "; queries per stage=" << kQueries
                 << "\nquery_policy=APS target=0.9; candidate_fraction=0.1; recompute_threshold=0.001; warmup=100 queries before first measured stage for each method\n"
                 << "shared_initial_index=" << initial_index_path.string()
                 << "\nTriad=PRD 0.05; load split and merge threshold 2.2T; capacity split 1.05*sqrt(N) while nlist<=sqrt(N); local reassignment radius 50, one iteration; global center move after each update; historical residual reference retained after reassignment\n"
                 << "Quake=split tau 250ns; delete tau 250ns; alpha 0.9; query window 1000; minimum partition 32; reassignment radius 50, one iteration\n"
                 << "LIRE=Quake source size-threshold policy; maximum partition size=" << lire_max_partition_size
                 << "; minimum partition size 32; short partition objects retired after routing their vectors to nearest surviving centroids; after splits, radius 50 including seed and one assignment pass\n"
                 << "LVQ=false; early_stop=false; exact stage truth reused\n";
    }

    AdaptiveMethodConfig triad;
    triad.name = "triad";
    triad.prd = true;
    triad.triad_policy = true;
    triad.aps_queries = true;
    triad.split_local_refinement_radius = 50;
    triad.split_local_refinement_iterations = 1;

    AdaptiveMethodConfig quake;
    quake.name = "quake";
    quake.quake = true;
    quake.aps_queries = true;
    quake.quake_split_threshold_ns = kQuakeSplitThresholdNs;
    quake.quake_delete_threshold_ns = kQuakeDeleteThresholdNs;

    AdaptiveMethodConfig lire;
    lire.name = "quake_lire";
    lire.quake_lire = true;
    lire.aps_queries = true;
    lire.lire_max_partition_size = lire_max_partition_size;

    for (const AdaptiveMethodConfig* method : {&triad, &quake, &lire}) {
        std::cout << "FULL_COMPARE_START method=" << method->name << '\n' << std::flush;
        run_adaptive_clean(output_root, data, queries, *method, initial_index_path);
        std::cout << "FULL_COMPARE_METHOD_DONE method=" << method->name << '\n' << std::flush;
    }

    score_existing_outputs(output_root, output_root, data, queries);
    score_existing_recall_at_10(output_root, output_root / "groundtruth_stage_top100.bin",
                                output_root / "recall_at_10");
    std::ofstream complete(output_root / "COMPLETE.txt", std::ios::trunc);
    if (!complete) throw std::runtime_error("cannot write three-method completion marker");
    complete << "status=complete\nmethods=triad,quake,quake_lire\ninsert_batches_per_method="
             << kInsertOps << "\nquery_stages_per_method=" << kQueryOps
             << "\nqueries_per_stage=" << kQueries
             << "\ncommon_initial_index_reused=true\nLIRE_refinement=one assignment pass, radius 50, max partition size="
             << lire_max_partition_size << "\nquery_policy=APS target 0.9; candidate_fraction=0.1; recompute_threshold=0.001; warmup 100 queries per method\n";
    std::cout << "FULL_COMPARE_COMPLETE output=" << output_root.string() << '\n' << std::flush;
    return 0;
}

void run_deep10m_final_only(const std::filesystem::path& data_path,
                            const std::filesystem::path& query_path,
                            const std::filesystem::path& truth_path,
                            const std::filesystem::path& output_root,
                            const MappedFbin& data, const MappedFbin& queries) {
    if (data.rows() != 10'000'000 || queries.rows() != 10'000 ||
        kDimension != 96 || kTopK != 100 || kInsertOps != 900 ||
        kQueryOps != 1 || kInitialRows != 1'000'000 || kInitialNlist != 1'000)
        throw std::runtime_error("Deep10M final-only runner was built with an unexpected profile");
    if (!std::filesystem::is_regular_file(truth_path) ||
        std::filesystem::file_size(truth_path) != 8ull + kQueries * 100ull * sizeof(std::int32_t))
        throw std::runtime_error("Deep10M exact truth must be the official 10000 x 100 ibin");

    std::vector<Ids> exact(kQueries);
    {
        std::ifstream input(truth_path, std::ios::binary);
        std::int32_t rows = 0, width = 0;
        input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        input.read(reinterpret_cast<char*>(&width), sizeof(width));
        if (!input || rows != static_cast<std::int32_t>(kQueries) || width != 100)
            throw std::runtime_error("Deep10M truth header must be [10000,100]");
        std::array<std::int32_t, 100> row{};
        for (std::size_t qi = 0; qi < kQueries; ++qi) {
            input.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(sizeof(row)));
            if (!input) throw std::runtime_error("truncated Deep10M exact truth");
            for (std::size_t rank = 0; rank < kTopK; ++rank) {
                if (row[rank] < 0 || static_cast<std::size_t>(row[rank]) >= data.rows())
                    throw std::runtime_error("Deep10M truth contains an ID outside the 10M base");
                exact[qi][rank] = static_cast<std::uint32_t>(row[rank]);
            }
        }
    }

    ensure_empty_or_create(output_root);
    std::filesystem::copy_file(truth_path, output_root / "groundtruth.public.10K.ibin");
    const auto initial_index_path = output_root / "initial_1m_nlist1000.index.bin";
    {
        AdaptiveHierarchicalConfig cfg;
        cfg.dimension = kDimension;
        cfg.initial_leaf_nlist = kInitialNlist;
        cfg.kmeans_iterations = kKmeansIterations;
        cfg.seed = kSeed;
        cfg.lvq8_storage = false;
        cfg.lvq4_storage = false;
        cfg.angular_metric = false;
        cfg.uint8_storage = false;
        AdaptiveHierarchicalIvf initial(cfg);
        const auto start = std::chrono::steady_clock::now();
        initial.train(data.row(0), kInitialRows);
        initial.save(initial_index_path.string());
        const double build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        std::ofstream build(output_root / "shared_initial_build.txt", std::ios::trunc);
        if (!build) throw std::runtime_error("cannot write shared initial-build record");
        build << "initial_vectors=" << kInitialRows << "\ninitial_nlist=" << kInitialNlist
              << "\ndimension=" << kDimension << "\nkmeans_iterations=" << kKmeansIterations
              << "\nseed=" << kSeed << "\nmetric=squared L2\ninitial_build_and_save_seconds="
              << std::setprecision(10) << build_s << "\nindex=" << initial_index_path.string() << '\n';
        std::cout << "DEEP10M_SHARED_INITIAL_INDEX_READY seconds=" << build_s
                  << " path=" << initial_index_path.string() << '\n' << std::flush;
    }
    {
        std::ofstream manifest(output_root / "manifest.txt", std::ios::trunc);
        if (!manifest) throw std::runtime_error("cannot create Deep10M experiment manifest");
        manifest << "experiment=Deep10M_2 Triad, Quake, and LIRE; final-only fixed-budget evaluation\n"
                 << "base=" << data_path.string() << "\nqueries=" << query_path.string()
                 << "\nexact_truth=" << truth_path.string()
                 << "\nbase=10000000x96 FP32; query_metric=squared L2; no normalization\n"
                 << "initial_vectors=1000000; initial_nlist=1000; inserts=900 batches x 10000, original file order\n"
                 << "shared_initial_index=" << initial_index_path.string() << "\n"
                 << "query_schedule=one final query stage after all insertions; all 10000 queries\n"
                 << "query_budget=500000 total distance computations/query (N/20), centroid distances included\n"
                 << "topk=100; LVQ=false; query_early_stop=false; APS=false; warmup=none\n"
                 << "Triad: PRD=0.075; load split/merge=2.2T and <0.2*sqrt(N); capacity split=1.05*sqrt(N) while nlist<=sqrt(N); direct PRD split; no local split reassignment; global center refresh to stored means after each insert batch\n"
                 << "Quake: split tau=250ns; delete tau=250ns; alpha=0.9; query hit window=1000; minimum partition=32; radius=50; one refinement iteration. No queries occur before the final stage, therefore no online hit feedback is available for maintenance.\n"
                 << "LIRE: Quake source size-threshold policy; maximum partition size=2000; minimum=32; radius=50; one assignment pass after splits; maintenance after each insert batch\n"
                 << "truth_source=provided Deep10M_2 official exact 10000x100 ibin, header and ID range validated\n"
                 << "final_indexes_saved=false\n";
    }

    std::array<AdaptiveMethodConfig, 3> methods{};
    methods[0].name = "triad";
    methods[0].prd = true;
    methods[0].triad_policy = true;
    methods[0].query_stages_override = 1;
    methods[0].insert_ops_per_query_override = kInsertOps;
    methods[0].override_split_local_refinement = true;
    methods[0].split_local_refinement_radius = 0;
    methods[0].prd_threshold_override = 0.075f;
    methods[0].short_list_sqrt_n_fraction_override = 0.20f;
    methods[1].name = "quake";
    methods[1].quake = true;
    methods[1].query_stages_override = 1;
    methods[1].insert_ops_per_query_override = kInsertOps;
    methods[1].quake_split_threshold_ns = 250.0;
    methods[1].quake_delete_threshold_ns = 250.0;
    methods[2].name = "quake_lire";
    methods[2].quake_lire = true;
    methods[2].query_stages_override = 1;
    methods[2].insert_ops_per_query_override = kInsertOps;
    methods[2].lire_max_partition_size = 2'000;

    std::ofstream summary(output_root / "final_recall_summary.csv", std::ios::trunc);
    if (!summary) throw std::runtime_error("cannot create final recall summary");
    summary << "method,active_vectors,query_count,topk,budget_per_query,recall_at_10,recall_at_100,final_nlist\n";
    for (const auto& method : methods) {
        std::cout << "DEEP10M_METHOD_START method=" << method.name << '\n' << std::flush;
        run_adaptive_clean(output_root, data, queries, method, initial_index_path);
        std::ifstream result(output_root / method.name / "results.bin", std::ios::binary);
        char magic[8]{};
        std::uint32_t shape[3]{};
        result.read(magic, sizeof(magic));
        result.read(reinterpret_cast<char*>(shape), sizeof(shape));
        if (!result || std::string(magic, sizeof(magic)) != "MSTIHR01" ||
            shape[0] != 1 || shape[1] != kQueries || shape[2] != kTopK)
            throw std::runtime_error("final result shape mismatch for method " + method.name);
        std::vector<Ids> found(kQueries);
        result.read(reinterpret_cast<char*>(found.data()),
                    static_cast<std::streamsize>(found.size() * sizeof(Ids)));
        if (!result) throw std::runtime_error("truncated final query results for " + method.name);
        std::uint64_t hits10 = 0, hits100 = 0;
        for (std::size_t qi = 0; qi < kQueries; ++qi) {
            for (std::size_t rank = 0; rank < 10; ++rank) {
                if (std::find(exact[qi].begin(), exact[qi].begin() + 10, found[qi][rank]) != exact[qi].begin() + 10)
                    ++hits10;
                if (std::find(exact[qi].begin(), exact[qi].end(), found[qi][rank]) != exact[qi].end())
                    ++hits100;
            }
            for (std::size_t rank = 10; rank < kTopK; ++rank)
                if (std::find(exact[qi].begin(), exact[qi].end(), found[qi][rank]) != exact[qi].end())
                    ++hits100;
        }
        const double recall10 = static_cast<double>(hits10) / (kQueries * 10.0);
        const double recall100 = static_cast<double>(hits100) / (kQueries * 100.0);
        std::size_t final_nlist = 0;
        {
            std::ifstream stages(output_root / method.name / "stage_summary.csv");
            std::string line;
            std::getline(stages, line);
            if (std::getline(stages, line)) {
                std::stringstream fields(line);
                std::string field;
                std::vector<std::string> row;
                while (std::getline(fields, field, ',')) row.push_back(field);
                if (row.size() >= 8) final_nlist = static_cast<std::size_t>(std::stoull(row[7]));
            }
        }
        std::ofstream method_recall(output_root / method.name / "final_recall.csv", std::ios::trunc);
        if (!method_recall) throw std::runtime_error("cannot write final recall for " + method.name);
        method_recall << "method,active_vectors,query_count,topk,budget_per_query,recall_at_10,recall_at_100,final_nlist\n"
                      << method.name << ",10000000," << kQueries << ',' << kTopK << ",500000,"
                      << std::setprecision(10) << recall10 << ',' << recall100 << ',' << final_nlist << '\n';
        summary << method.name << ",10000000," << kQueries << ',' << kTopK << ",500000,"
                << std::setprecision(10) << recall10 << ',' << recall100 << ',' << final_nlist << '\n';
        summary.flush();
        std::cout << "DEEP10M_METHOD_COMPLETE method=" << method.name
                  << " nlist=" << final_nlist << " recall@10=" << recall10
                  << " recall@100=" << recall100 << '\n' << std::flush;
    }
    std::ofstream complete(output_root / "COMPLETE.txt", std::ios::trunc);
    if (!complete) throw std::runtime_error("cannot write Deep10M completion marker");
    complete << "status=complete\nmethods=triad,quake,quake_lire\ninsert_batches_per_method=900\n"
             << "query_stages_per_method=1\nqueries_per_stage=10000\nquery_budget_per_query=500000\n"
             << "truth=official exact Deep10M_2 10000x100\nfinal_indexes_saved=false\n";
    std::cout << "DEEP10M_THREE_METHODS_COMPLETE output=" << output_root.string() << '\n' << std::flush;
}

void run_deep10m_lire_final_only(const std::filesystem::path& data_path,
                                 const std::filesystem::path& query_path,
                                 const std::filesystem::path& truth_path,
                                 const std::filesystem::path& initial_index_path,
                                 const std::filesystem::path& output_root,
                                 std::size_t max_partition_size,
                                 std::size_t budget_divisor,
                                 std::size_t query_threads,
                                 const MappedFbin& data,
                                 const MappedFbin& queries) {
    if (data.rows() != 10'000'000 || queries.rows() != 10'000 ||
        kDimension != 96 || kTopK != 100 || kInsertOps != 900 ||
        kInitialRows != 1'000'000 || kInitialNlist != 1'000)
        throw std::runtime_error("Deep10M LIRE final-only runner was built with an unexpected profile");
    if (max_partition_size == 0 || budget_divisor == 0 || query_threads == 0 ||
        !std::filesystem::is_regular_file(initial_index_path))
        throw std::invalid_argument("max partition size, budget divisor, query threads, and initial index must be valid");
    if (!std::filesystem::is_regular_file(truth_path) ||
        std::filesystem::file_size(truth_path) != 8ull + kQueries * kTopK * sizeof(std::int32_t))
        throw std::runtime_error("Deep10M exact truth must be the official 10000 x 100 ibin");

    std::vector<Ids> exact(kQueries);
    {
        std::ifstream input(truth_path, std::ios::binary);
        std::int32_t rows = 0, width = 0;
        input.read(reinterpret_cast<char*>(&rows), sizeof(rows));
        input.read(reinterpret_cast<char*>(&width), sizeof(width));
        if (!input || rows != static_cast<std::int32_t>(kQueries) ||
            width != static_cast<std::int32_t>(kTopK))
            throw std::runtime_error("Deep10M truth header must be [10000,100]");
        std::array<std::int32_t, kTopK> row{};
        for (std::size_t qi = 0; qi < kQueries; ++qi) {
            input.read(reinterpret_cast<char*>(row.data()), static_cast<std::streamsize>(sizeof(row)));
            if (!input) throw std::runtime_error("truncated Deep10M exact truth");
            for (std::size_t rank = 0; rank < kTopK; ++rank) {
                if (row[rank] < 0 || static_cast<std::size_t>(row[rank]) >= data.rows())
                    throw std::runtime_error("Deep10M truth contains an ID outside the 10M base");
                exact[qi][rank] = static_cast<std::uint32_t>(row[rank]);
            }
        }
    }

    AdaptiveHierarchicalIvf initial = AdaptiveHierarchicalIvf::load(initial_index_path.string());
    if (initial.size() != kInitialRows || initial.leaf_nlist() != kInitialNlist ||
        initial.config().dimension != kDimension || initial.config().lvq8_storage ||
        initial.config().lvq4_storage || initial.config().angular_metric ||
        initial.config().uint8_storage)
        throw std::runtime_error("reused initial index does not match Deep10M 1M/1000 FP32 squared-L2 configuration");

    ensure_empty_or_create(output_root);
    {
        std::ofstream manifest(output_root / "manifest.txt", std::ios::trunc);
        if (!manifest) throw std::runtime_error("cannot create Deep10M LIRE final-only manifest");
        manifest << "experiment=Deep10M_2 LIRE size-policy; insert all, then query once\n"
                 << "base=" << data_path.string() << "\nqueries=" << query_path.string()
                 << "\nexact_truth=" << truth_path.string()
                 << "\nbase=10000000x96 FP32; original file order; squared L2; no normalization\n"
                 << "initial_vectors=1000000; initial_nlist=1000; inserts=900 batches x 10000\n"
                 << "shared_initial_index=" << initial_index_path.string() << "\n"
                 << "query_schedule=one final query stage after all insertions; all 10000 queries\n"
                 << "query_budget=floor(final indexed vectors/" << budget_divisor
                 << "), including centroid and vector distances\n"
                 << "query_threads=" << query_threads
                 << "; topk=100; LVQ=false; early_stop=false; APS=false; seed=" << kSeed << '\n'
                 << "LIRE=max_partition_size=" << max_partition_size
                 << "; min_partition_size=" << kQuakeMinPartitionSize
                 << "; local refinement radius=" << kQuakeRefinementRadius
                 << "; one assignment pass after all splits in each insertion-batch maintenance\n"
                 << "maintenance=once after every insertion batch; no maintenance after final query\n"
                 << "final_index_saved=false\n";
    }

    AdaptiveMethodConfig lire;
    lire.name = "lire";
    lire.quake_lire = true;
    lire.initial_leaf_nlist = kInitialNlist;
    lire.lire_max_partition_size = max_partition_size;
    lire.final_query_only = true;
    lire.skip_final_aps_diagnostic = true;
    lire.query_budget_divisor_override = budget_divisor;
    lire.query_threads = query_threads;
    run_adaptive_clean(output_root, data, queries, lire, initial_index_path);

    const auto result_path = output_root / "lire" / "results.bin";
    std::ifstream result(result_path, std::ios::binary);
    char magic[8]{};
    std::uint32_t shape[3]{};
    result.read(magic, sizeof(magic));
    result.read(reinterpret_cast<char*>(shape), sizeof(shape));
    if (!result || std::string(magic, sizeof(magic)) != "MSTIHR01" ||
        shape[0] != 1 || shape[1] != kQueries || shape[2] != kTopK)
        throw std::runtime_error("Deep10M LIRE final-query result shape mismatch");
    std::vector<Ids> found(kQueries);
    result.read(reinterpret_cast<char*>(found.data()),
                static_cast<std::streamsize>(found.size() * sizeof(Ids)));
    if (!result) throw std::runtime_error("truncated Deep10M LIRE final-query results");

    std::uint64_t hits10 = 0, hits100 = 0;
    for (std::size_t qi = 0; qi < kQueries; ++qi) {
        for (std::size_t rank = 0; rank < 10; ++rank) {
            if (std::find(exact[qi].begin(), exact[qi].begin() + 10, found[qi][rank]) != exact[qi].begin() + 10)
                ++hits10;
            if (std::find(exact[qi].begin(), exact[qi].end(), found[qi][rank]) != exact[qi].end())
                ++hits100;
        }
        for (std::size_t rank = 10; rank < kTopK; ++rank)
            if (std::find(exact[qi].begin(), exact[qi].end(), found[qi][rank]) != exact[qi].end())
                ++hits100;
    }
    const double recall10 = static_cast<double>(hits10) / (kQueries * 10.0);
    const double recall100 = static_cast<double>(hits100) / (kQueries * 100.0);
    std::size_t final_nlist = 0;
    {
        std::ifstream stages(output_root / "lire" / "stage_summary.csv");
        std::string header, values;
        std::getline(stages, header);
        std::getline(stages, values);
        std::stringstream header_stream(header), value_stream(values);
        std::vector<std::string> headers, fields;
        std::string field;
        while (std::getline(header_stream, field, ',')) headers.push_back(field);
        while (std::getline(value_stream, field, ',')) fields.push_back(field);
        const auto nlist_column = std::find(headers.begin(), headers.end(), "nlist");
        if (nlist_column != headers.end()) {
            const std::size_t column = static_cast<std::size_t>(nlist_column - headers.begin());
            if (column < fields.size()) final_nlist = static_cast<std::size_t>(std::stoull(fields[column]));
        }
    }
    {
        std::ofstream recall(output_root / "final_recall.csv", std::ios::trunc);
        if (!recall) throw std::runtime_error("cannot write Deep10M LIRE final recall summary");
        recall << "method,active_vectors,query_count,topk,budget_per_query,recall_at_10,recall_at_100,final_nlist\n"
               << "lire,10000000," << kQueries << ',' << kTopK << ','
               << (10'000'000 / budget_divisor) << ',' << std::setprecision(10)
               << recall10 << ',' << recall100 << ',' << final_nlist << '\n';
    }
    {
        std::ofstream complete(output_root / "COMPLETE.txt", std::ios::trunc);
        if (!complete) throw std::runtime_error("cannot write Deep10M LIRE completion marker");
        complete << "status=complete\nmethod=LIRE\nmax_partition_size=" << max_partition_size
                 << "\ninsert_batches=900\nquery_stages=1\nqueries=10000\nbudget_divisor="
                 << budget_divisor << "\nquery_threads=" << query_threads
                 << "\nfinal_nlist=" << final_nlist << "\nrecall_at_10=" << std::setprecision(10)
                 << recall10 << "\nrecall_at_100=" << recall100 << "\nfinal_index_saved=false\n";
    }
    std::cout << "DEEP10M_LIRE_FINAL_COMPLETE nlist=" << final_nlist
              << " recall@10=" << recall10 << " recall@100=" << recall100
              << " output=" << output_root.string() << '\n' << std::flush;
}

template <typename Data, typename Queries>
void run_lire_full_staged(const std::filesystem::path& data_path,
                          const std::filesystem::path& query_path,
                          const std::filesystem::path& truth_path,
                          const std::filesystem::path& initial_index_path,
                          const std::filesystem::path& output_root,
                          std::size_t max_partition_size,
                          std::size_t budget_divisor,
                          std::size_t query_threads,
                          const Data& data,
                          const Queries& queries) {
    if (data.rows() != 10'000'000 || queries.rows() != kQueries ||
        kInitialRows != 1'000'000 || kInitialNlist != 1'000 ||
        kInsertBatchRows != 10'000 || kInsertOps != 900 || kQueryOps != 100 ||
        kInsertOpsPerQuery != 9 || kTopK != 100)
        throw std::runtime_error("LIRE full staged runner was built with an unexpected 10M profile");
    if (max_partition_size == 0 || budget_divisor == 0 || query_threads == 0 ||
        !std::filesystem::is_regular_file(truth_path) ||
        !std::filesystem::is_regular_file(initial_index_path))
        throw std::invalid_argument("LIRE full staged run requires valid cap, budget, threads, truth, and initial index");
    constexpr std::uint64_t truth_header = 8 + 3 * sizeof(std::uint32_t);
    const std::uint64_t truth_bytes = truth_header +
        static_cast<std::uint64_t>(kQueryOps) * kQueries * kTopK * sizeof(std::int32_t);
    if (std::filesystem::file_size(truth_path) != truth_bytes)
        throw std::runtime_error("stage truth file size does not match this 100x10000x100 profile");
    {
        std::ifstream truth(truth_path, std::ios::binary);
        char magic[8]{};
        std::uint32_t shape[3]{};
        truth.read(magic, sizeof(magic));
        truth.read(reinterpret_cast<char*>(shape), sizeof(shape));
        if (!truth || std::string(magic, sizeof(magic)) != "MSTGTH01" ||
            shape[0] != kQueryOps || shape[1] != kQueries || shape[2] != kTopK)
            throw std::runtime_error("expected exact stage truth MSTGTH01 [100,10000,100]");
    }
    AdaptiveHierarchicalIvf initial = AdaptiveHierarchicalIvf::load(initial_index_path.string());
    if (initial.size() != kInitialRows || initial.leaf_nlist() != kInitialNlist ||
        initial.config().dimension != kDimension || initial.config().lvq8_storage ||
        initial.config().lvq4_storage || initial.config().angular_metric ||
        initial.config().uint8_storage != uses_u8_input_v<Data>)
        throw std::runtime_error("reused initial index does not match this dataset's 1M/1000 exact-vector L2 profile");

    ensure_empty_or_create(output_root);
    {
        std::ofstream manifest(output_root / "manifest.txt", std::ios::trunc);
        if (!manifest) throw std::runtime_error("cannot create full staged LIRE manifest");
        manifest << "experiment=10M LIRE full insert-heavy staged workload\n"
                 << "dataset=" << data_path.string() << "\nqueries=" << query_path.string()
                 << "\nstage_truth=" << truth_path.string()
                 << "\nrows=10000000; dimension=" << kDimension << "; metric=" << kMetricName
                 << "; normalization=none\n"
                 << "initial_vectors=1000000; initial_nlist=1000; seed=" << kSeed << '\n'
                 << "inserts=900 batches x 10000; query after each 9 batches; 100 query stages\n"
                 << "queries_per_stage=" << kQueries << "; topk=100; exact prefix truth\n"
                 << "query_budget=floor(current indexed vectors/" << budget_divisor
                 << "), centroid and vector distances included\n"
                 << "query_threads=" << query_threads
                 << "; APS=false; early_stop=false; LVQ=false; warmup=none\n"
                 << "shared_initial_index=" << initial_index_path.string() << '\n'
                 << "LIRE=max_partition_size=" << max_partition_size
                 << "; min_partition_size=32; local refinement radius=50; one assignment pass after all splits in each insert-batch maintenance\n"
                 << "maintenance=once after each insert batch; short partition vectors are reassigned, not deleted; no query-time maintenance\n"
                 << "final_index_saved=false\n";
    }
    AdaptiveMethodConfig lire;
    lire.name = "lire";
    lire.quake_lire = true;
    lire.initial_leaf_nlist = kInitialNlist;
    lire.lire_max_partition_size = max_partition_size;
    lire.query_budget_divisor_override = budget_divisor;
    lire.query_threads = query_threads;
    run_adaptive_clean(output_root, data, queries, lire, initial_index_path);
    score_existing_outputs(output_root, output_root, data, queries, truth_path);
    score_existing_recall_at_10(output_root, truth_path, output_root / "recall_at_10");
    {
        std::ofstream complete(output_root / "COMPLETE.txt", std::ios::trunc);
        if (!complete) throw std::runtime_error("cannot write full staged LIRE completion marker");
        complete << "status=complete\nmethod=LIRE\nmax_partition_size=" << max_partition_size
                 << "\ninsert_batches=900\nquery_stages=100\nqueries_per_stage=" << kQueries
                 << "\nbudget_divisor=" << budget_divisor << "\nquery_threads=" << query_threads
                 << "\nfinal_index_saved=false\nrecall=exact posthoc Recall@10 and Recall@100 for all stages\n";
    }
    std::cout << "LIRE_FULL_STAGED_COMPLETE dataset=" << data_path.string()
              << " output=" << output_root.string() << '\n' << std::flush;
}

void run_deep10m_quake_staged(const std::filesystem::path& data_path,
                              const std::filesystem::path& query_path,
                              const std::filesystem::path& stage_truth_path,
                              const std::filesystem::path& official_truth_path,
                              const std::filesystem::path& initial_index_path,
                              const std::filesystem::path& output_root,
                              const MappedFbin& data, const MappedFbin& queries,
                              bool save_final_index = true,
                              std::size_t budget_divisor = 100,
                              bool build_initial_if_missing = false) {
    if (data.rows() != 10'000'000 || queries.rows() != kQueries ||
        kDimension != 96 || kTopK != 100 || kInsertOps != 900 ||
        kQueryOps != 100 || kInsertOpsPerQuery != 9 ||
        kInitialRows != 1'000'000 || kInitialNlist != 1'000)
        throw std::runtime_error("Deep10M Quake runner was built with an unexpected profile");
    if (!budget_divisor || !std::filesystem::is_regular_file(stage_truth_path) ||
        !std::filesystem::is_regular_file(official_truth_path))
        throw std::runtime_error("Deep10M Quake run requires valid truth files and a nonzero budget divisor");

    const std::uint64_t truth_header_bytes = 8 + 3 * sizeof(std::uint32_t);
    const std::uint64_t truth_row_bytes = static_cast<std::uint64_t>(kQueries) * sizeof(Ids);
    const std::uint64_t expected_stage_truth_bytes = truth_header_bytes +
        static_cast<std::uint64_t>(kQueryOps) * truth_row_bytes;
    {
        std::ifstream truth(stage_truth_path, std::ios::binary);
        char magic[8]{};
        std::uint32_t shape[3]{};
        truth.read(magic, sizeof(magic));
        truth.read(reinterpret_cast<char*>(shape), sizeof(shape));
        if (!truth || std::string(magic, sizeof(magic)) != "MSTGTH01" ||
            shape[0] != kQueryOps || shape[1] != kQueries || shape[2] != kTopK ||
            std::filesystem::file_size(stage_truth_path) != expected_stage_truth_bytes)
            throw std::runtime_error("Deep10M Quake stage truth must be MSTGTH01 [100,10000,100]");
    }
    {
        if (std::filesystem::file_size(official_truth_path) !=
            8ull + static_cast<std::uint64_t>(kQueries) * kTopK * sizeof(std::int32_t))
            throw std::runtime_error("official Deep10M truth must be 10000x100 with a two-int header");
        std::ifstream official(official_truth_path, std::ios::binary);
        std::int32_t official_rows = 0, official_width = 0;
        official.read(reinterpret_cast<char*>(&official_rows), sizeof(official_rows));
        official.read(reinterpret_cast<char*>(&official_width), sizeof(official_width));
        if (!official || official_rows != static_cast<std::int32_t>(kQueries) ||
            official_width != static_cast<std::int32_t>(kTopK))
            throw std::runtime_error("official Deep10M truth header must be [10000,100]");
        std::ifstream staged(stage_truth_path, std::ios::binary);
        staged.seekg(static_cast<std::streamoff>(truth_header_bytes +
            static_cast<std::uint64_t>(kQueryOps - 1) * truth_row_bytes));
        for (std::size_t qi = 0; qi < kQueries; ++qi) {
            std::array<std::int32_t, 100> official_row{};
            Ids staged_row{};
            official.read(reinterpret_cast<char*>(official_row.data()), sizeof(official_row));
            staged.read(reinterpret_cast<char*>(staged_row.data()), sizeof(staged_row));
            if (!official || !staged) throw std::runtime_error("truncated Deep10M final-stage truth");
            auto official_set = official_row;
            auto staged_set = staged_row;
            for (const std::int32_t id : official_row)
                if (id < 0) throw std::runtime_error("official Deep10M truth contains a negative ID");
            std::sort(official_set.begin(), official_set.end());
            std::sort(staged_set.begin(), staged_set.end());
            for (std::size_t rank = 0; rank < kTopK; ++rank)
                if (static_cast<std::uint32_t>(official_set[rank]) != staged_set[rank])
                    throw std::runtime_error("Deep10M final-stage truth neighbor set disagrees with official ground truth");
        }
    }
    ensure_empty_or_create(output_root);
    bool initial_index_created_here = false;
    if (std::filesystem::is_regular_file(initial_index_path)) {
        const AdaptiveHierarchicalIvf initial = AdaptiveHierarchicalIvf::load(initial_index_path.string());
        const auto& cfg = initial.config();
        if (initial.size() != kInitialRows || initial.leaf_nlist() != kInitialNlist ||
            cfg.dimension != kDimension || cfg.lvq8_storage || cfg.lvq4_storage ||
            cfg.angular_metric || cfg.uint8_storage)
            throw std::runtime_error("reused initial index does not match Deep10M 1M/1000 FP32 L2 configuration");
    } else if (build_initial_if_missing) {
        AdaptiveHierarchicalConfig cfg;
        cfg.dimension = kDimension;
        cfg.initial_leaf_nlist = kInitialNlist;
        cfg.kmeans_iterations = kKmeansIterations;
        cfg.seed = kSeed;
        cfg.lvq8_storage = false;
        cfg.lvq4_storage = false;
        cfg.angular_metric = false;
        cfg.uint8_storage = false;
        AdaptiveHierarchicalIvf initial(cfg);
        const auto build_start = std::chrono::steady_clock::now();
        initial.train(data.row(0), kInitialRows);
        initial.save(initial_index_path.string());
        const double build_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - build_start).count();
        std::ofstream build(output_root / "shared_initial_build.txt", std::ios::trunc);
        if (!build) throw std::runtime_error("cannot write shared initial-build record");
        build << "initial_vectors=" << kInitialRows << "\ninitial_nlist=" << kInitialNlist
              << "\ndimension=" << kDimension << "\nkmeans_iterations=" << kKmeansIterations
              << "\nseed=" << kSeed << "\nmetric=squared L2\ninitial_build_and_save_seconds="
              << std::setprecision(10) << build_s << "\nindex=" << initial_index_path.string() << '\n';
        initial_index_created_here = true;
        std::cout << "DEEP10M_SHARED_INITIAL_INDEX_READY seconds=" << build_s
                  << " path=" << initial_index_path.string() << '\n' << std::flush;
    } else {
        throw std::runtime_error("reused initial index is missing: " + initial_index_path.string());
    }

    std::filesystem::copy_file(stage_truth_path, output_root / "groundtruth_stage_top100.bin");
    std::filesystem::copy_file(official_truth_path, output_root / "groundtruth.public.10K.ibin");
    const auto final_index_path = save_final_index
        ? output_root / "quake_final.index.bin" : std::filesystem::path{};
    {
        std::ofstream manifest(output_root / "manifest.txt", std::ios::trunc);
        if (!manifest) throw std::runtime_error("cannot write Deep10M Quake manifest");
        manifest << "experiment=Deep10M_2 standalone Quake insert-heavy workload; 100 query stages\n"
                 << "base=" << data_path.string() << "\nqueries=" << query_path.string()
                 << "\nstage_truth=" << stage_truth_path.string()
                 << "\nofficial_final_truth=" << official_truth_path.string()
                 << "\nbase=10000000x96 FP32; squared L2; original file order; no normalization\n"
                 << "initial_vectors=1000000; initial_nlist=1000; inserts=900 batches x 10000\n"
                 << "query_schedule=100 stages, each after 9 insert batches; 10000 queries/stage\n"
                 << "query_budget=floor(current_active_vectors/" << budget_divisor
                 << "), centroid and vector distances included\n"
                 << "query_threads=16; topk=100; LVQ=false; early_stop=false; APS=false\n"
                 << "seed=1234; initial_index_reused=" << initial_index_path.string() << '\n'
                 << "Quake=split_tau=250ns; delete_tau=250ns; alpha=0.9; hit_window=1000; min_partition=32; local_refinement_radius=50; one iteration\n"
                 << "final_index_output=" << (save_final_index ? final_index_path.string() : "none")
                 << (save_final_index ? " (will be marked read-only after the run)\n" : "\n")
                 << "truth=exact brute-force prefix top-100; final prefix cross-checked with official Deep10M_2 truth\n";
    }

    AdaptiveMethodConfig quake;
    quake.name = "quake";
    quake.quake = true;
    quake.quake_split_threshold_ns = 250.0;
    quake.quake_delete_threshold_ns = 250.0;
    quake.query_budget_divisor_override = budget_divisor;
    quake.query_threads = 16;
    std::cout << "DEEP10M_QUAKE_STAGED_START budget=N/" << budget_divisor << " threads=16 final_index="
              << (save_final_index ? final_index_path.string() : "none") << '\n' << std::flush;
    run_adaptive_clean(output_root, data, queries, quake, initial_index_path,
                       save_final_index ? final_index_path : std::filesystem::path{});
    score_existing_outputs(output_root, output_root, data, queries);
    score_existing_recall_at_10(output_root, output_root / "groundtruth_stage_top100.bin",
                                output_root / "recall_at_10");
    if (initial_index_created_here) {
        std::error_code remove_error;
        std::filesystem::remove(initial_index_path, remove_error);
        if (remove_error)
            throw std::runtime_error("cannot remove temporary shared initial index: " + remove_error.message());
    }
    std::ofstream complete(output_root / "COMPLETE.txt", std::ios::trunc);
    if (!complete) throw std::runtime_error("cannot write Deep10M Quake completion marker");
    complete << "status=complete\nmethod=Quake\ninsert_batches=900\nquery_stages=100\n"
             << "queries_per_stage=10000\nquery_budget=floor(current_active_vectors/" << budget_divisor << ")\n"
             << "split_tau_ns=250\ndelete_tau_ns=250\nquery_threads=16\n"
             << "final_index_saved=" << (save_final_index ? "true" : "false")
             << "\ninitial_index_removed_after_run=" << (initial_index_created_here ? "true" : "false") << '\n';
    if (save_final_index) complete << "final_index=" << final_index_path.string() << '\n';
    std::cout << "DEEP10M_QUAKE_STAGED_COMPLETE output=" << output_root.string()
              << " final_index=" << (save_final_index ? final_index_path.string() : "none") << '\n' << std::flush;
}

void run_deep10m_staged(const std::filesystem::path& data_path,
                        const std::filesystem::path& query_path,
                        const std::filesystem::path& stage_truth_path,
                        const std::filesystem::path& official_truth_path,
                        const std::filesystem::path& initial_index_path,
                        const std::filesystem::path& output_root,
                        std::size_t lire_max_partition_size,
                        bool triad_only,
                        const MappedFbin& data, const MappedFbin& queries) {
    if (data.rows() != 10'000'000 || queries.rows() != kQueries ||
        kDimension != 96 || kTopK != 100 || kInsertOps != 900 ||
        kQueryOps != 100 || kInsertOpsPerQuery != 9 ||
        kInitialRows != 1'000'000 || kInitialNlist != 1'000)
        throw std::runtime_error("Deep10M staged runner was built with an unexpected profile");
    if (lire_max_partition_size == 0 ||
        !std::filesystem::is_regular_file(stage_truth_path) ||
        !std::filesystem::is_regular_file(official_truth_path) ||
        !std::filesystem::is_regular_file(initial_index_path))
        throw std::runtime_error("Deep10M run requires stage truth, official truth, and initial index files");

    const std::uint64_t truth_header_bytes = 8 + 3 * sizeof(std::uint32_t);
    const std::uint64_t truth_row_bytes = static_cast<std::uint64_t>(kQueries) * sizeof(Ids);
    const std::uint64_t expected_stage_truth_bytes = truth_header_bytes +
        static_cast<std::uint64_t>(kQueryOps) * truth_row_bytes;
    {
        std::ifstream truth(stage_truth_path, std::ios::binary);
        char magic[8]{};
        std::uint32_t shape[3]{};
        truth.read(magic, sizeof(magic));
        truth.read(reinterpret_cast<char*>(shape), sizeof(shape));
        if (!truth || std::string(magic, sizeof(magic)) != "MSTGTH01" ||
            shape[0] != kQueryOps || shape[1] != kQueries || shape[2] != kTopK ||
            std::filesystem::file_size(stage_truth_path) != expected_stage_truth_bytes)
            throw std::runtime_error("Deep10M exact stage truth must be MSTGTH01 [100,10000,100]");
    }
    {
        const AdaptiveHierarchicalIvf initial = AdaptiveHierarchicalIvf::load(initial_index_path.string());
        const auto& cfg = initial.config();
        if (initial.size() != kInitialRows || initial.leaf_nlist() != kInitialNlist ||
            cfg.dimension != kDimension || cfg.lvq8_storage || cfg.lvq4_storage ||
            cfg.angular_metric || cfg.uint8_storage)
            throw std::runtime_error("shared initial index does not match Deep10M 1M/1000 FP32 L2 configuration");
    }

    ensure_empty_or_create(output_root);
    std::filesystem::copy_file(stage_truth_path, output_root / "groundtruth_stage_top100.bin");
    std::filesystem::copy_file(official_truth_path, output_root / "groundtruth.public.10K.ibin");
    {
        std::ofstream manifest(output_root / "manifest.txt", std::ios::trunc);
        if (!manifest) throw std::runtime_error("cannot write Deep10M staged manifest");
        manifest << "experiment=" << (triad_only ? "full Triad only" : "Triad, Quake, and LIRE")
                 << " on Deep10M_2 insert-heavy workload\n"
                 << "base=" << data_path.string() << "\nqueries=" << query_path.string()
                 << "\nstage_truth=" << stage_truth_path.string()
                 << "\nofficial_final_truth=" << official_truth_path.string()
                 << "\nbase=10000000x96 FP32; query_metric=squared L2; original file order; no normalization\n"
                 << "initial_vectors=1000000; initial_nlist=1000; insert_batches=900 x 10000\n"
                 << "query_schedule=100 stages, each after 9 insert batches; 10000 queries/stage; same query set reused\n"
                 << "query_budget=floor(active_vectors/100) total distance computations/query, centroid computations included\n"
                 << "query_threads=16; topk=100; LVQ=false; early_stop=false; APS=false; warmup=none; seed=1234\n"
                 << "shared_initial_index=" << initial_index_path.string() << "\n"
                 << "Triad=full policy; PRD=0.075; load split/merge threshold=2.2T / short cutoff <0.2*sqrt(N); capacity split=1.05*sqrt(N) while nlist<=sqrt(N); PRD direct split; local split reassignment radius=50, one iteration; global centers refreshed to stored means after each insert batch\n"
                 << "Quake=split and delete tau=250ns; alpha=0.9; query hit window=1000; min partition=32; local refinement radius=50; one iteration\n"
                 << "LIRE=Quake source size policy; max partition size=" << lire_max_partition_size
                 << "; min partition=32; after any batch with splits, reassign locally around both child centers with radius=50 (including seeds), one assignment pass; small partition vectors deleted\n"
                 << "LIRE maintenance is invoked once after every insert batch; all splits in that maintenance call are performed before its local reassignment pass\n"
                 << "exact_truth=100 per-stage prefixes computed by brute-force squared L2; final stage checked against supplied official top-100\n"
                 << "final_indexes_saved=true\n";
    }

    AdaptiveMethodConfig triad;
    triad.name = "triad";
    triad.prd = true;
    triad.triad_policy = true;
    triad.split_local_refinement_radius = 50;
    triad.split_local_refinement_iterations = 1;
    triad.override_split_local_refinement = true;
    triad.prd_threshold_override = 0.075f;
    triad.short_list_sqrt_n_fraction_override = 0.20f;
    triad.query_budget_divisor_override = 100;
    triad.query_threads = 16;

    AdaptiveMethodConfig quake;
    quake.name = "quake";
    quake.quake = true;
    quake.quake_split_threshold_ns = 250.0;
    quake.quake_delete_threshold_ns = 250.0;
    quake.query_budget_divisor_override = 100;
    quake.query_threads = 16;

    AdaptiveMethodConfig lire;
    lire.name = "lire";
    lire.quake_lire = true;
    lire.lire_max_partition_size = lire_max_partition_size;
    lire.query_budget_divisor_override = 100;
    lire.query_threads = 16;

    if (triad_only) {
        std::cout << "DEEP10M_STAGE_RUN_START method=" << triad.name << '\n' << std::flush;
        run_adaptive_clean(output_root, data, queries, triad, initial_index_path);
        std::cout << "DEEP10M_STAGE_RUN_DONE method=" << triad.name << '\n' << std::flush;
    } else {
        for (const AdaptiveMethodConfig* method : {&triad, &quake, &lire}) {
            std::cout << "DEEP10M_STAGE_RUN_START method=" << method->name << '\n' << std::flush;
            run_adaptive_clean(output_root, data, queries, *method, initial_index_path,
                               output_root / (method->name + "_final.index.bin"));
            std::cout << "DEEP10M_STAGE_RUN_DONE method=" << method->name << '\n' << std::flush;
        }
    }

    score_existing_outputs(output_root, output_root, data, queries);
    score_existing_recall_at_10(output_root, output_root / "groundtruth_stage_top100.bin",
                                output_root / "recall_at_10_completed");
    std::ofstream complete(output_root / "COMPLETE.txt", std::ios::trunc);
    if (!complete) throw std::runtime_error("cannot write Deep10M completion marker");
    complete << "status=complete\nmethods=" << (triad_only ? "triad" : "triad,quake,lire")
             << "\ninsert_batches_per_method=900\n"
              << "query_stages_per_method=100\nqueries_per_stage=10000\nquery_budget=floor(current_active_vectors/100)\nquery_threads=16\n"
             << "Triad_complete_modules=PRD,load split/merge,capacity split,split-local reassignment radius=50 one iteration,global center refresh after every insertion batch\n"
             << "LIRE_max_partition_size=" << lire_max_partition_size
             << "\nLIRE_local_reassignment=radius 50, one pass after every insertion-batch split phase\n"
             << "truth=exact brute-force stage prefixes, final stage cross-checked against official Deep10M_2 truth\n"
             << "final_indexes_saved=true\n";
    complete.close();
    std::cout << (triad_only ? "DEEP10M_TRIAD_STAGED_COMPLETE output="
                             : "DEEP10M_THREE_METHODS_STAGED_COMPLETE output=")
              << output_root.string() << '\n' << std::flush;
}

int main(int argc, char** argv) {
    try {
#if defined(MSTURING_QUAKE_PROFILE_DEFAULT) || defined(DEEP10M_100STAGE_IH) || defined(YFCC10M_IH)
        if (argc > 0 && g_quake_latency_profile_path.empty()) {
            std::error_code ec;
            const auto executable = std::filesystem::absolute(argv[0], ec);
            if (!ec) {
#if defined(MSTURING_QUAKE_PROFILE_DEFAULT)
                g_quake_latency_profile_path = executable.parent_path() /
                    "quake_latency_profile_msturing_dim100_avx2_30runs_k10_20261031.csv";
#elif defined(DEEP10M_100STAGE_IH)
                g_quake_latency_profile_path = executable.parent_path() /
                    "quake_latency_profile_deep_dim96_fp32_avx2_30runs_k10_20261031.csv";
#elif defined(YFCC10M_IH)
                g_quake_latency_profile_path = executable.parent_path() /
                    "quake_latency_profile_yfcc_dim192_u8_avx2_30runs_k10_20261031.csv";
#endif
            }
        }
#endif
        if (argc >= 3 && std::string(argv[1]) == "--quake-latency-profile") {
            g_quake_latency_profile_path = argv[2];
            if (!std::filesystem::is_regular_file(g_quake_latency_profile_path))
                throw std::runtime_error("Quake latency profile does not exist: " +
                                         g_quake_latency_profile_path.string());
            argc -= 2;
            argv += 2;
        }
        if (argc == 2 && std::string(argv[1]) == "--validate-quake-latency-profile") {
            if (g_quake_latency_profile_path.empty() ||
                !std::filesystem::is_regular_file(g_quake_latency_profile_path))
                throw std::runtime_error("validation requires an available Quake latency profile CSV");
            ScanLatencyProfile profile({}, g_quake_latency_profile_path);
            std::cout << "QUAKE_LATENCY_PROFILE_VALID rows=60 sample_L_1024_10_ns="
                      << profile(1024, 10) << " sample_L_16384_10_ns="
                      << profile(16384, 10) << " sample_L_65536_10_ns="
                      << profile(65536, 10) << " source="
                      << profile.profile_source() << '\n';
            return 0;
        }
        if ((argc == 10 || argc == 11) && (std::string(argv[1]) == "--repeat-ih-method" ||
                                           std::string(argv[1]) == "--repeat-ih-insert-only")) {
            const bool insert_only = std::string(argv[1]) == "--repeat-ih-insert-only";
            const std::string method_name(argv[2]);
            const std::filesystem::path data_path(argv[3]);
            const std::filesystem::path query_path(argv[4]);
            const std::filesystem::path truth_path(argv[5]);
            const std::filesystem::path initial_index_path(argv[6]);
            const std::filesystem::path output_root(argv[7]);
            const std::size_t query_threads = static_cast<std::size_t>(std::stoull(argv[8]));
            const std::size_t budget_divisor = static_cast<std::size_t>(std::stoull(argv[9]));
            const std::size_t lire_max_partition_size = argc == 11
                ? static_cast<std::size_t>(std::stoull(argv[10])) : 1'000;
            if ((method_name != "quake" && method_name != "triad" && method_name != "lire") ||
                query_threads < 2 || budget_divisor == 0 ||
                (argc == 11 && (!insert_only || method_name != "lire" || lire_max_partition_size == 0)) ||
                !std::filesystem::is_regular_file(truth_path) ||
                !std::filesystem::is_regular_file(initial_index_path))
                throw std::runtime_error("repeat runner requires quake|triad|lire, existing truth/index, at least 2 query threads, and positive budget divisor");
            {
                std::ifstream truth(truth_path, std::ios::binary);
                char magic[8]{};
                std::uint32_t shape[3]{};
                truth.read(magic, sizeof(magic));
                truth.read(reinterpret_cast<char*>(shape), sizeof(shape));
                const std::uint64_t expected = 8 + 3 * sizeof(std::uint32_t) +
                    static_cast<std::uint64_t>(kQueryOps) * kQueries * sizeof(Ids);
                if (!truth || std::string(magic, sizeof(magic)) != "MSTGTH01" ||
                    shape[0] != kQueryOps || shape[1] != kQueries || shape[2] != kTopK ||
                    std::filesystem::file_size(truth_path) != expected)
                    throw std::runtime_error("repeat runner requires exact MSTGTH01 [100,10000,100] stage truth");
            }
            {
                AdaptiveHierarchicalIvf initial = AdaptiveHierarchicalIvf::load(initial_index_path.string());
                if (initial.size() != kInitialRows || initial.leaf_nlist() != kInitialNlist ||
                    initial.config().dimension != kDimension || initial.config().lvq8_storage ||
                    initial.config().lvq4_storage || initial.config().angular_metric)
                    throw std::runtime_error("shared initial index shape or metric does not match this runner");
            }
            ensure_empty_or_create(output_root);
            std::ofstream run_manifest(output_root / "manifest.txt", std::ios::trunc);
            if (!run_manifest) throw std::runtime_error("cannot create repeat-run manifest");
            run_manifest << "experiment=repeatability run for 10M insert-heavy workload\n"
                         << "method=" << method_name << "\ndata=" << data_path.string()
                         << "\nqueries=" << query_path.string() << "\nstage_truth=" << truth_path.string()
                         << "\nshared_initial_index=" << initial_index_path.string()
                         << "\ninitial_vectors=" << kInitialRows << "\ninitial_nlist=" << kInitialNlist
                         << "\ninsert_ops=" << kInsertOps << "\ninsert_batch=" << kInsertBatchRows
                         << "\nquery_stages=" << (insert_only ? 0 : kQueryOps)
                         << "\nqueries_per_stage=" << (insert_only ? 0 : kQueries)
                         << "\nquery_schedule=" << (insert_only ? "none; insertion and maintenance only" :
                                "after each " + std::to_string(kInsertOpsPerQuery) + " insert batches")
                         << "\nquery_budget=" << (insert_only ? "not applicable" :
                                "floor(active_vectors/" + std::to_string(budget_divisor) + ") including center and vector distances")
                         << "\nquery_threads=" << (insert_only ? 0 : query_threads)
                         << "\nLVQ=false\nearly_stop=false\nAPS=false\nfinal_index_saved=false"
                         << "\nper_query_csv=" << (insert_only ? "disabled" : "full query-level scan and latency details retained")
                         << "\nresult_ids=" << (insert_only ? "disabled; no query results generated" :
                                "results.bin; exact posthoc recall files are emitted") << '\n';
            run_manifest.close();

            AdaptiveMethodConfig method;
            method.name = method_name;
            method.initial_leaf_nlist = kInitialNlist;
            method.query_threads = query_threads;
            method.query_budget_divisor_override = budget_divisor;
            method.insertion_only = insert_only;
            method.record_per_query = !insert_only;
            if (method_name == "triad") {
                method.prd = true;
                method.triad_policy = true;
                method.prd_threshold_override = 0.075f;
                method.short_list_sqrt_n_fraction_override = 0.20f;
                method.triad_global_center_refresh = true;
                method.override_split_local_refinement = true;
                method.split_local_refinement_radius = 50;
                method.split_local_refinement_iterations = 1;
            } else if (method_name == "quake") {
                method.quake = true;
                method.quake_split_threshold_ns = 250.0;
                method.quake_delete_threshold_ns = 250.0;
                if (g_quake_latency_profile_path.empty() ||
                    !std::filesystem::is_regular_file(g_quake_latency_profile_path))
                    throw std::runtime_error("repeat Quake runner requires its fixed, dimension-matched latency profile");
            } else {
                method.quake_lire = true;
                method.lire_max_partition_size = lire_max_partition_size;
            }

            auto run_typed = [&](const auto& data, const auto& queries) {
                if (data.rows() != 10'000'000 || queries.rows() != kQueries)
                    throw std::runtime_error("repeat runner requires 10M base vectors and exactly 10K queries");
                run_adaptive_clean(output_root, data, queries, method, initial_index_path);
                if (!method.insertion_only) {
                    score_existing_outputs(output_root, output_root, data, queries, truth_path);
                    score_existing_recall_at_10(output_root, truth_path, output_root / "recall_at_10");
                }
            };
#ifdef YFCC10M_IH
            MappedU8bin data(data_path), query_bytes(query_path);
            FloatQueryRows queries(query_bytes);
            run_typed(data, queries);
#else
            MappedFbin data(data_path), queries(query_path);
            run_typed(data, queries);
#endif
            std::ofstream complete(output_root / "RUN_COMPLETE.txt", std::ios::trunc);
            complete << "status=complete\nmethod=" << method_name << "\nquery_threads=" << query_threads
                     << "\nbudget_divisor=" << budget_divisor << "\nquery_stages="
                     << (insert_only ? 0 : kQueryOps)
                     << "\nqueries_executed=" << (insert_only ? 0 : kQueryOps * kQueries)
                     << "\nfinal_index_saved=false\n";
            std::cout << (insert_only ? "REPEAT_METHOD_INSERT_ONLY_COMPLETE method=" : "REPEAT_METHOD_COMPLETE method=") << method_name
                      << " output=" << output_root.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 10 && std::string(argv[1]) == "--lire-full-100stage") {
            const std::filesystem::path data_path(argv[2]);
            const std::filesystem::path query_path(argv[3]);
            const std::filesystem::path truth_path(argv[4]);
            const std::filesystem::path initial_index_path(argv[5]);
            const std::filesystem::path output_root(argv[6]);
            const std::size_t max_partition_size = static_cast<std::size_t>(std::stoull(argv[7]));
            const std::size_t budget_divisor = static_cast<std::size_t>(std::stoull(argv[8]));
            const std::size_t query_threads = static_cast<std::size_t>(std::stoull(argv[9]));
#ifdef YFCC10M_IH
            MappedU8bin data(data_path.string());
            MappedU8bin query_bytes(query_path.string());
            FloatQueryRows queries(query_bytes);
            run_lire_full_staged(data_path, query_path, truth_path, initial_index_path,
                                 output_root, max_partition_size, budget_divisor,
                                 query_threads, data, queries);
#else
            MappedFbin data(data_path.string()), queries(query_path.string());
            run_lire_full_staged(data_path, query_path, truth_path, initial_index_path,
                                 output_root, max_partition_size, budget_divisor,
                                 query_threads, data, queries);
#endif
            return 0;
        }
#ifdef DEEP10M_100STAGE_IH
        if (argc == 10 && std::string(argv[1]) == "--deep10m-lire-final-only") {
            MappedFbin data(argv[2]), queries(argv[3]);
            if (data.rows() != 10'000'000 || queries.rows() != 10'000)
                throw std::runtime_error("expected Deep10M_2 base=10M x 96 and query=10K x 96");
            const std::size_t max_partition_size = static_cast<std::size_t>(std::stoull(argv[7]));
            const std::size_t budget_divisor = static_cast<std::size_t>(std::stoull(argv[8]));
            const std::size_t query_threads = static_cast<std::size_t>(std::stoull(argv[9]));
            run_deep10m_lire_final_only(argv[2], argv[3], argv[4], argv[5], argv[6],
                                        max_partition_size, budget_divisor, query_threads,
                                        data, queries);
            return 0;
        }
        if (argc == 8 && (std::string(argv[1]) == "--deep10m-quake-100stage" ||
                          std::string(argv[1]) == "--deep10m-quake-100stage-no-index")) {
            MappedFbin data(argv[2]), queries(argv[3]);
            if (data.rows() != 10'000'000 || queries.rows() != 10'000)
                throw std::runtime_error("expected Deep10M_2 base=10M x 96 and query=10K x 96");
            run_deep10m_quake_staged(argv[2], argv[3], argv[4], argv[5], argv[6], argv[7],
                                     data, queries,
                                     std::string(argv[1]) == "--deep10m-quake-100stage");
            return 0;
        }
        if (argc == 7 && std::string(argv[1]) == "--deep10m-quake-100stage-n20") {
            MappedFbin data(argv[2]), queries(argv[3]);
            const std::filesystem::path stage_truth_path(argv[4]);
            const std::filesystem::path official_truth_path(argv[5]);
            const std::filesystem::path output_root(argv[6]);
            if (data.rows() != 10'000'000 || queries.rows() != 10'000)
                throw std::runtime_error("expected Deep10M_2 base=10M x 96 and query=10K x 96");
            const auto temporary_initial_index = output_root / "temporary_shared_initial.index.bin";
            run_deep10m_quake_staged(argv[2], argv[3], stage_truth_path, official_truth_path,
                                     temporary_initial_index, output_root, data, queries,
                                     false, 20, true);
            return 0;
        }
        if (argc == 9 && (std::string(argv[1]) == "--deep10m-ih-100stage" ||
                          std::string(argv[1]) == "--deep10m-triad-full-ih-100stage")) {
            const bool triad_only = std::string(argv[1]) == "--deep10m-triad-full-ih-100stage";
            MappedFbin data(argv[2]), queries(argv[3]);
            const std::filesystem::path stage_truth_path(argv[4]);
            const std::filesystem::path official_truth_path(argv[5]);
            const std::filesystem::path initial_index_path(argv[6]);
            const std::filesystem::path output_root(argv[7]);
            const std::size_t lire_max_partition_size =
                static_cast<std::size_t>(std::stoull(argv[8]));
            if (data.rows() != 10'000'000 || queries.rows() != 10'000)
                throw std::runtime_error("expected Deep10M_2 base=10M x 96 and query=10K x 96");
            run_deep10m_staged(argv[2], argv[3], stage_truth_path, official_truth_path,
                               initial_index_path, output_root, lire_max_partition_size, triad_only,
                               data, queries);
            return 0;
        }
#endif
        if (argc == 7 && std::string(argv[1]) == "--frozen-final-only") {
#ifdef YFCC10M_IH
            MappedU8bin data(argv[2]), query_bytes(argv[3]);
            FloatQueryRows queries(query_bytes);
#else
            MappedFbin data(argv[2]), queries(argv[3]);
#endif
            const std::filesystem::path truth_path(argv[4]);
            const std::filesystem::path initial_index_path(argv[5]);
            const std::filesystem::path output_root(argv[6]);
            if (data.rows() != 10'000'000 || queries.rows() < kQueries ||
                !std::filesystem::is_regular_file(truth_path) ||
                !std::filesystem::is_regular_file(initial_index_path))
                throw std::runtime_error("frozen final-only run requires 10M data, queries, cached truth, and reusable initial index");
            ensure_empty_or_create(output_root);
            std::filesystem::copy_file(truth_path, output_root / "groundtruth_stage_top100.bin");
            AdaptiveMethodConfig method;
            method.name = "frozen_ivf";
            method.initial_leaf_nlist = kInitialNlist;
            method.final_query_only = true;
            run_adaptive_clean(output_root, data, queries, method, initial_index_path);

            std::ifstream truth_file(truth_path, std::ios::binary);
            char truth_magic[8]{};
            std::uint32_t truth_shape[3]{};
            truth_file.read(truth_magic, sizeof(truth_magic));
            truth_file.read(reinterpret_cast<char*>(truth_shape), sizeof(truth_shape));
            if (!truth_file || std::string(truth_magic, sizeof(truth_magic)) != "MSTGTH01" ||
                (truth_shape[0] != kQueryOps && truth_shape[0] != 1) ||
                truth_shape[1] != kQueries || truth_shape[2] != kTopK)
                throw std::runtime_error("frozen final-only run requires exact final-stage top-100 truth");
            constexpr std::uint64_t header_bytes = 8 + 3 * sizeof(std::uint32_t);
            constexpr std::uint64_t stage_bytes = static_cast<std::uint64_t>(kQueries) * sizeof(Ids);
            const std::size_t final_truth_stage = truth_shape[0] == 1 ? 0 : kQueryOps - 1;
            truth_file.seekg(static_cast<std::streamoff>(header_bytes + final_truth_stage * stage_bytes));
            std::vector<Ids> exact(kQueries), found(kQueries);
            truth_file.read(reinterpret_cast<char*>(exact.data()), static_cast<std::streamsize>(stage_bytes));
            if (!truth_file) throw std::runtime_error("cannot read final exact truth stage");
            std::ifstream result_file(output_root / method.name / "results.bin", std::ios::binary);
            char result_magic[8]{};
            std::uint32_t result_shape[3]{};
            result_file.read(result_magic, sizeof(result_magic));
            result_file.read(reinterpret_cast<char*>(result_shape), sizeof(result_shape));
            if (!result_file || std::string(result_magic, sizeof(result_magic)) != "MSTIHR01" ||
                result_shape[0] != 1 || result_shape[1] != kQueries || result_shape[2] != kTopK)
                throw std::runtime_error("frozen final-only result shape mismatch");
            result_file.read(reinterpret_cast<char*>(found.data()), static_cast<std::streamsize>(stage_bytes));
            if (!result_file) throw std::runtime_error("cannot read frozen final query results");
            std::uint64_t hits10 = 0, hits100 = 0;
            for (std::size_t qi = 0; qi < kQueries; ++qi) {
                for (std::size_t rank = 0; rank < 10; ++rank) {
                    if (std::find(exact[qi].begin(), exact[qi].begin() + 10, found[qi][rank]) != exact[qi].begin() + 10)
                        ++hits10;
                    if (std::find(exact[qi].begin(), exact[qi].end(), found[qi][rank]) != exact[qi].end())
                        ++hits100;
                }
                for (std::size_t rank = 10; rank < kTopK; ++rank)
                    if (std::find(exact[qi].begin(), exact[qi].end(), found[qi][rank]) != exact[qi].end())
                        ++hits100;
            }
            const double recall10 = static_cast<double>(hits10) / (static_cast<double>(kQueries) * 10.0);
            const double recall100 = static_cast<double>(hits100) / (static_cast<double>(kQueries) * 100.0);
            std::ofstream score(output_root / method.name / "final_recall.csv", std::ios::trunc);
            score << "dataset_rows,nlist,query_count,budget_per_query,recall_at_10,recall_at_100\n"
                  << data.rows() << ',' << kInitialNlist << ',' << kQueries << ','
                  << data.rows() / 20 << ',' << std::setprecision(10) << recall10 << ',' << recall100 << '\n';
            if (!score) throw std::runtime_error("cannot write frozen final-only recall metrics");
            std::cout << "FROZEN_FINAL_ONLY_COMPLETE recall@10=" << recall10
                      << " recall@100=" << recall100 << " output=" << output_root.string() << '\n' << std::flush;
            return 0;
        }
#ifdef YFCC10M_IH
        if (argc == 7 && std::string(argv[1]) == "--quake-only-u8-reuse-initial") {
            MappedU8bin data(argv[2]), query_bytes(argv[3]);
            FloatQueryRows queries(query_bytes);
            const std::filesystem::path stage_truth_path(argv[4]);
            const std::filesystem::path initial_index_path(argv[5]);
            const std::filesystem::path root(argv[6]);
            if (data.rows() != 10'000'000 || query_bytes.rows() != kQueries ||
                kDimension != 192 || kTopK != 100 || kInsertOps != 900 ||
                kQueryOps != 100 || kInsertOpsPerQuery != 9 ||
                kInitialRows != 1'000'000 || kInitialNlist != 1'000)
                throw std::runtime_error("YFCC Quake repeat runner was built with an unexpected profile");
            if (!std::filesystem::is_regular_file(stage_truth_path) ||
                !std::filesystem::is_regular_file(initial_index_path))
                throw std::runtime_error("YFCC Quake repeat run requires cached stage truth and shared initial index");
            const std::uint64_t header_bytes = 8 + 3 * sizeof(std::uint32_t);
            const std::uint64_t expected_truth_bytes = header_bytes +
                static_cast<std::uint64_t>(kQueryOps) * kQueries * sizeof(Ids);
            {
                std::ifstream truth(stage_truth_path, std::ios::binary);
                char magic[8]{};
                std::uint32_t shape[3]{};
                truth.read(magic, sizeof(magic));
                truth.read(reinterpret_cast<char*>(shape), sizeof(shape));
                if (!truth || std::string(magic, sizeof(magic)) != "MSTGTH01" ||
                    shape[0] != kQueryOps || shape[1] != kQueries || shape[2] != kTopK ||
                    std::filesystem::file_size(stage_truth_path) != expected_truth_bytes)
                    throw std::runtime_error("YFCC Quake stage truth must be MSTGTH01 [100,10000,100]");
            }
            {
                const AdaptiveHierarchicalIvf initial = AdaptiveHierarchicalIvf::load(initial_index_path.string());
                const auto& cfg = initial.config();
                if (initial.size() != kInitialRows || initial.leaf_nlist() != kInitialNlist ||
                    cfg.dimension != kDimension || cfg.lvq8_storage || cfg.lvq4_storage ||
                    cfg.angular_metric || !cfg.uint8_storage)
                    throw std::runtime_error("reused initial index does not match YFCC 1M/1000 uint8 L2 configuration");
            }
            ensure_empty_or_create(root);
            std::filesystem::copy_file(stage_truth_path, root / "groundtruth_stage_top100.bin");
            {
                std::ofstream manifest(root / "manifest.txt", std::ios::trunc);
                if (!manifest) throw std::runtime_error("cannot write YFCC Quake repeat manifest");
                manifest << "experiment=YFCC10M Quake IH repeatability run; 100 query stages\n"
                         << "base=" << argv[2] << "\nqueries=" << argv[3]
                         << "\nstage_truth=" << stage_truth_path.string()
                         << "\nbase=10000000x192 uint8; squared L2; original source order; no normalization\n"
                         << "query_source=official first 10000 queries\n"
                         << "initial_vectors=1000000; initial_nlist=1000; inserts=900 batches x 10000\n"
                         << "query_schedule=100 stages, each after 9 insert batches; 10000 queries/stage\n"
                         << "query_budget=floor(current_active_vectors/100), centroid and vector distances included\n"
                         << "query_threads=16; topk=100; LVQ=false; early_stop=false; APS=false\n"
                         << "seed=1234; initial_index_reused=" << initial_index_path.string() << '\n'
                         << "Quake=split_tau=250ns; delete_tau=250ns; alpha=0.9; hit_window=1000; min_partition=32; local_refinement_radius=50; one iteration\n"
                         << "final_index_saved=false\n";
            }
            AdaptiveMethodConfig quake;
            quake.name = "quake";
            quake.initial_leaf_nlist = kInitialNlist;
            quake.quake = true;
            quake.quake_split_threshold_ns = 250.0;
            quake.quake_delete_threshold_ns = 250.0;
            quake.query_budget_divisor_override = 100;
            quake.query_threads = 16;
            std::cout << "YFCC_QUAKE_REPEAT_START budget=N/100 threads=16 final_index=none output="
                      << root.string() << '\n' << std::flush;
            run_adaptive_clean(root, data, queries, quake, initial_index_path);
            score_existing_outputs(root, root, data, queries);
            score_existing_recall_at_10(root, root / "groundtruth_stage_top100.bin",
                                        root / "recall_at_10");
            std::ofstream complete(root / "COMPLETE.txt", std::ios::trunc);
            if (!complete) throw std::runtime_error("cannot write YFCC Quake repeat completion marker");
            complete << "status=complete\nmethod=Quake\ninsert_batches=900\nquery_stages=100\n"
                     << "queries_per_stage=10000\nquery_budget=floor(current_active_vectors/100)\n"
                     << "split_tau_ns=250\ndelete_tau_ns=250\nquery_threads=16\nfinal_index_saved=false\n";
            std::cout << "YFCC_QUAKE_REPEAT_COMPLETE output=" << root.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 9 && std::string(argv[1]) == "--yfcc-static-sample20-u8") {
            MappedU8bin data(argv[2]), query_bytes(argv[3]);
            const std::size_t nlist = static_cast<std::size_t>(std::stoull(argv[7]));
            const std::uint32_t seed = static_cast<std::uint32_t>(std::stoul(argv[8]));
            run_yfcc_static_sample20(argv[6], data, query_bytes, argv[4], argv[5], nlist, seed);
            return 0;
        }
#endif
        if (argc == 3 && std::string(argv[1]) == "--inspect-initial-index") {
            const auto index = AdaptiveHierarchicalIvf::load(argv[2]);
#ifdef YFCC10M_IH
            constexpr bool expected_uint8 = true;
#else
            constexpr bool expected_uint8 = false;
#endif
            if (index.size() != kInitialRows || index.leaf_nlist() != kInitialNlist ||
                index.config().dimension != kDimension || index.config().lvq8_storage ||
                index.config().lvq4_storage || index.config().angular_metric ||
                index.config().uint8_storage != expected_uint8) {
                throw std::runtime_error("initial index does not match compile-time dataset settings");
            }
            std::cout << "INITIAL_INDEX_OK rows=" << index.size()
                      << " nlist=" << index.leaf_nlist()
                      << " dimension=" << index.config().dimension
                      << " uint8=" << index.config().uint8_storage
                      << " prd_threshold=" << index.config().prd_threshold
                      << " short_list_sqrt_n_fraction=" << index.config().short_list_sqrt_n_fraction
                      << "\n" << std::flush;
            return 0;
        }
        if (argc == 7 && std::string(argv[1]) == "--triad-fixed-n20") {
#ifdef YFCC10M_IH
            MappedU8bin data(argv[2]), query_bytes(argv[3]);
            FloatQueryRows queries(query_bytes);
#else
            MappedFbin data(argv[2]), queries(argv[3]);
#endif
            run_triad_fixed_n20(argv[2], argv[3], argv[4], argv[5], argv[6], data, queries);
            return 0;
        }
        if (argc == 7 && std::string(argv[1]) == "--triad-final-n20-no-reassignment") {
#ifdef YFCC10M_IH
            MappedU8bin data(argv[2]), query_bytes(argv[3]);
            FloatQueryRows queries(query_bytes);
#else
            MappedFbin data(argv[2]), queries(argv[3]);
#endif
            run_triad_final_n20_no_reassignment(argv[2], argv[3], argv[4], argv[5], argv[6], data, queries);
            return 0;
        }
        if ((argc == 7 || argc == 8 || argc == 9 || argc == 10) &&
            std::string(argv[1]) == "--triad-threshold-robustness-final-n20") {
#ifdef YFCC10M_IH
            throw std::runtime_error("Triad threshold robustness runner expects FP32 MSTuring data");
#else
            MappedFbin data(argv[2]), queries(argv[3]);
            const unsigned detected_threads = std::thread::hardware_concurrency();
            const std::size_t query_threads = argc >= 8
                ? static_cast<std::size_t>(std::stoull(argv[7]))
                : std::max(1u, detected_threads);
            const std::size_t n_divisor = argc >= 9
                ? static_cast<std::size_t>(std::stoull(argv[8])) : 1000;
            const std::size_t capacity_cluster_count_gate = argc == 10
                ? static_cast<std::size_t>(std::stoull(argv[9])) : 1000;
            run_triad_threshold_robustness_final_n20(
                argv[2], argv[3], argv[4], argv[5], argv[6], data, queries,
                query_threads, n_divisor, capacity_cluster_count_gate);
            return 0;
#endif
        }
        if (argc == 5 && std::string(argv[1]) == "--prepare-truth-only-u8") {
            MappedU8bin data(argv[2]), queries(argv[3]);
            if (data.rows() != 10'000'000 || queries.rows() == 0)
                throw std::runtime_error("uint8 stage-truth preparation expects 10M base vectors and a non-empty query set");
            const auto truth = calculate_truth_u8(data, queries);
            const std::filesystem::path truth_path(argv[4]);
            std::ofstream out(truth_path, std::ios::binary);
            if (!out) throw std::runtime_error("cannot create stage-truth file: " + truth_path.string());
            const char magic[8] = {'M','S','T','G','T','H','0','1'};
            const std::uint32_t shape[3] = {static_cast<std::uint32_t>(kQueryOps),
                static_cast<std::uint32_t>(queries.rows()), static_cast<std::uint32_t>(kTopK)};
            out.write(magic, sizeof(magic));
            out.write(reinterpret_cast<const char*>(shape), sizeof(shape));
            for (const auto& ids : truth)
                out.write(reinterpret_cast<const char*>(ids.data()), static_cast<std::streamsize>(sizeof(ids)));
            if (!out) throw std::runtime_error("failed writing stage-truth file: " + truth_path.string());
            std::cout << "STAGE_TRUTH_U8_COMPLETE stages=" << kQueryOps << " queries=" << queries.rows()
                      << " topk=" << kTopK << " output=" << truth_path.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 5 && std::string(argv[1]) == "--resume-yfcc10m-lire") {
            MappedU8bin data(argv[2]), query_bytes(argv[3]);
            FloatQueryRows queries(query_bytes);
            const std::filesystem::path root(argv[4]);
            const auto truth_path = root / "groundtruth_stage_top100.bin";
            const auto initial_index_path = root / "shared_initial_1m_nlist1000.index.bin";
            if (data.rows() != 10'000'000 || queries.rows() != 10'000 || kDimension != 192 ||
                kQueryOps != 100 || kQueries != 10'000)
                throw std::runtime_error("YFCC LIRE resume requires 10M x 192 data and 10K-query build");
            if (!std::filesystem::is_regular_file(truth_path) ||
                std::filesystem::file_size(truth_path) != 20ull + 100ull * 10'000ull * 100ull * sizeof(std::uint32_t))
                throw std::runtime_error("YFCC staged truth missing or malformed in resume directory");
            if (!std::filesystem::is_regular_file(initial_index_path) ||
                !std::filesystem::is_regular_file(root / "triad_final.index.bin") ||
                !std::filesystem::is_regular_file(root / "quake_final.index.bin"))
                throw std::runtime_error("resume requires the shared initial index and completed Triad/Quake indexes");
            const auto expected_result_bytes = 20ull + 100ull * 10'000ull * 100ull * sizeof(std::uint32_t);
            for (const char* method : {"triad", "quake"}) {
                const auto result_path = root / method / "results.bin";
                if (!std::filesystem::is_regular_file(result_path) ||
                    std::filesystem::file_size(result_path) != expected_result_bytes)
                    throw std::runtime_error(std::string("cannot resume: incomplete result file for ") + method);
            }
            if (std::filesystem::exists(root / "lire") || std::filesystem::exists(root / "lire_final.index.bin"))
                throw std::runtime_error("LIRE output already exists; refusing to overwrite it during resume");

            AdaptiveMethodConfig lire;
            lire.name = "lire";
            lire.quake_lire = true;
            lire.lire_max_partition_size = 1'000;
            lire.query_budget_divisor_override = 100;
            lire.query_threads = 16;
            std::cout << "YFCC_FULL_IH_METHOD_START method=lire (resume from shared initial index)\n" << std::flush;
            run_adaptive_clean(root, data, queries, lire, initial_index_path,
                               root / "lire_final.index.bin");
            score_existing_outputs(root, root, data, queries);
            score_existing_recall_at_10(root, truth_path, root / "recall_at_10_completed");
            std::ofstream complete(root / "COMPLETE.txt", std::ios::trunc);
            if (!complete) throw std::runtime_error("cannot write YFCC full IH completion marker");
            complete << "status=complete\nmethods=triad,quake,lire\nquery_stages_per_method=100\n"
                     << "queries_per_stage=10000\nquery_budget=floor(active_vectors/100)\nquery_threads=16\n"
                     << "methods_run_sequentially=true\nshared_initial_index_reused=true\n"
                     << "final_indexes=triad_final.index.bin,quake_final.index.bin,lire_final.index.bin\n";
            std::cout << "YFCC_FULL_IH_COMPLETE output=" << root.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "--run-yfcc10m-full-ih-fixed-budget") {
            MappedU8bin data(argv[2]), query_bytes(argv[3]);
            FloatQueryRows queries(query_bytes);
            const std::filesystem::path truth_path(argv[4]);
            const std::filesystem::path root(argv[5]);
            if (data.rows() != 10'000'000 || queries.rows() != kQueries ||
                kDimension != 192 || kQueryOps != 100 || kQueries != 10'000)
                throw std::runtime_error("YFCC full IH requires 10M x 192 data, 100x10K-query profile, and the YFCC-10K build");
            if (!std::filesystem::is_regular_file(truth_path) ||
                std::filesystem::file_size(truth_path) !=
                    20ull + static_cast<std::uint64_t>(kQueryOps) * kQueries * kTopK * sizeof(std::uint32_t))
                throw std::runtime_error("YFCC exact staged top-100 truth is missing or has an unexpected size");
            {
                std::ifstream input(truth_path, std::ios::binary);
                char magic[8]{};
                std::uint32_t shape[3]{};
                input.read(magic, sizeof(magic));
                input.read(reinterpret_cast<char*>(shape), sizeof(shape));
                if (!input || std::string(magic, sizeof(magic)) != "MSTGTH01" ||
                    shape[0] != kQueryOps || shape[1] != kQueries || shape[2] != kTopK)
                    throw std::runtime_error("YFCC truth must have shape [100,10000,100]");
            }
            ensure_empty_or_create(root);
            std::filesystem::copy_file(truth_path, root / "groundtruth_stage_top100.bin");
            const auto initial_index_path = root / "shared_initial_1m_nlist1000.index.bin";
            AdaptiveHierarchicalConfig initial_cfg;
            initial_cfg.dimension = kDimension;
            initial_cfg.initial_leaf_nlist = kInitialNlist;
            initial_cfg.kmeans_iterations = kKmeansIterations;
            initial_cfg.seed = kSeed;
            initial_cfg.lvq8_storage = false;
            initial_cfg.lvq4_storage = false;
            initial_cfg.angular_metric = false;
            initial_cfg.uint8_storage = true;
            initial_cfg.automatic_maintenance = false;
            initial_cfg.prd_threshold = 0.075f;
            initial_cfg.drift_criterion = DriftCriterion::GlobalFitOnly;
            initial_cfg.load_split_fraction = 2.20f;
            initial_cfg.capacity_sqrt_n_split_multiplier = 1.05f;
            initial_cfg.short_list_sqrt_n_fraction = 0.20f;
            initial_cfg.split_local_refinement_radius = 50;
            initial_cfg.split_local_refinement_iterations = 1;
            AdaptiveHierarchicalIvf initial_index(initial_cfg);
            const auto initial_start = std::chrono::steady_clock::now();
            initial_index.train_u8(data.row(0), kInitialRows);
            initial_index.save(initial_index_path.string());
            const double initial_build_save_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - initial_start).count();
            {
                std::ofstream build(root / "shared_initial_build.txt", std::ios::trunc);
                if (!build) throw std::runtime_error("cannot record shared YFCC initial-index build");
                build << "initial_vectors=" << kInitialRows << "\ninitial_nlist=" << kInitialNlist
                      << "\ndimension=" << kDimension << "\nkmeans_iterations=" << kKmeansIterations
                      << "\nseed=" << kSeed << "\nmetric=squared L2; native uint8 payload"
                      << "\nbuild_and_save_seconds=" << std::setprecision(10) << initial_build_save_s
                      << "\nindex=" << initial_index_path.string()
                      << "\nindex_bytes=" << std::filesystem::file_size(initial_index_path) << '\n';
            }
            {
                std::ofstream manifest(root / "manifest.txt", std::ios::trunc);
                if (!manifest) throw std::runtime_error("cannot create YFCC full IH manifest");
                manifest << "experiment=YFCC10M Triad, Quake, and LIRE staged insert-heavy comparison\n"
                         << "base=" << argv[2] << "\nqueries=" << argv[3]
                         << "\ntruth=" << truth_path.string()
                         << "\nbase=10M x 192 uint8; original order; squared L2; no normalization\n"
                         << "query_selection=official query.public.100K rows 0..9999\n"
                         << "initial_vectors=1M; initial_nlist=1000; kmeans_seed=" << kSeed << '\n'
                         << "insertions=900 batches x 10000; query every 9 insert batches; 100 query stages\n"
                         << "active_vectors_at_query_stages=1.09M,1.18M,...,10M\n"
                         << "queries_per_stage=10000; topk=100; exact stage truth reused\n"
                         << "query_budget=floor(active_vectors/100), center and vector distances included\n"
                         << "query_threads=16; APS=false; early_stop=false; LVQ=false; query_warmup=none\n"
                         << "shared_initial_index=" << initial_index_path.string() << '\n'
                         << "Triad=PRD 0.075 direct split; load split 2.2T; merge below 0.2*sqrt(N); capacity split 1.05*sqrt(N) while nlist<=sqrt(N); split-local reassignment radius 50, one pass; move centers to stored cluster means after each insertion batch\n"
                         << "Quake=split/delete tau 250ns; alpha 0.9; query-hit window 1000; min cluster 32; local refinement radius 50, one pass\n"
                         << "LIRE=Quake size-threshold policy; max partition size 1000; min cluster 32; one unified local reassignment pass radius 50 after splits; short clusters and their vectors deleted\n"
                         << "methods_run_sequentially=Triad,Quake,LIRE\n";
            }

            AdaptiveMethodConfig triad;
            triad.name = "triad";
            triad.prd = true;
            triad.triad_policy = true;
            triad.query_budget_divisor_override = 100;
            triad.query_threads = 16;
            triad.prd_threshold_override = 0.075f;
            triad.short_list_sqrt_n_fraction_override = 0.20f;
            triad.override_split_local_refinement = true;
            triad.split_local_refinement_radius = 50;
            triad.split_local_refinement_iterations = 1;
            triad.triad_global_center_refresh = true;

            AdaptiveMethodConfig quake;
            quake.name = "quake";
            quake.quake = true;
            quake.query_budget_divisor_override = 100;
            quake.query_threads = 16;
            quake.quake_split_threshold_ns = 250.0;
            quake.quake_delete_threshold_ns = 250.0;

            AdaptiveMethodConfig lire;
            lire.name = "lire";
            lire.quake_lire = true;
            lire.lire_max_partition_size = 1'000;
            lire.query_budget_divisor_override = 100;
            lire.query_threads = 16;

            for (const AdaptiveMethodConfig* method : {&triad, &quake, &lire}) {
                std::cout << "YFCC_FULL_IH_METHOD_START method=" << method->name << '\n' << std::flush;
                run_adaptive_clean(root, data, queries, *method, initial_index_path,
                                   root / (method->name + "_final.index.bin"));
                score_existing_outputs(root, root, data, queries);
                std::cout << "YFCC_FULL_IH_METHOD_COMPLETE method=" << method->name << '\n' << std::flush;
            }
            score_existing_recall_at_10(root, root / "groundtruth_stage_top100.bin",
                                        root / "recall_at_10");
            std::ofstream complete(root / "COMPLETE.txt", std::ios::trunc);
            if (!complete) throw std::runtime_error("cannot write YFCC full IH completion marker");
            complete << "status=complete\nmethods=triad,quake,lire\nquery_stages_per_method=100\n"
                     << "queries_per_stage=10000\nquery_budget=floor(active_vectors/100)\nquery_threads=16\n"
                     << "methods_run_sequentially=true\nshared_initial_index_reused=true\n"
                     << "final_indexes=triad_final.index.bin,quake_final.index.bin,lire_final.index.bin\n";
            std::cout << "YFCC_FULL_IH_COMPLETE output=" << root.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "--quake-only-u8") {
            MappedU8bin data(argv[2]), query_bytes(argv[3]);
            FloatQueryRows queries(query_bytes);
            if (data.rows() != 10'000'000 || queries.rows() != kQueries)
                throw std::runtime_error("native uint8 Quake IH expects 10M base vectors and the configured query sample");
            const std::filesystem::path root(argv[5]);
            ensure_empty_or_create(root);
            const std::filesystem::path cached_truth(argv[4]);
            if (!std::filesystem::is_regular_file(cached_truth))
                throw std::runtime_error("cached stage truth does not exist: " + cached_truth.string());
            std::filesystem::copy_file(cached_truth, root / "groundtruth_stage_top100.bin");
            {
                std::ofstream manifest(root / "manifest.txt");
                if (!manifest) throw std::runtime_error("cannot create native uint8 Quake manifest");
                manifest << "experiment=YFCC10M Quake insert-heavy workload (IH), native uint8 storage\n"
                         << "dataset=" << argv[2] << "\nqueries=" << argv[3]
                         << "\nrows=10000000; dimension=" << kDimension << "; query_metric=" << kMetricName
                         << "; maintenance_metric=squared L2"
                         << "\nsource_vectors=uint8; index_vector_payload=exact uint8 bytes"
                         << "\ncentroids=FP32; sufficient_statistics=FP64; no FP32 copy of base vectors"
                         << "\ninitial_vectors=" << kInitialRows << "\ninitial_nlist=" << kInitialNlist
                         << "\ninsert_batch=" << kInsertBatchRows << "\ninsert_ops=" << kInsertOps
                         << "\nquery_ops=" << kQueryOps << "\nqueries_per_query_op=" << kQueries
                         << "\ntopk=" << kTopK
                         << "\nquery_budget=floor(current_active_vectors/20), including centroid and vector distances"
                         << "\nLVQ=false; early_stop=false; seed=" << kSeed
                         << "\nQuake split_threshold_ns=250"
                         << "; delete_threshold_ns=250"
                         << "\nalpha=" << kQuakeAlpha << "; min_partition_size=" << kQuakeMinPartitionSize
                         << "; query_hit_window=" << kQueryWindow
                         << "; local_refinement_radius=" << kQuakeRefinementRadius
                         << "; iterations=" << kQuakeRefinementIterations
                         << "\ngroundtruth=reused exact squared-L2 top-100 over each active prefix\n";
            }
            AdaptiveMethodConfig quake_method{"quake", false, true};
            quake_method.quake_split_threshold_ns = 250.0;
            quake_method.quake_delete_threshold_ns = 250.0;
            run_adaptive_clean(root, data, queries, quake_method);
            score_existing_outputs(root, root, data, queries);
            score_existing_recall_at_10(root, root / "groundtruth_stage_top100.bin",
                                        root / "recall_at_10");
            std::ofstream complete(root / "COMPLETE.txt");
            complete << "status=complete\nmethod=Quake only, native uint8 vector payload\n"
                     << "split_threshold_ns=250\ndelete_threshold_ns=250\n"
                     << "insert_ops=" << kInsertOps << "\nquery_stages=" << kQueryOps
                     << "\nqueries_per_stage=" << kQueries
                     << "\nrecall=post-hoc exact Recall@100 and Recall@10\n";
            complete.close();
            std::cout << "COMPLETE output=" << root.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 7 && std::string(argv[1]) == "--quake-lire-only-u8") {
            MappedU8bin data(argv[2]), query_bytes(argv[3]);
            FloatQueryRows queries(query_bytes);
            if (data.rows() != 10'000'000 || queries.rows() != kQueries)
                throw std::runtime_error("native uint8 Quake-LIRE IH expects 10M base vectors and exactly the configured query sample");
            const std::size_t max_partition_size = static_cast<std::size_t>(std::stoull(argv[6]));
            if (max_partition_size == 0)
                throw std::invalid_argument("max_partition_size must be positive");
            const std::filesystem::path root(argv[5]);
            ensure_empty_or_create(root);
            const std::filesystem::path cached_truth(argv[4]);
            if (!std::filesystem::is_regular_file(cached_truth))
                throw std::runtime_error("cached stage truth does not exist: " + cached_truth.string());
            std::filesystem::copy_file(cached_truth, root / "groundtruth_stage_top100.bin");
            {
                std::ofstream manifest(root / "manifest.txt", std::ios::trunc);
                if (!manifest) throw std::runtime_error("cannot create native uint8 Quake-LIRE manifest");
                manifest << "experiment=YFCC-10M insert-heavy workload (IH), Quake source LIRE size policy port\n"
                         << "dataset=" << argv[2] << "\nqueries=" << argv[3]
                         << "\nrows=10000000; dimension=" << kDimension << "; metric=squared L2"
                         << "\nsource_vectors=uint8; exact inline uint8 payload; centroids=FP32; sufficient statistics=FP64"
                         << "\ninitial_vectors=" << kInitialRows << "; initial_nlist=" << kInitialNlist
                         << "\ninsert_batch=" << kInsertBatchRows << "; insert_ops=" << kInsertOps
                         << "\nquery_ops=" << kQueryOps << "; queries_per_query_operation=" << kQueries
                         << "\ntopk=" << kTopK << "; max_partition_size=" << max_partition_size
                         << "; min_partition_size=" << kQuakeMinPartitionSize
                         << "; refinement_radius=" << kQuakeRefinementRadius
                         << "; refinement_iterations=one assignment pass"
                         << "\nmaintenance=after each insert batch; split all eligible partitions, then one unified local reassignment pass; query batches do not trigger maintenance"
                         << "\nquery_schedule=after every 9 insert batches; query_budget=floor(current_active_vectors/20), including centroid comparisons"
                         << "\nLVQ=false; early_stop=false; APS=false; seed=" << kSeed
                         << "\ngroundtruth=reused exact squared-L2 top-100 over each active prefix\n";
            }
            AdaptiveMethodConfig method;
            method.name = "quake_lire";
            method.quake_lire = true;
            method.lire_max_partition_size = max_partition_size;
            run_adaptive_clean(root, data, queries, method);
            score_existing_outputs(root, root, data, queries);
            score_existing_recall_at_10(root, root / "groundtruth_stage_top100.bin",
                                        root / "recall_at_10");
            std::ofstream complete(root / "COMPLETE.txt", std::ios::trunc);
            complete << "status=complete\nmethod=Quake source size-threshold LIRE policy port, native uint8 payload"
                     << "\nmax_partition_size=" << max_partition_size
                     << "\nmin_partition_size=" << kQuakeMinPartitionSize
                     << "\nrefinement_radius=" << kQuakeRefinementRadius
                     << "\nrefinement_iterations=one assignment pass"
                     << "\ninsert_ops=" << kInsertOps << "\nquery_stages=" << kQueryOps
                     << "\nqueries_per_stage=" << kQueries
                     << "\nrecall=post-hoc exact Recall@100 and Recall@10\n";
            complete.close();
            std::cout << "COMPLETE output=" << root.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "--lire-only-u8") {
            MappedU8bin data(argv[2]), query_bytes(argv[3]);
            FloatQueryRows queries(query_bytes);
            if (data.rows() != 10'000'000 || queries.rows() != kQueries)
                throw std::runtime_error("native uint8 LIRE IH expects 10M base vectors and the configured 1K query sample");
            const std::filesystem::path root(argv[5]);
            ensure_empty_or_create(root);
            const std::filesystem::path cached_truth(argv[4]);
            if (!std::filesystem::is_regular_file(cached_truth))
                throw std::runtime_error("cached stage truth does not exist: " + cached_truth.string());
            std::filesystem::copy_file(cached_truth, root / "groundtruth_stage_top100.bin");
            {
                std::ofstream manifest(root / "manifest.txt");
                if (!manifest) throw std::runtime_error("cannot create native uint8 LIRE manifest");
                manifest << "experiment=YFCC-10M insert-heavy workload (IH), SPFresh-LIRE local port\n"
                         << "dataset=" << argv[2] << "\nqueries=" << argv[3]
                         << "\nrows=10000000; dimension=" << kDimension << "; query_metric=" << kMetricName
                         << "; maintenance_metric=squared L2"
                         << "\nsource_vectors=uint8; index_vector_payload=exact inline uint8 bytes"
                         << "\ncentroids=FP32; sufficient_statistics=FP64; no FP32 copy of base vectors"
                         << "\ninitial_vectors=" << kInitialRows << "; initial_nlist=" << kInitialNlist
                         << "\ninsert_batch=" << kInsertBatchRows << "; insert_ops=" << kInsertOps
                         << "\nquery_ops=" << kQueryOps << "; queries_per_query_op=" << kQueries
                         << "\ntopk=" << kTopK
                         << "\nquery_budget=floor(current_active_vectors/20), including centroid and vector distances"
                         << "\nLVQ=false; early_stop=false; seed=" << kSeed
                         << "\ngroundtruth=reused exact squared-L2 top-100 over each active prefix\n";
            }
            run_lire(root, data, queries);
            score_existing_outputs(root, root, data, queries);
            score_existing_recall_at_10(root, root / "groundtruth_stage_top100.bin",
                                        root / "recall_at_10");
            std::ofstream complete(root / "COMPLETE.txt");
            complete << "status=complete\nmethod=SPFresh-LIRE local port, native uint8 inline vector payload\n"
                     << "insert_ops=" << kInsertOps << "\nquery_stages=" << kQueryOps
                     << "\nqueries_per_stage=" << kQueries
                     << "\nrecall=post-hoc exact Recall@100 and Recall@10\n";
            complete.close();
            std::cout << "COMPLETE output=" << root.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 8 && std::string(argv[1]) == "--prd-final-ih-u8") {
            MappedU8bin data(argv[2]), query_bytes(argv[3]);
            FloatQueryRows queries(query_bytes);
            if (data.rows() != 10'000'000 || queries.rows() != kQueries)
                throw std::runtime_error("native uint8 PRD final-only IH expects 10M base vectors and the configured 1K query sample");
            run_prd_final_ih(argv[5], argv[2], argv[3], argv[4], data, queries,
                             std::stof(argv[6]), std::stof(argv[7]), false);
            return 0;
        }
        if (argc == 9 && std::string(argv[1]) == "--prd-full-ih-u8") {
            MappedU8bin data(argv[2]), query_bytes(argv[3]);
            FloatQueryRows queries(query_bytes);
            if (data.rows() != 10'000'000 || queries.rows() != kQueries)
                throw std::runtime_error("native uint8 PRD full IH expects 10M base vectors and the configured 1K query sample");
            if (!std::filesystem::is_regular_file(argv[8]))
                throw std::runtime_error("reusable initial PRD index does not exist: " + std::string(argv[8]));
            run_prd_final_ih(argv[5], argv[2], argv[3], argv[4], data, queries,
                             std::stof(argv[6]), std::stof(argv[7]), true, argv[8]);
            return 0;
        }
        if (argc == 5 && std::string(argv[1]) == "--prepare-truth-only") {
            MappedFbin data(argv[2]), queries(argv[3]);
            if (data.rows() != 10'000'000 || queries.rows() != kQueries)
                throw std::runtime_error("stage-truth preparation expects 10M vectors and exactly the configured query count");
            const auto truth = calculate_truth(data, queries);
            const std::filesystem::path truth_path(argv[4]);
            std::ofstream out(truth_path, std::ios::binary);
            if (!out) throw std::runtime_error("cannot create stage-truth file: " + truth_path.string());
            const char magic[8] = {'M','S','T','G','T','H','0','1'};
            const std::uint32_t shape[3] = {static_cast<std::uint32_t>(kQueryOps),
                static_cast<std::uint32_t>(kQueries), static_cast<std::uint32_t>(kTopK)};
            out.write(magic, sizeof(magic));
            out.write(reinterpret_cast<const char*>(shape), sizeof(shape));
            for (const auto& ids : truth)
                out.write(reinterpret_cast<const char*>(ids.data()), static_cast<std::streamsize>(sizeof(ids)));
            if (!out) throw std::runtime_error("failed writing stage-truth file: " + truth_path.string());
            std::cout << "STAGE_TRUTH_COMPLETE stages=" << kQueryOps << " queries=" << kQueries
                      << " topk=" << kTopK << " output=" << truth_path.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "--prd-only") {
            MappedFbin data(argv[2]), queries(argv[3]);
            if (data.rows() != 10'000'000 || queries.rows() != kQueries)
                throw std::runtime_error("PRD IH expects 10M vectors and exactly the configured query sample");
            const std::filesystem::path root(argv[5]);
            ensure_empty_or_create(root);
            const std::filesystem::path cached_truth(argv[4]);
            if (!std::filesystem::is_regular_file(cached_truth))
                throw std::runtime_error("cached stage truth does not exist: " + cached_truth.string());
            std::filesystem::copy_file(cached_truth, root / "groundtruth_stage_top100.bin");
            {
                std::ofstream manifest(root / "manifest.txt");
                if (!manifest) throw std::runtime_error("cannot create standalone PRD manifest");
                manifest << "experiment=YandexText-to-Image10M insert-heavy workload (IH), PRD-IVF\n"
                         << "dataset=" << argv[2] << "\nqueries=" << argv[3]
                         << "\nrows=10000000; dimension=" << kDimension
                         << "; query_metric=" << kMetricName << "; maintenance_metric=squared L2"
                         << "\ninitial_vectors=" << kInitialRows << "; initial_nlist=" << kInitialNlist
                         << "\ninsert_batch=" << kInsertBatchRows << "; insert_ops=" << kInsertOps
                         << "\nquery_ops=" << kQueryOps << "; queries_per_query_op=" << kQueries
                         << "\ntopk=" << kTopK
                         << "\nquery_budget=floor(current_active_vectors/20), including centroid and vector distances"
                         << "\nLVQ=false; early_stop=false; seed=" << kSeed
                         << "\nprd_threshold=0.05; load_split_fraction=2.2; capacity_sqrt_n_split_multiplier=1.05"
                         << "\nlocal_refinement=disabled; per-batch centroid refresh enabled"
                         << "\ngroundtruth=reused exact maximum-inner-product top-100 over each active prefix\n";
            }
            run_adaptive_clean(root, data, queries,
                               {"prd_ivf_no_reassignment", true, false, 0, 1});
            score_existing_outputs(root, root, data, queries);
            score_existing_recall_at_10(root, root / "groundtruth_stage_top100.bin",
                                        root / "recall_at_10");
            std::ofstream complete(root / "COMPLETE.txt");
            complete << "status=complete\nmethod=PRD-IVF (no local reassignment)\n"
                     << "insert_ops=" << kInsertOps << "\nquery_stages=" << kQueryOps
                     << "\nqueries_per_stage=" << kQueries
                     << "\nrecall=post-hoc exact Recall@100 and Recall@10\n";
            complete.close();
            std::cout << "COMPLETE output=" << root.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "--quake-aps-only") {
            MappedFbin data(argv[2]), queries(argv[3]);
#ifdef MSTURING_IH_SMOKE
            if (data.rows() < kInitialRows + kInsertOps * kInsertBatchRows ||
                queries.rows() < kQueries)
#else
            if (data.rows() != 10'000'000 || queries.rows() < kQueries)
#endif
                throw std::runtime_error("expected 10M data vectors and at least the configured query count");
            const std::filesystem::path root(argv[5]);
            ensure_empty_or_create(root);
            const std::filesystem::path cached_truth(argv[4]);
            if (!std::filesystem::is_regular_file(cached_truth))
                throw std::runtime_error("cached stage truth does not exist: " + cached_truth.string());
            std::filesystem::copy_file(cached_truth, root / "groundtruth_stage_top100.bin");
            {
                std::ofstream manifest(root / "manifest.txt");
                if (!manifest) throw std::runtime_error("cannot create standalone Quake APS manifest");
                manifest << "experiment=Quake Table 3 insert-heavy workload (IH), standalone Quake with APS queries\n"
                         << "dataset=" << argv[2] << "\nqueries=" << argv[3]
                         << "\nrows=10000000; dimension=" << kDimension << "; query_metric=" << kMetricName
                         << "; maintenance_metric=squared L2"
                         << "\ninitial_vectors=" << kInitialRows << "\ninitial_nlist=" << kInitialNlist
                         << "\ninsert_batch=" << kInsertBatchRows << "\ninsert_ops=" << kInsertOps
                         << "\nquery_ops=" << kQueryOps << "\nqueries_per_query_op=" << kQueries
                         << "\ntopk=" << kTopK
                         << "\nquery_policy=APS target recall 0.9; candidate_fraction=0.1; recompute_threshold=0.001; 100-query warmup before first stage"
                         << "\nLVQ=false; early_stop=false; seed=" << kSeed
                         << "\nQuake paper cost threshold tau=250ns, applied to both split and delete decisions"
                         << "\nalpha=" << kQuakeAlpha << "; min_partition_size=" << kQuakeMinPartitionSize
                         << "; query_hit_window=" << kQueryWindow
                         << "; local_refinement_radius=" << kQuakeRefinementRadius
                         << "; iterations=" << kQuakeRefinementIterations
                         << "\ngroundtruth=reused cached exact top-100 for each active prefix; actual Recall@10/100 scored post hoc\n";
            }
            AdaptiveMethodConfig quake_method{"quake", false, true};
            quake_method.quake_split_threshold_ns = 250.0;
            quake_method.quake_delete_threshold_ns = 250.0;
            quake_method.aps_queries = true;
            run_adaptive_clean(root, data, queries, quake_method);
            score_existing_outputs(root, root, data, queries);
            score_existing_recall_at_10(root, root / "groundtruth_stage_top100.bin",
                                        root / "recall_at_10");
            std::ofstream complete(root / "COMPLETE.txt");
            complete << "status=complete\nmethod=Quake only, APS target 0.9 at every query stage\n"
                     << "split_threshold_ns=250\ndelete_threshold_ns=250\n"
                     << "insert_ops=" << kInsertOps << "\nquery_stages=" << kQueryOps
                     << "\nqueries_per_stage=" << kQueries
                     << "\nquery_policy=APS target=0.9; candidate_fraction=0.1; recompute_threshold=0.001\n"
                     << "recall=post-hoc exact Recall@100 and Recall@10\n";
            complete.close();
            std::cout << "COMPLETE output=" << root.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "--triad-aps-only") {
            MappedFbin data(argv[2]), queries(argv[3]);
#ifdef MSTURING_IH_SMOKE
            if (data.rows() < kInitialRows + kInsertOps * kInsertBatchRows ||
                queries.rows() < kQueries)
#else
            if (data.rows() != 10'000'000 || queries.rows() < kQueries)
#endif
                throw std::runtime_error("expected 10M data vectors and at least the configured query count");
            const std::filesystem::path root(argv[5]);
            ensure_empty_or_create(root);
            const std::filesystem::path cached_truth(argv[4]);
            if (!std::filesystem::is_regular_file(cached_truth))
                throw std::runtime_error("cached stage truth does not exist: " + cached_truth.string());
            std::filesystem::copy_file(cached_truth, root / "groundtruth_stage_top100.bin");
            {
                std::ofstream manifest(root / "manifest.txt");
                if (!manifest) throw std::runtime_error("cannot create standalone Triad APS manifest");
                manifest << "experiment=Quake Table 3 insert-heavy workload (IH), Triad PRD-IVF with APS queries\n"
                         << "dataset=" << argv[2] << "\nqueries=" << argv[3]
                         << "\nrows=10000000; dimension=" << kDimension << "; query_metric=" << kMetricName
                         << "; maintenance_metric=squared L2"
                         << "\ninitial_vectors=" << kInitialRows << "; initial_nlist=" << kInitialNlist
                         << "\ninsert_batch=" << kInsertBatchRows << "; insert_ops=" << kInsertOps
                         << "\nquery_ops=" << kQueryOps << "; queries_per_query_op=" << kQueries
                         << "\ntopk=" << kTopK
                         << "\nquery_policy=APS target=0.9; candidate_fraction=0.1; recompute_threshold=0.001; 100-query warmup before first stage"
                         << "\nprd_threshold=0.05; drift_criterion=GlobalFitOnly; load_split_fraction=2.2"
                         << "; capacity_sqrt_n_split_multiplier=1.05; local_refinement=disabled"
                         << "; per-insert-batch global center refresh enabled"
                         << "\nLVQ=false; early_stop=false; seed=" << kSeed << '\n'
                         << "\ngroundtruth=reused cached exact top-100 for every active prefix; actual Recall@10/100 scored post hoc\n";
            }
            AdaptiveMethodConfig triad_method;
            triad_method.name = "triad_prd";
            triad_method.prd = true;
            triad_method.aps_queries = true;
            run_adaptive_clean(root, data, queries, triad_method);
            score_existing_outputs(root, root, data, queries);
            score_existing_recall_at_10(root, root / "groundtruth_stage_top100.bin", root / "recall_at_10");
            std::ofstream complete(root / "COMPLETE.txt");
            complete << "status=complete\nmethod=Triad PRD-IVF with APS target 0.9 at every query stage\n"
                     << "insert_ops=" << kInsertOps << "\nquery_stages=" << kQueryOps
                     << "\nqueries_per_stage=" << kQueries
                     << "\nquery_policy=APS target=0.9; candidate_fraction=0.1; recompute_threshold=0.001\n"
                     << "recall=post-hoc exact Recall@100 and Recall@10\n";
            complete.close();
            std::cout << "COMPLETE output=" << root.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--save-triad-initial-index") {
            MappedFbin data(argv[2]);
#ifdef MSTURING_IH_SMOKE
            if (data.rows() < kInitialRows)
#else
            if (data.rows() != 10'000'000)
#endif
                throw std::runtime_error("initial Triad snapshot expects the 10M shuffled MSTuring file");
            const std::filesystem::path index_path(argv[3]);
            if (std::filesystem::exists(index_path))
                throw std::runtime_error("refusing to overwrite existing initial index: " + index_path.string());
            if (!std::filesystem::is_directory(index_path.parent_path()))
                throw std::runtime_error("initial index parent directory does not exist: " + index_path.parent_path().string());

            AdaptiveHierarchicalConfig cfg;
            cfg.dimension = kDimension;
            cfg.initial_leaf_nlist = kInitialNlist;
            cfg.kmeans_iterations = kKmeansIterations;
            cfg.seed = kSeed;
            cfg.lvq8_storage = false;
            cfg.lvq4_storage = false;
            cfg.angular_metric = false;
            cfg.uint8_storage = false;
            cfg.automatic_maintenance = true;
            cfg.prd_threshold = 0.05f;
            cfg.drift_criterion = DriftCriterion::GlobalFitOnly;
            cfg.load_split_fraction = 2.20f;
            cfg.capacity_growth_split_factor = kApsCapacityGrowthSplitFactor;
            cfg.capacity_sqrt_n_split_multiplier = 1.05f;
            cfg.split_local_refinement_radius = 50;
            cfg.split_local_refinement_iterations = 1;

            AdaptiveHierarchicalIvf index(cfg);
            const auto build_start = std::chrono::steady_clock::now();
            index.train(data.row(0), kInitialRows);
            const double build_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - build_start).count();
            const auto save_start = std::chrono::steady_clock::now();
            index.save(index_path.string());
            const double save_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - save_start).count();
            const std::uintmax_t index_bytes = std::filesystem::file_size(index_path);
            std::ofstream manifest(index_path.string() + ".manifest.txt", std::ios::trunc);
            if (!manifest) throw std::runtime_error("cannot write initial-index manifest");
            manifest << "index=Triad-compatible shuffled MSTuring-10M initial snapshot\n"
                     << "dataset=" << argv[2] << "\nrows=" << kInitialRows
                     << "\ndimension=" << kDimension << "\nnlist=" << index.leaf_nlist()
                     << "\nshuffle_seed=1234\nkmeans_seed=" << kSeed
                     << "\nkmeans_iterations=" << kKmeansIterations
                     << "\nmetric=squared L2\nLVQ=false\nprd_threshold=0.05"
                     << "\nload_split_fraction=2.2\ncapacity_sqrt_n_split_multiplier=1.05"
                     << "\nsplit_local_refinement_radius=50\nsplit_local_refinement_iterations=1"
                     << "\ninitial_build_ms=" << std::setprecision(10) << build_ms
                     << "\nsave_ms=" << save_ms << "\nindex_bytes=" << index_bytes
                     << "\ncontains_vectors=true\nstatus=complete\n";
            std::cout << "INITIAL_INDEX_SAVED path=" << index_path.string()
                      << " bytes=" << index_bytes << " build_ms=" << build_ms
                      << " save_ms=" << save_ms << " nlist=" << index.leaf_nlist()
                      << "\n" << std::flush;
            return 0;
        }
        if ((argc == 7 || argc == 8) && std::string(argv[1]) == "--triad-ablation-final-aps") {
            MappedFbin data(argv[2]), queries(argv[3]);
#ifdef MSTURING_IH_SMOKE
            if (data.rows() < kInitialRows + kInsertOps * kInsertBatchRows ||
                queries.rows() < kQueries)
#else
            if (data.rows() != 10'000'000 || queries.rows() < kQueries)
#endif
                throw std::runtime_error("Triad ablation expects 10M shuffled vectors and at least 10K queries");

            const std::filesystem::path truth_path(argv[4]);
            const std::filesystem::path permutation_path(argv[5]);
            if (!std::filesystem::is_regular_file(truth_path) ||
                !std::filesystem::is_regular_file(permutation_path))
                throw std::runtime_error("final exact truth and shuffle permutation must both exist");

            const bool reuse_initial_index = argc == 8;
            const std::filesystem::path reusable_initial_index = reuse_initial_index
                ? std::filesystem::path(argv[6]) : std::filesystem::path{};
            const std::filesystem::path root(argv[reuse_initial_index ? 7 : 6]);
            if (reuse_initial_index) {
                if (!std::filesystem::is_regular_file(reusable_initial_index))
                    throw std::runtime_error("reusable Triad initial index does not exist: " +
                                             reusable_initial_index.string());
                const AdaptiveHierarchicalIvf initial =
                    AdaptiveHierarchicalIvf::load(reusable_initial_index.string());
                const auto& cfg = initial.config();
                if (initial.size() != kInitialRows || initial.leaf_nlist() != kInitialNlist ||
                    cfg.dimension != kDimension || cfg.lvq8_storage || cfg.lvq4_storage ||
                    cfg.angular_metric || cfg.uint8_storage ||
                    std::abs(cfg.prd_threshold - 0.05f) > 1e-6f ||
                    std::abs(cfg.load_split_fraction - 2.20f) > 1e-6f ||
                    std::abs(cfg.capacity_sqrt_n_split_multiplier - 1.05f) > 1e-6f)
                    throw std::runtime_error("reusable initial index metadata does not match the MSTuring Triad ablation");
            }
            ensure_empty_or_create(root);
            {
                std::ofstream manifest(root / "manifest.txt", std::ios::trunc);
                std::ofstream configs(root / "ablation_config.csv", std::ios::trunc);
                std::ofstream summary(root / "ablation_summary.csv", std::ios::trunc);
                if (!manifest || !configs || !summary)
                    throw std::runtime_error("cannot create Triad ablation reports");
                manifest << "experiment=Triad component ablation on shuffled MSTuring-10M; insert all, then query once\n"
                         << "dataset=" << argv[2] << "\nqueries=" << argv[3]
                         << "\noriginal_order_final_truth=" << truth_path.string()
                         << "\nshuffled_row_to_original_id_permutation=" << permutation_path.string()
                         << "\nrows=10000000; dimension=" << kDimension << "; metric=squared L2"
                         << "\nshuffle_seed=1234; initial_vectors=" << kInitialRows
                         << "; initial_nlist=" << kInitialNlist
                         << "\ninsert_batch=" << kInsertBatchRows << "; insert_ops=" << kInsertOps
                         << "\nquery_ops=1; queries_after_all_insertions=" << kQueries
                         << "\nquery_policy=APS target=0.9; candidate_fraction=0.1; recompute_threshold=0.001; warmup=100"
                         << "\ncommon_config=PRD threshold 0.05; load maintenance (merge + split) uses 2.2T; capacity split 1.05*sqrt(N) only while nlist<=sqrt(N); empty cleanup is common"
                         << "\nLVQ=false; early_stop=false; kmeans_seed=" << kSeed
                         << "\nrecall=final exact top-100 from original-order truth, IDs remapped through the fixed shuffle permutation"
                         << "\ninitial_index_reused=" << (reuse_initial_index ? "true" : "false")
                         << (reuse_initial_index ? "\ninitial_index_source=" + reusable_initial_index.string() : std::string{})
                         << "\nfinal_index_saved=false\n";
                configs << "method,prd_splits,load_splits,capacity_splits,split_local_reassignment_radius,iterations,global_center_refresh,notes\n"
                        << "triad_full_r50_i1,1,1,1,50,1,1,complete Triad\n"
                        << "triad_no_prd_r50_i1,0,1,1,50,1,1,PRD-triggered split disabled\n"
                        << "triad_no_load_r50_i1,1,0,1,50,1,1,load maintenance disabled (both merge and load split)\n"
                        << "triad_no_capacity_r50_i1,1,1,0,50,1,1,capacity-triggered split disabled\n";
                if (!reuse_initial_index)
                    configs << "triad_no_reassignment,1,1,1,0,1,1,optional split-local reassignment disabled; required split and merge routing retained\n"
                            << "triad_no_global_center_refresh_r50_i1,1,1,1,50,1,0,per-batch global center refresh disabled\n";
                summary << "method,insert_seconds,maintenance_seconds,load_splits,capacity_splits,prd_splits,merges,empty_removed,local_refinement_vectors,local_reassigned_vectors,global_center_refresh_seconds,final_nlist,aps_mean_estimated_recall,aps_mean_candidate_partitions,aps_mean_scanned_partitions,aps_mean_scanned_vectors,aps_mean_query_ms,aps_p95_query_ms,aps_query_seconds,actual_recall_at_10,actual_recall_at_100\n";
            }

            struct AblationVariant {
                const char* name;
                bool prd;
                bool load;
                bool capacity;
                std::size_t reassignment_radius;
                bool global_center_refresh;
            };
            const std::array<AblationVariant, 6> variants{{
                {"triad_full_r50_i1", true, true, true, 50, true},
                {"triad_no_prd_r50_i1", false, true, true, 50, true},
                {"triad_no_load_r50_i1", true, false, true, 50, true},
                {"triad_no_capacity_r50_i1", true, true, false, 50, true},
                {"triad_no_reassignment", true, true, true, 0, true},
                {"triad_no_global_center_refresh_r50_i1", true, true, true, 50, false}
            }};

            const auto csv_fields = [](const std::string& line) {
                std::vector<std::string> fields;
                std::stringstream stream(line);
                std::string field;
                while (std::getline(stream, field, ',')) fields.push_back(field);
                return fields;
            };
            const std::size_t variant_count = reuse_initial_index ? 4 : variants.size();
            for (std::size_t variant_index = 0; variant_index < variant_count; ++variant_index) {
                const AblationVariant& variant = variants[variant_index];
                std::cout << "ABLATION_START method=" << variant.name << "\n" << std::flush;
                AdaptiveMethodConfig method;
                method.name = variant.name;
                method.prd = true;
                method.final_query_only = true;
                method.aps_queries = true;
                method.triad_policy = true;
                method.triad_enable_prd_splits = variant.prd;
                method.triad_enable_load_splits = variant.load;
                method.triad_enable_capacity_splits = variant.capacity;
                method.triad_enable_load_merges = variant.load;
                method.split_local_refinement_radius = variant.reassignment_radius;
                method.split_local_refinement_iterations = 1;
                method.triad_global_center_refresh = variant.global_center_refresh;
                run_adaptive_clean(root, data, queries, method, reusable_initial_index);

                const std::filesystem::path method_root = root / variant.name;
                score_final_aps_recall(method_root, truth_path, permutation_path, variant.name);

                double insert_ms = 0.0, maintenance_ms = 0.0, refresh_ms = 0.0;
                std::uint64_t load_splits = 0, capacity_splits = 0, prd_splits = 0;
                std::uint64_t merges = 0, empty_removed = 0, refinement_vectors = 0;
                std::uint64_t reassigned_vectors = 0;
                std::size_t final_nlist = 0;
                {
                    std::ifstream updates(method_root / "insert_operations.csv");
                    std::string line;
                    std::getline(updates, line);
                    while (std::getline(updates, line)) {
                        const auto f = csv_fields(line);
                        if (f.size() < 29) continue;
                        insert_ms += std::stod(f[6]);
                        maintenance_ms += std::stod(f[7]);
                        final_nlist = static_cast<std::size_t>(std::stoull(f[9]));
                        load_splits += std::stoull(f[10]);
                        capacity_splits += std::stoull(f[11]);
                        prd_splits += std::stoull(f[12]);
                        merges += std::stoull(f[13]);
                        empty_removed += std::stoull(f[14]);
                        refinement_vectors += std::stoull(f[16]);
                        reassigned_vectors += std::stoull(f[17]);
                        refresh_ms += std::stod(f[26]);
                    }
                    if (!updates.eof()) throw std::runtime_error("failed reading per-batch Triad ablation log");
                }
                std::vector<std::string> aps, recall;
                {
                    std::ifstream in(method_root / "aps09_stage_summary.csv");
                    std::string line;
                    std::getline(in, line);
                    if (std::getline(in, line)) aps = csv_fields(line);
                }
                {
                    std::ifstream in(method_root / "final_aps_recall.csv");
                    std::string line;
                    std::getline(in, line);
                    if (std::getline(in, line)) recall = csv_fields(line);
                }
                if (aps.size() < 12 || recall.size() < 5)
                    throw std::runtime_error("missing final APS or exact recall summary for ablation variant");
                std::ofstream summary(root / "ablation_summary.csv", std::ios::app);
                if (!summary) throw std::runtime_error("cannot append Triad ablation summary");
                summary << variant.name << ',' << insert_ms / 1000.0 << ',' << maintenance_ms / 1000.0 << ','
                        << load_splits << ',' << capacity_splits << ',' << prd_splits << ',' << merges << ','
                        << empty_removed << ',' << refinement_vectors << ',' << reassigned_vectors << ','
                        << refresh_ms / 1000.0 << ',' << final_nlist << ',' << aps[5] << ',' << aps[6] << ','
                        << aps[7] << ',' << aps[8] << ',' << aps[9] << ',' << aps[10] << ',' << aps[11] << ','
                        << recall[3] << ',' << recall[4] << '\n' << std::flush;
                std::ofstream complete(method_root / "COMPLETE.txt", std::ios::trunc);
                complete << "status=complete\nmethod=" << variant.name
                         << "\ninsert_ops=" << kInsertOps << "\nquery_ops=1\nqueries=" << kQueries
                         << "\nquery_policy=APS target=0.9; candidate_fraction=0.1; recompute_threshold=0.001\n"
                         << "\nfinal_index_saved=false\nrecall=exact final top-100, remapped to shuffled row IDs\n";
                std::cout << "ABLATION_COMPLETE method=" << variant.name << " final_nlist=" << final_nlist
                          << " insert_s=" << insert_ms / 1000.0
                          << " maintenance_s=" << maintenance_ms / 1000.0
                          << " refresh_s=" << refresh_ms / 1000.0
                          << " recall10=" << recall[3] << " recall100=" << recall[4] << "\n" << std::flush;
            }
            std::ofstream complete(root / "COMPLETE.txt", std::ios::trunc);
            complete << "status=complete\nvariants=" << variant_count << "\ninsert_ops_per_variant=" << kInsertOps
                     << "\nquery_ops_per_variant=1\nqueries_per_query_op=" << kQueries
                     << "\nquery_after_all_insertions=true\nfinal_query=APS target 0.9\n"
                     << "\ninitial_index_reused=" << (reuse_initial_index ? "true" : "false")
                     << (reuse_initial_index ? "\ninitial_index_source=" + reusable_initial_index.string() : std::string{})
                     << "\nsummary=ablation_summary.csv\nfinal_index_saved=false\n";
            std::cout << "ABLATION_ALL_COMPLETE output=" << root.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 8 && std::string(argv[1]) == "--msturing-triad-quake-lire-full") {
            return run_msturing_triad_quake_lire_full(
                argv[2], argv[3], argv[4], argv[5], argv[6],
                static_cast<std::size_t>(std::stoull(argv[7])));
        }
        if ((argc == 6 || argc == 7 || argc == 8) && std::string(argv[1]) == "--quake-only") {
            MappedFbin data(argv[2]), queries(argv[3]);
#ifdef MSTURING_IH_SMOKE
            if (data.rows() < kInitialRows + kInsertOps * kInsertBatchRows ||
                queries.rows() < kQueries)
#else
            if (data.rows() != 10'000'000 || queries.rows() < kQueries)
#endif
                throw std::runtime_error("expected 10M data vectors and at least the configured query count");
            const std::filesystem::path root(argv[5]);
            ensure_empty_or_create(root);
            const std::filesystem::path cached_truth(argv[4]);
            const std::filesystem::path reusable_initial_index = argc == 7
                ? std::filesystem::path(argv[6]) :
                  (argc == 8 ? std::filesystem::path(argv[6]) : std::filesystem::path{});
            const std::filesystem::path final_index_output = argc == 8
                ? std::filesystem::path(argv[7]) : std::filesystem::path{};
            if (!std::filesystem::is_regular_file(cached_truth))
                throw std::runtime_error("cached stage truth does not exist: " + cached_truth.string());
            if (!reusable_initial_index.empty() &&
                !std::filesystem::is_regular_file(reusable_initial_index))
                throw std::runtime_error("reusable initial index does not exist: " + reusable_initial_index.string());
            std::filesystem::copy_file(cached_truth, root / "groundtruth_stage_top100.bin");
            {
                std::ofstream manifest(root / "manifest.txt");
                if (!manifest) throw std::runtime_error("cannot create standalone Quake manifest");
                manifest << "experiment=Quake Table 3 insert-heavy workload (IH), standalone Quake\n"
                         << "dataset=" << argv[2] << "\nqueries=" << argv[3]
                         << "\nrows=10000000; dimension=" << kDimension << "; query_metric=" << kMetricName
                         << "; maintenance_metric=squared L2"
                         << "\ninitial_vectors=" << kInitialRows << "\ninitial_nlist=" << kInitialNlist
                         << "\ninsert_batch=" << kInsertBatchRows << "\ninsert_ops=" << kInsertOps
                         << "\nquery_ops=" << kQueryOps << "\nqueries_per_query_op=" << kQueries
                         << "\ntopk=" << kTopK
                         << "\nquery_budget=floor(current_active_vectors/20), including centroid and vector distances"
                         << "\nLVQ=false; early_stop=false; seed=" << kSeed
                         << "\nQuake paper cost threshold tau=250ns, applied to both split and delete decisions"
                         << "\nalpha=" << kQuakeAlpha << "; min_partition_size=" << kQuakeMinPartitionSize
                         << "; query_hit_window=" << kQueryWindow
                         << "; local_refinement_radius=" << kQuakeRefinementRadius
                         << "; iterations=" << kQuakeRefinementIterations
                         << (reusable_initial_index.empty() ? "\ninitial_index_reused=false\n"
                                                            : "\ninitial_index_reused=true\ninitial_index_source=" + reusable_initial_index.string() + "\n")
                         << "\ngroundtruth=reused cached exact top-100 for each active prefix\n";
            }
            run_adaptive_clean(root, data, queries, {"quake", false, true},
                               reusable_initial_index, final_index_output);
            score_existing_outputs(root, root, data, queries);
            score_existing_recall_at_10(root, root / "groundtruth_stage_top100.bin",
                                        root / "recall_at_10");
            std::ofstream complete(root / "COMPLETE.txt");
            complete << "status=complete\nmethod=Quake only\n"
                     << "split_threshold_ns=250\ndelete_threshold_ns=250\n"
                     << "insert_ops=" << kInsertOps << "\nquery_stages=" << kQueryOps
                     << "\nqueries_per_stage=" << kQueries
                     << "\ninitial_index_reused=" << (!reusable_initial_index.empty())
                     << "\nfinal_index_saved=" << (!final_index_output.empty())
                     << (final_index_output.empty() ? std::string{} : "\nfinal_index=" + final_index_output.string())
                     << "\nrecall=post-hoc exact Recall@100 and Recall@10\n";
            complete.close();
            std::cout << "COMPLETE output=" << root.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "--lire-only") {
            MappedFbin data(argv[2]), queries(argv[3]);
#ifdef MSTURING_IH_SMOKE
            if (data.rows() < kInitialRows + kInsertOps * kInsertBatchRows ||
                queries.rows() < kQueries)
#else
            if (data.rows() != 10'000'000 || queries.rows() < kQueries)
#endif
                throw std::runtime_error("expected 10M data vectors and at least the configured query count");
            const std::filesystem::path root(argv[5]);
            ensure_empty_or_create(root);
            const std::filesystem::path cached_truth(argv[4]);
            if (!std::filesystem::is_regular_file(cached_truth))
                throw std::runtime_error("cached stage truth does not exist: " + cached_truth.string());
            std::filesystem::copy_file(cached_truth, root / "groundtruth_stage_top100.bin");
            {
                std::ofstream manifest(root / "manifest.txt");
                if (!manifest) throw std::runtime_error("cannot create standalone LIRE manifest");
                manifest << "experiment=Quake Table 3 insert-heavy workload (IH), standalone SPFresh-LIRE\n"
                         << "dataset=" << argv[2] << "\nqueries=" << argv[3]
                         << "\ninitial_vectors=" << kInitialRows << "\ninitial_nlist=" << kInitialNlist
                         << "\ninsert_batch=" << kInsertBatchRows << "\ninsert_ops=" << kInsertOps
                         << "\nquery_ops=" << kQueryOps << "\nqueries_per_query_op=" << kQueries
                         << "\ntopk=" << kTopK
                         << "\nquery_budget=floor(current_active_vectors/20), including centroid and vector distances"
                         << "\nLVQ=false; early_stop=false; seed=" << kSeed
                         << "\nraw_vector_storage=inline contiguous FP32 vectors in each posting"
                         << "\ngroundtruth=reused cached exact top-100 for each active prefix\n";
            }
            run_lire(root, data, queries);
            score_existing_outputs(root, root, data, queries);
            score_existing_recall_at_10(root, root / "groundtruth_stage_top100.bin",
                                        root / "recall_at_10");
            std::ofstream complete(root / "COMPLETE.txt");
            complete << "status=complete\nmethod=SPFresh-LIRE with inline raw FP32 vector storage\n"
                     << "insert_ops=" << kInsertOps << "\nquery_stages=" << kQueryOps
                     << "\nqueries_per_stage=" << kQueries
                     << "\nrecall=post-hoc exact Recall@100 and Recall@10\n";
            std::cout << "COMPLETE output=" << root.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "--lire-aps-only") {
            MappedFbin data(argv[2]), queries(argv[3]);
#ifdef MSTURING_IH_SMOKE
            if (data.rows() < kInitialRows + kInsertOps * kInsertBatchRows ||
                queries.rows() < kQueries)
#else
            if (data.rows() != 10'000'000 || queries.rows() < kQueries)
#endif
                throw std::runtime_error("expected 10M data vectors and at least the configured query count");
            const std::filesystem::path root(argv[5]);
            ensure_empty_or_create(root);
            const std::filesystem::path cached_truth(argv[4]);
            if (!std::filesystem::is_regular_file(cached_truth))
                throw std::runtime_error("cached stage truth does not exist: " + cached_truth.string());
            std::filesystem::copy_file(cached_truth, root / "groundtruth_stage_top100.bin");
            {
                std::ofstream manifest(root / "manifest.txt");
                if (!manifest) throw std::runtime_error("cannot create standalone LIRE APS manifest");
                manifest << "experiment=Quake Table 3 insert-heavy workload (IH), standalone SPFresh-LIRE with APS queries\n"
                         << "dataset=" << argv[2] << "\nqueries=" << argv[3]
                         << "\ninitial_vectors=" << kInitialRows << "; initial_nlist=" << kInitialNlist
                         << "\ninsert_batch=" << kInsertBatchRows << "; insert_ops=" << kInsertOps
                         << "\nquery_ops=" << kQueryOps << "; queries_per_query_op=" << kQueries
                         << "\ntopk=" << kTopK
                         << "\nquery_policy=APS target=0.9; candidate_fraction=0.1; recompute_threshold=0.001; 100-query warmup before first stage"
                         << "\nraw_vector_storage=inline contiguous exact FP32; LVQ=false; early_stop=false; seed=" << kSeed << '\n'
                         << "\ngroundtruth=reused cached exact top-100 for every active prefix; actual Recall@10/100 scored post hoc\n";
            }
            run_lire(root, data, queries, true);
            score_existing_outputs(root, root, data, queries);
            score_existing_recall_at_10(root, root / "groundtruth_stage_top100.bin", root / "recall_at_10");
            std::ofstream complete(root / "COMPLETE.txt");
            complete << "status=complete\nmethod=SPFresh-LIRE with APS target 0.9 at every query stage\n"
                     << "insert_ops=" << kInsertOps << "\nquery_stages=" << kQueryOps
                     << "\nqueries_per_stage=" << kQueries
                     << "\nquery_policy=APS target=0.9; candidate_fraction=0.1; recompute_threshold=0.001\n"
                     << "recall=post-hoc exact Recall@100 and Recall@10\n";
            complete.close();
            std::cout << "COMPLETE output=" << root.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 8 && std::string(argv[1]) == "--quake-lire-final-only") {
            MappedFbin data(argv[2]), queries(argv[3]);
#ifdef MSTURING_IH_SMOKE
            if (data.rows() < kInitialRows + kInsertOps * kInsertBatchRows ||
                queries.rows() < kQueries)
#else
            if (data.rows() != 10'000'000 || queries.rows() < kQueries)
#endif
                throw std::runtime_error("expected 10M shuffled FP32 vectors and at least the configured query count");
            const std::filesystem::path truth_path(argv[4]);
            const std::filesystem::path permutation_path(argv[5]);
            if (!std::filesystem::is_regular_file(truth_path) ||
                !std::filesystem::is_regular_file(permutation_path))
                throw std::runtime_error("final truth and row permutation files must exist");
            const std::filesystem::path root(argv[6]);
            ensure_empty_or_create(root);
            const std::size_t max_partition_size = static_cast<std::size_t>(std::stoull(argv[7]));
            if (max_partition_size == 0)
                throw std::invalid_argument("max_partition_size must be positive");
            {
                std::ofstream manifest(root / "manifest.txt", std::ios::trunc);
                if (!manifest) throw std::runtime_error("cannot create final-only Quake-LIRE manifest");
                manifest << "experiment=Quake source size-threshold LIRE policy on shuffled MSTuring-10M; insert all, then query once\n"
                         << "dataset=" << argv[2] << "\nqueries=" << argv[3]
                         << "\noriginal_order_final_truth=" << truth_path.string()
                         << "\nshuffled_row_to_original_id_permutation=" << permutation_path.string()
                         << "\nrows=10000000; dimension=" << kDimension << "; metric=" << kMetricName
                         << "\nshuffle_seed=1234; initial_vectors=" << kInitialRows
                         << "; initial_nlist=" << kInitialNlist
                         << "\ninsert_batch=" << kInsertBatchRows << "; insert_ops=" << kInsertOps
                         << "\nquery_ops=1; queries_after_all_insertions=" << kQueries
                         << "\ntopk=" << kTopK << "; query_budget=floor(final_indexed_vectors/20), including centroid distances"
                         << "\nmax_partition_size=" << max_partition_size
                         << "; min_partition_size=" << kQuakeMinPartitionSize
                         << "; refinement_radius=" << kQuakeRefinementRadius
                         << "; refinement_iterations=0 (one assignment pass)"
                         << "\nmaintenance=after every insert batch; no query-time maintenance"
                         << "\nLVQ=false; APS=false; early_stop=false; compression=false\n"
                         << "recall=post-hoc against exact final top-100; truth IDs remapped through the fixed shuffle permutation\n";
            }
            AdaptiveMethodConfig method;
            method.name = "quake_lire";
            method.quake_lire = true;
            method.lire_max_partition_size = max_partition_size;
            method.final_query_only = true;
            run_adaptive_clean(root, data, queries, method);
            score_shuffled_final_lire_recall(root, truth_path, permutation_path);
            std::ofstream complete(root / "COMPLETE.txt", std::ios::trunc);
            complete << "status=complete\nmethod=Quake source size-threshold LIRE policy port\n"
                     << "insert_ops=" << kInsertOps << "\nquery_ops=1\nqueries=" << kQueries
                     << "\nmax_partition_size=" << max_partition_size
                     << "\nquery_after_all_insertions=true\n"
                     << "recall=post-hoc exact Recall@100 and Recall@10, shuffled IDs remapped from original-order truth\n";
            complete.close();
            std::cout << "COMPLETE output=" << root.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 7 && std::string(argv[1]) == "--prd-final-only-shuffled") {
            MappedFbin data(argv[2]), queries(argv[3]);
#ifdef MSTURING_IH_SMOKE
            if (data.rows() < kInitialRows + kInsertOps * kInsertBatchRows ||
                queries.rows() < kQueries)
#else
            if (data.rows() != 10'000'000 || queries.rows() < kQueries)
#endif
                throw std::runtime_error("expected 10M shuffled FP32 vectors and at least the configured query count");
            const std::filesystem::path truth_path(argv[4]);
            const std::filesystem::path permutation_path(argv[5]);
            if (!std::filesystem::is_regular_file(truth_path) ||
                !std::filesystem::is_regular_file(permutation_path))
                throw std::runtime_error("final truth and row permutation files must exist");
            const std::filesystem::path root(argv[6]);
            ensure_empty_or_create(root);
            {
                std::ofstream manifest(root / "manifest.txt", std::ios::trunc);
                if (!manifest) throw std::runtime_error("cannot create final-only shuffled PRD manifest");
                manifest << "experiment=PRD-IVF on shuffled MSTuring-10M; insert all, then query once\n"
                         << "dataset=" << argv[2] << "\nqueries=" << argv[3]
                         << "\noriginal_order_final_truth=" << truth_path.string()
                         << "\nshuffled_row_to_original_id_permutation=" << permutation_path.string()
                         << "\nrows=10000000; dimension=" << kDimension << "; metric=" << kMetricName
                         << "\nshuffle_seed=1234; initial_vectors=" << kInitialRows
                         << "; initial_nlist=" << kInitialNlist
                         << "; kmeans_iterations=" << kKmeansIterations << "; kmeans_seed=" << kSeed
                         << "\ninsert_batch=" << kInsertBatchRows << "; insert_ops=" << kInsertOps
                         << "\nquery_ops=1; queries_after_all_insertions=" << kQueries
                         << "\ntopk=" << kTopK << "; fixed_query_budget=floor(final_indexed_vectors/20), including centroid distances"
                         << "\nprd_threshold=0.05; drift_criterion=GlobalFitOnly (mean SSE)"
                         << "; load_split_fraction=2.2; capacity_split=1.05*sqrt(N) only while nlist<=sqrt(N)"
                         << "\nmaintenance=after every insert batch; global center refresh from stored cluster statistics after maintenance"
                         << "\nlocal_refinement_radius=0 (disabled); vector reassignment=disabled; historical PRD reference retained"
                         << "\nLVQ=false; fixed_budget_early_stop=false; APS_target=0.9; APS_candidate_fraction=0.1; APS_recompute_threshold=0.001; APS_warmup=100"
                         << "\nrecall=post-hoc against exact final top-100; truth IDs remapped through the fixed shuffle permutation\n";
            }
            AdaptiveMethodConfig method;
            method.name = "prd_ivf_no_reassignment";
            method.prd = true;
            method.final_query_only = true;
            run_adaptive_clean(root, data, queries, method);
            score_shuffled_final_lire_recall(root, truth_path, permutation_path,
                                             "prd_ivf_no_reassignment");
            std::ofstream complete(root / "COMPLETE.txt", std::ios::trunc);
            complete << "status=complete\nmethod=PRD-IVF\ninsert_ops=" << kInsertOps
                     << "\nquery_ops=1\nqueries=" << kQueries
                     << "\nquery_after_all_insertions=true\nfinal_index_saved=false\n"
                     << "recall=post-hoc exact Recall@100 and Recall@10, shuffled IDs remapped from original-order truth\n";
            complete.close();
            std::cout << "COMPLETE output=" << root.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 9 && std::string(argv[1]) == "--prd-final-only-shuffled-refined") {
            MappedFbin data(argv[2]), queries(argv[3]);
#ifdef MSTURING_IH_SMOKE
            if (data.rows() < kInitialRows + kInsertOps * kInsertBatchRows ||
                queries.rows() < kQueries)
#else
            if (data.rows() != 10'000'000 || queries.rows() < kQueries)
#endif
                throw std::runtime_error("expected 10M shuffled FP32 vectors and at least the configured query count");
            const std::filesystem::path truth_path(argv[4]);
            const std::filesystem::path permutation_path(argv[5]);
            if (!std::filesystem::is_regular_file(truth_path) ||
                !std::filesystem::is_regular_file(permutation_path))
                throw std::runtime_error("final truth and row permutation files must exist");
            const std::filesystem::path root(argv[6]);
            ensure_empty_or_create(root);
            const std::size_t refinement_radius = static_cast<std::size_t>(std::stoull(argv[7]));
            const std::size_t refinement_iterations = static_cast<std::size_t>(std::stoull(argv[8]));
            if (refinement_radius == 0 || refinement_iterations == 0)
                throw std::invalid_argument("refinement radius and iterations must be positive");
            const std::string method_name = "prd_ivf_localref_r" +
                std::to_string(refinement_radius) + "_i" + std::to_string(refinement_iterations);
            {
                std::ofstream manifest(root / "manifest.txt", std::ios::trunc);
                if (!manifest) throw std::runtime_error("cannot create refined final-only shuffled PRD manifest");
                manifest << "experiment=PRD-IVF with local vector reassignment on shuffled MSTuring-10M; insert all, then query once\n"
                         << "dataset=" << argv[2] << "\nqueries=" << argv[3]
                         << "\noriginal_order_final_truth=" << truth_path.string()
                         << "\nshuffled_row_to_original_id_permutation=" << permutation_path.string()
                         << "\nrows=10000000; dimension=" << kDimension << "; metric=" << kMetricName
                         << "\nshuffle_seed=1234; initial_vectors=" << kInitialRows
                         << "; initial_nlist=" << kInitialNlist
                         << "; kmeans_iterations=" << kKmeansIterations << "; kmeans_seed=" << kSeed
                         << "\ninsert_batch=" << kInsertBatchRows << "; insert_ops=" << kInsertOps
                         << "\nquery_ops=1; queries_after_all_insertions=" << kQueries
                         << "\ntopk=" << kTopK << "; fixed_query_budget=floor(final_indexed_vectors/20), including centroid distances"
                         << "\nprd_threshold=0.05; drift_criterion=GlobalFitOnly (mean SSE)"
                         << "; load_split_fraction=2.2; capacity_split=1.05*sqrt(N) only while nlist<=sqrt(N)"
                         << "\nmaintenance=after every insert batch; global center refresh from stored cluster statistics after maintenance"
                         << "\nlocal_refinement_radius=" << refinement_radius
                         << " additional neighboring partitions per split-child seed; iterations=" << refinement_iterations
                         << "; one refinement pass over the union of affected partitions per maintenance batch"
                         << "\nLVQ=false; fixed_budget_early_stop=false; APS_target=0.9; APS_candidate_fraction=0.1; APS_recompute_threshold=0.001; APS_warmup=100"
                         << "\nrecall=post-hoc against exact final top-100; truth IDs remapped through the fixed shuffle permutation\n";
            }
            AdaptiveMethodConfig method;
            method.name = method_name;
            method.prd = true;
            method.split_local_refinement_radius = refinement_radius;
            method.split_local_refinement_iterations = refinement_iterations;
            method.final_query_only = true;
            run_adaptive_clean(root, data, queries, method);
            score_shuffled_final_lire_recall(root, truth_path, permutation_path, method_name);
            std::ofstream complete(root / "COMPLETE.txt", std::ios::trunc);
            complete << "status=complete\nmethod=PRD-IVF with local vector reassignment\n"
                     << "insert_ops=" << kInsertOps << "\nquery_ops=1\nqueries=" << kQueries
                     << "\nrefinement_radius=" << refinement_radius
                     << "\nrefinement_iterations=" << refinement_iterations
                     << "\nquery_after_all_insertions=true\nfinal_index_saved=false\n"
                     << "recall=post-hoc exact Recall@100 and Recall@10, shuffled IDs remapped from original-order truth\n";
            complete.close();
            std::cout << "COMPLETE output=" << root.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 7 && std::string(argv[1]) == "--quake-lire-only") {
            MappedFbin data(argv[2]), queries(argv[3]);
#ifdef MSTURING_IH_SMOKE
            if (data.rows() < kInitialRows + kInsertOps * kInsertBatchRows ||
                queries.rows() < kQueries)
#else
            if (data.rows() != 10'000'000 || queries.rows() < kQueries)
#endif
                throw std::runtime_error("expected 10M FP32 vectors and at least the configured query count");
            const std::size_t max_partition_size = static_cast<std::size_t>(std::stoull(argv[6]));
            if (max_partition_size == 0)
                throw std::invalid_argument("max_partition_size must be positive");
            const std::filesystem::path root(argv[5]);
            ensure_empty_or_create(root);
            const std::filesystem::path cached_truth(argv[4]);
            if (!std::filesystem::is_regular_file(cached_truth))
                throw std::runtime_error("cached stage truth does not exist: " + cached_truth.string());
            std::filesystem::copy_file(cached_truth, root / "groundtruth_stage_top100.bin");
            {
                std::ofstream manifest(root / "manifest.txt", std::ios::trunc);
                if (!manifest) throw std::runtime_error("cannot create Quake-LIRE manifest");
                manifest << "experiment=Quake source LIRE size-threshold policy, insert-heavy workload\n"
                         << "dataset=" << argv[2] << "\nqueries=" << argv[3]
                         << "\nrows=10000000; dimension=" << kDimension << "; metric=" << kMetricName
                         << "\ninitial_vectors=" << kInitialRows << "; initial_nlist=" << kInitialNlist
                         << "\ninsert_batch=" << kInsertBatchRows << "; insert_ops=" << kInsertOps
                         << "\nquery_ops=" << kQueryOps << "; queries_per_query_operation=" << kQueries
                         << "\ntopk=" << kTopK
                         << "\nmax_partition_size=" << max_partition_size
                         << "; min_partition_size=" << kQuakeMinPartitionSize
                         << "; refinement_radius=" << kQuakeRefinementRadius
                         << "; refinement_iterations=0 (one assignment pass)"
                         << "\nsmall partition handling=reassign vectors to nearest surviving centroids before retiring empty partition objects"
                         << "\nmaintenance frequency=after each insert batch; query batches do not trigger maintenance"
                         << "\nquery_budget=floor(dataset prefix size/20), including centroid distances"
                         << "\nAPS=false; early_stop=false; compression=false; seed=" << kSeed
                         << "\nquery_threads=" << std::max(1u, std::thread::hardware_concurrency())
                         << "\ngroundtruth=cached exact top-100 over every dataset prefix\n";
            }
            AdaptiveMethodConfig method;
            method.name = "quake_lire";
            method.quake_lire = true;
            method.lire_max_partition_size = max_partition_size;
            method.query_threads = std::max(1u, std::thread::hardware_concurrency());
            run_adaptive_clean(root, data, queries, method);
            score_existing_outputs(root, root, data, queries);
            score_existing_recall_at_10(root, root / "groundtruth_stage_top100.bin",
                                        root / "recall_at_10");
            std::ofstream complete(root / "COMPLETE.txt", std::ios::trunc);
            complete << "status=complete\nmethod=Quake source size-threshold LIRE policy port\n"
                     << "max_partition_size=" << max_partition_size
                     << "\nmin_partition_size=" << kQuakeMinPartitionSize
                     << "\nrefinement_radius=" << kQuakeRefinementRadius
                     << "\nrefinement_iterations=one assignment pass\n"
                     << "query_threads=" << method.query_threads << " (read-only parallel query execution)\n"
                     << "insert_ops=" << kInsertOps << "\nquery_stages=" << kQueryOps
                     << "\nqueries_per_stage=" << kQueries
                     << "\nrecall=post-hoc exact Recall@100 and Recall@10\n";
            complete.close();
            std::cout << "COMPLETE output=" << root.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 8 && (std::string(argv[1]) == "--prd-final-ih" ||
                           std::string(argv[1]) == "--prd-full-ih")) {
            MappedFbin data(argv[2]), queries(argv[3]);
            const bool query_every_stage = std::string(argv[1]) == "--prd-full-ih";
            const float load_split_fraction = std::stof(argv[6]);
            const float capacity_sqrt_n_split_multiplier = std::stof(argv[7]);
            run_prd_final_ih(argv[5], argv[2], argv[3], argv[4], data, queries,
                             load_split_fraction, capacity_sqrt_n_split_multiplier,
                             query_every_stage);
            return 0;
        }
        if (argc == 8 && std::string(argv[1]) == "--requery-prd-final-ih") {
            MappedFbin data(argv[3]), queries(argv[4]);
            const std::size_t budget = static_cast<std::size_t>(std::stoull(argv[7]));
            recompute_saved_prd_final_query(argv[2], argv[3], argv[4], argv[5], argv[6],
                                            budget, queries, data);
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "--prd-localref-only-u8") {
            MappedU8bin data(argv[2]), query_bytes(argv[3]);
            FloatQueryRows queries(query_bytes);
            if (data.rows() != 10'000'000 || queries.rows() != kQueries)
                throw std::runtime_error("native uint8 PRD local-refinement IH expects 10M vectors and the configured 1K query sample");
            const std::filesystem::path root(argv[4]);
            ensure_empty_or_create(root);
            const std::filesystem::path cached_truth(argv[5]);
            if (!std::filesystem::is_regular_file(cached_truth))
                throw std::runtime_error("cached stage truth does not exist: " + cached_truth.string());
            std::filesystem::copy_file(cached_truth, root / "groundtruth_stage_top100.bin");
            {
                std::ofstream manifest(root / "manifest.txt");
                if (!manifest) throw std::runtime_error("cannot create native uint8 PRD local-refinement manifest");
                manifest << "experiment=YFCC-10M insert-heavy workload (IH), PRD-IVF local-refinement variant\n"
                         << "dataset=" << argv[2] << "\nqueries=" << argv[3]
                         << "\nrows=10000000; dimension=" << kDimension << "; query_metric=" << kMetricName
                         << "; maintenance_metric=squared L2"
                         << "\nsource_vectors=uint8; index_vector_payload=exact uint8 bytes"
                         << "\ncentroids=FP32; sufficient_statistics=FP64; no FP32 copy of base vectors"
                         << "\ninitial_vectors=" << kInitialRows << "; initial_nlist=" << kInitialNlist
                         << "\ninsert_batch=" << kInsertBatchRows << "; insert_ops=" << kInsertOps
                         << "\nquery_ops=" << kQueryOps << "; queries_per_query_op=" << kQueries
                         << "\ntopk=" << kTopK
                         << "\nquery_budget=floor(current_active_vectors/20), including centroid and vector distances"
                         << "\nLVQ=false; early_stop=false; seed=" << kSeed
                         << "\nmethod=PRD-IVF; local_refinement_radius=25; iterations=1"
                         << "\ngroundtruth=reused exact top-100 over each active prefix\n";
            }
            run_adaptive_clean(root, data, queries,
                               {"prd_ivf_localref_r25_i1", true, false, 25, 1});
            score_existing_outputs(root, root, data, queries);
            score_existing_recall_at_10(root, root / "groundtruth_stage_top100.bin",
                                        root / "recall_at_10");
            std::ofstream complete(root / "COMPLETE.txt");
            complete << "status=complete\nmethod=PRD-IVF local reassignment, native uint8 payload\n"
                     << "local_refinement_radius=25\niterations=1\n"
                     << "insert_ops=" << kInsertOps << "\nquery_stages=" << kQueryOps
                     << "\nqueries_per_stage=" << kQueries
                     << "\nrecall=post-hoc exact Recall@100 and Recall@10\n";
            complete.close();
            std::cout << "COMPLETE output=" << root.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "--prd-localref-only") {
            MappedFbin data(argv[2]), queries(argv[3]);
#ifdef MSTURING_IH_SMOKE
            if (data.rows() < kInitialRows + kInsertOps * kInsertBatchRows ||
                queries.rows() < kQueries)
#else
            if (data.rows() != 10'000'000 || queries.rows() < kQueries)
#endif
                throw std::runtime_error("expected 10M data vectors and at least the configured query count");
            const std::filesystem::path root(argv[4]);
            ensure_empty_or_create(root);
            const std::filesystem::path cached_truth(argv[5]);
            if (!std::filesystem::is_regular_file(cached_truth))
                throw std::runtime_error("cached stage truth does not exist: " + cached_truth.string());
            std::filesystem::copy_file(cached_truth, root / "groundtruth_stage_top100.bin");
            {
                std::ofstream manifest(root / "manifest.txt");
                if (!manifest) throw std::runtime_error("cannot create PRD local-reassignment manifest");
                manifest << "experiment=Quake Table 3 insert-heavy workload (IH), PRD local-reassignment variant\n"
                         << "dataset=" << argv[2] << "\nqueries=" << argv[3]
                         << "\ninitial_vectors=" << kInitialRows << "\ninitial_nlist=" << kInitialNlist
                         << "\ninsert_batch=" << kInsertBatchRows << "\ninsert_ops=" << kInsertOps
                         << "\nquery_ops=" << kQueryOps << "\nqueries_per_query_op=" << kQueries
                         << "\ntopk=" << kTopK
                         << "\nquery_budget=floor(current_active_vectors/20), including centroid and vector distances"
                         << "\nLVQ=false; early_stop=false; seed=" << kSeed
                         << "\nmethod=PRD-IVF; local_refinement_radius=25; iterations=1"
                         << "\ngroundtruth=reused cached exact top-100 for each active prefix\n";
            }
            run_adaptive_clean(root, data, queries,
                               {"prd_ivf_localref_r25_i1", true, false, 25, 1});
            score_existing_outputs(root, root, data, queries);
            score_existing_recall_at_10(root, root / "groundtruth_stage_top100.bin",
                                        root / "recall_at_10");
            std::ofstream complete(root / "COMPLETE.txt");
            complete << "status=complete\nmethod=PRD-IVF with local refinement radius 25, one iteration\n"
                     << "insert_ops=" << kInsertOps << "\nquery_stages=" << kQueryOps
                     << "\nqueries_per_stage=" << kQueries
                     << "\nrecall=post-hoc exact Recall@100 and Recall@10\n";
            std::cout << "COMPLETE output=" << root.string() << '\n' << std::flush;
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--check-prd-index") {
            AdaptiveHierarchicalIvf index = AdaptiveHierarchicalIvf::load(argv[2]);
            MappedFbin queries(argv[3]);
            if (queries.rows() == 0 || index.config().dimension != kDimension)
                throw std::runtime_error("index/query dimensions do not match");
            const auto found = index.search(queries.row(0), 10);
            if (found.empty()) throw std::runtime_error("loaded index returned no results");
            std::cout << "INDEX_LOAD_OK active=" << index.size()
                      << " nlist=" << index.leaf_nlist()
                      << " prd_threshold=" << index.config().prd_threshold
                      << " capacity_growth_split_factor="
                      << index.config().capacity_growth_split_factor
                      << " probe_results=" << found.size() << '\n' << std::flush;
            return 0;
        }
        if (argc == 5 && std::string(argv[1]) == "--prd-aps-initial") {
            MappedFbin data(argv[2]), queries(argv[3]);
            run_prd_aps_initial(argv[4], argv[2], argv[3], data, queries);
            return 0;
        }
        if (argc == 7 && std::string(argv[1]) == "--prd-aps-full-build") {
            MappedFbin data(argv[2]), queries(argv[3]);
            const std::size_t nlist = static_cast<std::size_t>(std::stoull(argv[6]));
            run_prd_aps_full_build(argv[5], argv[2], argv[3], argv[4],
                                   data, queries, nlist);
            return 0;
        }
        if (argc == 7 && std::string(argv[1]) == "--prd-full-build-only") {
            MappedFbin data(argv[2]), queries(argv[3]);
            const std::size_t nlist = static_cast<std::size_t>(std::stoull(argv[6]));
            run_prd_aps_full_build(argv[5], argv[2], argv[3], argv[4],
                                   data, queries, nlist, false);
            return 0;
        }
        if (argc == 7 && std::string(argv[1]) == "--prd-full-training-build-only") {
            MappedFbin data(argv[2]), queries(argv[3]);
            const std::size_t nlist = static_cast<std::size_t>(std::stoull(argv[6]));
            run_prd_aps_full_build(argv[5], argv[2], argv[3], argv[4],
                                   data, queries, nlist, false, 1.0);
            return 0;
        }
        if (argc == 9 && std::string(argv[1]) == "--prd-sample-training-build-only") {
            MappedFbin data(argv[2]), queries(argv[3]);
            const std::size_t nlist = static_cast<std::size_t>(std::stoull(argv[6]));
            const double training_fraction = std::stod(argv[7]);
            const std::uint32_t training_seed =
                static_cast<std::uint32_t>(std::stoul(argv[8]));
            run_prd_aps_full_build(argv[5], argv[2], argv[3], argv[4],
                                   data, queries, nlist, false, training_fraction,
                                   training_seed);
            return 0;
        }
        if (argc == 9 && std::string(argv[1]) == "--fixed-nprobe") {
            MappedFbin queries(argv[5]);
            const std::size_t nprobe = static_cast<std::size_t>(std::stoull(argv[8]));
            run_fixed_nprobe_query(argv[7], argv[2], argv[3], argv[4], argv[5], argv[6],
                                   nprobe, queries);
            return 0;
        }
        if (argc == 10 && std::string(argv[1]) == "--fixed-nprobe-sweep-sample") {
            MappedFbin queries(argv[5]);
            const std::size_t sample_count = static_cast<std::size_t>(std::stoull(argv[8]));
            const std::uint32_t sample_seed = static_cast<std::uint32_t>(std::stoul(argv[9]));
            const std::vector<std::size_t> nprobes{16, 32, 64};
            run_fixed_nprobe_sweep_sample(argv[7], argv[2], argv[3], argv[4], argv[5], argv[6],
                                          sample_count, sample_seed, nprobes, queries);
            return 0;
        }
        if ((argc == 9 || argc == 10) && std::string(argv[1]) == "--fixed-budget") {
            MappedFbin queries(argv[5]);
            const std::size_t budget = static_cast<std::size_t>(std::stoull(argv[8]));
            const std::size_t query_threads = argc == 10
                ? static_cast<std::size_t>(std::stoull(argv[9])) : 1;
            run_fixed_budget_query(argv[7], argv[2], argv[3], argv[4], argv[5], argv[6],
                                   budget, queries, query_threads);
            return 0;
        }
        if ((argc == 7 || argc == 8) && std::string(argv[1]) == "--fixed-budget-snapshot") {
            const std::size_t budget = static_cast<std::size_t>(std::stoull(argv[6]));
            const std::size_t query_threads = argc == 8
                ? static_cast<std::size_t>(std::stoull(argv[7])) : 1;
            run_fixed_budget_snapshot_eval(argv[5], argv[2], argv[3], argv[4], budget, query_threads);
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--cluster-size-summary") {
            write_cluster_size_summary(argv[3], argv[2]);
            return 0;
        }
        if (argc == 11 && std::string(argv[1]) == "--fixed-nprobe-single-sample") {
            MappedFbin queries(argv[5]);
            const std::size_t sample_count = static_cast<std::size_t>(std::stoull(argv[8]));
            const std::uint32_t sample_seed = static_cast<std::uint32_t>(std::stoul(argv[9]));
            const std::size_t nprobe = static_cast<std::size_t>(std::stoull(argv[10]));
            const std::vector<std::size_t> nprobes{nprobe};
            run_fixed_nprobe_sweep_sample(argv[7], argv[2], argv[3], argv[4], argv[5], argv[6],
                                          sample_count, sample_seed, nprobes, queries);
            return 0;
        }
        if (argc >= 11 && std::string(argv[1]) == "--aps-sample") {
            MappedFbin queries(argv[5]);
            const std::size_t sample_count = static_cast<std::size_t>(std::stoull(argv[8]));
            const std::uint32_t sample_seed = static_cast<std::uint32_t>(std::stoul(argv[9]));
            const std::size_t candidate_partition_limit = static_cast<std::size_t>(std::stoull(argv[10]));
            run_aps_sample_query(argv[7], argv[2], argv[3], argv[4], argv[5], argv[6],
                                 sample_count, sample_seed, candidate_partition_limit, queries);
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "--prd-aps") {
            MappedFbin data(argv[2]), queries(argv[3]);
            run_prd_aps(argv[5], argv[2], argv[3], argv[4], data, queries);
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "--prd-aps-final") {
            MappedFbin data(argv[2]), queries(argv[3]);
            run_prd_aps(argv[5], argv[2], argv[3], argv[4], data, queries, true);
            return 0;
        }
        if (argc == 9 && std::string(argv[1]) == "--prd-aps-final-local-refinement") {
            MappedFbin data(argv[2]), queries(argv[3]);
            const std::size_t radius = static_cast<std::size_t>(std::stoull(argv[7]));
            const std::size_t iterations = static_cast<std::size_t>(std::stoull(argv[8]));
            run_prd_aps(argv[6], argv[2], argv[3], argv[4], data, queries, true,
                        argv[5], radius, iterations);
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--score-final") {
            score_previous_final_recall(argv[2], argv[3]);
            std::cout << "FINAL_RECALL output=" << (std::filesystem::path(argv[2]) /
                "previous_experiment_final_recall.csv").string() << '\n';
            return 0;
        }
        if (argc == 5 && std::string(argv[1]) == "--score-k10") {
            score_existing_recall_at_10(argv[2], argv[3], argv[4]);
            std::cout << "RECALL_AT_10_COMPLETE output=" << argv[4] << '\n' << std::flush;
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "--score-existing") {
            MappedFbin data(argv[4]), queries(argv[5]);
            if (data.rows() < kInitialRows + kInsertOps * kInsertBatchRows || queries.rows() < kQueries)
                throw std::runtime_error("source data/query files are too small for this experiment configuration");
            score_existing_outputs(argv[2], argv[3], data, queries);
            std::cout << "RECALL_COMPLETE output=" << argv[3] << '\n' << std::flush;
            return 0;
        }
#ifdef DEEP1M_FREEWAYML_ABLATION
        const std::string triad_nine_mode = argc > 1 ? std::string(argv[1]) : std::string();
        const bool triad_nine_prd_splits_enabled =
            triad_nine_mode == "--deep1m-freewayml-triad-nine-fixed50k";
        const bool triad_nine_prd_splits_disabled =
            triad_nine_mode == "--deep1m-freewayml-triad-nine-fixed50k-no-prd";
        const bool triad_nine_args_valid = triad_nine_prd_splits_enabled
            ? (argc == 5 || argc == 7 || argc == 8)
            : (argc == 5 || argc == 6 || argc == 7);
        if (triad_nine_args_valid &&
            (triad_nine_prd_splits_enabled || triad_nine_prd_splits_disabled)) {
            const std::filesystem::path data_root(argv[2]);
            const std::filesystem::path initial_index_path(argv[3]);
            const std::filesystem::path output_root(argv[4]);
            const float triad_prd_threshold = triad_nine_prd_splits_enabled && argc >= 7
                ? std::stof(argv[5]) : 0.05f;
            const std::size_t query_threads = triad_nine_prd_splits_enabled
                ? (argc >= 7 ? static_cast<std::size_t>(std::stoull(argv[6])) : 1)
                : (argc >= 6 ? static_cast<std::size_t>(std::stoull(argv[5])) : 1);
            const std::size_t query_budget_divisor = triad_nine_prd_splits_enabled
                ? (argc == 8 ? static_cast<std::size_t>(std::stoull(argv[7])) : 0)
                : (argc == 7 ? static_cast<std::size_t>(std::stoull(argv[6])) : 0);
            if ((triad_nine_prd_splits_enabled && !(triad_prd_threshold > 0.0f)) || query_threads == 0 ||
                ((triad_nine_prd_splits_enabled && argc == 8) ||
                 (triad_nine_prd_splits_disabled && argc == 7)) && query_budget_divisor == 0)
                throw std::invalid_argument("invalid PRD threshold, query thread count, or budget divisor");
            ensure_empty_or_create(output_root);
            if (!std::filesystem::is_regular_file(initial_index_path))
                throw std::runtime_error("shared initial DEEP-1M index does not exist: " + initial_index_path.string());
            constexpr std::array<const char*, 9> case_names{
                "G", "A1", "A2", "B1", "B2", "B3", "B4", "C1", "C2"};
            MappedFbin reference_data(data_root / "G" / "base.fbin");
            if (reference_data.rows() != kInitialRows + kInsertOps * kInsertBatchRows)
                throw std::runtime_error("DEEP-1M FreewayML scenario vectors must contain 1M rows");
            const std::size_t shared_prefix_bytes = kInitialRows * kDimension * sizeof(float);
            {
                const AdaptiveHierarchicalIvf initial = AdaptiveHierarchicalIvf::load(initial_index_path.string());
                if (initial.size() != kInitialRows || initial.leaf_nlist() != kInitialNlist ||
                    initial.config().dimension != kDimension || initial.config().lvq8_storage ||
                    initial.config().lvq4_storage || initial.config().angular_metric ||
                    initial.config().uint8_storage)
                    throw std::runtime_error("shared initial index metadata does not match DEEP-1M Triad run");
            }
            {
                std::ofstream manifest(output_root / "manifest.txt", std::ios::trunc);
                if (!manifest) throw std::runtime_error("cannot create Triad manifest");
                manifest << "experiment=DEEP-1M FreewayML-inspired nine-scenario Triad; matched staged queries\n"
                         << "data_root=" << data_root.string() << "\nshared_initial_index=" << initial_index_path.string() << '\n'
                         << "cases=G,A1,A2,B1,B2,B3,B4,C1,C2\n"
                         << "initial_vectors=500000; initial_nlist=512; inserts=100 batches x 5000; final_vectors=1000000\n"
                         << "queries=stage 000 initial index, then each insertion batch; 1000 matched queries at every stage\n"
                         << (query_budget_divisor > 0
                             ? "query_budget=floor(current active vectors/" + std::to_string(query_budget_divisor) + ") exact total distance computations per query, including all live-centroid and vector distances\n"
                             : "query_budget=50000 exact total distance computations per query, including all live-centroid and vector distances\n")
                         << (query_budget_divisor > 0
                             ? "query_budget_examples=stage 0: 5000; final stage: 10000\n" : "")
                         << "query_threads=" << query_threads << " (parallel read-only fixed-budget queries)\n"
                          << (triad_nine_prd_splits_enabled
                              ? "PRD=" + std::to_string(triad_prd_threshold) + "; drift criterion=GlobalFitOnly; PRD-marked partitions split directly without recenter attempt\n"
                              : "geometric_maintenance=disabled; PRD-triggered split checks are off\n")
                         << "load_split=leaf size > 2.2*T; merge=leaf size < 0.2*sqrt(current active N)\n"
                         << "capacity_split=leaf size > 1.05*sqrt(current active N), enabled only while live nlist <= sqrt(current active N)\n"
                         << "split_local_reassignment=radius 50, one pass; global center refresh=after each insertion batch from stored per-cluster mean statistics\n"
                         << "historical residual reference retained; LVQ=false; early_stop=false; metric=squared L2 on unit-normalized vectors; seed=1234\n"
                         << "initial_index_reused=true; final_indexes_saved=false; warmup=none (matches aligned Quake/LIRE fixed-budget run)\n";
            }
            for (const char* case_name : case_names) {
                const std::filesystem::path case_dir = data_root / case_name;
                MappedFbin data(case_dir / "base.fbin");
                if (data.rows() != kInitialRows + kInsertOps * kInsertBatchRows ||
                    !std::filesystem::is_regular_file(data_root / "truth" / case_name / "groundtruth.npz"))
                    throw std::runtime_error(std::string("missing or malformed DEEP-1M scenario inputs for ") + case_name);
                if (std::memcmp(reference_data.row(0), data.row(0), shared_prefix_bytes) != 0)
                    throw std::runtime_error(std::string("initial 500K vectors differ in ") + case_name +
                        "; refusing to reuse the shared initial index");
                std::vector<std::filesystem::path> stage_files;
                stage_files.reserve(kInsertOps + 1);
                for (std::size_t stage = 0; stage <= kInsertOps; ++stage) {
                    std::ostringstream name;
                    name << "query_" << std::setw(3) << std::setfill('0') << stage << ".fbin";
                    stage_files.push_back(case_dir / name.str());
                }
                for (const auto& path : stage_files)
                    if (!std::filesystem::is_regular_file(path))
                        throw std::runtime_error("missing stage query file: " + path.string());
                MappedFbin initial_queries(stage_files.front());
                const std::filesystem::path case_output = output_root / case_name;
                std::filesystem::create_directories(case_output);
                AdaptiveMethodConfig triad;
                triad.name = "triad";
                triad.prd = true;
                triad.triad_policy = true;
                triad.triad_enable_prd_splits = triad_nine_prd_splits_enabled;
                triad.include_initial_query_stage = true;
                triad.query_stages_override = kInsertOps + 1;
                triad.insert_ops_per_query_override = 1;
                if (query_budget_divisor > 0)
                    triad.query_budget_divisor_override = query_budget_divisor;
                else
                    triad.fixed_query_budget_override = 50'000;
                triad.query_stage_files = std::move(stage_files);
                triad.query_threads = query_threads;
                triad.prd_threshold_override = triad_prd_threshold;
                triad.short_list_sqrt_n_fraction_override = 0.20f;
                triad.split_local_refinement_radius = 50;
                triad.split_local_refinement_iterations = 1;
                triad.override_split_local_refinement = true;
                triad.triad_global_center_refresh = true;
                std::cout << "TRIAD_NINE_START case=" << case_name
                          << " budget=" << (query_budget_divisor > 0
                              ? "floor(active/" + std::to_string(query_budget_divisor) + ")"
                              : "50000") << "\n" << std::flush;
                run_adaptive_clean(case_output, data, initial_queries, triad, initial_index_path);
                std::ofstream case_manifest(case_output / "triad" / "manifest.txt", std::ios::trunc);
                if (!case_manifest) throw std::runtime_error("cannot write per-case Triad manifest");
                case_manifest << "case=" << case_name << "\nmethod=Triad\n"
                              << "initial_index=" << initial_index_path.string() << "\n"
                              << "initial_vectors=500000; initial_nlist=512; batches=100; batch_size=5000\n"
                              << "query_stages=101 including initial stage; queries_per_stage=1000; "
                              << (query_budget_divisor > 0
                                  ? "budget=floor(current_active_vectors/" + std::to_string(query_budget_divisor) + ") (stage 0=5000; final=10000)\n"
                                  : "fixed_budget=50000\n")
                               << (triad_nine_prd_splits_enabled
                                   ? "PRD=" + std::to_string(triad_prd_threshold) + "; "
                                   : "geometric_maintenance=disabled; ")
                               << "load_split=2.2T; merge_cutoff=0.2*sqrt(N); capacity_split=1.05*sqrt(N) while nlist<=sqrt(N)\n"
                              << "local_reassignment_radius=50; iterations=1; global_center_refresh=each batch from stored mean stats\n"
                              << "query_threads=" << query_threads << "\n"
                              << "LVQ=false; early_stop=false; metric=squared L2; warmup=none\n";
                std::cout << "TRIAD_NINE_CASE_COMPLETE case=" << case_name << "\n" << std::flush;
            }
            std::ofstream complete(output_root / "COMPLETE.txt", std::ios::trunc);
            if (!complete) throw std::runtime_error("cannot write Triad completion marker");
             complete << "status=complete\ncases=9\nquery_stages_per_case=101\nqueries_per_stage=1000\n"
                     << (query_budget_divisor > 0
                         ? "query_budget=floor(current_active_vectors/" + std::to_string(query_budget_divisor) + "); stage0=5000; final=10000\n"
                         : "fixed_distance_budget_per_query=50000\n")
                      << "query_threads=" << query_threads << '\n'
                      << (triad_nine_prd_splits_enabled
                          ? "PRD=" + std::to_string(triad_prd_threshold) + '\n'
                          : "geometric_maintenance=disabled\n")
                      << "initial_index_reused=true\nfinal_indexes_saved=false\n";
            std::cout << "TRIAD_NINE_ALL_COMPLETE output=" << output_root.string() << "\n" << std::flush;
            return 0;
        }
        if (argc == 6 && std::string(argv[1]) == "--deep1m-freewayml-component-ablation-rerun-no-reassignment") {
            const std::filesystem::path data_root(argv[2]);
            const std::filesystem::path shared_initial_index(argv[3]);
            const std::filesystem::path final_root(argv[4]);
            const std::filesystem::path checkpoint_root(argv[5]);
            if (!std::filesystem::is_directory(final_root) || !std::filesystem::is_directory(checkpoint_root))
                throw std::runtime_error("existing final and checkpoint result roots are required for targeted rerun");
            constexpr std::array<const char*, 9> all_cases{
                "G", "A1", "A2", "B1", "B2", "B3", "B4", "C1", "C2"};
            constexpr std::array<const char*, 4> checkpoint_cases{"G", "B1", "B3", "C2"};
            constexpr std::array<const char*, 5> query_names{
                "query_020.fbin", "query_040.fbin", "query_060.fbin", "query_080.fbin", "query_100.fbin"};
            MappedFbin reference_data(data_root / "B1" / "base.fbin");
            const std::size_t initial_bytes = kInitialRows * kDimension * sizeof(float);
            auto make_method = [] {
                AdaptiveMethodConfig method;
                method.name = "triad_no_reassignment";
                method.prd = true;
                method.aps_queries = true;
                method.triad_policy = true;
                method.triad_enable_prd_splits = true;
                method.triad_enable_load_splits = true;
                method.triad_enable_capacity_splits = true;
                method.triad_enable_load_merges = true;
                method.split_local_refinement_radius = 0;
                method.split_local_refinement_iterations = 1;
                method.triad_global_center_refresh = true;
                method.override_split_local_refinement = true;
                return method;
            };
            {
                std::ofstream note(final_root / "no_reassignment_rerun.txt", std::ios::trunc);
                note << "Rerun reason: loaded snapshots retain their serialized local-refinement radius. The runner now overrides it to 0 after loading the common initial index. Only triad_no_reassignment was rerun; all other arms are unchanged.\n"
                     << "effective_split_local_refinement_radius=0\nmandatory_merge_vector_routing=enabled\n";
            }
            for (const char* case_name : all_cases) {
                MappedFbin data(data_root / case_name / "base.fbin");
                MappedFbin queries(data_root / case_name / "query_100.fbin");
                if (data.rows() != kInitialRows + kInsertOps * kInsertBatchRows || queries.rows() != kQueries ||
                    std::memcmp(reference_data.row(0), data.row(0), initial_bytes) != 0)
                    throw std::runtime_error(std::string("invalid/common-initial-data mismatch in ") + case_name);
                const auto case_root = final_root / case_name;
                std::filesystem::create_directories(case_root);
                std::cout << "RERUN_NO_REASSIGNMENT_FINAL_START case=" << case_name << "\n" << std::flush;
                auto method = make_method();
                run_adaptive_clean(case_root, data, queries, method, shared_initial_index);
                std::ofstream manifest(case_root / method.name / "manifest.txt", std::ios::trunc);
                manifest << "case=" << case_name << "\nmethod=triad_no_reassignment\n"
                         << "initial_index=" << shared_initial_index.string() << "\ninitial_nlist=512\n"
                         << "insert_batches=100; batch_size=5000\nPRD=0.05; load_threshold=2.2T; capacity=1.05*sqrt(N) while nlist<=sqrt(N)\n"
                         << "enabled=PRD,load split,load merge,capacity split,global center refresh; split-local reassignment=disabled\n"
                         << "split_local_refinement_radius=0; mandatory merge vector routing=enabled\n"
                         << "query=final 1000 stage-matched queries; APS=0.9; candidate_fraction=0.1; recompute_threshold=0.001; warmup=100\n"
                         << "LVQ=false; early_stop=false; metric=squared L2\n";
            }
            for (const char* case_name : checkpoint_cases) {
                MappedFbin data(data_root / case_name / "base.fbin");
                std::vector<std::filesystem::path> stage_files;
                for (const char* query_name : query_names)
                    stage_files.push_back(data_root / case_name / query_name);
                const auto case_root = checkpoint_root / case_name;
                std::filesystem::create_directories(case_root);
                std::cout << "RERUN_NO_REASSIGNMENT_CHECKPOINT_START case=" << case_name << "\n" << std::flush;
                auto method = make_method();
                method.query_stages_override = query_names.size();
                method.insert_ops_per_query_override = kInsertOps / query_names.size();
                method.query_stage_files = stage_files;
                MappedFbin final_queries(stage_files.back());
                run_adaptive_clean(case_root, data, final_queries, method, shared_initial_index);
                std::ofstream manifest(case_root / method.name / "manifest.txt", std::ios::trunc);
                manifest << "case=" << case_name << "\nmethod=triad_no_reassignment\n"
                         << "initial_index=" << shared_initial_index.string() << "\ninitial_nlist=512\n"
                         << "insert_batches=100; batch_size=5000; measured_checkpoints=20,40,60,80,100\n"
                         << "enabled=PRD,load split,load merge,capacity split,global center refresh; split-local reassignment=disabled\n"
                         << "split_local_refinement_radius=0; mandatory merge vector routing=enabled\n"
                         << "query=1000 matching queries per checkpoint; APS=0.9; candidate_fraction=0.1; recompute_threshold=0.001; warmup=100 before first checkpoint\n"
                         << "LVQ=false; early_stop=false; metric=squared L2\n";
            }
            std::cout << "RERUN_NO_REASSIGNMENT_COMPLETE\n" << std::flush;
            return 0;
        }
        if (argc == 5 && std::string(argv[1]) == "--deep1m-freewayml-component-ablation-final-aps") {
            const std::filesystem::path data_root(argv[2]);
            const std::filesystem::path shared_initial_index(argv[3]);
            const std::filesystem::path root(argv[4]);
            ensure_empty_or_create(root);
            if (!std::filesystem::is_regular_file(shared_initial_index))
                throw std::runtime_error("shared initial DEEP-1M index does not exist: " + shared_initial_index.string());

            constexpr std::array<const char*, 9> case_names{
                "G", "A1", "A2", "B1", "B2", "B3", "B4", "C1", "C2"};
            MappedFbin reference_data(data_root / "B1" / "base.fbin");
            if (reference_data.rows() != kInitialRows + kInsertOps * kInsertBatchRows)
                throw std::runtime_error("DEEP-1M scenario data must contain 1M rows");
            const std::size_t initial_bytes = kInitialRows * kDimension * sizeof(float);
            {
                auto initial = AdaptiveHierarchicalIvf::load(shared_initial_index.string());
                if (initial.size() != kInitialRows || initial.leaf_nlist() != kInitialNlist ||
                    initial.config().dimension != kDimension || initial.config().lvq8_storage ||
                    initial.config().lvq4_storage || initial.config().angular_metric ||
                    initial.config().uint8_storage)
                    throw std::runtime_error("shared initial index metadata does not match the component ablation");
            }

            const std::filesystem::path matched_index_path =
                root / "shared_initial_500k_nlist1001_seed1234.index.bin";
            AdaptiveHierarchicalConfig matched_cfg;
            matched_cfg.dimension = kDimension;
            matched_cfg.initial_leaf_nlist = 1001;
            matched_cfg.kmeans_iterations = kKmeansIterations;
            matched_cfg.seed = kSeed;
            matched_cfg.lvq8_storage = false;
            matched_cfg.lvq4_storage = false;
            matched_cfg.angular_metric = false;
            matched_cfg.uint8_storage = false;
            matched_cfg.automatic_maintenance = false;
            matched_cfg.prd_threshold = 0.05f;
            matched_cfg.drift_criterion = DriftCriterion::GlobalFitOnly;
            matched_cfg.load_split_fraction = 2.20f;
            matched_cfg.capacity_sqrt_n_split_multiplier = 1.05f;
            matched_cfg.split_local_refinement_radius = 0;
            AdaptiveHierarchicalIvf matched_index(matched_cfg);
            const auto matched_build_start = std::chrono::steady_clock::now();
            matched_index.train(reference_data.row(0), kInitialRows);
            const double matched_build_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - matched_build_start).count();
            matched_index.save(matched_index_path.string());

            struct ComponentVariant {
                const char* name;
                bool triad;
                bool prd;
                bool load_splits;
                bool capacity;
                bool load_merges;
                bool local_reassignment;
                bool global_center_refresh;
                std::size_t initial_nlist;
                const char* description;
            };
            const std::array<ComponentVariant, 10> variants{{
                {"frozen_ivf_nlist512", false, false, false, false, false, false, false,
                    kInitialNlist, "no maintenance; shared 512-center initial index"},
                {"triad_full_r50_i1", true, true, true, true, true, true, true,
                    kInitialNlist, "complete Triad"},
                {"triad_no_prd_r50_i1", true, false, true, true, true, true, true,
                    kInitialNlist, "PRD-triggered splits disabled"},
                {"triad_no_load_r50_i1", true, true, false, true, false, true, true,
                    kInitialNlist, "load splits and load merges disabled"},
                {"triad_no_load_split_r50_i1", true, true, false, true, true, true, true,
                    kInitialNlist, "load splits disabled; load merges retained"},
                {"triad_no_merge_r50_i1", true, true, true, true, false, true, true,
                    kInitialNlist, "load merges disabled; load splits retained"},
                {"triad_no_capacity_r50_i1", true, true, true, false, true, true, true,
                    kInitialNlist, "capacity-triggered splits disabled"},
                {"triad_no_reassignment", true, true, true, true, true, false, true,
                    kInitialNlist, "split-local reassignment disabled; mandatory merge routing retained"},
                {"triad_no_global_center_refresh_r50_i1", true, true, true, true, true, true, false,
                    kInitialNlist, "per-batch global center refresh disabled"},
                {"frozen_ivf_nlist1001", false, false, false, false, false, false, false,
                    1001, "no maintenance; fixed 1001-center static control, not a per-scenario matched count"}
            }};

            {
                std::ofstream manifest(root / "manifest.txt", std::ios::trunc);
                if (!manifest) throw std::runtime_error("cannot create component ablation manifest");
                manifest << "experiment=DEEP-1M FreewayML-inspired nine-scenario Triad component ablation; final query only\n"
                         << "data_root=" << data_root.string() << "\nshared_initial_index=" << shared_initial_index.string() << "\n"
                         << "initial_vectors=500000; insert_batches=100; insert_batch=5000; dimension=96\n"
                         << "query=final 1000 queries after all insertions; APS target=0.9; candidate_fraction=0.1; recompute_threshold=0.001; warmup=100 per method excluded\n"
                         << "common_config=PRD=0.05; load threshold=2.2T; capacity=1.05*sqrt(N) only while nlist<=sqrt(N); local refinement radius=50; iterations=1; global mean refresh after each batch\n"
                         << "baseline=frozen IVF using shared 512-center index; secondary cluster-count control=frozen IVF initialized with 1001 centers\n"
                         << "cases=G,A1,A2,B1,B2,B3,B4,C1,C2\nLVQ=false; early_stop=false; metric=squared L2\n"
                         << "matched_nlist_initial_build_ms=" << std::setprecision(10) << matched_build_ms
                         << "\nmatched_nlist_index=" << matched_index_path.string() << "\n"
                         << "final_indexes_saved=false (shared initial indexes retained)\n";
            }
            {
                std::ofstream matched_manifest(root / "matched_nlist_initial_index.txt", std::ios::trunc);
                if (!matched_manifest) throw std::runtime_error("cannot write matched-nlist index manifest");
                matched_manifest << "snapshot=static control index initialized with 1001 centers; common across scenarios, not per-scenario matched\n"
                                 << "source=" << (data_root / "B1" / "base.fbin").string() << "\nrows=" << kInitialRows
                                 << "\ndimension=" << kDimension << "\nnlist=1001\nseed=" << kSeed
                                 << "\nmetric=squared L2\nLVQ=false\nautomatic_maintenance=false\n"
                                 << "initial_build_ms=" << std::setprecision(10) << matched_build_ms << "\n";
            }

            for (const char* case_name : case_names) {
                MappedFbin data(data_root / case_name / "base.fbin");
                MappedFbin queries(data_root / case_name / "query_100.fbin");
                if (data.rows() != kInitialRows + kInsertOps * kInsertBatchRows || queries.rows() != kQueries)
                    throw std::runtime_error(std::string("DEEP-1M ") + case_name +
                        " must contain 1M data rows and 1K final queries");
                if (std::memcmp(reference_data.row(0), data.row(0), initial_bytes) != 0)
                    throw std::runtime_error(std::string("DEEP-1M ") + case_name +
                        " initial 500K rows differ; refusing to reuse common initial indexes");

                const std::filesystem::path case_root = root / case_name;
                std::filesystem::create_directories(case_root);
                for (const ComponentVariant& variant : variants) {
                    std::cout << "COMPONENT_ABLATION_START case=" << case_name
                              << " method=" << variant.name << "\n" << std::flush;
                    AdaptiveMethodConfig method;
                    method.name = variant.name;
                    method.prd = variant.triad;
                    method.final_query_only = true;
                    method.aps_queries = true;
                    method.triad_policy = variant.triad;
                    method.triad_enable_prd_splits = variant.prd;
                    method.triad_enable_load_splits = variant.load_splits;
                    method.triad_enable_capacity_splits = variant.capacity;
                    method.triad_enable_load_merges = variant.load_merges;
                    method.initial_leaf_nlist = variant.initial_nlist;
                    method.split_local_refinement_radius = variant.local_reassignment ? 50 : 0;
                    method.split_local_refinement_iterations = 1;
                    method.triad_global_center_refresh = variant.global_center_refresh;
                    method.override_split_local_refinement = true;
                    const std::filesystem::path initial_path = variant.initial_nlist == 1001 ?
                        matched_index_path : shared_initial_index;
                    run_adaptive_clean(case_root, data, queries, method, initial_path);
                    std::ofstream method_manifest(case_root / variant.name / "manifest.txt", std::ios::trunc);
                    if (!method_manifest) throw std::runtime_error("cannot write component method manifest");
                    method_manifest << "case=" << case_name << "\nmethod=" << variant.name << "\n"
                                    << "description=" << variant.description << "\n"
                                    << "initial_index=" << initial_path.string() << "\n"
                                    << "initial_vectors=500000; insert_batches=100; insert_batch=5000\n"
                                    << "initial_nlist=" << variant.initial_nlist
                                    << "\nPRD=0.05; load_threshold=2.2T; capacity=1.05*sqrt(N) while nlist<=sqrt(N)\n"
                                    << "enabled=triad:" << variant.triad << ",PRD:" << variant.prd
                                    << ",load_split:" << variant.load_splits << ",load_merge:" << variant.load_merges
                                    << ",capacity:" << variant.capacity
                                    << ",local_reassignment:" << variant.local_reassignment
                                    << ",global_center_refresh:" << variant.global_center_refresh << "\n"
                                    << "local_refinement_radius=" << (variant.local_reassignment ? 50 : 0)
                                    << "; iterations=1\n"
                                    << "query=final stage; 1000 queries; APS=0.9; candidate_fraction=0.1; recompute_threshold=0.001\n"
                                    << "warmup=100 (excluded); LVQ=false; early_stop=false; metric=squared L2\n";
                    std::cout << "COMPONENT_ABLATION_METHOD_COMPLETE case=" << case_name
                              << " method=" << variant.name << "\n" << std::flush;
                }
            }
            std::ofstream complete(root / "COMPLETE.txt", std::ios::trunc);
            if (!complete) throw std::runtime_error("cannot write component ablation completion marker");
            complete << "status=complete\ncases=9\nvariants_per_case=10\n"
                     << "warmup_queries_per_variant=100\nmeasured_queries_per_variant=1000\n"
                     << "shared_initial_index_reused=true\nquery_after_all_insertions=true\nfinal_indexes_saved=false\n";
            std::cout << "COMPONENT_ABLATION_ALL_COMPLETE output=" << root.string() << "\n" << std::flush;
            return 0;
        }
        if (argc == 5 && std::string(argv[1]) == "--deep1m-freewayml-component-ablation-checkpoints-aps") {
            const std::filesystem::path data_root(argv[2]);
            const std::filesystem::path shared_initial_index(argv[3]);
            const std::filesystem::path root(argv[4]);
            ensure_empty_or_create(root);
            if (!std::filesystem::is_regular_file(shared_initial_index))
                throw std::runtime_error("shared initial DEEP-1M index does not exist: " + shared_initial_index.string());

            constexpr std::array<const char*, 4> case_names{"G", "B1", "B3", "C2"};
            constexpr std::array<const char*, 5> query_names{
                "query_020.fbin", "query_040.fbin", "query_060.fbin", "query_080.fbin", "query_100.fbin"};
            struct CheckpointVariant {
                const char* name;
                bool triad;
                bool prd;
                bool load_splits;
                bool capacity;
                bool load_merges;
                bool local_reassignment;
                bool global_center_refresh;
            };
            constexpr std::array<CheckpointVariant, 7> variants{{
                {"frozen_ivf_nlist512", false, false, false, false, false, false, false},
                {"triad_full_r50_i1", true, true, true, true, true, true, true},
                {"triad_no_prd_r50_i1", true, false, true, true, true, true, true},
                {"triad_no_load_r50_i1", true, true, false, true, false, true, true},
                {"triad_no_capacity_r50_i1", true, true, true, false, true, true, true},
                {"triad_no_reassignment", true, true, true, true, true, false, true},
                {"triad_no_global_center_refresh_r50_i1", true, true, true, true, true, true, false}
            }};
            {
                auto initial = AdaptiveHierarchicalIvf::load(shared_initial_index.string());
                if (initial.size() != kInitialRows || initial.leaf_nlist() != kInitialNlist ||
                    initial.config().dimension != kDimension || initial.config().lvq8_storage ||
                    initial.config().lvq4_storage || initial.config().angular_metric ||
                    initial.config().uint8_storage)
                    throw std::runtime_error("shared initial index metadata does not match staged component ablation");
            }
            {
                std::ofstream manifest(root / "manifest.txt", std::ios::trunc);
                if (!manifest) throw std::runtime_error("cannot create staged component ablation manifest");
                manifest << "experiment=DEEP-1M FreewayML-inspired component ablation; five matched query checkpoints\n"
                         << "data_root=" << data_root.string() << "\nshared_initial_index=" << shared_initial_index.string() << "\n"
                         << "cases=G,B1,B3,C2\ncheckpoints=20,40,60,80,100 inserted batches; query files are stage-specific\n"
                         << "initial_vectors=500000; insert_batches=100; insert_batch=5000; dimension=96\n"
                         << "query=1000 queries per checkpoint; APS target=0.9; candidate_fraction=0.1; recompute_threshold=0.001; 100 warmup queries before first measured checkpoint\n"
                         << "common_config=initial_nlist=512; seed=1234; PRD=0.05; load threshold=2.2T; capacity=1.05*sqrt(N) only while nlist<=sqrt(N); local refinement radius=50; iterations=1; global mean refresh after each batch\n"
                         << "methods=frozen_ivf_nlist512,triad_full_r50_i1,triad_no_prd_r50_i1,triad_no_load_r50_i1,triad_no_capacity_r50_i1,triad_no_reassignment,triad_no_global_center_refresh_r50_i1\n"
                         << "LVQ=false; early_stop=false; metric=squared L2; exact truth=neighbor_ids at matching inserted-batch checkpoint\n";
            }

            for (const char* case_name : case_names) {
                const std::filesystem::path case_data_root = data_root / case_name;
                MappedFbin data(case_data_root / "base.fbin");
                if (data.rows() != kInitialRows + kInsertOps * kInsertBatchRows)
                    throw std::runtime_error(std::string("DEEP-1M ") + case_name + " must contain 1M rows");
                std::vector<std::filesystem::path> stage_files;
                for (const char* query_name : query_names) {
                    const auto query_path = case_data_root / query_name;
                    if (!std::filesystem::is_regular_file(query_path))
                        throw std::runtime_error("missing stage-specific query file: " + query_path.string());
                    stage_files.push_back(query_path);
                }
                const std::filesystem::path case_root = root / case_name;
                std::filesystem::create_directories(case_root);
                for (const CheckpointVariant& variant : variants) {
                    std::cout << "COMPONENT_CHECKPOINT_START case=" << case_name
                              << " method=" << variant.name << "\n" << std::flush;
                    AdaptiveMethodConfig method;
                    method.name = variant.name;
                    method.prd = variant.triad;
                    method.aps_queries = true;
                    method.triad_policy = variant.triad;
                    method.triad_enable_prd_splits = variant.prd;
                    method.triad_enable_load_splits = variant.load_splits;
                    method.triad_enable_capacity_splits = variant.capacity;
                    method.triad_enable_load_merges = variant.load_merges;
                    method.split_local_refinement_radius = variant.local_reassignment ? 50 : 0;
                    method.split_local_refinement_iterations = 1;
                    method.triad_global_center_refresh = variant.global_center_refresh;
                    method.query_stages_override = query_names.size();
                    method.insert_ops_per_query_override = kInsertOps / query_names.size();
                    method.query_stage_files = stage_files;
                    method.override_split_local_refinement = true;
                    MappedFbin final_queries(stage_files.back());
                    run_adaptive_clean(case_root, data, final_queries, method, shared_initial_index);
                    std::ofstream method_manifest(case_root / variant.name / "manifest.txt", std::ios::trunc);
                    if (!method_manifest) throw std::runtime_error("cannot write staged method manifest");
                    method_manifest << "case=" << case_name << "\nmethod=" << variant.name << "\n"
                                    << "initial_index=" << shared_initial_index.string() << "\ninitial_nlist=512\n"
                                    << "insert_batches=100; batch_size=5000; measured_checkpoints=20,40,60,80,100\n"
                                    << "PRD=0.05; load_threshold=2.2T; capacity=1.05*sqrt(N) while nlist<=sqrt(N)\n"
                                    << "enabled=triad:" << variant.triad << ",PRD:" << variant.prd
                                    << ",load_split:" << variant.load_splits << ",load_merge:" << variant.load_merges
                                    << ",capacity:" << variant.capacity
                                    << ",local_reassignment:" << variant.local_reassignment
                                    << ",global_center_refresh:" << variant.global_center_refresh << "\n"
                                    << "local_refinement_radius=" << (variant.local_reassignment ? 50 : 0)
                                    << "; iterations=1\n"
                                    << "queries=1000 per checkpoint from matching query_XXX.fbin; APS=0.9; candidate_fraction=0.1; recompute_threshold=0.001\n"
                                    << "warmup=100 before first checkpoint (excluded); LVQ=false; early_stop=false; metric=squared L2\n";
                    std::cout << "COMPONENT_CHECKPOINT_METHOD_COMPLETE case=" << case_name
                              << " method=" << variant.name << "\n" << std::flush;
                }
            }
            std::ofstream complete(root / "COMPLETE.txt", std::ios::trunc);
            if (!complete) throw std::runtime_error("cannot write staged completion marker");
            complete << "status=complete\ncases=4\nvariants_per_case=7\ncheckpoints_per_variant=5\n"
                     << "warmup_queries_per_variant=100\nmeasured_queries_per_checkpoint=1000\n"
                     << "shared_initial_index_reused=true\ntruth_recall_scored_separately=true\n";
            std::cout << "COMPONENT_CHECKPOINT_ALL_COMPLETE output=" << root.string() << "\n" << std::flush;
            return 0;
        }
        if (argc == 4 && std::string(argv[1]) == "--deep1m-freewayml-triad-ablation-final-aps") {
            const std::filesystem::path data_root(argv[2]);
            const std::filesystem::path root(argv[3]);
            ensure_empty_or_create(root);
            MappedFbin b1_data(data_root / "B1" / "base.fbin");
            MappedFbin c2_data(data_root / "C2" / "base.fbin");
            MappedFbin b1_queries(data_root / "B1" / "query_100.fbin");
            MappedFbin c2_queries(data_root / "C2" / "query_100.fbin");
            if (b1_data.rows() != 1'000'000 || c2_data.rows() != 1'000'000 ||
                b1_queries.rows() != kQueries || c2_queries.rows() != kQueries)
                throw std::runtime_error("DEEP-1M B1/C2 files must each contain 1M rows and 1K final queries");
            const std::size_t initial_bytes = kInitialRows * kDimension * sizeof(float);
            if (std::memcmp(b1_data.row(0), c2_data.row(0), initial_bytes) != 0)
                throw std::runtime_error("B1 and C2 initial 500K rows differ; refusing to reuse one initial index");

            const std::filesystem::path initial_index_path =
                root / "shared_initial_500k_nlist512_seed1234.index.bin";
            AdaptiveHierarchicalConfig initial_cfg;
            initial_cfg.dimension = kDimension;
            initial_cfg.initial_leaf_nlist = kInitialNlist;
            initial_cfg.kmeans_iterations = kKmeansIterations;
            initial_cfg.seed = kSeed;
            initial_cfg.lvq8_storage = false;
            initial_cfg.lvq4_storage = false;
            initial_cfg.angular_metric = false;
            initial_cfg.uint8_storage = false;
            initial_cfg.automatic_maintenance = false;
            initial_cfg.prd_threshold = 0.05f;
            initial_cfg.drift_criterion = DriftCriterion::GlobalFitOnly;
            initial_cfg.load_split_fraction = 2.20f;
            initial_cfg.capacity_growth_split_factor = kApsCapacityGrowthSplitFactor;
            initial_cfg.capacity_sqrt_n_split_multiplier = 1.05f;
            initial_cfg.split_local_refinement_radius = 50;
            initial_cfg.split_local_refinement_iterations = 1;
            AdaptiveHierarchicalIvf initial_index(initial_cfg);
            const auto build_start = std::chrono::steady_clock::now();
            initial_index.train(b1_data.row(0), kInitialRows);
            const double initial_build_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - build_start).count();
            initial_index.save(initial_index_path.string());
            {
                std::ofstream manifest(root / "manifest.txt", std::ios::trunc);
                std::ofstream initial_manifest(root / "shared_initial_index.txt", std::ios::trunc);
                if (!manifest || !initial_manifest)
                    throw std::runtime_error("cannot write DEEP-1M drift ablation manifest");
                manifest << "experiment=DEEP-1M FreewayML-inspired B1/C2; four Triad ablations; final query only\n"
                         << "data_root=" << data_root.string() << "\ninitial_vectors=500000\ninsert_batch=5000\ninsert_batches=100\n"
                         << "dimension=" << kDimension << "\nqueries_per_case=1000\nmetric=squared L2 on individually normalized vectors\n"
                         << "common_initial_index=" << initial_index_path.string() << "\n"
                         << "common_config=initial_nlist=512; kmeans_seed=1234; kmeans_iterations=10; PRD=0.05; load_threshold=2.2T; capacity=1.05*sqrt(N) while nlist<=sqrt(N); local_refinement_radius=50; local_refinement_iterations=1; global_mean_refresh=after each insert batch\n"
                         << "query=final stage after all inserts and maintenance; APS target=0.9; candidate_fraction=0.1; recompute_threshold=0.001; warmup=100 queries per method, excluded from measurements\n"
                         << "methods=triad_full,triad_no_prd,triad_no_load,triad_no_capacity\n"
                         << "truth=B1/C2 truth/groundtruth.npz, neighbor_ids[100], exact top-10\n"
                         << "initial_build_ms=" << std::setprecision(10) << initial_build_ms << "\nfinal_indexes_saved=false\n";
                initial_manifest << "snapshot=shared initial index for DEEP-1M B1/C2 drift ablations\n"
                                 << "source=" << (data_root / "B1" / "base.fbin").string() << "\n"
                                 << "rows=" << kInitialRows << "\ndimension=" << kDimension
                                 << "\nnlist=" << initial_index.leaf_nlist() << "\nseed=" << kSeed
                                 << "\nmetric=squared L2\nLVQ=false\nautomatic_maintenance=false\n"
                                 << "initial_build_ms=" << std::setprecision(10) << initial_build_ms << "\n";
            }

            struct DriftVariant {
                const char* name;
                bool prd;
                bool load;
                bool capacity;
            };
            const std::array<DriftVariant, 4> variants{{
                {"triad_full", true, true, true},
                {"triad_no_prd", false, true, true},
                {"triad_no_load", true, false, true},
                {"triad_no_capacity", true, true, false}
            }};
            const auto run_case = [&](const char* case_name, const MappedFbin& data,
                                      const MappedFbin& queries) {
                const std::filesystem::path case_root = root / case_name;
                std::filesystem::create_directories(case_root);
                for (const DriftVariant& variant : variants) {
                    std::cout << "DRIFT_ABLATION_START case=" << case_name
                              << " method=" << variant.name << "\n" << std::flush;
                    AdaptiveMethodConfig method;
                    method.name = variant.name;
                    method.prd = true;
                    method.final_query_only = true;
                    method.aps_queries = true;
                    method.triad_policy = true;
                    method.triad_enable_prd_splits = variant.prd;
                    method.triad_enable_load_splits = variant.load;
                    method.triad_enable_capacity_splits = variant.capacity;
                    method.triad_enable_load_merges = variant.load;
                    method.split_local_refinement_radius = 50;
                    method.split_local_refinement_iterations = 1;
                    method.triad_global_center_refresh = true;
                    run_adaptive_clean(case_root, data, queries, method, initial_index_path);
                    std::ofstream method_manifest(case_root / variant.name / "manifest.txt", std::ios::trunc);
                    if (!method_manifest)
                        throw std::runtime_error("cannot write drift ablation method manifest");
                    method_manifest << "case=" << case_name << "\nmethod=" << variant.name << "\n"
                                    << "initial_index_reused=" << initial_index_path.string() << "\n"
                                    << "initial_vectors=500000\ninsert_batch=5000\ninsert_batches=100\n"
                                    << "initial_nlist=512\nPRD=0.05\nload_threshold=2.2T\n"
                                    << "capacity_threshold=1.05*sqrt(N), active only while nlist<=sqrt(N)\n"
                                    << "local_refinement_radius=50\nlocal_refinement_iterations=1\n"
                                    << "global_mean_refresh=after each insert batch\n"
                                    << "query_stage=after all inserts and maintenance\nquery_count=1000\n"
                                    << "query_policy=APS target 0.9; candidate fraction 0.1; recompute threshold 0.001\n"
                                    << "warmup_queries=100 before measured queries; excluded from query CSV and APS summary\n"
                                    << "LVQ=false\nearly_stop=false\nmetric=squared L2\n"
                                    << "enabled_splits=PRD:" << variant.prd << ",load:" << variant.load
                                    << ",capacity:" << variant.capacity << "\n";
                    std::cout << "DRIFT_ABLATION_METHOD_COMPLETE case=" << case_name
                              << " method=" << variant.name << "\n" << std::flush;
                }
            };
            run_case("B1", b1_data, b1_queries);
            run_case("C2", c2_data, c2_queries);
            std::ofstream complete(root / "COMPLETE.txt", std::ios::trunc);
            if (!complete) throw std::runtime_error("cannot write DEEP-1M drift ablation completion marker");
            complete << "status=complete\ncases=2\nvariants_per_case=4\n"
                     << "warmup_queries_per_variant=100\nmeasured_queries_per_variant=1000\n"
                     << "query_after_all_insertions=true\nfinal_indexes_saved=false\n";
            std::cout << "DRIFT_ABLATION_ALL_COMPLETE output=" << root.string() << "\n" << std::flush;
            return 0;
        }
        if (argc == 5 && std::string(argv[1]) == "--deep1m-freewayml-single-trigger-final-aps") {
            const std::filesystem::path data_root(argv[2]);
            const std::filesystem::path initial_index_path(argv[3]);
            const std::filesystem::path root(argv[4]);
            ensure_empty_or_create(root);
            if (!std::filesystem::is_regular_file(initial_index_path))
                throw std::runtime_error("shared initial DEEP-1M index does not exist: " + initial_index_path.string());
            constexpr std::array<const char*, 9> case_names{
                "G", "A1", "A2", "B1", "B2", "B3", "B4", "C1", "C2"};
            MappedFbin reference_data(data_root / "B1" / "base.fbin");
            if (reference_data.rows() != 1'000'000)
                throw std::runtime_error("DEEP-1M scenario base files must contain 1M rows");
            const std::size_t initial_bytes = kInitialRows * kDimension * sizeof(float);
            {
                auto initial = AdaptiveHierarchicalIvf::load(initial_index_path.string());
                if (initial.size() != kInitialRows || initial.leaf_nlist() != kInitialNlist ||
                    initial.config().dimension != kDimension || initial.config().lvq8_storage ||
                    initial.config().lvq4_storage || initial.config().angular_metric ||
                    initial.config().uint8_storage)
                    throw std::runtime_error("shared initial index metadata does not match the DEEP-1M ablation");
            }
            struct DriftVariant {
                const char* name;
                bool prd;
                bool load;
                bool capacity;
            };
            const std::array<DriftVariant, 3> variants{{
                {"triad_prd_only", true, false, false},
                {"triad_load_only", false, true, false},
                {"triad_capacity_only", false, false, true}
            }};
            {
                std::ofstream manifest(root / "manifest.txt", std::ios::trunc);
                if (!manifest) throw std::runtime_error("cannot create single-trigger experiment manifest");
                manifest << "experiment=DEEP-1M nine-scenario single-trigger ablations; final query only\n"
                         << "data_root=" << data_root.string() << "\nshared_initial_index=" << initial_index_path.string() << "\n"
                         << "initial_vectors=500000; insert_batches=100; insert_batch=5000; dimension=96\n"
                         << "common_maintenance=empty cleanup and global mean refresh retained; load merge+split is enabled only for load-only; split-local reassignment enabled in all variants\n"
                         << "query=final 1000 queries after insertion; APS target=0.9; warmup=100 per method excluded\n"
                         << "methods=triad_prd_only,triad_load_only,triad_capacity_only\n"
                         << "cases=G,A1,A2,B1,B2,B3,B4,C1,C2\n"
                         << "LVQ=false; early_stop=false; metric=squared L2\n";
            }
            const auto run_case = [&](const char* case_name) {
                MappedFbin data(data_root / case_name / "base.fbin");
                MappedFbin queries(data_root / case_name / "query_100.fbin");
                if (data.rows() != 1'000'000 || queries.rows() != kQueries)
                    throw std::runtime_error(std::string("DEEP-1M ") + case_name +
                        " must contain 1M base rows and 1K final queries");
                if (std::memcmp(reference_data.row(0), data.row(0), initial_bytes) != 0)
                    throw std::runtime_error(std::string("DEEP-1M ") + case_name +
                        " initial 500K rows differ; refusing to reuse the shared index");
                const std::filesystem::path case_root = root / case_name;
                std::filesystem::create_directories(case_root);
                for (const DriftVariant& variant : variants) {
                    std::cout << "SINGLE_TRIGGER_START case=" << case_name
                              << " method=" << variant.name << "\n" << std::flush;
                    AdaptiveMethodConfig method;
                    method.name = variant.name;
                    method.prd = true;
                    method.final_query_only = true;
                    method.aps_queries = true;
                    method.triad_policy = true;
                    method.triad_enable_prd_splits = variant.prd;
                    method.triad_enable_load_splits = variant.load;
                    method.triad_enable_capacity_splits = variant.capacity;
                    method.triad_enable_load_merges = variant.load;
                    method.split_local_refinement_radius = 50;
                    method.split_local_refinement_iterations = 1;
                    method.triad_global_center_refresh = true;
                    run_adaptive_clean(case_root, data, queries, method, initial_index_path);
                    std::ofstream method_manifest(case_root / variant.name / "manifest.txt", std::ios::trunc);
                    if (!method_manifest) throw std::runtime_error("cannot write single-trigger method manifest");
                    method_manifest << "case=" << case_name << "\nmethod=" << variant.name << "\n"
                                    << "initial_index_reused=" << initial_index_path.string() << "\n"
                                    << "initial_vectors=500000; insert_batches=100; insert_batch=5000\n"
                                    << "initial_nlist=512; PRD=0.05; load_threshold=2.2T; capacity=1.05*sqrt(N) while nlist<=sqrt(N)\n"
                                    << "local_refinement_radius=50; iterations=1; global_mean_refresh=after every insert batch\n"
                                    << "enabled_split_trigger=PRD:" << variant.prd << ",load:" << variant.load
                                    << ",capacity:" << variant.capacity << "\n"
                                    << "enabled_load_merge=" << variant.load << "\n"
                                    << "final APS target=0.9; candidate_fraction=0.1; recompute_threshold=0.001\n"
                                    << "warmup_queries=100 (excluded); measured_queries=1000\n"
                                    << "LVQ=false; early_stop=false; metric=squared L2\n";
                    std::cout << "SINGLE_TRIGGER_METHOD_COMPLETE case=" << case_name
                              << " method=" << variant.name << "\n" << std::flush;
                }
            };
            for (const char* case_name : case_names) run_case(case_name);
            std::ofstream complete(root / "COMPLETE.txt", std::ios::trunc);
            if (!complete) throw std::runtime_error("cannot write single-trigger completion marker");
            complete << "status=complete\ncases=9\nvariants_per_case=3\n"
                     << "warmup_queries_per_variant=100\nmeasured_queries_per_variant=1000\n"
                     << "shared_initial_index_reused=true\nquery_after_all_insertions=true\nfinal_indexes_saved=false\n";
            std::cout << "SINGLE_TRIGGER_ALL_COMPLETE output=" << root.string() << "\n" << std::flush;
            return 0;
        }
#endif
        if (argc != 4 && argc != 5) {
            std::cerr << "Usage prefix: msturing_ih_runner --quake-latency-profile <profile.csv> <existing command and arguments>\n"
                      << "   or: msturing_ih_runner --quake-latency-profile <profile.csv> --validate-quake-latency-profile\n"
                      << "   or: msturing_ih_runner --prepare-truth-only <base.fbin> <queries.fbin> <stage_truth_top100.bin>\n"
                      << "   or: deep1m_triad_fixed50k_runner --deep1m-freewayml-triad-nine-fixed50k <data_root> <initial_index.bin> <output_dir> [prd_threshold query_threads [query_budget_divisor]]\n"
                      << "   or: deep1m_triad_fixed50k_runner --deep1m-freewayml-triad-nine-fixed50k-no-prd <data_root> <initial_index.bin> <output_dir> [query_threads [query_budget_divisor]]\n"
                      << "   or: msturing_ih_runner --triad-fixed-n20 <base> <queries> <cached_stage_truth_top100.bin> <initial_index.bin> <new_output_dir>\n"
                      << "   or: msturing_ih_runner --triad-final-n20-no-reassignment <base> <queries> <cached_stage_truth_top100.bin> <initial_index.bin> <new_output_dir>\n"
                      << "   or: msturing_ih_runner --triad-threshold-robustness-final-n20 <shuffled_base.fbin> <queries.fbin> <cached_stage_truth_top100.bin> <initial_index.bin> <new_output_dir> <query_threads> <N_divisor> <capacity_K_gate>\n"
                      << "   or: yfcc10m_ih_runner --prepare-truth-only-u8 <base.u8bin> <queries.u8bin> <stage_truth_top100.bin>\n"
                      << "   or: yfcc10m_ih_runner --quake-only-u8 <base.u8bin> <queries.u8bin> <cached_stage_truth_top100.bin> <new_output_dir>\n"
                      << "   or: yfcc10m_ih_runner --quake-only-u8-reuse-initial <base.u8bin> <queries.u8bin> <stage_truth_top100.bin> <initial_index.bin> <new_output_dir>\n"
                      << "   or: yfcc10m_ih_runner --lire-only-u8 <base.u8bin> <queries.u8bin> <cached_stage_truth_top100.bin> <new_output_dir>\n"
                      << "   or: yfcc10m_ih_runner --prd-final-ih-u8 <base.u8bin> <queries.u8bin> <cached_stage_truth_top100.bin> <new_output_dir> <load_split_fraction> <capacity_sqrt_n_multiplier>\n"
                      << "   or: yfcc10m_ih_runner --prd-full-ih-u8 <base.u8bin> <queries.u8bin> <cached_stage_truth_top100.bin> <new_output_dir> <load_split_fraction> <capacity_sqrt_n_multiplier> <initial_index.bin>\n"
                      << "   or: msturing_ih_runner <msturing-10M-clustered.fbin> <testQuery10K.fbin> <new_output_dir> [cached_stage_truth_top100.bin]\n"
                      << "   or: msturing_ih_runner --quake-only <msturing-10M-clustered.fbin> <queries.fbin> <cached_stage_truth_top100.bin> <new_output_dir> [reusable_initial_index.bin [final_index_output.bin]]\n"
                      << "   or: deep10m_triad_quake_lire_runner --deep10m-quake-100stage <base.fbin> <queries.fbin> <stage_truth_top100.bin> <official_truth.ibin> <initial_index.bin> <new_output_dir>\n"
                      << "   or: deep10m_triad_quake_lire_runner --deep10m-quake-100stage-no-index <base.fbin> <queries.fbin> <stage_truth_top100.bin> <official_truth.ibin> <initial_index.bin> <new_output_dir>\n"
                      << "   or: deep10m_triad_quake_lire_runner --deep10m-quake-100stage-n20 <base.fbin> <queries.fbin> <stage_truth_top100.bin> <official_truth.ibin> <new_output_dir>\n"
                      << "   or: deep10m_triad_quake_lire_runner --deep10m-lire-final-only <base.fbin> <queries.fbin> <official_truth.ibin> <initial_index.bin> <new_output_dir> <max_partition_size> <budget_divisor> <query_threads>\n"
                      << "   or: 10M_ih_runner --lire-full-100stage <base> <queries> <stage_truth_top100.bin> <initial_index.bin> <new_output_dir> <max_partition_size> <budget_divisor> <query_threads>\n"
                      << "   or: msturing_ih_runner --quake-aps-only <msturing-10M-clustered.fbin> <testQuery10K.fbin> <cached_stage_truth_top100.bin> <new_output_dir>\n"
                      << "   or: msturing_ih_runner --lire-only <msturing-10M-clustered.fbin> <testQuery10K.fbin> <cached_stage_truth_top100.bin> <new_output_dir>\n"
                      << "   or: msturing_ih_runner --lire-aps-only <msturing-10M-clustered.fbin> <testQuery10K.fbin> <cached_stage_truth_top100.bin> <new_output_dir>\n"
                      << "   or: msturing_ih_runner --triad-aps-only <msturing-10M-clustered.fbin> <testQuery10K.fbin> <cached_stage_truth_top100.bin> <new_output_dir>\n"
                      << "   or: msturing_ih_runner --save-triad-initial-index <shuffled_base.fbin> <output_index.bin>\n"
                    << "   or: msturing_ih_runner --triad-ablation-final-aps <shuffled_base.fbin> <queries.fbin> <original_final_gt100> <permutation.bin> <new_output_dir>\n"
                    << "   or: msturing_ih_runner --triad-ablation-final-aps <shuffled_base.fbin> <queries.fbin> <original_final_gt100> <permutation.bin> <initial_index.bin> <new_output_dir>\n"
                      << "   or: msturing_ih_runner --quake-lire-only <base.fbin> <queries.fbin> <cached_stage_truth_top100.bin> <new_output_dir> <max_partition_size>\n"
                      << "   or: yfcc10m_ih_runner --quake-lire-only-u8 <base.u8bin> <queries.u8bin> <cached_stage_truth_top100.bin> <new_output_dir> <max_partition_size>\n"
                      << "   or: msturing_ih_runner --quake-lire-final-only <shuffled_base.fbin> <queries.fbin> <original_final_gt100> <permutation.bin> <new_output_dir> <max_partition_size>\n"
                      << "   or: msturing_ih_runner --prd-final-only-shuffled <shuffled_base.fbin> <queries.fbin> <original_final_gt100> <permutation.bin> <new_output_dir>\n"
                      << "   or: msturing_ih_runner --prd-final-only-shuffled-refined <shuffled_base.fbin> <queries.fbin> <original_final_gt100> <permutation.bin> <new_output_dir> <refinement_radius> <iterations>\n"
                      << "   or: msturing_ih_runner --prd-final-ih <msturing-10M-clustered.fbin> <testQuery10K.fbin> <cached_stage_truth_top100.bin> <new_output_dir> <load_split_fraction> <capacity_sqrt_n_multiplier>\n"
                      << "   or: msturing_ih_runner --prd-full-ih <msturing-10M-clustered.fbin> <testQuery10K.fbin> <cached_stage_truth_top100.bin> <new_output_dir> <load_split_fraction> <capacity_sqrt_n_multiplier>\n"
                      << "   or: msturing_ih_runner --requery-prd-final-ih <saved_index.bin> <msturing-10M-clustered.fbin> <testQuery10K.fbin> <cached_stage_truth_top100.bin> <existing_output_dir> <budget>\n"
                      << "   or: msturing_ih_runner --prd-localref-only <msturing-10M-clustered.fbin> <testQuery10K.fbin> <new_output_dir> <cached_stage_truth_top100.bin>\n"
                      << "   or: msturing_ih_runner --prd-aps-initial <msturing-10M-clustered.fbin> <testQuery10K.fbin> <new_output_dir>\n"
                      << "   or: msturing_ih_runner --prd-aps-full-build <msturing-10M-clustered.fbin> <testQuery10K.fbin> <stage_truth_top100.bin> <new_output_dir> <nlist>\n"
                      << "   or: msturing_ih_runner --prd-full-build-only <msturing-10M-clustered.fbin> <testQuery10K.fbin> <stage_truth_top100.bin> <new_output_dir> <nlist>\n"
                      << "   or: msturing_ih_runner --prd-full-training-build-only <msturing-10M-clustered.fbin> <testQuery10K.fbin> <stage_truth_top100.bin> <new_output_dir> <nlist>\n"
                      << "   or: msturing_ih_runner --prd-sample-training-build-only <msturing-10M-clustered.fbin> <testQuery10K.fbin> <stage_truth_top100.bin> <new_output_dir> <nlist> <training_fraction> <seed>\n"
                      << "   or: msturing_ih_runner --fixed-nprobe <index.bin> <id_mapping.u32bin> <msturing-10M-clustered.fbin> <testQuery10K.fbin> <stage_truth_top100.bin> <new_output_dir> <nprobe>\n"
                      << "   or: msturing_ih_runner --fixed-budget <index.bin> <id_mapping.u32bin> <msturing-10M-shuffled.fbin> <testQuery10K.fbin> <stage_truth_top100.bin> <new_output_dir> <budget> [query_threads]\n"
                      << "   or: msturing_ih_runner --fixed-budget-snapshot <index.bin> <queries.fbin> <final_or_staged_truth.bin> <new_output_dir> <budget> [query_threads]\n"
                      << "   or: msturing_ih_runner --fixed-nprobe-sweep-sample <index.bin> <id_mapping.u32bin> <msturing-10M-clustered.fbin> <testQuery10K.fbin> <stage_truth_top100.bin> <new_output_dir> <sample_count> <sample_seed>\n"
                      << "   or: msturing_ih_runner --fixed-nprobe-single-sample <index.bin> <id_mapping.u32bin> <msturing-10M-clustered.fbin> <testQuery10K.fbin> <stage_truth_top100.bin> <new_output_dir> <sample_count> <sample_seed> <nprobe>\n"
                      << "   or: msturing_ih_runner --aps-sample <index.bin> <id_mapping.u32bin> <msturing-10M-clustered.fbin> <testQuery10K.fbin> <stage_truth_top100.bin> <new_output_dir> <sample_count> <sample_seed> <candidate_partition_limit>\n"
                      << "   or: msturing_ih_runner --prd-aps <msturing-10M-clustered.fbin> <testQuery10K.fbin> <stage_truth_top100.bin> <new_output_dir>\n"
                      << "   or: msturing_ih_runner --prd-aps-final <msturing-10M-clustered.fbin> <testQuery10K.fbin> <stage_truth_top100.bin> <new_output_dir>\n"
                      << "   or: msturing_ih_runner --prd-aps-final-local-refinement <msturing-10M-clustered.fbin> <testQuery10K.fbin> <stage_truth_top100.bin> <initial_index.bin> <new_output_dir> <radius> <iterations>\n"
                      << "   or: msturing_ih_runner --score-existing <results_dir> <new_recall_dir> <msturing-10M-clustered.fbin> <testQuery10K.fbin>\n"
                      << "   or: msturing_ih_runner --score-k10 <results_dir> <stage_truth_top100.bin> <new_output_dir>\n";
            return 2;
        }
        const std::filesystem::path root(argv[3]); ensure_empty_or_create(root);
        MappedFbin data(argv[1]), queries(argv[2]);
#ifdef MSTURING_IH_SMOKE
        if (data.rows() < kInitialRows + kInsertOps * kInsertBatchRows || queries.rows() < kQueries)
#else
        if (data.rows() != 10'000'000 || queries.rows() < kQueries)
#endif
            throw std::runtime_error("expected 10M base vectors and a query file containing the configured query count");
        if (argc == 5) {
            const std::filesystem::path cached_truth(argv[4]);
            if (!std::filesystem::is_regular_file(cached_truth))
                throw std::runtime_error("cached stage truth does not exist: " + cached_truth.string());
            std::filesystem::copy_file(cached_truth, root / "groundtruth_stage_top100.bin");
            std::cout << "cached_stage_truth_copied path=" << cached_truth.string() << '\n' << std::flush;
        }
        std::ofstream manifest(root / "manifest.txt");
        if (!manifest) throw std::runtime_error("cannot create root manifest");
        manifest << "experiment=Quake Table 3 insert-heavy workload (IH)\n"
                 << "dataset=" << argv[1] << "\nqueries=" << argv[2] << "\nrows=10000000\ndimension=" << kDimension << "\nmetric=squared L2\n"
                 << "initial_vectors=" << kInitialRows << "\ninitial_nlist=" << kInitialNlist << "\ninsert_batch=" << kInsertBatchRows
                 << "\ninsert_ops=" << kInsertOps << "\nquery_ops=" << kQueryOps << "\nqueries_per_query_op=" << kQueries
                 << "\ntopk=" << kTopK << "\nquery_budget=floor(current_active_vectors/20), including all centroid and vector distances\n"
                 << "each query op follows nine insert ops; no LVQ; seed=1234\n"
                 << "methods=PRD-IVF(no local reassignment),Quake(split threshold=" << kQuakeSplitThresholdNs
                 << "ns; delete+reassignment threshold=" << kQuakeDeleteThresholdNs
                 << "ns),SPFresh-LIRE,PRD-IVF(local refinement radius 25, iterations 1)\n"
                 << "groundtruth=" << (argc == 5 ? "reused cached exact top-100 for each active prefix" : "calculated after all method runs; exact top-100 over active prefix") << '\n'
                 << "recall_metrics=post-hoc Recall@100 and Recall@10 against the same exact stage truth\n";
        manifest.close();
        run_adaptive_clean(root, data, queries, {"prd_ivf_no_reassignment", true, false, 0, 1});
        run_adaptive_clean(root, data, queries, {"quake", false, true});
        run_lire(root, data, queries);
        run_adaptive_clean(root, data, queries, {"prd_ivf_localref_r25_i1", true, false, 25, 1});
        write_truth_and_recall(root, data, queries);
        score_existing_recall_at_10(root, root / "groundtruth_stage_top100.bin", root / "recall_at_10");
        std::ofstream complete(root / "COMPLETE.txt");
        complete << "completed_methods=4\ncompleted_insert_ops_per_method=" << kInsertOps
                 << "\ncompleted_query_ops_per_method=" << kQueryOps << "\nqueries_per_operation=" << kQueries
                 << "\nmethod_order=prd_ivf_no_reassignment,quake,spfresh_lire,prd_ivf_localref_r25_i1"
                 << "\ngroundtruth=post-hoc exact, all 100 stages\n";
        complete.close();
        std::cout << "COMPLETE output=" << root.string() << '\n' << std::flush;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
