#include "../adaptive_hierarchical_ivf.h"
#include "../simd_l2.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
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
constexpr std::size_t kDimension = 96;
constexpr std::size_t kTopK = 10;
constexpr std::size_t kInitialNlist = 512;
constexpr std::size_t kInitialRows = 500'000;
constexpr std::size_t kBatchRows = 5'000;
constexpr std::size_t kStages = 100;
constexpr std::size_t kQueries = 1'000;
constexpr std::size_t kQueryWindow = 1'000;
constexpr std::size_t kDefaultBudget = 50'000;
constexpr std::size_t kKmeansIterations = 10;
constexpr std::size_t kMinPartitionSize = 32;
constexpr std::size_t kDefaultRefinementRadius = 25;
constexpr std::size_t kRefinementIterations = 1;
constexpr double kAlpha = 0.9;
constexpr double kDefaultSplitThresholdNs = 560.0;
constexpr std::uint32_t kSeed = 1234;
constexpr std::array<const char*, 9> kCases{"G", "A1", "A2", "B1", "B2", "B3", "B4", "C1", "C2"};

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
        if (mapped_ == MAP_FAILED) { mapped_ = nullptr; close(); throw std::runtime_error("cannot view fbin mapping"); }
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
    std::size_t dimension() const noexcept { return dimension_; }
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

using Truth = std::array<std::uint32_t, kTopK>;
struct QueryTruth { std::vector<Truth> rows; };

QueryTruth read_truth(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read prepared truth: " + path.string());
    char magic[8]{};
    std::uint32_t shape[3]{};
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char*>(shape), sizeof(shape));
    if (!in || std::string(magic, sizeof(magic)) != "DLQTRTH1" ||
        shape[0] != kStages + 1 || shape[1] != kQueries || shape[2] != kTopK)
        throw std::runtime_error("prepared truth header mismatch: " + path.string());
    QueryTruth truth;
    truth.rows.resize((kStages + 1) * kQueries);
    in.read(reinterpret_cast<char*>(truth.rows.data()),
            static_cast<std::streamsize>(truth.rows.size() * sizeof(Truth)));
    if (!in || in.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("prepared truth payload size mismatch: " + path.string());
    return truth;
}

const Truth& truth_at(const QueryTruth& truth, std::size_t stage, std::size_t query) {
    return truth.rows.at(stage * kQueries + query);
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t i = std::min(values.size() - 1,
        static_cast<std::size_t>(std::ceil(p * values.size()) - 1.0));
    return values[i];
}

class ScanLatencyProfile {
public:
    ScanLatencyProfile(const MappedFbin& data, const MappedFbin& queries) {
        constexpr std::array<std::size_t, 10> grid{1,2,4,16,64,256,1024,4096,16384,65536};
        volatile float sink = 0.0f;
        for (const auto n : grid) {
            double total = 0.0;
            for (std::size_t trial = 0; trial < 5; ++trial) {
                const float* q = queries.row((trial * 193) % queries.rows());
                std::priority_queue<float> top;
                const auto start = std::chrono::steady_clock::now();
                for (std::size_t i = 0; i < n; ++i) {
                    const float d = simd_squared_l2(q, data.row(i), kDimension);
                    if (top.size() < kTopK) top.push(d);
                    else if (d < top.top()) { top.pop(); top.push(d); }
                }
                total += std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count();
                if (!top.empty()) sink += top.top();
            }
            n_.push_back(n); ns_.push_back(total / 5.0);
        }
        if (sink == -1.0f) std::cerr << "";
    }
    double operator()(std::size_t n) const {
        if (!n) return 0.0;
        if (n <= n_.front()) return ns_.front() * n / n_.front();
        auto it = std::lower_bound(n_.begin(), n_.end(), n);
        std::size_t hi = it == n_.end() ? n_.size() - 1 : static_cast<std::size_t>(it - n_.begin());
        if (n_[hi] == n) return ns_[hi];
        const std::size_t lo = hi - 1;
        const double f = static_cast<double>(n - n_[lo]) / static_cast<double>(n_[hi] - n_[lo]);
        return ns_[lo] + f * (ns_[hi] - ns_[lo]);
    }
private:
    std::vector<std::size_t> n_;
    std::vector<double> ns_;
};

struct SplitCandidate { std::size_t leaf, size, hits; double rate, delta; bool selected; };
std::vector<SplitCandidate> choose_splits(const AdaptiveHierarchicalIvf& index,
        const std::deque<std::vector<std::size_t>>& window, const ScanLatencyProfile& latency,
        double split_threshold_ns) {
    const auto sizes = index.leaf_sizes();
    std::vector<std::size_t> hits(sizes.size(), 0);
    for (const auto& list : window) for (const auto leaf : list) if (leaf < hits.size()) ++hits[leaf];
    std::vector<SplitCandidate> result;
    const double denom = static_cast<double>(window.size());
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        if (sizes[i] < kMinPartitionSize) continue;
        const double rate = denom ? static_cast<double>(hits[i]) / denom : 0.0;
        const double delta = latency(sizes.size() + 1) - latency(sizes.size())
            + 2.0 * kAlpha * rate * latency(sizes[i] / 2) - rate * latency(sizes[i]);
        result.push_back({i, sizes[i], hits[i], rate, delta,
                          denom >= kQueryWindow && delta < -split_threshold_ns});
    }
    return result;
}

struct QueryStats { std::uint64_t hits=0, center=0, vectors=0, total=0, partitions=0; std::vector<double> ms; };
struct StageSummary {
    std::size_t active=0, inserted=0, nlist_before=0, nlist_after=0, max_list=0, candidates=0, splits=0;
    double insert_ms=0, maintenance_ms=0, sse=0;
    QueryStats query;
};

double measure_sse(const MappedFbin& data, const AdaptiveHierarchicalIvf& index) {
    const auto assignments = index.vector_leaf_assignments();
    const auto centers = index.leaf_centroids();
    if (assignments.size() != index.size() || centers.size() != index.leaf_nlist() * kDimension)
        throw std::runtime_error("SSE inputs do not match Quake index population");
    double sse = 0.0;
    for (const auto& item : assignments) {
        if (item.first < 0 || static_cast<std::size_t>(item.first) >= data.rows() || item.second >= index.leaf_nlist())
            throw std::runtime_error("invalid vector-to-list assignment");
        const float* x = data.row(static_cast<std::size_t>(item.first));
        const float* c = centers.data() + item.second * kDimension;
        for (std::size_t d = 0; d < kDimension; ++d) { const double dx = x[d] - c[d]; sse += dx * dx; }
    }
    return sse;
}

bool is_truth_hit(std::int64_t id, const Truth& truth) {
    return id >= 0 && std::find(truth.begin(), truth.end(), static_cast<std::uint32_t>(id)) != truth.end();
}

void run_case(const std::filesystem::path& data_root, const std::filesystem::path& out_root,
              const std::string& name, std::ofstream& all_summary,
              std::size_t budget_divisor, std::size_t query_threads, double split_threshold_ns,
              std::size_t refinement_radius) {
    const auto case_dir = data_root / name;
    MappedFbin data(case_dir / "base.fbin");
    if (data.rows() != kInitialRows + kStages * kBatchRows) throw std::runtime_error("base row count mismatch for " + name);
    const auto truth = read_truth(out_root / "_prepared_truth" / (name + ".u32bin"));
    const auto case_out = out_root / name;
    std::filesystem::create_directories(case_out);
    std::ofstream summary(case_out / "stage_summary.csv"), query_rows(case_out / "per_query.csv"),
        candidates(case_out / "maintenance_candidates.csv"), manifest(case_out / "manifest.txt");
    if (!summary || !query_rows || !candidates || !manifest) throw std::runtime_error("cannot create output files for " + name);
    summary << "case,stage,active_vectors,inserted_vectors,nlist_before,nlist_after,max_list_size,insert_ms,maintenance_ms,update_ms,maintenance_window_queries,split_candidates,splits,recall_at_10,top1_hit_rate,query_mean_ms,query_p50_ms,query_p95_ms,query_core_mean_ms,query_core_p50_ms,query_core_p95_ms,mean_center_distance_computations,mean_vector_distance_computations,mean_total_distance_computations,mean_partitions_scanned,scan_budget_per_query,total_sse,mean_sse_per_vector\n";
    query_rows << "stage,query_index,scan_budget,center_distance_computations,vector_distance_computations,total_distance_computations,partitions_scanned,recall_at_10,query_ms,search_core_ms\n";
    candidates << "stage,leaf_id,leaf_size,window_hits,hit_rate,estimated_delta_ns,selected\n";
    manifest << "method=Quake query-cost partition-splitting policy adapted to the project IVF engine\n"
        << "case=" << name << "\ndataset=DEEP-1M FreewayML-style controlled drift; 96-D FP32; squared L2 on unit-normalized vectors\n"
        << "initial_vectors=500000\ninsert_batch_vectors=5000\ninsertion_batches=100\nqueries_per_stage=1000\n"
        << "seed=1234\ninitial_nlist=512\nquery_budget="
        << (budget_divisor ? "floor(current_active_vectors/" + std::to_string(budget_divisor) + ")" : std::to_string(kDefaultBudget) + " fixed")
        << " total distance computations per query; query_threads=" << query_threads << "\n"
        << "early_stopping=false\nLVQ8=false\nQuake alpha=0.9; split_threshold_ns=" << split_threshold_ns
        << "; delete_threshold_ns=" << split_threshold_ns << " (unused: insert-only workload); min_partition_size=32; local_refinement_radius="
        << refinement_radius << "; local_refinement_iterations="
        << kRefinementIterations << "\n"
        << "Quake uses the previous stage's up-to-1000 query partition-hit window; current stage queries follow insertion and maintenance\n"
        << "maintenance rule is Quake cost model, not PRD threshold/gates; APS disabled for fixed-budget comparability\n"
        << "query timing: per-query steady_clock around the native search_with_options call; excludes truth evaluation and file output\n";

    AdaptiveHierarchicalConfig config;
    config.dimension = kDimension; config.initial_leaf_nlist = kInitialNlist; config.fixed_leaf_nprobe = 0;
    config.kmeans_iterations = kKmeansIterations; config.maintenance_fraction = .20f;
    config.automatic_maintenance = false; config.lvq8_storage = false; config.seed = kSeed;
    AdaptiveHierarchicalIvf index(config);
    const auto build_start = std::chrono::steady_clock::now();
    index.train(data.row(0), kInitialRows);
    const double build_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - build_start).count();
    MappedFbin q0(case_dir / "query_000.fbin");
    ScanLatencyProfile latency(data, q0);
    std::deque<std::vector<std::size_t>> hit_window;
    std::size_t total_splits = 0;

    for (std::size_t stage = 0; stage <= kStages; ++stage) {
        StageSummary m;
        m.active = kInitialRows + stage * kBatchRows;
        m.nlist_before = index.leaf_nlist();
        if (stage == 0) {
            m.inserted = kInitialRows;
        } else {
            const auto update_start = std::chrono::steady_clock::now();
            index.add(data.row(kInitialRows + (stage - 1) * kBatchRows), kBatchRows);
            m.insert_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - update_start).count();
            m.inserted = kBatchRows;
            const auto maintenance_start = std::chrono::steady_clock::now();
            const auto decisions = choose_splits(index, hit_window, latency, split_threshold_ns);
            std::vector<std::size_t> selected;
            for (const auto& c : decisions) {
                candidates << stage << ',' << c.leaf << ',' << c.size << ',' << c.hits << ','
                           << std::fixed << std::setprecision(8) << c.rate << ',' << std::setprecision(3)
                           << c.delta << ',' << (c.selected ? 1 : 0) << '\n';
                if (c.selected) selected.push_back(c.leaf);
            }
            m.candidates = selected.size();
            if (!selected.empty()) {
                const auto report = index.maintain_selected(selected, SelectedMaintenanceAction::Split,
                                                            refinement_radius, kRefinementIterations);
                m.splits = report.split_partitions;
            }
            total_splits += m.splits;
            m.maintenance_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - maintenance_start).count();
        }
        m.nlist_after = index.leaf_nlist();
        if (index.size() != m.active) throw std::runtime_error("active population mismatch in " + name);
        const auto query_path = case_dir / ("query_" + [&] { std::ostringstream s; s << std::setw(3) << std::setfill('0') << stage; return s.str(); }() + ".fbin");
        MappedFbin queries(query_path);
        if (queries.rows() != kQueries) throw std::runtime_error("query row count mismatch: " + query_path.string());
        const std::size_t budget = budget_divisor ? m.active / budget_divisor : kDefaultBudget;
        QuerySearchOptions options; options.enable_early_stop = false; options.max_total_distance_computations = budget;
        m.query.ms.reserve(kQueries);
        struct QueryOutcome {
            std::vector<std::size_t> probed;
            QuerySearchStats stats;
            double ms = 0.0;
            std::size_t hits = 0;
            bool top1_hit = false;
        };
        std::vector<QueryOutcome> outcomes(kQueries);
        std::atomic<std::size_t> next_query{0};
        std::exception_ptr worker_error;
        std::mutex worker_error_mutex;
        const std::size_t thread_count = std::min(query_threads, kQueries);
        auto worker = [&]() {
            while (true) {
                const std::size_t qi = next_query.fetch_add(1);
                if (qi >= kQueries) return;
                try {
                    QueryOutcome& outcome = outcomes[qi];
                    const auto start = std::chrono::steady_clock::now();
                    const auto found = index.search_with_options(queries.row(qi), kTopK, options,
                                                                  &outcome.stats, &outcome.probed);
                    outcome.ms = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start).count();
                    if (outcome.stats.total_distance_computations != budget ||
                        outcome.stats.center_distance_computations != index.leaf_nlist() ||
                        outcome.stats.total_distance_computations != outcome.stats.center_distance_computations + outcome.stats.scanned_vectors)
                        throw std::runtime_error("fixed query budget mismatch in " + name);
                    const Truth& truth_ids = truth_at(truth, stage, qi);
                    for (const auto& item : found) if (is_truth_hit(item.id, truth_ids)) ++outcome.hits;
                    outcome.top1_hit = !found.empty() && found.front().id >= 0 &&
                        found.front().id == truth_ids[0];
                } catch (...) {
                    std::lock_guard<std::mutex> lock(worker_error_mutex);
                    if (!worker_error) worker_error = std::current_exception();
                    return;
                }
            }
        };
        std::vector<std::thread> workers;
        workers.reserve(thread_count);
        for (std::size_t i = 0; i < thread_count; ++i) workers.emplace_back(worker);
        for (auto& thread : workers) thread.join();
        if (worker_error) std::rethrow_exception(worker_error);
        std::size_t top1_hits = 0;
        for (std::size_t qi = 0; qi < kQueries; ++qi) {
            QueryOutcome& outcome = outcomes[qi];
            const double recall = static_cast<double>(outcome.hits) / kTopK;
            query_rows << stage << ',' << qi << ',' << budget << ',' << outcome.stats.center_distance_computations << ','
                       << outcome.stats.scanned_vectors << ',' << outcome.stats.total_distance_computations << ',' << outcome.stats.scanned_partitions << ','
                       << std::fixed << std::setprecision(8) << recall << ',' << std::setprecision(6) << outcome.ms << ','
                       << outcome.ms << '\n';
            m.query.hits += outcome.hits; m.query.center += outcome.stats.center_distance_computations;
            m.query.vectors += outcome.stats.scanned_vectors; m.query.total += outcome.stats.total_distance_computations;
            m.query.partitions += outcome.stats.scanned_partitions; m.query.ms.push_back(outcome.ms);
            if (outcome.top1_hit) ++top1_hits;
            hit_window.push_back(std::move(outcome.probed));
            while (hit_window.size() > kQueryWindow) hit_window.pop_front();
        }
        m.sse = measure_sse(data, index);
        const double nq = static_cast<double>(kQueries);
        const double recall = static_cast<double>(m.query.hits) / (nq * kTopK);
        std::size_t max_size = 0; for (const auto size : index.leaf_sizes()) max_size = std::max(max_size, size);
        m.max_list = max_size;
        const double mean_ms = std::accumulate(m.query.ms.begin(), m.query.ms.end(), 0.0) / nq;
        summary << std::fixed << std::setprecision(8) << name << ',' << stage << ',' << m.active << ',' << m.inserted << ','
            << m.nlist_before << ',' << m.nlist_after << ',' << m.max_list << ',' << std::setprecision(4)
            << m.insert_ms << ',' << m.maintenance_ms << ',' << m.insert_ms + m.maintenance_ms << ','
            << (stage ? kQueries : 0) << ',' << m.candidates << ',' << m.splits << ',' << std::setprecision(6)
            << recall << ',' << static_cast<double>(top1_hits) / nq << ',' << mean_ms << ','
            << percentile(m.query.ms, .50) << ',' << percentile(m.query.ms, .95) << ','
            << mean_ms << ',' << percentile(m.query.ms, .50) << ',' << percentile(m.query.ms, .95) << ','
            << m.query.center / nq << ',' << m.query.vectors / nq << ',' << m.query.total / nq << ','
            << m.query.partitions / nq << ',' << budget << ',' << m.sse << ',' << m.sse / m.active << '\n';
        summary.flush(); query_rows.flush(); candidates.flush();
        if (stage == kStages) {
            all_summary << name << ',' << recall << ',' << top1_hits / nq << ',' << m.nlist_after << ','
                << mean_ms << ',' << mean_ms << ',' << percentile(m.query.ms, .50) << ','
                << percentile(m.query.ms, .95) << ',' << (m.insert_ms + m.maintenance_ms) << ','
                << total_splits << '\n';
        }
        std::cout << "Quake case=" << name << " stage=" << std::setw(3) << std::setfill('0') << stage
                  << '/' << kStages << " active=" << m.active << " recall@10=" << std::fixed << std::setprecision(4) << recall
                  << " nlist=" << m.nlist_after << " splits=" << m.splits << " query_mean_ms=" << std::setprecision(3)
                  << mean_ms << " budget=" << budget << " threads=" << thread_count << '\n' << std::flush;
    }
    const auto final_index_path = case_out / "final.index.bin";
    index.save(final_index_path.string());
    manifest << "initial_build_ms=" << build_ms << "\nfinal_nlist=" << index.leaf_nlist()
             << "\nfinal_index_file=" << final_index_path.filename().string()
             << "\nfinal_index_bytes=" << std::filesystem::file_size(final_index_path) << "\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3 && argc != 5 && argc != 6 && argc != 7) {
            std::cerr << "Usage: deep1m_quake_freewayml_runner DATA_ROOT OUTPUT_DIR [budget_divisor query_threads [split_threshold_ns [refinement_radius]]]\n";
            return 2;
        }
        const std::filesystem::path data_root(argv[1]), out_root(argv[2]);
        const std::size_t budget_divisor = argc >= 5 ? static_cast<std::size_t>(std::stoull(argv[3])) : 0;
        const std::size_t query_threads = argc >= 5 ? static_cast<std::size_t>(std::stoull(argv[4])) : 1;
        const double split_threshold_ns = argc >= 6 ? std::stod(argv[5]) : kDefaultSplitThresholdNs;
        const std::size_t refinement_radius = argc >= 7 ? static_cast<std::size_t>(std::stoull(argv[6])) : kDefaultRefinementRadius;
        if ((argc >= 5 && budget_divisor == 0) || query_threads == 0 || refinement_radius == 0 ||
            !std::isfinite(split_threshold_ns) || split_threshold_ns < 0.0)
            throw std::invalid_argument("budget divisor and query thread count must be positive");
        if (std::filesystem::exists(out_root)) {
            if (!std::filesystem::is_directory(out_root)) throw std::runtime_error("output path is not a directory");
            for (const auto& entry : std::filesystem::directory_iterator(out_root))
                if (entry.path().filename() != "_prepared_truth")
                    throw std::runtime_error("output directory contains unexpected files: " + entry.path().string());
        } else {
            std::filesystem::create_directories(out_root);
        }
        if (!std::filesystem::is_directory(out_root / "_prepared_truth"))
            throw std::runtime_error("missing _prepared_truth; run prepare_deep1m_quake_truth.py first");
        std::ofstream all(out_root / "all_cases_summary.csv");
        std::ofstream progress(out_root / "progress.txt");
        if (!all || !progress) throw std::runtime_error("cannot create summary files");
        all << "case,final_recall_at_10,final_top1_hit_rate,final_nlist,final_query_mean_ms,final_query_core_mean_ms,final_query_core_p50_ms,final_query_core_p95_ms,final_update_ms,total_splits\n";
        progress << "method=Quake; split_threshold_ns=" << split_threshold_ns
                 << "; delete_threshold_ns=" << split_threshold_ns << " (unused: insert-only workload); local_refinement_radius=" << refinement_radius
                 << "; local_refinement_iterations=" << kRefinementIterations << "; initial_nlist=512; query_budget="
                 << (budget_divisor ? "floor(current_active_vectors/" + std::to_string(budget_divisor) + ")" : std::to_string(kDefaultBudget) + " fixed")
                 << "; query_threads=" << query_threads << "; LVQ8=false; early_stop=false\n";
        for (const char* name : kCases) {
            std::cout << "BEGIN Quake case=" << name << "\n" << std::flush;
            run_case(data_root, out_root, name, all, budget_divisor, query_threads, split_threshold_ns, refinement_radius);
            progress << name << " complete\n"; progress.flush(); all.flush();
        }
        std::ofstream complete(out_root / "COMPLETE.txt");
        complete << "completed_cases=9\nstages_per_case=101\nqueries_per_stage=1000\nquery_budget="
                 << (budget_divisor ? "floor(current_active_vectors/" + std::to_string(budget_divisor) + ")" : std::to_string(kDefaultBudget) + " fixed")
                 << "\nquery_threads=" << query_threads << "\nsplit_threshold_ns=" << split_threshold_ns
                 << "\nlocal_refinement_radius=" << refinement_radius << "\n"
                 << "query_timing=per-query native steady_clock around search_with_options; excludes recall and file output\n";
        std::cout << "ALL CASES COMPLETE: " << out_root.string() << '\n' << std::flush;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
