#include "../adaptive_hierarchical_ivf.h"
#include "../ivf_flat.h"

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
#include <sstream>
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
constexpr std::size_t kTopK = 10;
constexpr std::size_t kInitialNlist = 512;
constexpr std::size_t kInitialRows = 500'000;
constexpr std::size_t kBatchRows = 5'000;
constexpr std::size_t kStages = 100;
constexpr std::size_t kQueries = 1'000;
constexpr std::uint32_t kSeed = 1234;
constexpr std::array<const char*, 9> kCases{"G", "A1", "A2", "B1", "B2", "B3", "B4", "C1", "C2"};
using TruthRow = std::array<std::uint32_t, kTopK>;

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
        if (!mapped_) { close(); throw std::runtime_error("cannot view fbin"); }
#else
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) throw std::runtime_error("cannot open fbin: " + path.string());
        struct stat st {};
        if (::fstat(fd_, &st) != 0 || st.st_size < 8) { close(); throw std::runtime_error("cannot stat fbin"); }
        bytes_ = static_cast<std::uint64_t>(st.st_size);
        mapped_ = static_cast<const std::uint8_t*>(::mmap(nullptr, static_cast<std::size_t>(bytes_), PROT_READ, MAP_SHARED, fd_, 0));
        if (mapped_ == MAP_FAILED) { mapped_ = nullptr; close(); throw std::runtime_error("cannot mmap fbin"); }
#endif
        std::uint32_t shape[2]{};
        std::memcpy(shape, mapped_, sizeof(shape));
        rows_ = shape[0]; dimension_ = shape[1];
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

std::vector<TruthRow> read_truth(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open prepared truth: " + path.string());
    char magic[8]{};
    std::uint32_t shape[3]{};
    in.read(magic, sizeof(magic));
    in.read(reinterpret_cast<char*>(shape), sizeof(shape));
    if (!in || std::string(magic, sizeof(magic)) != "DLQTRTH1" ||
        shape[0] != kStages + 1 || shape[1] != kQueries || shape[2] != kTopK)
        throw std::runtime_error("prepared truth header mismatch: " + path.string());
    std::vector<TruthRow> truth((kStages + 1) * kQueries);
    in.read(reinterpret_cast<char*>(truth.data()), static_cast<std::streamsize>(truth.size() * sizeof(TruthRow)));
    if (!in || in.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("prepared truth payload size mismatch: " + path.string());
    return truth;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const std::size_t i = std::min(values.size() - 1,
        static_cast<std::size_t>(std::ceil(p * values.size()) - 1.0));
    return values[i];
}

void run_case(const std::filesystem::path& data_root,
              const std::filesystem::path& truth_root,
              const IvfFlat& shared_initial_index,
              const MappedFbin& reference_data,
              const std::filesystem::path& output_root,
              const char* case_name, std::size_t thread_count,
              std::ofstream& all_summary) {
    const std::filesystem::path case_dir = data_root / case_name;
    MappedFbin data(case_dir / "base.fbin");
    if (data.rows() != kInitialRows + kStages * kBatchRows)
        throw std::runtime_error(std::string("unexpected base row count for ") + case_name);
    if (std::memcmp(data.row(0), reference_data.row(0),
                    kInitialRows * kDimension * sizeof(float)) != 0)
        throw std::runtime_error(std::string("initial 500K prefix differs in ") + case_name);
    const auto truth = read_truth(truth_root / (std::string(case_name) + ".u32bin"));

    const auto case_output = output_root / case_name;
    std::filesystem::create_directories(case_output);
    std::ofstream stages(case_output / "stage_summary.csv", std::ios::trunc);
    std::ofstream per_query(case_output / "per_query.csv", std::ios::trunc);
    std::ofstream manifest(case_output / "manifest.txt", std::ios::trunc);
    if (!stages || !per_query || !manifest) throw std::runtime_error("cannot create case outputs");
    stages << "case,stage,active_vectors,nlist,budget_per_query,inserted_vectors,insert_ms,cumulative_insert_s,recall_at_10,top1_hit_rate,query_mean_ms,query_p50_ms,query_p95_ms,mean_center_distances,mean_scanned_vectors,mean_total_distances,mean_partitions_scanned,query_batch_wall_s\n";
    per_query << "stage,query_id,active_vectors,budget_per_query,center_distances,scanned_vectors,total_distances,partitions_scanned,hits_at_10,recall_at_10,query_ms\n";

    manifest << "method=project ordinary IVF-Flat; no maintenance\n"
             << "case=" << case_name << "\ndataset=DEEP-1M FreewayML-inspired controlled scenario\n"
             << "initial_index=shared ordinary IVF-Flat built from the shared centroid codebook\ninitial_vectors=500000; initial_nlist=512\n"
             << "insertion=100 batches x 5000; final_vectors=1000000\n"
             << "queries=stage 0 initial index, then after each insertion batch; 1000 matched queries per stage\n"
             << "query_budget=floor(current_active_vectors/100) exact total distance computations, including centers\n"
             << "query_threads=" << thread_count << "\nseed=" << kSeed
             << "\nmetric=squared L2 on independently normalized vectors; top_k=10\n"
             << "LVQ=false; early_stop=false; APS=false; maintenance=false; warmup=none\n";

    const auto copy_start = std::chrono::steady_clock::now();
    IvfFlat index = shared_initial_index;
    const double copy_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - copy_start).count();
    if (index.size() != kInitialRows || index.nlist() != kInitialNlist)
        throw std::runtime_error("shared initial IVF-Flat snapshot has unexpected size or nlist");
    manifest << "initial_index_copy_ms=" << std::setprecision(10) << copy_ms << '\n';

    std::uint64_t stage_recall_sum = 0;
    double total_insert_ms = 0.0, total_query_wall_s = 0.0;
    double final_recall = 0.0, final_top1 = 0.0;
    for (std::size_t stage = 0; stage <= kStages; ++stage) {
        std::size_t inserted = 0;
        double insert_ms = 0.0;
        if (stage > 0) {
            const std::size_t begin = kInitialRows + (stage - 1) * kBatchRows;
            const auto insert_start = std::chrono::steady_clock::now();
            index.add(data.row(begin), kBatchRows);
            insert_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - insert_start).count();
            inserted = kBatchRows;
            total_insert_ms += insert_ms;
        }
        const std::size_t active = kInitialRows + stage * kBatchRows;
        if (index.size() != active) throw std::runtime_error("active vector count mismatch");
        std::ostringstream query_name;
        query_name << "query_" << std::setw(3) << std::setfill('0') << stage << ".fbin";
        MappedFbin queries(case_dir / query_name.str());
        if (queries.rows() != kQueries) throw std::runtime_error("unexpected query count at stage " + std::to_string(stage));
        const std::size_t budget = active / 100;
        struct Outcome { IvfBudgetSearchStats stats; double ms = 0.0; std::size_t hits = 0; bool top1 = false; };
        std::vector<Outcome> outcomes(kQueries);
        std::atomic<std::size_t> next{0};
        std::exception_ptr error;
        std::mutex error_mutex;
        const auto batch_start = std::chrono::steady_clock::now();
        auto worker = [&]() {
            while (true) {
                const std::size_t qi = next.fetch_add(1);
                if (qi >= kQueries) return;
                try {
                    Outcome& out = outcomes[qi];
                    const auto query_start = std::chrono::steady_clock::now();
                    const auto found = index.search_with_distance_budget(queries.row(qi), kTopK, budget, &out.stats);
                    out.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - query_start).count();
                    if (found.size() != kTopK || out.stats.total_distance_computations != budget ||
                        out.stats.center_distance_computations != index.nlist() ||
                        out.stats.total_distance_computations != out.stats.center_distance_computations + out.stats.scanned_vectors)
                        throw std::runtime_error("fixed-budget query accounting mismatch");
                    const auto& expected = truth[stage * kQueries + qi];
                    for (const auto& item : found)
                        if (item.id >= 0 && std::find(expected.begin(), expected.end(), static_cast<std::uint32_t>(item.id)) != expected.end())
                            ++out.hits;
                    out.top1 = found.front().id >= 0 && static_cast<std::uint32_t>(found.front().id) == expected[0];
                } catch (...) {
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (!error) error = std::current_exception();
                    return;
                }
            }
        };
        std::vector<std::thread> workers;
        workers.reserve(thread_count);
        for (std::size_t i = 0; i < thread_count; ++i) workers.emplace_back(worker);
        for (auto& t : workers) t.join();
        if (error) std::rethrow_exception(error);
        const double batch_wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - batch_start).count();
        total_query_wall_s += batch_wall_s;

        std::uint64_t hits = 0, top1_hits = 0, centers = 0, vectors = 0, total = 0, partitions = 0;
        std::vector<double> query_ms;
        query_ms.reserve(kQueries);
        for (std::size_t qi = 0; qi < kQueries; ++qi) {
            const Outcome& out = outcomes[qi];
            hits += out.hits;
            top1_hits += out.top1 ? 1 : 0;
            centers += out.stats.center_distance_computations;
            vectors += out.stats.scanned_vectors;
            total += out.stats.total_distance_computations;
            partitions += out.stats.scanned_lists;
            query_ms.push_back(out.ms);
            per_query << stage << ',' << qi << ',' << active << ',' << budget << ','
                      << out.stats.center_distance_computations << ',' << out.stats.scanned_vectors << ','
                      << out.stats.total_distance_computations << ',' << out.stats.scanned_lists << ','
                      << out.hits << ',' << static_cast<double>(out.hits) / kTopK << ','
                      << std::setprecision(8) << out.ms << '\n';
        }
        const double recall = static_cast<double>(hits) / (kQueries * kTopK);
        const double top1_rate = static_cast<double>(top1_hits) / kQueries;
        stage_recall_sum += hits;
        final_recall = recall;
        final_top1 = top1_rate;
        stages << case_name << ',' << stage << ',' << active << ',' << index.nlist() << ',' << budget << ','
               << inserted << ',' << std::fixed << std::setprecision(4) << insert_ms << ','
               << std::setprecision(8) << total_insert_ms / 1000.0 << ',' << recall << ',' << top1_rate << ','
               << std::setprecision(6)
               << std::accumulate(query_ms.begin(), query_ms.end(), 0.0) / kQueries << ','
               << percentile(query_ms, .50) << ',' << percentile(query_ms, .95) << ','
               << static_cast<double>(centers) / kQueries << ',' << static_cast<double>(vectors) / kQueries << ','
               << static_cast<double>(total) / kQueries << ',' << static_cast<double>(partitions) / kQueries << ','
               << std::setprecision(6) << batch_wall_s << '\n';
        stages.flush();
        if ((stage + 1) % 10 == 0 || stage == kStages) {
            per_query.flush();
            std::cout << "IVF-Flat case=" << case_name << " stage=" << stage << '/' << kStages
                      << " active=" << active << " recall@10=" << std::fixed << std::setprecision(4) << recall
                      << " nlist=" << index.nlist() << " budget=" << budget
                      << " insert_total_s=" << std::setprecision(3) << total_insert_ms / 1000.0
                      << " query_batch_s=" << batch_wall_s << " threads=" << thread_count << '\n' << std::flush;
        }
    }
    const double mean_recall = static_cast<double>(stage_recall_sum) / ((kStages + 1) * kQueries * kTopK);
    all_summary << case_name << ',' << kInitialRows << ',' << index.size() << ',' << kInitialNlist << ','
                << std::setprecision(10) << mean_recall << ',' << final_recall << ',' << final_top1 << ','
                << total_insert_ms / 1000.0 << ',' << total_query_wall_s << ',' << copy_ms << '\n';
    all_summary.flush();
    manifest << "status=complete\nmean_recall_at_10=" << mean_recall << "\nfinal_recall_at_10=" << final_recall
             << "\nfinal_top1_hit_rate=" << final_top1 << "\ntotal_insert_s=" << total_insert_ms / 1000.0
             << "\ntotal_query_wall_s=" << total_query_wall_s << "\nfinal_nlist=" << index.nlist()
             << "\nfinal_index_saved=false\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 5 && argc != 6) {
            std::cerr << "Usage: deep1m_ivf_flat_freewayml_runner DATA_ROOT CENTROID_SNAPSHOT TRUTH_ROOT OUTPUT_ROOT [QUERY_THREADS]\n";
            return 2;
        }
        const std::filesystem::path data_root(argv[1]);
        const std::filesystem::path initial_index(argv[2]);
        const std::filesystem::path truth_root(argv[3]);
        const std::filesystem::path output_root(argv[4]);
        const std::size_t threads = argc == 6 ? static_cast<std::size_t>(std::stoull(argv[5])) : 16;
        if (threads == 0 || !std::filesystem::is_regular_file(initial_index) ||
            !std::filesystem::is_directory(data_root) || !std::filesystem::is_directory(truth_root))
            throw std::invalid_argument("invalid paths or query thread count");
        if (std::filesystem::exists(output_root)) {
            if (!std::filesystem::is_directory(output_root) || !std::filesystem::is_empty(output_root))
                throw std::runtime_error("output directory must be new or empty: " + output_root.string());
        } else {
            std::filesystem::create_directories(output_root);
        }
        std::ofstream summary(output_root / "all_cases_summary.csv", std::ios::trunc);
        std::ofstream manifest(output_root / "manifest.txt", std::ios::trunc);
        if (!summary || !manifest) throw std::runtime_error("cannot create experiment summary");
        summary << "case,initial_vectors,final_vectors,final_nlist,mean_recall_at_10,final_recall_at_10,final_top1_hit_rate,total_insert_s,total_query_wall_s,initial_index_copy_ms\n";
        manifest << "experiment=DEEP-1M FreewayML-inspired nine-scenario project ordinary IVF-Flat; full staged queries\n"
                 << "cases=G,A1,A2,B1,B2,B3,B4,C1,C2\ndata_root=" << data_root.string()
                 << "\ntruth_root=" << truth_root.string() << "\ncentroid_snapshot=" << initial_index.string()
                 << "\ninitial_vectors=500000; nlist=512; insert_batches=100; batch_size=5000; final_vectors=1000000"
                 << "\nquery_schedule=stage 0 initial index, then after every batch; 101 stages; 1000 queries per stage"
                 << "\nquery_budget=floor(current active N/100) total distance computations per query"
                 << "\nquery_threads=" << threads << "\nmaintenance=none; LVQ=false; early_stop=false; APS=false; warmup=none"
                 << "\nmetric=squared L2 on independently normalized vectors; top_k=10\nseed=" << kSeed << '\n';
        MappedFbin reference_data(data_root / "G" / "base.fbin");
        AdaptiveHierarchicalIvf centroid_snapshot = AdaptiveHierarchicalIvf::load(initial_index.string());
        const auto& cfg = centroid_snapshot.config();
        if (centroid_snapshot.size() != kInitialRows || centroid_snapshot.leaf_nlist() != kInitialNlist ||
            cfg.dimension != kDimension || cfg.lvq8_storage || cfg.lvq4_storage ||
            cfg.angular_metric || cfg.uint8_storage)
            throw std::runtime_error("centroid snapshot metadata does not match the shared DEEP-1M initial index");
        const std::vector<float> centroids = centroid_snapshot.leaf_centroids();
        IvfFlat shared_initial_index(kDimension, kInitialNlist, kSeed);
        shared_initial_index.train_with_centroids(centroids.data());
        const auto initial_import_start = std::chrono::steady_clock::now();
        shared_initial_index.add(reference_data.row(0), kInitialRows);
        const double initial_import_s = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - initial_import_start).count();
        if (shared_initial_index.size() != kInitialRows)
            throw std::runtime_error("could not import shared initial vectors into project IVF-Flat");
        manifest << "shared_ivf_flat_initial_import_s=" << std::setprecision(10) << initial_import_s << '\n';
        manifest.flush();
        for (const char* name : kCases) {
            std::cout << "BEGIN IVF-Flat case=" << name << '\n' << std::flush;
            run_case(data_root, truth_root, shared_initial_index, reference_data, output_root, name, threads, summary);
            manifest << name << "=complete\n";
            manifest.flush();
        }
        std::ofstream complete(output_root / "COMPLETE.txt", std::ios::trunc);
        complete << "status=complete\ncases=9\nstages_per_case=101\nqueries_per_stage=1000\nquery_threads=" << threads
                 << "\nquery_budget=floor(current_active_N/100)\nmaintenance=none; method=project IVF-Flat\n";
        std::cout << "ALL IVF-Flat CASES COMPLETE output=" << output_root.string() << '\n' << std::flush;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
