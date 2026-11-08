#include "../ivf_quantized_common.h"
#include "../simd_l2.h"

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
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
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
constexpr std::size_t kDimension = 96;
constexpr std::size_t kTopK = 10;
constexpr std::size_t kInitialNlist = 512;
constexpr std::size_t kInitialRows = 500'000;
constexpr std::size_t kBatchRows = 5'000;
constexpr std::size_t kStages = 100;
constexpr std::size_t kQueries = 1'000;
constexpr std::size_t kMaxPostingLength = 1'000;
constexpr std::size_t kReassignRange = 64;
constexpr std::size_t kDefaultBudget = 50'000;
constexpr std::size_t kKmeansIterations = 10;
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

struct Hit { float distance; std::uint32_t id; };
bool hit_better(const Hit& a, const Hit& b) {
    return a.distance == b.distance ? a.id < b.id : a.distance < b.distance;
}
void offer(std::vector<Hit>& best, Hit candidate) {
    auto at = std::lower_bound(best.begin(), best.end(), candidate, hit_better);
    if (best.size() == kTopK && at == best.end()) return;
    best.insert(at, candidate);
    if (best.size() > kTopK) best.pop_back();
}
float point_distance(const float* a, const float* b) {
    return simd_squared_l2(a, b, kDimension);
}

struct Partition {
    std::vector<std::uint32_t> ids;
    // Keep vector payload in posting order, matching the packed per-partition
    // layout used by the other in-memory IVF implementations in this suite.
    std::vector<float> vectors;
    std::vector<float> center;
};
struct Move { std::size_t from; std::size_t to; std::uint32_t id; };
struct Maintenance { std::size_t splits = 0, candidates = 0, moved = 0; };
struct Search { std::size_t centers = 0, vectors = 0, partitions = 0; std::vector<std::uint32_t> ids; };

class LireIndex {
public:
    explicit LireIndex(const MappedFbin& data) : data_(data) {}

    void initialize(const std::vector<float>& centers, std::size_t count) {
        parts_.resize(centers.size() / kDimension);
        for (std::size_t c = 0; c < parts_.size(); ++c)
            parts_[c].center.assign(centers.data() + c * kDimension,
                                    centers.data() + (c + 1) * kDimension);
        for (std::size_t id = 0; id < count; ++id)
            append_vector(parts_[nearest(data_.row(id))], static_cast<std::uint32_t>(id));
        active_ = count;
    }

    void add(std::size_t begin, std::size_t count) {
        for (std::size_t id = begin; id < begin + count; ++id)
            append_vector(parts_[nearest(data_.row(id))], static_cast<std::uint32_t>(id));
        active_ += count;
    }

    Maintenance maintain_lire() {
        Maintenance stats;
        std::vector<std::size_t> pending;
        std::vector<std::uint8_t> queued(parts_.size(), 0);
        auto enqueue_oversized = [&]() {
            if (queued.size() < parts_.size()) queued.resize(parts_.size(), 0);
            for (std::size_t i = 0; i < parts_.size(); ++i)
                if (parts_[i].ids.size() > kMaxPostingLength && !queued[i]) {
                    pending.push_back(i); queued[i] = 1;
                }
        };
        enqueue_oversized();
        while (!pending.empty()) {
            const std::size_t id = pending.back(); pending.pop_back();
            queued[id] = 0;
            if (parts_[id].ids.size() <= kMaxPostingLength) continue;
            split_and_reassign(id, stats);
            if (stats.splits % 100 == 0)
                std::cout << "SPFresh-LIRE maintenance splits=" << stats.splits
                          << " pending=" << pending.size() << " nlist=" << parts_.size() << '\n' << std::flush;
            enqueue_oversized();
        }
        return stats;
    }

    std::size_t nlist() const { return parts_.size(); }
    std::size_t size() const { return active_; }
    std::size_t max_size() const {
        std::size_t value = 0;
        for (const auto& part : parts_) value = std::max(value, part.ids.size());
        return value;
    }
    double total_sse() const {
        double result = 0.0;
        for (const auto& part : parts_)
            for (std::size_t position = 0; position < part.ids.size(); ++position)
                result += point_distance(part.vectors.data() + position * kDimension,
                                         part.center.data());
        return result;
    }
    void validate_population() const {
        std::vector<std::uint8_t> seen(active_, 0);
        std::size_t total = 0;
        for (const auto& part : parts_) {
            if (part.vectors.size() != part.ids.size() * kDimension)
                throw std::runtime_error("LIRE posting IDs and packed vector payload are misaligned");
            for (const auto id : part.ids) {
                if (id >= active_ || seen[id]++) throw std::runtime_error("LIRE has invalid or duplicate ownership");
                ++total;
            }
        }
        if (total != active_ || std::find(seen.begin(), seen.end(), 0) != seen.end())
            throw std::runtime_error("LIRE posting ownership does not cover active vectors exactly once");
    }

    Search search_budget(const float* query, std::size_t budget) const {
        Search result;
        std::vector<std::pair<float, std::size_t>> ranked;
        ranked.reserve(parts_.size());
        for (std::size_t i = 0; i < parts_.size(); ++i)
            ranked.emplace_back(point_distance(query, parts_[i].center.data()), i);
        result.centers = ranked.size();
        if (budget <= result.centers) throw std::runtime_error("scan budget smaller than current nlist");
        std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
            return a.first == b.first ? a.second < b.second : a.first < b.first;
        });
        const std::size_t vector_budget = budget - result.centers;
        std::vector<Hit> best;
        best.reserve(kTopK + 1);
        for (const auto& item : ranked) {
            const Partition& part = parts_[item.second];
            for (std::size_t position = 0; position < part.ids.size(); ++position) {
                offer(best, Hit{point_distance(query,
                        part.vectors.data() + position * kDimension), part.ids[position]});
                if (++result.vectors == vector_budget) { ++result.partitions; break; }
            }
            if (result.vectors == vector_budget) break;
            if (!part.ids.empty()) ++result.partitions;
        }
        if (result.vectors != vector_budget) throw std::runtime_error("LIRE could not consume exact scan budget");
        for (const auto& hit : best) result.ids.push_back(hit.id);
        return result;
    }

private:
    void append_vector(Partition& part, std::uint32_t id) {
        const float* vector = data_.row(id);
        part.ids.push_back(id);
        part.vectors.insert(part.vectors.end(), vector, vector + kDimension);
    }

    void rebuild_payload(Partition& part) {
        part.vectors.clear();
        part.vectors.reserve(part.ids.size() * kDimension);
        for (const auto id : part.ids) {
            const float* vector = data_.row(id);
            part.vectors.insert(part.vectors.end(), vector, vector + kDimension);
        }
    }

    std::size_t nearest(const float* vector) const {
        std::size_t best = 0;
        float best_distance = point_distance(vector, parts_[0].center.data());
        for (std::size_t c = 1; c < parts_.size(); ++c) {
            const float distance = point_distance(vector, parts_[c].center.data());
            if (distance < best_distance) { best = c; best_distance = distance; }
        }
        return best;
    }

    struct Split { std::array<std::vector<std::uint32_t>, 2> ids; std::array<std::vector<float>, 2> centers; };
    Split balanced_split(const std::vector<std::uint32_t>& ids, std::uint32_t seed) const {
        if (ids.size() < 2) throw std::runtime_error("cannot split a posting with fewer than two vectors");
        std::mt19937 rng(seed);
        const std::size_t first = static_cast<std::size_t>(rng()) % ids.size();
        const float* p0 = data_.row(ids[first]);
        std::size_t second = first;
        float farthest = -1.0f;
        for (std::size_t i = 0; i < ids.size(); ++i) {
            const float distance = point_distance(p0, data_.row(ids[i]));
            if (distance > farthest) { farthest = distance; second = i; }
        }
        Split result;
        result.centers[0].assign(p0, p0 + kDimension);
        const float* p1 = data_.row(ids[second]);
        result.centers[1].assign(p1, p1 + kDimension);
        std::vector<std::size_t> assignment(ids.size(), 0);
        for (std::size_t iteration = 0; iteration < 5; ++iteration) {
            std::vector<std::pair<float, float>> distances(ids.size());
            std::vector<std::pair<float, std::size_t>> regrets(ids.size());
            for (std::size_t i = 0; i < ids.size(); ++i) {
                const float* point = data_.row(ids[i]);
                const float d0 = point_distance(point, result.centers[0].data());
                const float d1 = point_distance(point, result.centers[1].data());
                distances[i] = {d0, d1};
                regrets[i] = {std::abs(d0 - d1), i};
            }
            std::sort(regrets.begin(), regrets.end(), [](const auto& a, const auto& b) {
                return a.first == b.first ? a.second < b.second : a.first > b.first;
            });
            result.ids[0].clear(); result.ids[1].clear();
            const std::array<std::size_t, 2> capacity{(ids.size() + 1) / 2, ids.size() / 2};
            std::array<std::size_t, 2> sizes{};
            std::array<std::vector<double>, 2> sums{std::vector<double>(kDimension, 0.0),
                                                    std::vector<double>(kDimension, 0.0)};
            for (const auto& item : regrets) {
                const std::size_t i = item.second;
                const std::size_t preferred = distances[i].first <= distances[i].second ? 0 : 1;
                const std::size_t chosen = sizes[preferred] < capacity[preferred] ? preferred : 1 - preferred;
                assignment[i] = chosen;
                ++sizes[chosen];
                const float* point = data_.row(ids[i]);
                for (std::size_t d = 0; d < kDimension; ++d) sums[chosen][d] += point[d];
            }
            for (std::size_t c = 0; c < 2; ++c) {
                result.centers[c].resize(kDimension);
                for (std::size_t d = 0; d < kDimension; ++d)
                    result.centers[c][d] = static_cast<float>(sums[c][d] / sizes[c]);
            }
        }
        for (std::size_t i = 0; i < ids.size(); ++i) result.ids[assignment[i]].push_back(ids[i]);
        if (result.ids[0].empty() || result.ids[1].empty()) throw std::runtime_error("balanced split produced an empty child");
        return result;
    }

    void split_and_reassign(std::size_t leaf, Maintenance& stats) {
        const auto old_ids = parts_[leaf].ids;
        const auto old_center = parts_[leaf].center;
        std::vector<std::pair<float, std::size_t>> neighbor_rank;
        for (std::size_t i = 0; i < parts_.size(); ++i)
            if (i != leaf && !parts_[i].ids.empty())
                neighbor_rank.emplace_back(point_distance(old_center.data(), parts_[i].center.data()), i);
        const std::size_t neighbor_count = std::min(kReassignRange, neighbor_rank.size());
        std::partial_sort(neighbor_rank.begin(), neighbor_rank.begin() + neighbor_count, neighbor_rank.end(),
            [](const auto& a, const auto& b) { return a.first == b.first ? a.second < b.second : a.first < b.first; });
        std::vector<std::size_t> neighbors;
        for (std::size_t i = 0; i < neighbor_count; ++i) neighbors.push_back(neighbor_rank[i].second);

        const std::uint32_t seed = kSeed + static_cast<std::uint32_t>(active_) * 17u +
                                   static_cast<std::uint32_t>(leaf * 31u + stats.splits);
        auto split = balanced_split(old_ids, seed);
        parts_[leaf].ids = std::move(split.ids[0]);
        parts_[leaf].center = std::move(split.centers[0]);
        rebuild_payload(parts_[leaf]);
        const std::size_t child = parts_.size();
        Partition child_partition;
        child_partition.ids = std::move(split.ids[1]);
        child_partition.center = std::move(split.centers[1]);
        rebuild_payload(child_partition);
        parts_.push_back(std::move(child_partition));
        std::unordered_set<std::uint32_t> left_ids(parts_[leaf].ids.begin(), parts_[leaf].ids.end());

        std::vector<std::pair<std::size_t, std::uint32_t>> candidates;
        for (const auto id : old_ids) {
            const float* point = data_.row(id);
            const float d0 = point_distance(point, old_center.data());
            const float d1 = point_distance(point, parts_[leaf].center.data());
            const float d2 = point_distance(point, parts_[child].center.data());
            if (d0 <= d1 && d0 <= d2) candidates.emplace_back(left_ids.count(id) ? leaf : child, id);
        }
        for (const auto neighbor : neighbors) {
            const auto ids = parts_[neighbor].ids;
            for (const auto id : ids) {
                const float* point = data_.row(id);
                const float old_distance = point_distance(point, old_center.data());
                const float d1 = point_distance(point, parts_[leaf].center.data());
                const float d2 = point_distance(point, parts_[child].center.data());
                if (std::min(d1, d2) <= old_distance) candidates.emplace_back(neighbor, id);
            }
        }
        stats.candidates += candidates.size();
        std::vector<Move> moves;
        moves.reserve(candidates.size());
        for (const auto& candidate : candidates) {
            const std::size_t from = candidate.first;
            const std::uint32_t id = candidate.second;
            const float* point = data_.row(id);
            const float current = point_distance(point, parts_[from].center.data());
            std::size_t best = from;
            float best_distance = current;
            for (std::size_t c = 0; c < parts_.size(); ++c) {
                const float distance = point_distance(point, parts_[c].center.data());
                if (distance < best_distance) { best = c; best_distance = distance; }
            }
            if (best != from) moves.push_back({from, best, id});
        }
        std::unordered_map<std::size_t, std::unordered_set<std::uint32_t>> moved_by_source;
        for (const auto& move : moves) moved_by_source[move.from].insert(move.id);
        for (const auto& item : moved_by_source) {
            Partition& source = parts_[item.first];
            std::vector<std::uint32_t> kept_ids;
            std::vector<float> kept_vectors;
            kept_ids.reserve(source.ids.size());
            kept_vectors.reserve(source.vectors.size());
            for (std::size_t position = 0; position < source.ids.size(); ++position) {
                if (item.second.count(source.ids[position]) != 0) continue;
                kept_ids.push_back(source.ids[position]);
                const float* vector = source.vectors.data() + position * kDimension;
                kept_vectors.insert(kept_vectors.end(), vector, vector + kDimension);
            }
            source.ids.swap(kept_ids);
            source.vectors.swap(kept_vectors);
        }
        for (const auto& move : moves) append_vector(parts_[move.to], move.id);
        stats.moved += moves.size();
        ++stats.splits;
    }

    const MappedFbin& data_;
    std::vector<Partition> parts_;
    std::size_t active_ = 0;
};

struct QueryMetrics {
    std::uint64_t hits = 0, top1_hits = 0, centers = 0, vectors = 0, total = 0, partitions = 0;
    std::vector<double> ms;
};

void run_case(const std::filesystem::path& data_root, const std::filesystem::path& truth_root,
              const std::filesystem::path& out_root, const char* case_name, std::ofstream& all_summary,
              std::size_t budget_divisor, std::size_t query_threads) {
    const std::string name(case_name);
    const auto case_dir = data_root / name;
    MappedFbin data(case_dir / "base.fbin");
    if (data.rows() != kInitialRows + kStages * kBatchRows)
        throw std::runtime_error("base row count mismatch for " + name);
    const auto truth = read_truth(truth_root / "_prepared_truth" / (name + ".u32bin"));
    const auto case_out = out_root / name;
    std::filesystem::create_directories(case_out);
    std::ofstream summary(case_out / "stage_summary.csv"), query_rows(case_out / "per_query.csv"), manifest(case_out / "manifest.txt");
    if (!summary || !query_rows || !manifest) throw std::runtime_error("cannot create LIRE output files for " + name);
    summary << "case,stage,active_vectors,inserted_vectors,nlist_before,nlist_after,max_list_size,build_ms,insert_ms,maintenance_ms,update_ms,initial_or_stage_splits,npa_candidates,reassigned_vectors,recall_at_10,top1_hit_rate,query_mean_ms,query_p50_ms,query_p95_ms,query_core_mean_ms,query_core_p50_ms,query_core_p95_ms,mean_center_distance_computations,mean_vector_distance_computations,mean_total_distance_computations,mean_partitions_scanned,scan_budget_per_query,total_sse,mean_sse_per_vector\n";
    query_rows << "stage,query_index,scan_budget,center_distance_computations,vector_distance_computations,total_distance_computations,partitions_scanned,recall_at_10,query_ms,search_core_ms\n";
    manifest << "method=SPFresh-LIRE local split and reassignment protocol adapted to in-memory IVF\n"
        << "case=" << name << "\ndataset=DEEP-1M FreewayML-style controlled-drift scenarios; 96-D FP32; squared L2 on normalized vectors\n"
        << "initial_vectors=500000; batches=100; inserted_per_batch=5000; queries_per_stage=1000; seed=1234\n"
        << "initial_nlist=512; kmeans_iterations=10; split when posting size > 1000; balanced binary split\n"
        << "reassign_range=64 nearest postings; SPFresh two NPA candidate conditions; exact nearest-centroid rerouting\n"
        << "insert then LIRE maintenance then query; initial posting-size repair before stage 0\n"
        << "total scan budget="
        << (budget_divisor ? "floor(current_active_vectors/" + std::to_string(budget_divisor) + ")" : std::to_string(kDefaultBudget) + " fixed")
        << " per query, including center and vector distance calculations; query_threads=" << query_threads
        << "; early stopping=false; LVQ8=false\n"
        << "storage=each posting stores IDs and a contiguous FP32 vector payload in matching order; queries scan the in-index payload, not the source matrix by ID\n"
        << "query timing: per-query steady_clock around the native search_budget call; excludes truth evaluation and file output\n"
        << "queries and exact stage top-10 truth are reused from the aligned Quake FreewayML run\n"
        << "index snapshots are not saved\n";

    const auto build_start = std::chrono::steady_clock::now();
    std::vector<float> initial(kInitialRows * kDimension);
    for (std::size_t i = 0; i < kInitialRows; ++i)
        std::copy_n(data.row(i), kDimension, initial.data() + i * kDimension);
    const auto centers = ivf_quantized_detail::train_kmeans(initial, kInitialRows, kDimension,
        kInitialNlist, kKmeansIterations, kSeed);
    LireIndex index(data);
    index.initialize(centers, kInitialRows);
    const double build_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - build_start).count();
    initial.clear(); initial.shrink_to_fit();
    auto start = std::chrono::steady_clock::now();
    const auto initial_maintenance = index.maintain_lire();
    const double initial_maintenance_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    index.validate_population();

    double final_recall = 0.0, final_query_mean_ms = 0.0, final_query_p50_ms = 0.0,
        final_query_p95_ms = 0.0, final_update_ms = 0.0, final_sse = 0.0;
    std::size_t final_stage_splits = 0, total_splits = initial_maintenance.splits;
    for (std::size_t stage = 0; stage <= kStages; ++stage) {
        const std::size_t active = kInitialRows + stage * kBatchRows;
        const std::size_t nlist_before = index.nlist();
        std::size_t inserted = 0, splits = 0, candidates = 0, moved = 0;
        double insert_ms = 0.0, maintenance_ms = 0.0;
        if (stage == 0) {
            splits = initial_maintenance.splits;
            candidates = initial_maintenance.candidates;
            moved = initial_maintenance.moved;
            maintenance_ms = initial_maintenance_ms;
        } else {
            start = std::chrono::steady_clock::now();
            index.add(kInitialRows + (stage - 1) * kBatchRows, kBatchRows);
            insert_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            inserted = kBatchRows;
            start = std::chrono::steady_clock::now();
            const auto maintenance = index.maintain_lire();
            maintenance_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            splits = maintenance.splits;
            candidates = maintenance.candidates;
            moved = maintenance.moved;
        }
        total_splits += stage ? splits : 0;
        if (index.size() != active) throw std::runtime_error("LIRE active population mismatch in " + name);
        index.validate_population();

        std::ostringstream query_filename;
        query_filename << "query_" << std::setw(3) << std::setfill('0') << stage << ".fbin";
        MappedFbin queries(case_dir / query_filename.str());
        if (queries.rows() != kQueries) throw std::runtime_error("query count mismatch in " + name + " stage " + std::to_string(stage));
        QueryMetrics qmetric;
        qmetric.ms.reserve(kQueries);
        const std::size_t budget = budget_divisor ? active / budget_divisor : kDefaultBudget;
        std::vector<Search> found_rows(kQueries);
        std::vector<double> elapsed_ms(kQueries, 0.0);
        std::exception_ptr worker_error;
        std::mutex worker_error_mutex;
        std::atomic<std::size_t> next_query{0};
        const std::size_t thread_count = std::min(query_threads, kQueries);
        auto worker = [&]() {
            while (true) {
                const std::size_t qi = next_query.fetch_add(1);
                if (qi >= kQueries) return;
                try {
                    const auto query_start = std::chrono::steady_clock::now();
                    found_rows[qi] = index.search_budget(queries.row(qi), budget);
                    elapsed_ms[qi] = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - query_start).count();
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
        for (std::size_t qi = 0; qi < kQueries; ++qi) {
            const Search& found = found_rows[qi];
            const double elapsed = elapsed_ms[qi];
            std::size_t hits = 0;
            for (const auto id : found.ids)
                if (std::find(truth_at(truth, stage, qi).begin(), truth_at(truth, stage, qi).end(), id) !=
                    truth_at(truth, stage, qi).end()) ++hits;
            if (!found.ids.empty() && found.ids.front() == truth_at(truth, stage, qi)[0]) ++qmetric.top1_hits;
            qmetric.hits += hits;
            qmetric.centers += found.centers;
            qmetric.vectors += found.vectors;
            qmetric.total += found.centers + found.vectors;
            qmetric.partitions += found.partitions;
            qmetric.ms.push_back(elapsed);
            query_rows << stage << ',' << qi << ',' << budget << ',' << found.centers << ','
                << found.vectors << ',' << found.centers + found.vectors << ',' << found.partitions << ','
                << std::fixed << std::setprecision(8) << static_cast<double>(hits) / kTopK << ','
                << std::setprecision(6) << elapsed << ',' << elapsed << '\n';
        }
        const double nq = static_cast<double>(kQueries);
        const double recall = static_cast<double>(qmetric.hits) / (nq * kTopK);
        const double top1_rate = static_cast<double>(qmetric.top1_hits) / nq;
        const double query_mean = std::accumulate(qmetric.ms.begin(), qmetric.ms.end(), 0.0) / nq;
        const double sse = index.total_sse();
        const std::size_t max_list = index.max_size();
        summary << std::fixed << std::setprecision(8) << name << ',' << stage << ',' << active << ',' << inserted << ','
            << nlist_before << ',' << index.nlist() << ',' << max_list << ',' << std::setprecision(4) << build_ms << ','
            << insert_ms << ',' << maintenance_ms << ',' << insert_ms + maintenance_ms << ',' << splits << ','
            << candidates << ',' << moved << ',' << std::setprecision(6) << recall << ',' << top1_rate << ','
            << query_mean << ',' << percentile(qmetric.ms, .50) << ',' << percentile(qmetric.ms, .95) << ','
            << query_mean << ',' << percentile(qmetric.ms, .50) << ',' << percentile(qmetric.ms, .95) << ','
            << static_cast<double>(qmetric.centers) / nq << ',' << static_cast<double>(qmetric.vectors) / nq << ','
            << static_cast<double>(qmetric.total) / nq << ',' << static_cast<double>(qmetric.partitions) / nq << ','
            << budget << ',' << sse << ',' << sse / active << '\n';
        summary.flush(); query_rows.flush();
        if (stage == kStages) {
            final_recall = recall;
            final_query_mean_ms = query_mean;
            final_query_p50_ms = percentile(qmetric.ms, .50);
            final_query_p95_ms = percentile(qmetric.ms, .95);
            final_update_ms = insert_ms + maintenance_ms;
            final_sse = sse;
            final_stage_splits = splits;
        }
        std::cout << "SPFresh-LIRE case=" << name << " stage=" << std::setw(3) << std::setfill('0') << stage
            << '/' << kStages << " active=" << active << " nlist=" << index.nlist() << " splits=" << splits
            << " recall@10=" << std::fixed << std::setprecision(4) << recall << " query_mean_ms="
            << std::setprecision(3) << query_mean << " budget=" << budget << " threads=" << thread_count
            << " update_ms=" << std::setprecision(1) << insert_ms + maintenance_ms << '\n' << std::flush;
    }
    all_summary << std::fixed << std::setprecision(6) << name << ',' << final_recall << ','
        << final_query_mean_ms << ',' << final_query_mean_ms << ','
        << final_query_p50_ms << ',' << final_query_p95_ms << ','
        << final_update_ms << ',' << index.nlist() << ','
        << index.max_size() << ',' << total_splits << ',' << final_stage_splits << ',' << final_sse << ','
        << build_ms << ",0\n";
    manifest << "final_nlist=" << index.nlist() << "\nfinal_max_posting_size=" << index.max_size()
        << "\nfinal_active_vectors=" << index.size() << "\nindex_file_saved=false\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 4 && argc != 6) {
            std::cerr << "Usage: deep1m_spfresh_lire_freewayml_runner DATA_ROOT TRUTH_RUN_ROOT OUTPUT_DIR [budget_divisor query_threads]\n";
            return 2;
        }
        const std::filesystem::path data_root(argv[1]), truth_root(argv[2]), out_root(argv[3]);
        const std::size_t budget_divisor = argc == 6 ? static_cast<std::size_t>(std::stoull(argv[4])) : 0;
        const std::size_t query_threads = argc == 6 ? static_cast<std::size_t>(std::stoull(argv[5])) : 1;
        if ((argc == 6 && budget_divisor == 0) || query_threads == 0)
            throw std::invalid_argument("budget divisor and query thread count must be positive");
        if (!std::filesystem::is_directory(data_root) || !std::filesystem::is_directory(truth_root / "_prepared_truth"))
            throw std::runtime_error("data root or prepared truth directory is missing");
        if (std::filesystem::exists(out_root)) {
            if (!std::filesystem::is_directory(out_root) ||
                std::filesystem::directory_iterator(out_root) != std::filesystem::directory_iterator())
                throw std::runtime_error("output directory must be absent or empty: " + out_root.string());
        } else {
            std::filesystem::create_directories(out_root);
        }
        std::ofstream all(out_root / "all_cases_summary.csv"), progress(out_root / "progress.txt");
        if (!all || !progress) throw std::runtime_error("cannot create LIRE run summaries");
        all << "case,final_recall_at_10,final_query_mean_ms,final_query_core_mean_ms,final_query_core_p50_ms,final_query_core_p95_ms,final_update_ms,final_nlist,final_max_posting_size,total_splits,final_stage_splits,final_sse,initial_build_ms,index_files_saved\n";
        progress << "method=SPFresh-LIRE; initial_nlist=512; query_budget="
                 << (budget_divisor ? "floor(current_active_vectors/" + std::to_string(budget_divisor) + ")" : std::to_string(kDefaultBudget) + " fixed")
                 << "; query_threads=" << query_threads << "; LVQ8=false; early_stop=false; index_snapshots_saved=false\n"
                  << "query_timing=per-query native steady_clock around search_budget; excludes recall and file output\n";
        for (const char* name : kCases) {
            std::cout << "BEGIN SPFresh-LIRE case=" << name << '\n' << std::flush;
            run_case(data_root, truth_root, out_root, name, all, budget_divisor, query_threads);
            progress << name << " complete\n"; progress.flush(); all.flush();
        }
        std::ofstream complete(out_root / "COMPLETE.txt");
        complete << "completed_cases=9\nstages_per_case=101\nqueries_per_stage=1000\nquery_budget="
                 << (budget_divisor ? "floor(current_active_vectors/" + std::to_string(budget_divisor) + ")" : std::to_string(kDefaultBudget) + " fixed")
                 << "\nquery_threads=" << query_threads << "\nindex_snapshots_saved=false\n"
                 << "query_timing=per-query native steady_clock around search_budget; excludes recall and file output\n";
        std::cout << "ALL 9 SPFresh-LIRE CASES COMPLETE: " << out_root.string() << '\n' << std::flush;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
