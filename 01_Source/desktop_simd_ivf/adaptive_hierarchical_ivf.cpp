#include "adaptive_hierarchical_ivf.h"
#include "simd_l2.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace {
constexpr char kAdaptiveMagic[] = "PRDIFV01";
constexpr std::uint32_t kAdaptiveVersion = 17;

bool uses_lvq_storage(const AdaptiveHierarchicalConfig& config) {
    return config.lvq8_storage || config.lvq4_storage;
}

bool uses_u8_storage(const AdaptiveHierarchicalConfig& config) {
    return config.uint8_storage;
}

float dynamic_load_target_size(std::size_t active_vectors,
                              std::size_t live_leaf_count) noexcept {
    if (active_vectors == 0 || live_leaf_count == 0) return 0.0f;
    // The balanced load target uses every live partition, independently of
    // the sqrt(N) condition that controls capacity-driven growth.
    return static_cast<float>(static_cast<double>(active_vectors) /
                              static_cast<double>(live_leaf_count));
}

std::size_t lvq_code_bytes_per_vector(const AdaptiveHierarchicalConfig& config) {
    return config.lvq4_storage ? config.dimension / 2 : config.dimension;
}

std::size_t effective_worker_count(std::size_t count, std::size_t requested) {
    return std::min(count, std::max<std::size_t>(1, requested));
}

template <typename Function>
void parallel_for_static(std::size_t count, std::size_t requested_workers,
                         Function&& function) {
    const std::size_t workers = effective_worker_count(count, requested_workers);
    if (workers <= 1) {
        for (std::size_t i = 0; i < count; ++i) function(0, i);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(workers - 1);
    for (std::size_t worker = 1; worker < workers; ++worker) {
        const std::size_t begin = count * worker / workers;
        const std::size_t end = count * (worker + 1) / workers;
        threads.emplace_back([&, worker, begin, end] {
            for (std::size_t i = begin; i < end; ++i) function(worker, i);
        });
    }
    const std::size_t end = count / workers;
    for (std::size_t i = 0; i < end; ++i) function(0, i);
    for (auto& thread : threads) thread.join();
}

template <typename T>
void write_scalar(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("failed writing adaptive IVF index");
}

template <typename T>
T read_scalar(std::ifstream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("truncated adaptive IVF index");
    return value;
}

void write_size(std::ofstream& out, std::size_t value) {
    write_scalar(out, static_cast<std::uint64_t>(value));
}

std::size_t read_size(std::ifstream& in) {
    const std::uint64_t value = read_scalar<std::uint64_t>(in);
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error("adaptive IVF index size does not fit this platform");
    return static_cast<std::size_t>(value);
}

template <typename T>
void write_vector(std::ofstream& out, const std::vector<T>& values) {
    write_size(out, values.size());
    if (!values.empty()) {
        out.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(values.size() * sizeof(T)));
        if (!out) throw std::runtime_error("failed writing adaptive IVF vector");
    }
}

template <typename T>
std::vector<T> read_vector(std::ifstream& in, std::uintmax_t file_size) {
    const std::size_t count = read_size(in);
    if (count > file_size / sizeof(T)) throw std::runtime_error("invalid vector length in adaptive IVF index");
    std::vector<T> values(count);
    if (!values.empty()) {
        in.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(values.size() * sizeof(T)));
        if (!in) throw std::runtime_error("truncated vector in adaptive IVF index");
    }
    return values;
}

void write_config(std::ofstream& out, const AdaptiveHierarchicalConfig& c) {
    write_size(out, c.dimension); write_size(out, c.initial_leaf_nlist);
    write_size(out, c.top_nlist); write_size(out, c.top_nprobe);
    write_size(out, c.leaf_nprobe_ratio); write_size(out, c.kmeans_iterations);
    write_scalar(out, c.maintenance_fraction);
    write_scalar(out, static_cast<std::uint8_t>(c.automatic_maintenance));
    write_scalar(out, c.min_partition_change_fraction);
    write_size(out, c.min_list_size_for_split); write_size(out, c.local_neighbor_count);
    write_scalar(out, c.prd_threshold);
    write_scalar(out, static_cast<std::uint32_t>(c.drift_criterion));
    write_scalar(out, c.tail_fraction_threshold); write_scalar(out, c.recenter_gain_threshold);
    write_scalar(out, c.short_list_fraction); write_scalar(out, c.load_split_fraction);
    write_scalar(out, c.load_target_size);
    write_scalar(out, static_cast<std::uint8_t>(c.sqrt_load_target));
    write_scalar(out, c.sqrt_load_target_multiplier); write_scalar(out, c.seed);
    write_scalar(out, static_cast<std::uint8_t>(c.lvq8_storage));
    write_scalar(out, static_cast<std::uint8_t>(c.lvq4_storage));
    write_size(out, c.fixed_leaf_nprobe);
    write_scalar(out, static_cast<std::uint8_t>(c.angular_metric));
    write_scalar(out, c.capacity_growth_split_factor);
    write_scalar(out, c.capacity_sqrt_n_split_multiplier);
    write_scalar(out, c.residual_tail_reassign_fraction);
    write_scalar(out, static_cast<std::uint8_t>(c.residual_tail_reassign_selected_only));
    write_size(out, c.split_local_refinement_radius);
    write_size(out, c.split_local_refinement_iterations);
    write_scalar(out, static_cast<std::uint8_t>(c.uint8_storage));
    write_scalar(out, c.short_list_sqrt_n_fraction);
    write_scalar(out, c.capacity_n_divisor);
    write_scalar(out, c.short_list_n_fraction);
    write_size(out, c.capacity_cluster_count_gate);
}

AdaptiveHierarchicalConfig read_config(std::ifstream& in, std::uint32_t version) {
    AdaptiveHierarchicalConfig c;
    c.dimension = read_size(in); c.initial_leaf_nlist = read_size(in);
    c.top_nlist = read_size(in); c.top_nprobe = read_size(in);
    c.leaf_nprobe_ratio = read_size(in); c.kmeans_iterations = read_size(in);
    c.maintenance_fraction = read_scalar<float>(in);
    c.automatic_maintenance = read_scalar<std::uint8_t>(in) != 0;
    c.min_partition_change_fraction = read_scalar<float>(in);
    c.min_list_size_for_split = read_size(in); c.local_neighbor_count = read_size(in);
    c.prd_threshold = read_scalar<float>(in);
    const auto criterion = read_scalar<std::uint32_t>(in);
    if (criterion > static_cast<std::uint32_t>(DriftCriterion::InsertOnly))
        throw std::runtime_error("invalid drift criterion in adaptive IVF index");
    c.drift_criterion = static_cast<DriftCriterion>(criterion);
    c.tail_fraction_threshold = read_scalar<float>(in);
    c.recenter_gain_threshold = read_scalar<float>(in);
    c.short_list_fraction = read_scalar<float>(in); c.load_split_fraction = read_scalar<float>(in);
    c.load_target_size = read_scalar<float>(in);
    c.sqrt_load_target = read_scalar<std::uint8_t>(in) != 0;
    c.sqrt_load_target_multiplier = read_scalar<float>(in); c.seed = read_scalar<std::uint32_t>(in);
    if (version >= 4) c.lvq8_storage = read_scalar<std::uint8_t>(in) != 0;
    if (version >= 6) c.lvq4_storage = read_scalar<std::uint8_t>(in) != 0;
    if (version >= 5) c.fixed_leaf_nprobe = read_size(in);
    if (version >= 7) c.angular_metric = read_scalar<std::uint8_t>(in) != 0;
    // Version 8 introduced the capacity-growth split factor; keep loading
    // those experiment indexes with their original threshold.
    if (version >= 8) c.capacity_growth_split_factor = read_scalar<float>(in);
    if (version >= 13) c.capacity_sqrt_n_split_multiplier = read_scalar<float>(in);
    if (version >= 10) c.residual_tail_reassign_fraction = read_scalar<float>(in);
    if (version >= 11) c.residual_tail_reassign_selected_only = read_scalar<std::uint8_t>(in) != 0;
    if (version >= 12) {
        c.split_local_refinement_radius = read_size(in);
        c.split_local_refinement_iterations = read_size(in);
    }
    if (version >= 14) c.uint8_storage = read_scalar<std::uint8_t>(in) != 0;
    if (version >= 15) c.short_list_sqrt_n_fraction = read_scalar<float>(in);
    if (version >= 16) {
        c.capacity_n_divisor = read_scalar<float>(in);
        c.short_list_n_fraction = read_scalar<float>(in);
    }
    if (version >= 17) c.capacity_cluster_count_gate = read_size(in);
    return c;
}

float percentile95(std::vector<float> values) {
    if (values.empty()) return 0.0f;
    const std::size_t position = static_cast<std::size_t>(std::ceil(0.95 * values.size())) - 1;
    std::nth_element(values.begin(), values.begin() + position, values.end());
    return values[position];
}

double aps_incomplete_beta(double a, double b, double x) {
    constexpr double stop = 1.0e-8;
    constexpr double tiny = 1.0e-30;
    if (x <= 0.0) return 0.0;
    if (x >= 1.0) return 1.0;
    if (x > (a + 1.0) / (a + b + 2.0))
        return 1.0 - aps_incomplete_beta(b, a, 1.0 - x);
    const double lbeta = std::lgamma(a) + std::lgamma(b) - std::lgamma(a + b);
    const double front = std::exp(std::log(x) * a + std::log(1.0 - x) * b - lbeta) / a;
    double f = 1.0, c = 1.0, d = 0.0;
    for (int i = 0; i <= 200; ++i) {
        const int m = i / 2;
        double numerator = 1.0;
        if (i != 0 && i % 2 == 0) {
            numerator = (m * (b - m) * x) /
                ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        } else if (i % 2 != 0) {
            numerator = -((a + m) * (a + b + m) * x) /
                ((a + 2.0 * m) * (a + 2.0 * m + 1.0));
        }
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

double aps_incomplete_beta_lookup(double x, std::size_t dimension) {
    constexpr std::size_t table_size = 1001;
    struct LookupTable {
        std::size_t dimension = 0;
        std::array<double, table_size> values{};
    };
    static LookupTable table;
    if (table.dimension != dimension) {
        const double a = (static_cast<double>(dimension) + 1.0) / 2.0;
        for (std::size_t i = 0; i < table_size; ++i)
            table.values[i] = aps_incomplete_beta(a, 0.5, static_cast<double>(i) / (table_size - 1));
        table.dimension = dimension;
    }
    x = std::clamp(x, 0.0, 1.0);
    const double scaled = x * (table_size - 1);
    const std::size_t lo = std::min<std::size_t>(static_cast<std::size_t>(scaled), table_size - 2);
    const double x0 = static_cast<double>(lo) / (table_size - 1);
    return table.values[lo] + (x - x0) * (table.values[lo + 1] - table.values[lo]) * (table_size - 1);
}

double aps_cap_volume(double radius, double boundary, std::size_t dimension) {
    if (boundary >= radius) return 0.0;
    const double x = std::clamp(std::sqrt(std::max(0.0, 1.0 - (boundary / radius) * (boundary / radius))), 0.0, 1.0);
    return std::clamp(0.5 * aps_incomplete_beta_lookup(x, dimension), 0.0, 0.5);
}

std::vector<float> aps_recall_profile(const std::vector<float>& boundaries,
                                      float radius, std::size_t dimension) {
    const std::size_t n = boundaries.size();
    if (n == 0) return {};
    std::vector<float> probabilities(n, 0.0f);
    if (n == 1 || radius <= 1e-9f) {
        probabilities[0] = 1.0f;
        return probabilities;
    }
    std::vector<double> raw(n, 0.0);
    double sum = 0.0;
    for (std::size_t i = 1; i < n; ++i) {
        raw[i] = aps_cap_volume(radius, std::max(0.0f, boundaries[i]), dimension);
        sum += raw[i];
    }
    if (sum <= 1e-9) {
        probabilities[0] = 1.0f;
        return probabilities;
    }
    double survival = 1.0;
    for (std::size_t i = 1; i < n; ++i) survival *= (1.0 - raw[i] / sum);
    probabilities[0] = static_cast<float>(std::clamp(survival, 0.0, 1.0));
    const double target = 1.0 - probabilities[0];
    for (std::size_t i = 1; i < n; ++i)
        probabilities[i] = static_cast<float>(target * raw[i] / sum);
    return probabilities;
}
}

AdaptiveHierarchicalIvf::AdaptiveHierarchicalIvf(AdaptiveHierarchicalConfig config)
    : config_(config) {
    if (config_.dimension == 0 || config_.initial_leaf_nlist == 0 || config_.leaf_nprobe_ratio == 0 ||
        config_.short_list_fraction <= 0.0f || config_.short_list_fraction >= 1.0f ||
        config_.load_split_fraction <= 1.0f || config_.load_target_size < 0.0f ||
        !std::isfinite(config_.short_list_sqrt_n_fraction) ||
        config_.short_list_sqrt_n_fraction < 0.0f ||
        !std::isfinite(config_.capacity_n_divisor) ||
        config_.capacity_n_divisor < 0.0f ||
        !std::isfinite(config_.short_list_n_fraction) ||
        config_.short_list_n_fraction < 0.0f ||
        !std::isfinite(config_.residual_tail_reassign_fraction) ||
        config_.residual_tail_reassign_fraction < 0.0f ||
        config_.residual_tail_reassign_fraction > 1.0f ||
        !std::isfinite(config_.capacity_growth_split_factor) ||
        config_.capacity_growth_split_factor <= 1.0f) {
        throw std::invalid_argument("all hierarchy dimensions and probe parameters must be positive");
    }
    if ((config_.lvq8_storage && config_.lvq4_storage) ||
        (config_.uint8_storage && uses_lvq_storage(config_)) ||
        (config_.uint8_storage && config_.angular_metric) ||
        (config_.lvq4_storage && config_.dimension % 2 != 0))
        throw std::invalid_argument("adaptive IVF supports one storage format; uint8 storage is unsigned, L2-only, and cannot be combined with LVQ");
}

void AdaptiveHierarchicalIvf::save(const std::string& path) const {
    if (!trained_) throw std::logic_error("cannot save an untrained adaptive IVF index");
    if (update_batch_open_) throw std::logic_error("cannot save while an update batch is open");
    if (path.empty()) throw std::invalid_argument("index path must not be empty");
    const std::filesystem::path destination(path);
    if (destination.has_parent_path()) std::filesystem::create_directories(destination.parent_path());
    const std::filesystem::path temporary = destination.string() + ".tmp";
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot create adaptive IVF index file");
    out.write(kAdaptiveMagic, sizeof(kAdaptiveMagic) - 1);
    write_scalar(out, kAdaptiveVersion);
    write_config(out, config_);
    write_scalar(out, next_id_);
    write_size(out, active_count_);
    write_size(out, changes_since_maintenance_);
    write_size(out, next_maintenance_budget_);
    write_vector(out, leaf_centroids_);
    write_size(out, leaves_.size());
    for (const Leaf& leaf : leaves_) {
        write_vector(out, leaf.ids);
        if (uses_lvq_storage(config_)) {
            write_vector(out, leaf.lvq_codes);
            write_vector(out, leaf.lvq_lower);
            write_vector(out, leaf.lvq_delta);
        } else if (uses_u8_storage(config_)) {
            write_vector(out, leaf.uint8_vectors);
        } else {
            write_vector(out, leaf.vectors);
        }
        write_vector(out, leaf.vector_sum);
        write_scalar(out, leaf.squared_norm_sum);
        write_scalar(out, leaf.ref_mean);
        write_scalar(out, leaf.ref_p95);
        write_size(out, leaf.window_inserts);
        write_size(out, leaf.window_deletes);
        if (config_.drift_criterion != DriftCriterion::GlobalFitOnly)
            write_scalar(out, leaf.insert_residual_sum);
        write_size(out, leaf.window_tail_count);
        write_size(out, leaf.capacity_reference_size);
    }
    out.close();
    if (!out) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw std::runtime_error("failed to finish adaptive IVF index file");
    }
    std::error_code error;
    std::filesystem::rename(temporary, destination, error);
    if (error) {
        std::filesystem::remove(temporary);
        throw std::runtime_error("cannot activate adaptive IVF index file: " + error.message());
    }
}

AdaptiveHierarchicalIvf AdaptiveHierarchicalIvf::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open adaptive IVF index file");
    const std::uintmax_t file_size = std::filesystem::file_size(path);
    char magic[sizeof(kAdaptiveMagic) - 1]{};
    in.read(magic, sizeof(magic));
    if (!in || !std::equal(std::begin(magic), std::end(magic), std::begin(kAdaptiveMagic)))
        throw std::runtime_error("not an adaptive PRD-IVF index file");
    const auto version = read_scalar<std::uint32_t>(in);
    if (version < 1 || version > kAdaptiveVersion)
        throw std::runtime_error("unsupported adaptive IVF index version");

    AdaptiveHierarchicalIvf index(read_config(in, version));
    index.next_id_ = read_scalar<std::int64_t>(in);
    index.active_count_ = read_size(in);
    index.changes_since_maintenance_ = read_size(in);
    (void)read_size(in);  // legacy periodic-maintenance budget; no longer used
    index.next_maintenance_budget_ = 0;
    if (index.active_count_ > file_size / sizeof(std::int64_t))
        throw std::runtime_error("active vector count is impossible for adaptive IVF file size");
    index.leaf_centroids_ = read_vector<float>(in, file_size);
    const std::size_t leaf_count = read_size(in);
    if (leaf_count > file_size / sizeof(std::uint64_t))
        throw std::runtime_error("invalid leaf count in adaptive IVF index");
    index.leaves_.resize(leaf_count);
    std::size_t observed_vectors = 0;
    std::int64_t max_id = -1;
    std::unordered_set<std::int64_t> seen_ids;
    seen_ids.reserve(index.active_count_);
    for (Leaf& leaf : index.leaves_) {
        if (version >= 3) {
            leaf.ids = read_vector<std::uint32_t>(in, file_size);
        } else {
            const auto old_ids = read_vector<std::int64_t>(in, file_size);
            leaf.ids.reserve(old_ids.size());
            for (const auto id : old_ids) {
                if (id < 0 || static_cast<std::uint64_t>(id) > std::numeric_limits<std::uint32_t>::max())
                    throw std::runtime_error("legacy vector ID does not fit compact PRD-IVF storage");
                leaf.ids.push_back(static_cast<std::uint32_t>(id));
            }
        }
        if (version >= 4 && uses_lvq_storage(index.config_)) {
            leaf.lvq_codes = read_vector<std::uint8_t>(in, file_size);
            leaf.lvq_lower = read_vector<float>(in, file_size);
            leaf.lvq_delta = read_vector<float>(in, file_size);
        } else if (version >= 14 && uses_u8_storage(index.config_)) {
            leaf.uint8_vectors = read_vector<std::uint8_t>(in, file_size);
        } else {
            leaf.vectors = read_vector<float>(in, file_size);
        }
        leaf.vector_sum = read_vector<double>(in, file_size);
        leaf.squared_norm_sum = read_scalar<double>(in);
        leaf.ref_mean = read_scalar<float>(in);
        leaf.ref_p95 = read_scalar<float>(in);
        leaf.window_inserts = read_size(in);
        leaf.window_deletes = read_size(in);
        if (version == 1 || index.config_.drift_criterion != DriftCriterion::GlobalFitOnly)
            leaf.insert_residual_sum = read_scalar<double>(in);
        leaf.window_tail_count = read_size(in);
        if (version >= 8) leaf.capacity_reference_size = read_size(in);
        else leaf.capacity_reference_size = leaf.ids.size();
        if (leaf.ids.size() > std::numeric_limits<std::size_t>::max() / index.config_.dimension)
            throw std::runtime_error("inconsistent leaf payload in adaptive IVF index");
        const bool payload_valid = version >= 4 && uses_lvq_storage(index.config_)
            ? leaf.lvq_codes.size() == leaf.ids.size() * lvq_code_bytes_per_vector(index.config_) &&
              leaf.lvq_lower.size() == leaf.ids.size() && leaf.lvq_delta.size() == leaf.ids.size()
            : (version >= 14 && uses_u8_storage(index.config_)
                ? leaf.uint8_vectors.size() == leaf.ids.size() * index.config_.dimension
                : leaf.vectors.size() == leaf.ids.size() * index.config_.dimension);
        if (!payload_valid || leaf.vector_sum.size() != index.config_.dimension)
            throw std::runtime_error("inconsistent leaf payload in adaptive IVF index");
        observed_vectors += leaf.ids.size();
        for (const auto id : leaf.ids) {
            if (!seen_ids.insert(id).second)
                throw std::runtime_error("duplicate or invalid vector id in adaptive IVF index");
            max_id = std::max(max_id, static_cast<std::int64_t>(id));
        }
    }
    std::vector<float> legacy_top_centroids;
    std::vector<std::size_t> legacy_leaf_to_top;
    if (version < 3) {
        legacy_top_centroids = read_vector<float>(in, file_size);
        const std::size_t route_count = read_size(in);
        legacy_leaf_to_top.resize(route_count);
        for (auto& id : legacy_leaf_to_top) id = read_size(in);
    }
    if (in.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("unexpected trailing data in adaptive IVF index");
    const std::size_t dimension = index.config_.dimension;
    const std::size_t top_count = dimension == 0 ? 0 : legacy_top_centroids.size() / dimension;
    if (leaf_count > std::numeric_limits<std::size_t>::max() / dimension ||
        index.active_count_ != observed_vectors || index.next_id_ < 0 ||
        (max_id >= 0 && index.next_id_ <= max_id) ||
        index.leaf_centroids_.size() != leaf_count * dimension ||
        (version < 3 && legacy_top_centroids.size() != top_count * dimension) ||
        (version < 3 && legacy_leaf_to_top.size() != leaf_count) ||
        (version < 3 && leaf_count != 0 && top_count == 0))
        throw std::runtime_error("inconsistent index metadata in adaptive IVF file");
    for (const auto id : legacy_leaf_to_top)
        if (id >= top_count) throw std::runtime_error("invalid top-level route in adaptive IVF file");
    index.trained_ = true;
    index.update_batch_open_ = false;
    return index;
}

float AdaptiveHierarchicalIvf::squared_l2(const float* a, const float* b) const {
    return simd_squared_l2(a, b, config_.dimension);
}

float AdaptiveHierarchicalIvf::retrieval_distance(const float* a, const float* b) const {
    return config_.angular_metric
        ? simd_cosine_distance(a, b, config_.dimension)
        : squared_l2(a, b);
}

void AdaptiveHierarchicalIvf::decode_vector(std::size_t leaf_id, std::size_t position,
                                             float* output) const {
    const Leaf& leaf = leaves_.at(leaf_id);
    if (uses_u8_storage(config_)) {
        const std::uint8_t* input = leaf.uint8_vectors.data() + position * config_.dimension;
        for (std::size_t d = 0; d < config_.dimension; ++d)
            output[d] = static_cast<float>(input[d]);
        return;
    }
    if (!uses_lvq_storage(config_)) {
        std::copy_n(leaf.vectors.data() + position * config_.dimension,
                    config_.dimension, output);
        return;
    }
    const float* centroid = leaf_centroids_.data() + leaf_id * config_.dimension;
    const float lower = leaf.lvq_lower.at(position);
    const float delta = leaf.lvq_delta.at(position);
    const std::size_t code_bytes = lvq_code_bytes_per_vector(config_);
    const std::uint8_t* codes = leaf.lvq_codes.data() + position * code_bytes;
    if (config_.lvq4_storage) {
        for (std::size_t d = 0; d < config_.dimension; d += 2) {
            const std::uint8_t packed = codes[d / 2];
            output[d] = centroid[d] + lower + static_cast<float>(packed & 0x0fu) * delta;
            output[d + 1] = centroid[d + 1] + lower + static_cast<float>(packed >> 4) * delta;
        }
    } else {
        for (std::size_t d = 0; d < config_.dimension; ++d)
            output[d] = centroid[d] + lower + static_cast<float>(codes[d]) * delta;
    }
}

bool AdaptiveHierarchicalIvf::reconstruct_vector(std::int64_t id, float* output) const {
    if (!trained_ || output == nullptr || id < 0 ||
        static_cast<std::uint64_t>(id) > std::numeric_limits<std::uint32_t>::max())
        return false;
    const auto wanted = static_cast<std::uint32_t>(id);
    for (std::size_t leaf_id = 0; leaf_id < leaves_.size(); ++leaf_id) {
        const auto& ids = leaves_[leaf_id].ids;
        const auto found = std::find(ids.begin(), ids.end(), wanted);
        if (found == ids.end()) continue;
        decode_vector(leaf_id, static_cast<std::size_t>(found - ids.begin()), output);
        return true;
    }
    return false;
}

void AdaptiveHierarchicalIvf::store_vector(std::size_t leaf_id, Leaf& leaf,
                                           const float* vector) {
    if (uses_u8_storage(config_)) {
        for (std::size_t d = 0; d < config_.dimension; ++d) {
            const float value = vector[d];
            if (!std::isfinite(value) || value < 0.0f || value > 255.0f ||
                std::fabs(value - std::round(value)) > 1e-4f)
                throw std::invalid_argument("uint8 storage requires exact integer vector values in [0,255]");
            leaf.uint8_vectors.push_back(static_cast<std::uint8_t>(std::lround(value)));
        }
        return;
    }
    if (!uses_lvq_storage(config_)) {
        leaf.vectors.insert(leaf.vectors.end(), vector, vector + config_.dimension);
        return;
    }
    const float* centroid = leaf_centroids_.data() + leaf_id * config_.dimension;
    float lower = std::numeric_limits<float>::infinity();
    float upper = -std::numeric_limits<float>::infinity();
    for (std::size_t d = 0; d < config_.dimension; ++d) {
        const float residual = vector[d] - centroid[d];
        lower = std::min(lower, residual);
        upper = std::max(upper, residual);
    }
    const unsigned max_code = config_.lvq4_storage ? 15u : 255u;
    const float delta = (upper - lower) / static_cast<float>(max_code);
    leaf.lvq_lower.push_back(lower);
    leaf.lvq_delta.push_back(delta);
    if (config_.lvq4_storage) {
        const std::size_t offset = leaf.lvq_codes.size();
        leaf.lvq_codes.resize(offset + config_.dimension / 2, 0);
        for (std::size_t d = 0; d < config_.dimension; ++d) {
            const float residual = vector[d] - centroid[d];
            const unsigned code = delta > 0.0f
                ? static_cast<unsigned>(std::clamp(
                    static_cast<int>(std::floor((residual - lower) / delta + 0.5f)), 0, 15)) : 0u;
            if ((d & 1u) == 0) leaf.lvq_codes[offset + d / 2] |= static_cast<std::uint8_t>(code);
            else leaf.lvq_codes[offset + d / 2] |= static_cast<std::uint8_t>(code << 4);
        }
        return;
    }
    for (std::size_t d = 0; d < config_.dimension; ++d) {
        const float residual = vector[d] - centroid[d];
        const int code = delta > 0.0f
            ? static_cast<int>(std::floor((residual - lower) / delta + 0.5f)) : 0;
        leaf.lvq_codes.push_back(static_cast<std::uint8_t>(std::clamp(code, 0, 255)));
    }
}

void AdaptiveHierarchicalIvf::store_vector_u8(Leaf& leaf, const std::uint8_t* vector) {
    if (!uses_u8_storage(config_) || uses_lvq_storage(config_))
        throw std::logic_error("store_vector_u8 requires exact uint8 storage mode");
    leaf.uint8_vectors.insert(leaf.uint8_vectors.end(), vector, vector + config_.dimension);
}

void AdaptiveHierarchicalIvf::clear_vector_payload(Leaf& leaf) {
    leaf.vectors.clear();
    leaf.uint8_vectors.clear();
    leaf.lvq_codes.clear();
    leaf.lvq_lower.clear();
    leaf.lvq_delta.clear();
}

void AdaptiveHierarchicalIvf::remove_vector_payload(Leaf& leaf, std::size_t position,
                                                     std::size_t last) {
    if (uses_u8_storage(config_)) {
        const std::size_t dimension = config_.dimension;
        if (position != last)
            std::copy_n(leaf.uint8_vectors.data() + last * dimension, dimension,
                        leaf.uint8_vectors.data() + position * dimension);
        leaf.uint8_vectors.resize(last * dimension);
        return;
    }
    if (!uses_lvq_storage(config_)) {
        if (position != last)
            simd_copy_floats(leaf.vectors.data() + position * config_.dimension,
                             leaf.vectors.data() + last * config_.dimension, config_.dimension);
        leaf.vectors.resize(last * config_.dimension);
        return;
    }
    const std::size_t code_bytes = lvq_code_bytes_per_vector(config_);
    if (position != last) {
        std::copy_n(leaf.lvq_codes.data() + last * code_bytes, code_bytes,
                    leaf.lvq_codes.data() + position * code_bytes);
        leaf.lvq_lower[position] = leaf.lvq_lower[last];
        leaf.lvq_delta[position] = leaf.lvq_delta[last];
    }
    leaf.lvq_codes.resize(last * code_bytes);
    leaf.lvq_lower.resize(last);
    leaf.lvq_delta.resize(last);
}

std::vector<std::size_t> AdaptiveHierarchicalIvf::nearest(
        const float* query, const std::vector<float>& centers, std::size_t center_count,
        std::size_t wanted) const {
    if (center_count == 0 || wanted == 0) return {};
    std::vector<std::pair<float, std::size_t>> distances;
    distances.reserve(center_count);
    for (std::size_t i = 0; i < center_count; ++i) {
        distances.emplace_back(retrieval_distance(query, centers.data() + i * config_.dimension), i);
    }
    wanted = std::min(wanted, center_count);
    std::partial_sort(distances.begin(), distances.begin() + wanted, distances.end(),
                      [](const auto& a, const auto& b) {
                          return a.first == b.first ? a.second < b.second : a.first < b.first;
                      });
    std::vector<std::size_t> result;
    result.reserve(wanted);
    for (std::size_t i = 0; i < wanted; ++i) result.push_back(distances[i].second);
    return result;
}

void AdaptiveHierarchicalIvf::train_kmeans(
        const float* vectors, std::size_t count, std::size_t clusters, std::size_t iterations,
        std::vector<float>* centers, const std::vector<float>* weights) const {
    if (vectors == nullptr || count < clusters || clusters == 0 || iterations == 0) {
        throw std::invalid_argument("invalid k-means input");
    }
    if (weights != nullptr && weights->size() != count) {
        throw std::invalid_argument("k-means weights do not match vector count");
    }
    std::vector<std::size_t> choices(count);
    std::iota(choices.begin(), choices.end(), 0);
    std::mt19937 rng(config_.seed);
    std::shuffle(choices.begin(), choices.end(), rng);
    centers->assign(clusters * config_.dimension, 0.0f);
    for (std::size_t c = 0; c < clusters; ++c) {
        std::copy_n(vectors + choices[c] * config_.dimension, config_.dimension,
                    centers->data() + c * config_.dimension);
    }
    std::vector<std::size_t> assignment(count);
    for (std::size_t step = 0; step < iterations; ++step) {
        std::vector<double> sums(clusters * config_.dimension, 0.0);
        std::vector<double> masses(clusters, 0.0);
        for (std::size_t i = 0; i < count; ++i) {
            const float* point = vectors + i * config_.dimension;
            const auto best = nearest(point, *centers, clusters, 1).front();
            assignment[i] = best;
            const double weight = weights == nullptr ? 1.0 : static_cast<double>((*weights)[i]);
            masses[best] += weight;
            for (std::size_t d = 0; d < config_.dimension; ++d) {
                sums[best * config_.dimension + d] += weight * point[d];
            }
        }
        for (std::size_t c = 0; c < clusters; ++c) {
            float* center = centers->data() + c * config_.dimension;
            if (masses[c] == 0.0) {
                const std::size_t replacement = choices[rng() % count];
                std::copy_n(vectors + replacement * config_.dimension, config_.dimension, center);
            } else {
                for (std::size_t d = 0; d < config_.dimension; ++d) {
                    center[d] = static_cast<float>(sums[c * config_.dimension + d] / masses[c]);
                }
            }
        }
    }
}

void AdaptiveHierarchicalIvf::train_kmeans_u8(
        const std::uint8_t* vectors, std::size_t count, std::size_t clusters,
        std::size_t iterations, std::vector<float>* centers) const {
    if (vectors == nullptr || centers == nullptr || count < clusters || clusters == 0 || iterations == 0)
        throw std::invalid_argument("invalid uint8 k-means input");
    std::vector<std::size_t> choices(count);
    std::iota(choices.begin(), choices.end(), 0);
    std::mt19937 rng(config_.seed);
    std::shuffle(choices.begin(), choices.end(), rng);
    centers->assign(clusters * config_.dimension, 0.0f);
    for (std::size_t c = 0; c < clusters; ++c) {
        const std::uint8_t* point = vectors + choices[c] * config_.dimension;
        float* center = centers->data() + c * config_.dimension;
        for (std::size_t d = 0; d < config_.dimension; ++d)
            center[d] = static_cast<float>(point[d]);
    }
    std::vector<std::size_t> assignment(count);
    const std::size_t worker_count = effective_worker_count(count, config_.worker_threads);
    const std::size_t sums_per_worker = clusters * config_.dimension;
    std::vector<double> local_sums(worker_count * sums_per_worker, 0.0);
    std::vector<std::size_t> local_masses(worker_count * clusters, 0);
    std::vector<double> sums(sums_per_worker, 0.0);
    std::vector<std::size_t> masses(clusters, 0);
    for (std::size_t step = 0; step < iterations; ++step) {
        std::fill(local_sums.begin(), local_sums.end(), 0.0);
        std::fill(local_masses.begin(), local_masses.end(), 0);
        parallel_for_static(count, worker_count, [&](std::size_t worker, std::size_t i) {
            const std::uint8_t* point = vectors + i * config_.dimension;
            std::size_t best = 0;
            float best_distance = simd_squared_l2_u8_f32(
                point, centers->data(), config_.dimension);
            for (std::size_t c = 1; c < clusters; ++c) {
                const float distance = simd_squared_l2_u8_f32(
                    point, centers->data() + c * config_.dimension, config_.dimension);
                if (distance < best_distance) { best = c; best_distance = distance; }
            }
            assignment[i] = best;
            ++local_masses[worker * clusters + best];
            double* sum = local_sums.data() + worker * sums_per_worker + best * config_.dimension;
            for (std::size_t d = 0; d < config_.dimension; ++d) sum[d] += point[d];
        });
        std::fill(sums.begin(), sums.end(), 0.0);
        std::fill(masses.begin(), masses.end(), 0);
        for (std::size_t worker = 0; worker < worker_count; ++worker) {
            const double* worker_sums = local_sums.data() + worker * sums_per_worker;
            const std::size_t* worker_masses = local_masses.data() + worker * clusters;
            for (std::size_t c = 0; c < clusters; ++c) masses[c] += worker_masses[c];
            for (std::size_t j = 0; j < sums_per_worker; ++j) sums[j] += worker_sums[j];
        }
        for (std::size_t c = 0; c < clusters; ++c) {
            float* center = centers->data() + c * config_.dimension;
            if (masses[c] == 0) {
                const std::uint8_t* replacement = vectors + choices[rng() % count] * config_.dimension;
                for (std::size_t d = 0; d < config_.dimension; ++d)
                    center[d] = static_cast<float>(replacement[d]);
            } else {
                const double inverse = 1.0 / static_cast<double>(masses[c]);
                for (std::size_t d = 0; d < config_.dimension; ++d)
                    center[d] = static_cast<float>(sums[c * config_.dimension + d] * inverse);
            }
        }
    }
}

void AdaptiveHierarchicalIvf::reset_reference(Leaf& leaf, const float* centroid) {
    rebuild_aggregates(leaf);
    const std::size_t count = leaf.ids.size();
    leaf.capacity_reference_size = count;
    if (count == 0) {
        leaf.ref_mean = 0.0f;
        leaf.ref_p95 = 0.0f;
    } else {
        std::vector<float> residuals;
        residuals.reserve(count);
        double sum = 0.0;
        std::vector<float> decoded(config_.dimension);
        for (std::size_t i = 0; i < count; ++i) {
            decode_vector(static_cast<std::size_t>(&leaf - leaves_.data()), i, decoded.data());
            const float distance = squared_l2(decoded.data(), centroid);
            residuals.push_back(distance);
            sum += distance;
        }
        leaf.ref_mean = static_cast<float>(sum / count);
        leaf.ref_p95 = percentile95(std::move(residuals));
    }
    reset_window(leaf);
}

bool AdaptiveHierarchicalIvf::capacity_growth_split_due(const Leaf& leaf) const noexcept {
    if (leaf.capacity_reference_size == 0) return false;
    return static_cast<double>(leaf.ids.size()) >
           static_cast<double>(leaf.capacity_reference_size) *
               static_cast<double>(config_.capacity_growth_split_factor);
}

void AdaptiveHierarchicalIvf::reset_window(Leaf& leaf) {
    leaf.window_inserts = 0;
    leaf.window_deletes = 0;
    leaf.insert_residual_sum = 0.0;
    leaf.window_tail_count = 0;
}

void AdaptiveHierarchicalIvf::rebuild_aggregates(Leaf& leaf) {
    leaf.vector_sum.assign(config_.dimension, 0.0);
    leaf.squared_norm_sum = 0.0;
    std::vector<float> decoded(config_.dimension);
    for (std::size_t i = 0; i < leaf.ids.size(); ++i) {
        const std::size_t leaf_id = static_cast<std::size_t>(&leaf - leaves_.data());
        decode_vector(leaf_id, i, decoded.data());
        apply_vector_delta(leaf, decoded.data(), 1.0);
    }
}

void AdaptiveHierarchicalIvf::refresh_membership_state_preserve_residual_reference(Leaf& leaf) {
    // Reassignment changes the partition membership, but must not erase the
    // historical PRD residual baseline. Keep aggregate and capacity state
    // consistent with the new membership while retaining ref_mean/ref_p95.
    rebuild_aggregates(leaf);
    leaf.capacity_reference_size = leaf.ids.size();
    reset_window(leaf);
}

void AdaptiveHierarchicalIvf::apply_vector_delta(Leaf& leaf, const float* vector, double sign) {
    if (leaf.vector_sum.size() != config_.dimension) leaf.vector_sum.assign(config_.dimension, 0.0);
    simd_apply_vector_delta(leaf.vector_sum.data(), &leaf.squared_norm_sum,
                            vector, config_.dimension, sign);
    // Avoid a small negative value after many floating-point subtractions.
    if (leaf.squared_norm_sum < 0.0 && leaf.squared_norm_sum > -1e-6) leaf.squared_norm_sum = 0.0;
}

float AdaptiveHierarchicalIvf::current_mean_residual(const Leaf& leaf, const float* centroid) const {
    return simd_mean_residual_from_aggregates(leaf.vector_sum.data(), leaf.squared_norm_sum,
                                               centroid, config_.dimension, leaf.ids.size());
}

float AdaptiveHierarchicalIvf::centroid_shift(const Leaf& leaf, const float* centroid) const {
    if (leaf.ids.empty() || leaf.ref_mean <= 0.0f) return 0.0f;
    double shift = 0.0;
    for (std::size_t d = 0; d < config_.dimension; ++d) {
        const double mean = leaf.vector_sum[d] / leaf.ids.size();
        const double delta = mean - centroid[d];
        shift += delta * delta;
    }
    return static_cast<float>(shift / std::max(leaf.ref_mean, 1e-12f));
}

float AdaptiveHierarchicalIvf::insert_residual_drift(const Leaf& leaf) const {
    if (config_.drift_criterion == DriftCriterion::GlobalFitOnly) return 0.0f;
    if (leaf.window_inserts == 0 || leaf.ref_mean <= 0.0f) return 0.0f;
    const float recent = static_cast<float>(leaf.insert_residual_sum / leaf.window_inserts);
    return std::max(0.0f, (recent - leaf.ref_mean) / std::max(leaf.ref_mean, 1e-12f));
}

float AdaptiveHierarchicalIvf::partition_representation_drift(const Leaf& leaf, const float* centroid) const {
    if (leaf.ref_mean <= 0.0f) return 0.0f;
    const float global_fit = std::max(0.0f, (current_mean_residual(leaf, centroid) - leaf.ref_mean) /
                                                std::max(leaf.ref_mean, 1e-12f));
    return std::max({insert_residual_drift(leaf), global_fit, centroid_shift(leaf, centroid)});
}

float AdaptiveHierarchicalIvf::drift_score(const Leaf& leaf, const float* centroid) const {
    if (leaf.ref_mean <= 0.0f) return 0.0f;
    if (config_.drift_criterion == DriftCriterion::GlobalFitOnly) {
        return std::max(0.0f, (current_mean_residual(leaf, centroid) - leaf.ref_mean) /
                                  std::max(leaf.ref_mean, 1e-12f));
    }
    if (config_.drift_criterion == DriftCriterion::InsertOnly) {
        return insert_residual_drift(leaf);
    }
    return partition_representation_drift(leaf, centroid);
}

void AdaptiveHierarchicalIvf::assign_initial(const float* vectors, std::size_t count) {
    leaves_.assign(config_.initial_leaf_nlist, Leaf{});
    for (std::size_t i = 0; i < count; ++i) {
        const float* vector = vectors + i * config_.dimension;
        const auto leaf_id = nearest(vector, leaf_centroids_, leaves_.size(), 1).front();
        Leaf& leaf = leaves_[leaf_id];
        leaf.ids.push_back(static_cast<std::uint32_t>(next_id_++));
        store_vector(leaf_id, leaf, vector);
        ++active_count_;
    }
    for (std::size_t i = 0; i < leaves_.size(); ++i) {
        reset_reference(leaves_[i], leaf_centroids_.data() + i * config_.dimension);
    }
}

void AdaptiveHierarchicalIvf::assign_initial_u8(const std::uint8_t* vectors, std::size_t count) {
    leaves_.assign(config_.initial_leaf_nlist, Leaf{});
    std::vector<std::size_t> assignments(count);
    parallel_for_static(count, config_.worker_threads, [&](std::size_t, std::size_t i) {
        const std::uint8_t* source = vectors + i * config_.dimension;
        std::size_t best = 0;
        float best_distance = simd_squared_l2_u8_f32(
            source, leaf_centroids_.data(), config_.dimension);
        for (std::size_t c = 1; c < leaves_.size(); ++c) {
            const float distance = simd_squared_l2_u8_f32(
                source, leaf_centroids_.data() + c * config_.dimension, config_.dimension);
            if (distance < best_distance) { best = c; best_distance = distance; }
        }
        assignments[i] = best;
    });
    std::vector<float> point(config_.dimension);
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint8_t* source = vectors + i * config_.dimension;
        const std::size_t best = assignments[i];
        Leaf& leaf = leaves_[best];
        leaf.ids.push_back(static_cast<std::uint32_t>(next_id_++));
        store_vector_u8(leaf, source);
        for (std::size_t d = 0; d < config_.dimension; ++d)
            point[d] = static_cast<float>(source[d]);
        apply_vector_delta(leaf, point.data(), 1.0);
        ++active_count_;
    }
    for (std::size_t i = 0; i < leaves_.size(); ++i)
        reset_reference(leaves_[i], leaf_centroids_.data() + i * config_.dimension);
}

void AdaptiveHierarchicalIvf::train(const float* vectors, std::size_t count) {
    if (vectors == nullptr || count < config_.initial_leaf_nlist) {
        throw std::invalid_argument("training needs at least initial_leaf_nlist vectors");
    }
    if (static_cast<std::uint64_t>(count) >
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1ULL) {
        throw std::overflow_error("training IDs exceed compact 32-bit storage capacity");
    }
    leaf_centroids_.clear();
    leaves_.clear();
    next_id_ = 0;
    active_count_ = 0;
    changes_since_maintenance_ = 0;
    train_kmeans(vectors, count, config_.initial_leaf_nlist, config_.kmeans_iterations, &leaf_centroids_);
    assign_initial(vectors, count);
    trained_ = true;
    next_maintenance_budget_ = 0;
}

void AdaptiveHierarchicalIvf::train_u8(const std::uint8_t* vectors, std::size_t count) {
    if (!config_.uint8_storage || uses_lvq_storage(config_))
        throw std::logic_error("train_u8 requires exact uint8 storage mode");
    if (vectors == nullptr || count < config_.initial_leaf_nlist)
        throw std::invalid_argument("training needs at least initial_leaf_nlist uint8 vectors");
    if (static_cast<std::uint64_t>(count) >
        static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1ULL)
        throw std::overflow_error("training IDs exceed compact 32-bit storage capacity");
    leaf_centroids_.clear();
    leaves_.clear();
    next_id_ = 0;
    active_count_ = 0;
    changes_since_maintenance_ = 0;
    train_kmeans_u8(vectors, count, config_.initial_leaf_nlist,
                    config_.kmeans_iterations, &leaf_centroids_);
    assign_initial_u8(vectors, count);
    trained_ = true;
    next_maintenance_budget_ = 0;
}

void AdaptiveHierarchicalIvf::record_insert(Leaf& leaf, const float* vector, const float* centroid) {
    const float residual = squared_l2(vector, centroid);
    ++leaf.window_inserts;
    if (config_.drift_criterion != DriftCriterion::GlobalFitOnly)
        leaf.insert_residual_sum += residual;
    if (residual > leaf.ref_p95) ++leaf.window_tail_count;
}

void AdaptiveHierarchicalIvf::add(const float* vectors, std::size_t count) {
    if (!trained_) throw std::logic_error("train must be called before add");
    if (vectors == nullptr) throw std::invalid_argument("vectors must not be null");
    const std::uint64_t id_capacity = static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1ULL;
    if (static_cast<std::uint64_t>(next_id_) > id_capacity ||
        static_cast<std::uint64_t>(count) > id_capacity - static_cast<std::uint64_t>(next_id_)) {
        throw std::overflow_error("insert IDs exceed compact 32-bit storage capacity");
    }
    for (std::size_t i = 0; i < count; ++i) {
        const float* vector = vectors + i * config_.dimension;
        const auto leaf_id = nearest(vector, leaf_centroids_, leaves_.size(), 1).front();
        Leaf& leaf = leaves_[leaf_id];
        leaf.ids.push_back(static_cast<std::uint32_t>(next_id_++));
        store_vector(leaf_id, leaf, vector);
        apply_vector_delta(leaf, vector, 1.0);
        record_insert(leaf, vector, leaf_centroids_.data() + leaf_id * config_.dimension);
        ++active_count_;
        ++changes_since_maintenance_;
    }
    if (config_.automatic_maintenance && !update_batch_open_ && maintenance_due()) {
        maintain();
    }
}

void AdaptiveHierarchicalIvf::add_u8(const std::uint8_t* vectors, std::size_t count) {
    if (!trained_) throw std::logic_error("train must be called before add_u8");
    if (!config_.uint8_storage || uses_lvq_storage(config_))
        throw std::logic_error("add_u8 requires exact uint8 storage mode");
    if (vectors == nullptr) throw std::invalid_argument("vectors must not be null");
    const std::uint64_t id_capacity = static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1ULL;
    if (static_cast<std::uint64_t>(next_id_) > id_capacity ||
        static_cast<std::uint64_t>(count) > id_capacity - static_cast<std::uint64_t>(next_id_))
        throw std::overflow_error("insert IDs exceed compact 32-bit storage capacity");
    std::vector<std::size_t> assignments(count);
    parallel_for_static(count, config_.worker_threads, [&](std::size_t, std::size_t i) {
        const std::uint8_t* source = vectors + i * config_.dimension;
        std::size_t best = 0;
        float best_distance = simd_squared_l2_u8_f32(
            source, leaf_centroids_.data(), config_.dimension);
        for (std::size_t c = 1; c < leaves_.size(); ++c) {
            const float distance = simd_squared_l2_u8_f32(
                source, leaf_centroids_.data() + c * config_.dimension, config_.dimension);
            if (distance < best_distance) { best = c; best_distance = distance; }
        }
        assignments[i] = best;
    });
    std::vector<float> point(config_.dimension);
    for (std::size_t i = 0; i < count; ++i) {
        const std::uint8_t* source = vectors + i * config_.dimension;
        const std::size_t best = assignments[i];
        Leaf& leaf = leaves_[best];
        leaf.ids.push_back(static_cast<std::uint32_t>(next_id_++));
        store_vector_u8(leaf, source);
        for (std::size_t d = 0; d < config_.dimension; ++d)
            point[d] = static_cast<float>(source[d]);
        apply_vector_delta(leaf, point.data(), 1.0);
        record_insert(leaf, point.data(), leaf_centroids_.data() + best * config_.dimension);
        ++active_count_;
        ++changes_since_maintenance_;
    }
    if (config_.automatic_maintenance && !update_batch_open_ && maintenance_due()) maintain();
}

void AdaptiveHierarchicalIvf::append_to_leaf_for_diagnostic(
        std::size_t leaf_id, const float* vectors, std::size_t count) {
    if (!trained_) throw std::logic_error("train must be called before diagnostic append");
    if (leaf_id >= leaves_.size()) throw std::out_of_range("diagnostic leaf id is out of range");
    if (vectors == nullptr) throw std::invalid_argument("vectors must not be null");
    const std::uint64_t id_capacity = static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) + 1ULL;
    if (static_cast<std::uint64_t>(next_id_) > id_capacity ||
        static_cast<std::uint64_t>(count) > id_capacity - static_cast<std::uint64_t>(next_id_)) {
        throw std::overflow_error("diagnostic IDs exceed compact 32-bit storage capacity");
    }
    Leaf& leaf = leaves_[leaf_id];
    const float* centroid = leaf_centroids_.data() + leaf_id * config_.dimension;
    for (std::size_t i = 0; i < count; ++i) {
        const float* vector = vectors + i * config_.dimension;
        leaf.ids.push_back(static_cast<std::uint32_t>(next_id_++));
        store_vector(leaf_id, leaf, vector);
        apply_vector_delta(leaf, vector, 1.0);
        record_insert(leaf, vector, centroid);
        ++active_count_;
        ++changes_since_maintenance_;
    }
}

bool AdaptiveHierarchicalIvf::is_marked(std::size_t leaf_id) const {
    const Leaf& leaf = leaves_[leaf_id];
    if (leaf.ids.empty() || leaf.ref_mean <= 0.0f) return false;
    const float score = drift_score(leaf, leaf_centroids_.data() + leaf_id * config_.dimension);
    if (config_.drift_criterion == DriftCriterion::GlobalFitOnly) {
        // Global-only PRD uses only relative global-fit degradation. Per-leaf
        // change-volume gates are intentionally not part of maintenance.
        return score > config_.prd_threshold;
    }
    const float tail = leaf.window_inserts == 0 ? 0.0f :
            static_cast<float>(leaf.window_tail_count) / leaf.window_inserts;
    const bool evidence = leaf.window_deletes > 0 || tail > config_.tail_fraction_threshold;
    return score > config_.prd_threshold && evidence;
}

std::vector<std::size_t> AdaptiveHierarchicalIvf::local_neighbours(
        std::size_t leaf_id, std::size_t neighbor_count) const {
    const auto candidates = nearest(leaf_centroids_.data() + leaf_id * config_.dimension,
                                    leaf_centroids_, leaves_.size(),
                                    std::min(leaves_.size(), neighbor_count + 1));
    std::vector<std::size_t> result;
    for (const auto id : candidates) if (id != leaf_id) result.push_back(id);
    return result;
}

void AdaptiveHierarchicalIvf::locally_reassign(const std::vector<std::size_t>& seeds,
                                                std::size_t neighbor_count,
                                                std::size_t iterations,
                                                MaintenanceReport* report) {
    if (iterations == 0) return;
    std::unordered_set<std::size_t> selected(seeds.begin(), seeds.end());
    for (const auto seed : seeds) {
        for (const auto neighbour : local_neighbours(seed, neighbor_count)) selected.insert(neighbour);
    }
    std::vector<std::size_t> ids(selected.begin(), selected.end());
    if (ids.empty()) return;
    if (uses_lvq_storage(config_) || uses_u8_storage(config_)) {
        std::vector<std::uint32_t> all_ids;
        std::vector<std::size_t> source_leaves;
        std::vector<float> all_vectors;
        std::vector<float> decoded(config_.dimension);
        for (const auto id : ids) {
            Leaf& leaf = leaves_[id];
            all_ids.insert(all_ids.end(), leaf.ids.begin(), leaf.ids.end());
            for (std::size_t i = 0; i < leaf.ids.size(); ++i) {
                source_leaves.push_back(id);
                decode_vector(id, i, decoded.data());
                all_vectors.insert(all_vectors.end(), decoded.begin(), decoded.end());
            }
            leaf.ids.clear();
            clear_vector_payload(leaf);
        }
        const std::size_t count = all_ids.size();
        std::vector<std::size_t> assignments(count, ids.front());
        for (std::size_t pass = 0; pass < iterations; ++pass) {
            std::vector<double> sums(ids.size() * config_.dimension, 0.0);
            std::vector<std::size_t> sizes(ids.size(), 0);
            for (std::size_t i = 0; i < count; ++i) {
                const float* point = all_vectors.data() + i * config_.dimension;
                float best_distance = std::numeric_limits<float>::max();
                std::size_t best_slot = 0;
                for (std::size_t slot = 0; slot < ids.size(); ++slot) {
                    const float distance = retrieval_distance(point,
                        leaf_centroids_.data() + ids[slot] * config_.dimension);
                    if (distance < best_distance) { best_distance = distance; best_slot = slot; }
                }
                assignments[i] = ids[best_slot];
                ++sizes[best_slot];
                double* sum = sums.data() + best_slot * config_.dimension;
                for (std::size_t d = 0; d < config_.dimension; ++d) sum[d] += point[d];
            }
            if (pass + 1 < iterations) {
                for (std::size_t slot = 0; slot < ids.size(); ++slot) {
                    if (sizes[slot] == 0) continue;
                    float* center = leaf_centroids_.data() + ids[slot] * config_.dimension;
                    const double inverse = 1.0 / static_cast<double>(sizes[slot]);
                    for (std::size_t d = 0; d < config_.dimension; ++d)
                        center[d] = static_cast<float>(sums[slot * config_.dimension + d] * inverse);
                }
            }
        }
        for (std::size_t i = 0; i < count; ++i) {
            Leaf& destination = leaves_[assignments[i]];
            destination.ids.push_back(all_ids[i]);
            store_vector(assignments[i], destination,
                         all_vectors.data() + i * config_.dimension);
        }
        if (uses_lvq_storage(config_)) lvq8_reencoded_vectors_ += count;
        if (report != nullptr) {
            report->local_refinement_partitions += ids.size();
            report->local_refinement_vectors += count;
            for (std::size_t i = 0; i < count; ++i)
                if (assignments[i] != source_leaves[i])
                    ++report->local_refinement_reassigned_vectors;
        }
        for (const auto id : ids)
            refresh_membership_state_preserve_residual_reference(leaves_[id]);
        return;
    }
    std::vector<std::uint32_t> all_ids;
    std::vector<std::size_t> source_leaves;
    std::vector<float> all_vectors;
    for (const auto id : ids) {
        all_ids.insert(all_ids.end(), leaves_[id].ids.begin(), leaves_[id].ids.end());
        source_leaves.insert(source_leaves.end(), leaves_[id].ids.size(), id);
        all_vectors.insert(all_vectors.end(), leaves_[id].vectors.begin(), leaves_[id].vectors.end());
        leaves_[id].ids.clear();
        leaves_[id].vectors.clear();
    }
    const std::size_t count = all_ids.size();
    std::vector<std::size_t> assignments(count, ids.front());
    // Local Lloyd-style passes over the union of neighborhoods. Crucially this
    // runs once for the whole candidate batch, rather than resetting other
    // candidates after each local action.
    for (std::size_t pass = 0; pass < iterations; ++pass) {
        for (const auto id : ids) { leaves_[id].ids.clear(); leaves_[id].vectors.clear(); }
        for (std::size_t i = 0; i < count; ++i) {
            const float* point = all_vectors.data() + i * config_.dimension;
            float best_distance = std::numeric_limits<float>::max();
            std::size_t best = ids.front();
            for (const auto target : ids) {
                const float distance = retrieval_distance(point, leaf_centroids_.data() + target * config_.dimension);
                if (distance < best_distance) { best_distance = distance; best = target; }
            }
            assignments[i] = best;
            leaves_[best].ids.push_back(all_ids[i]);
            leaves_[best].vectors.insert(leaves_[best].vectors.end(), point, point + config_.dimension);
        }
        if (pass + 1 < iterations) {
            for (const auto id : ids) {
                Leaf& leaf = leaves_[id];
                if (leaf.ids.empty()) continue;
                float* center = leaf_centroids_.data() + id * config_.dimension;
                std::fill(center, center + config_.dimension, 0.0f);
                for (std::size_t i = 0; i < leaf.ids.size(); ++i)
                    for (std::size_t d = 0; d < config_.dimension; ++d)
                        center[d] += leaf.vectors[i * config_.dimension + d];
                for (std::size_t d = 0; d < config_.dimension; ++d) center[d] /= static_cast<float>(leaf.ids.size());
            }
        }
    }
    if (report != nullptr) {
        report->local_refinement_partitions += ids.size();
        report->local_refinement_vectors += count;
        for (std::size_t i = 0; i < count; ++i)
            if (assignments[i] != source_leaves[i])
                ++report->local_refinement_reassigned_vectors;
    }
    for (const auto id : ids)
        refresh_membership_state_preserve_residual_reference(leaves_[id]);
}

void AdaptiveHierarchicalIvf::reassign_residual_tail(
        float fraction, MaintenanceReport& report,
        const std::vector<std::size_t>* selected_leaves) {
    if (!(fraction > 0.0f) || leaves_.empty()) return;

    std::vector<bool> selected_mask(leaves_.size(), selected_leaves == nullptr);
    if (selected_leaves != nullptr) {
        for (const std::size_t leaf_id : *selected_leaves)
            if (leaf_id < selected_mask.size()) selected_mask[leaf_id] = true;
    }

    struct Candidate {
        float residual;
        std::size_t position;
    };
    struct Move {
        std::size_t source;
        std::size_t destination;
        std::size_t position;
        std::uint32_t id;
        std::vector<float> vector;
    };

    std::vector<Move> moves;
    std::vector<float> decoded(config_.dimension);
    for (std::size_t source = 0; source < leaves_.size(); ++source) {
        if (!selected_mask[source]) continue;
        const Leaf& leaf = leaves_[source];
        if (leaf.ids.empty()) continue;
        ++report.residual_tail_selected_partitions;
        const std::size_t tail_count = std::min(leaf.ids.size(),
            std::max<std::size_t>(1, static_cast<std::size_t>(
                std::ceil(static_cast<double>(fraction) * leaf.ids.size()))));
        std::vector<Candidate> residuals;
        residuals.reserve(leaf.ids.size());
        const float* source_center = leaf_centroids_.data() + source * config_.dimension;
        for (std::size_t position = 0; position < leaf.ids.size(); ++position) {
            decode_vector(source, position, decoded.data());
            residuals.push_back({squared_l2(decoded.data(), source_center), position});
        }
        const auto tail_begin = residuals.begin() +
            static_cast<std::ptrdiff_t>(residuals.size() - tail_count);
        std::nth_element(residuals.begin(), tail_begin, residuals.end(),
            [](const Candidate& a, const Candidate& b) { return a.residual < b.residual; });
        report.residual_tail_candidates += tail_count;

        for (auto candidate = tail_begin; candidate != residuals.end(); ++candidate) {
            const std::size_t position = candidate->position;
            decode_vector(source, position, decoded.data());
            // Start with the current leaf so ties keep the existing assignment.
            float best_distance = retrieval_distance(decoded.data(), source_center);
            std::size_t destination = source;
            for (std::size_t target = 0; target < leaves_.size(); ++target) {
                if (target == source) continue;
                const float distance = retrieval_distance(decoded.data(),
                    leaf_centroids_.data() + target * config_.dimension);
                if (distance < best_distance) {
                    best_distance = distance;
                    destination = target;
                }
            }
            if (destination == source) continue;
            Move move{source, destination, position, leaf.ids[position], {}};
            move.vector.assign(decoded.begin(), decoded.end());
            moves.push_back(std::move(move));
        }
    }
    if (moves.empty()) return;

    // Remove each source's selected positions from high to low, so swap-removal
    // cannot invalidate positions that are still waiting to be processed.
    std::sort(moves.begin(), moves.end(), [](const Move& a, const Move& b) {
        if (a.source != b.source) return a.source < b.source;
        return a.position > b.position;
    });
    std::vector<bool> touched(leaves_.size(), false);
    for (const Move& move : moves) {
        Leaf& source = leaves_[move.source];
        if (move.position >= source.ids.size() || source.ids[move.position] != move.id)
            throw std::logic_error("residual-tail reassignment position became stale");
        apply_vector_delta(source, move.vector.data(), -1.0);
        const std::size_t last = source.ids.size() - 1;
        if (move.position != last) source.ids[move.position] = source.ids[last];
        remove_vector_payload(source, move.position, last);
        source.ids.pop_back();

        Leaf& destination = leaves_[move.destination];
        destination.ids.push_back(move.id);
        store_vector(move.destination, destination, move.vector.data());
        apply_vector_delta(destination, move.vector.data(), 1.0);
        touched[move.source] = true;
        touched[move.destination] = true;
    }

    // Reassignment changes membership without moving centers. Rebuild the
    // aggregates/capacity reference, but preserve historical PRD residual refs.
    for (std::size_t leaf_id = 0; leaf_id < touched.size(); ++leaf_id) {
        if (touched[leaf_id])
            refresh_membership_state_preserve_residual_reference(leaves_[leaf_id]);
    }
    report.residual_tail_reassigned_vectors += moves.size();
    report.removed_empty_partitions += remove_empty_leaves();
}

void AdaptiveHierarchicalIvf::recenter(std::size_t leaf_id) {
    Leaf& leaf = leaves_[leaf_id];
    if (leaf.ids.empty()) return;
    float* centroid = leaf_centroids_.data() + leaf_id * config_.dimension;
    if (uses_u8_storage(config_)) {
        const double inverse = 1.0 / static_cast<double>(leaf.ids.size());
        for (std::size_t d = 0; d < config_.dimension; ++d)
            centroid[d] = static_cast<float>(leaf.vector_sum[d] * inverse);
        reset_reference(leaf, centroid);
        return;
    }
    if (uses_lvq_storage(config_) || uses_u8_storage(config_)) {
        std::vector<float> decoded_all;
        decoded_all.reserve(leaf.ids.size() * config_.dimension);
        std::vector<float> decoded(config_.dimension);
        for (std::size_t i = 0; i < leaf.ids.size(); ++i) {
            decode_vector(leaf_id, i, decoded.data());
            decoded_all.insert(decoded_all.end(), decoded.begin(), decoded.end());
        }
        std::fill(centroid, centroid + config_.dimension, 0.0f);
        for (std::size_t i = 0; i < leaf.ids.size(); ++i)
            for (std::size_t d = 0; d < config_.dimension; ++d)
                centroid[d] += decoded_all[i * config_.dimension + d];
        for (std::size_t d = 0; d < config_.dimension; ++d)
            centroid[d] /= static_cast<float>(leaf.ids.size());
        clear_vector_payload(leaf);
        for (std::size_t i = 0; i < leaf.ids.size(); ++i)
            store_vector(leaf_id, leaf, decoded_all.data() + i * config_.dimension);
        lvq8_reencoded_vectors_ += leaf.ids.size();
        reset_reference(leaf, centroid);
        return;
    }
    std::fill(centroid, centroid + config_.dimension, 0.0f);
    for (std::size_t i = 0; i < leaf.ids.size(); ++i) {
        for (std::size_t d = 0; d < config_.dimension; ++d) centroid[d] += leaf.vectors[i * config_.dimension + d];
    }
    for (std::size_t d = 0; d < config_.dimension; ++d) centroid[d] /= static_cast<float>(leaf.ids.size());
    reset_reference(leaf, centroid);
}

void AdaptiveHierarchicalIvf::split(std::size_t leaf_id, bool recenter_on_degenerate) {
    Leaf& source = leaves_[leaf_id];
    if (source.ids.size() < 2) return;
    if (uses_lvq_storage(config_) || uses_u8_storage(config_)) {
        std::vector<float> points(source.ids.size() * config_.dimension);
        std::vector<float> decoded(config_.dimension);
        const std::vector<std::uint32_t> source_ids = source.ids;
        for (std::size_t i = 0; i < source.ids.size(); ++i) {
            decode_vector(leaf_id, i, decoded.data());
            std::copy(decoded.begin(), decoded.end(), points.begin() +
                      static_cast<std::ptrdiff_t>(i * config_.dimension));
        }
        std::vector<float> centers;
        train_kmeans(points.data(), source_ids.size(), 2,
                     std::min<std::size_t>(5, config_.kmeans_iterations), &centers);
        std::vector<std::uint32_t> left_ids, right_ids;
        std::vector<float> left_points, right_points;
        for (std::size_t i = 0; i < source_ids.size(); ++i) {
            const float* point = points.data() + i * config_.dimension;
            const bool choose_left = retrieval_distance(point, centers.data()) <=
                                     retrieval_distance(point, centers.data() + config_.dimension);
            auto& target_ids = choose_left ? left_ids : right_ids;
            auto& target_points = choose_left ? left_points : right_points;
            target_ids.push_back(source_ids[i]);
            target_points.insert(target_points.end(), point, point + config_.dimension);
        }
        if (left_ids.empty() || right_ids.empty()) {
            if (recenter_on_degenerate) recenter(leaf_id);
            return;
        }
        source = Leaf{};
        source.ids = std::move(left_ids);
        std::copy_n(centers.data(), config_.dimension,
                    leaf_centroids_.data() + leaf_id * config_.dimension);
        leaves_.push_back(Leaf{});
        leaves_.back().ids = std::move(right_ids);
        leaf_centroids_.insert(leaf_centroids_.end(), centers.begin() + config_.dimension,
                               centers.end());
        Leaf& left_leaf = leaves_[leaf_id];
        for (std::size_t i = 0; i < left_leaf.ids.size(); ++i)
            store_vector(leaf_id, left_leaf, left_points.data() + i * config_.dimension);
        const std::size_t right_id = leaves_.size() - 1;
        for (std::size_t i = 0; i < leaves_.back().ids.size(); ++i)
            store_vector(right_id, leaves_.back(), right_points.data() + i * config_.dimension);
        if (uses_lvq_storage(config_)) lvq8_reencoded_vectors_ += source_ids.size();
        reset_reference(left_leaf, leaf_centroids_.data() + leaf_id * config_.dimension);
        reset_reference(leaves_.back(), leaf_centroids_.data() + right_id * config_.dimension);
        return;
    }
    std::vector<float> centers;
    train_kmeans(source.vectors.data(), source.ids.size(), 2,
                 std::min<std::size_t>(5, config_.kmeans_iterations), &centers);
    Leaf left, right;
    for (std::size_t i = 0; i < source.ids.size(); ++i) {
        const float* point = source.vectors.data() + i * config_.dimension;
        Leaf& destination = retrieval_distance(point, centers.data()) <= retrieval_distance(point, centers.data() + config_.dimension)
                                ? left : right;
        destination.ids.push_back(source.ids[i]);
        destination.vectors.insert(destination.vectors.end(), point, point + config_.dimension);
    }
    if (left.ids.empty() || right.ids.empty()) {
        if (recenter_on_degenerate) recenter(leaf_id);
        return;
    }
    source = std::move(left);
    std::copy_n(centers.data(), config_.dimension, leaf_centroids_.data() + leaf_id * config_.dimension);
    leaves_.push_back(std::move(right));
    leaf_centroids_.insert(leaf_centroids_.end(), centers.begin() + config_.dimension, centers.end());
    reset_reference(leaves_[leaf_id], leaf_centroids_.data() + leaf_id * config_.dimension);
    reset_reference(leaves_.back(), leaf_centroids_.data() + (leaves_.size() - 1) * config_.dimension);
}

std::size_t AdaptiveHierarchicalIvf::remove_empty_leaves(
        std::vector<std::size_t>* old_to_new) {
    const std::size_t old_count = leaves_.size();
    if (old_to_new != nullptr)
        old_to_new->assign(old_count, std::numeric_limits<std::size_t>::max());
    const std::size_t removed = static_cast<std::size_t>(std::count_if(
            leaves_.begin(), leaves_.end(), [](const Leaf& leaf) { return leaf.ids.empty(); }));
    if (removed == 0) {
        if (old_to_new != nullptr)
            std::iota(old_to_new->begin(), old_to_new->end(), std::size_t{0});
        return 0;
    }
    std::vector<Leaf> kept_leaves;
    std::vector<float> kept_centroids;
    kept_leaves.reserve(leaves_.size());
    kept_centroids.reserve(leaf_centroids_.size());
    for (std::size_t i = 0; i < leaves_.size(); ++i) {
        if (leaves_[i].ids.empty()) continue;
        if (old_to_new != nullptr) (*old_to_new)[i] = kept_leaves.size();
        kept_leaves.push_back(std::move(leaves_[i]));
        kept_centroids.insert(kept_centroids.end(),
                              leaf_centroids_.begin() + i * config_.dimension,
                              leaf_centroids_.begin() + (i + 1) * config_.dimension);
    }
    leaves_ = std::move(kept_leaves);
    leaf_centroids_ = std::move(kept_centroids);
    return removed;
}

bool AdaptiveHierarchicalIvf::merge_short_leaf(
        std::size_t leaf_id, float short_cutoff,
        std::vector<std::size_t>* merged_into) {
    if (leaf_id >= leaves_.size() || leaves_[leaf_id].ids.empty()) return false;
    const std::size_t source_size = leaves_[leaf_id].ids.size();
    if (static_cast<float>(source_size) >= short_cutoff) return false;
    const auto capacity_reference = [](const Leaf& leaf) {
        return leaf.capacity_reference_size == 0 ? leaf.ids.size() : leaf.capacity_reference_size;
    };
    const std::size_t source_capacity_reference = capacity_reference(leaves_[leaf_id]);
    std::size_t live_leaf_count = 0;
    for (const Leaf& leaf : leaves_)
        if (!leaf.ids.empty()) ++live_leaf_count;
    if (live_leaf_count <= 1) return false;

    // Snapshot the source vectors, then route each one independently to its
    // nearest other live centroid. Subsequent load/capacity checks in this
    // maintenance round handle any recipient that becomes oversized.
    std::vector<std::uint32_t> source_ids = leaves_[leaf_id].ids;
    std::vector<float> source_vectors(source_size * config_.dimension);
    std::vector<float> decoded(config_.dimension);
    for (std::size_t i = 0; i < source_size; ++i) {
        const float* point = nullptr;
        if (uses_lvq_storage(config_) || uses_u8_storage(config_)) {
            decode_vector(leaf_id, i, decoded.data());
            point = decoded.data();
        } else {
            point = leaves_[leaf_id].vectors.data() + i * config_.dimension;
        }
        std::copy_n(point, config_.dimension,
                    source_vectors.data() + i * config_.dimension);
    }

    std::vector<std::size_t> assignments(source_size, leaves_.size());
    std::vector<std::size_t> moved_counts(leaves_.size(), 0);
    std::vector<std::size_t> previous_capacity_references(leaves_.size(), 0);
    for (std::size_t i = 0; i < leaves_.size(); ++i)
        if (i != leaf_id && !leaves_[i].ids.empty())
            previous_capacity_references[i] = capacity_reference(leaves_[i]);

    for (std::size_t i = 0; i < source_size; ++i) {
        const float* point = source_vectors.data() + i * config_.dimension;
        float best_distance = std::numeric_limits<float>::max();
        std::size_t best_target = leaves_.size();
        for (std::size_t candidate = 0; candidate < leaves_.size(); ++candidate) {
            if (candidate == leaf_id || leaves_[candidate].ids.empty()) continue;
            const float distance = retrieval_distance(
                point, leaf_centroids_.data() + candidate * config_.dimension);
            if (distance < best_distance) {
                best_distance = distance;
                best_target = candidate;
            }
        }
        if (best_target == leaves_.size()) return false;
        assignments[i] = best_target;
        ++moved_counts[best_target];
    }

    Leaf& source = leaves_[leaf_id];
    source.ids.clear();
    clear_vector_payload(source);
    source.vector_sum.assign(config_.dimension, 0.0);
    source.squared_norm_sum = 0.0;
    source.capacity_reference_size = 0;

    for (std::size_t i = 0; i < source_size; ++i) {
        const std::size_t target = assignments[i];
        Leaf& destination = leaves_[target];
        destination.ids.push_back(source_ids[i]);
        store_vector(target, destination,
                     source_vectors.data() + i * config_.dimension);
    }

    std::size_t remaining_reference = source_capacity_reference;
    std::size_t remaining_vectors = source_size;
    for (std::size_t target = 0; target < moved_counts.size(); ++target) {
        const std::size_t moved = moved_counts[target];
        if (moved == 0) continue;
        const std::size_t reference_share = moved == remaining_vectors
            ? remaining_reference
            : static_cast<std::size_t>(
                (static_cast<long double>(source_capacity_reference) * moved) /
                static_cast<long double>(source_size));
        const std::size_t old_reference = previous_capacity_references[target];
        recenter(target);  // update the receiver mean and rebuild its statistics
        leaves_[target].capacity_reference_size =
            reference_share > std::numeric_limits<std::size_t>::max() - old_reference
                ? std::numeric_limits<std::size_t>::max()
                : old_reference + reference_share;
        remaining_reference -= reference_share;
        remaining_vectors -= moved;
        if (merged_into != nullptr) merged_into->push_back(target);
    }
    return true;
}

MaintenanceReport AdaptiveHierarchicalIvf::maintain() {
    return maintain_impl(true, true, true, true);
}

MaintenanceReport AdaptiveHierarchicalIvf::maintain_load_only() {
    return maintain_impl(false, true, false, true);
}

MaintenanceReport AdaptiveHierarchicalIvf::maintain_with_policy(
        bool enable_prd_splits, bool enable_load_splits,
        bool enable_capacity_splits, bool enable_load_merges) {
    if (!trained_) throw std::logic_error("train must be called before maintain_with_policy");
    if (update_batch_open_)
        throw std::logic_error("custom maintenance must run after the update batch is closed");
    return maintain_impl(enable_prd_splits, enable_load_splits,
                         enable_capacity_splits, enable_load_merges);
}

MaintenanceReport AdaptiveHierarchicalIvf::maintain_quake_lire_policy(
        std::size_t max_partition_size, std::size_t min_partition_size,
        std::size_t refinement_radius, std::size_t refinement_iterations) {
    if (!trained_) throw std::logic_error("train must be called before maintain_quake_lire_policy");
    if (update_batch_open_)
        throw std::logic_error("Quake LIRE maintenance must run after the update batch is closed");
    if (max_partition_size == 0)
        throw std::invalid_argument("Quake LIRE max_partition_size must be positive");

    const std::size_t nlist_before = leaves_.size();
    const std::size_t vector_count_before = active_count_;
    std::vector<std::size_t> split_ids;
    std::vector<std::size_t> delete_ids_by_leaf;
    for (std::size_t leaf_id = 0; leaf_id < leaves_.size(); ++leaf_id) {
        const std::size_t size = leaves_[leaf_id].ids.size();
        // This mirrors Quake's size-based branch, including its else-if:
        // a list is either split or deleted, never both in one pass.
        if (size > max_partition_size) {
            split_ids.push_back(leaf_id);
        } else if (size < min_partition_size) {
            delete_ids_by_leaf.push_back(leaf_id);
        }
    }

    struct RestoreAutomaticMaintenance {
        bool& setting;
        bool previous;
        ~RestoreAutomaticMaintenance() { setting = previous; }
    } restore{config_.automatic_maintenance, config_.automatic_maintenance};
    // Vector routing can trigger automatic PRD maintenance. The Quake LIRE
    // policy performs one explicit size-based pass, so suppress that callback.
    config_.automatic_maintenance = false;

    MaintenanceReport report;
    report.checked_partitions = nlist_before;
    report.marked_partitions = split_ids.size() + delete_ids_by_leaf.size();
    report.leaf_nlist_before = nlist_before;

    // Quake's delete_partitions() defaults to reassign=true: retiring a small
    // partition must not discard its vectors. Route every such vector to its
    // nearest surviving centroid before removing the now-empty leaf objects.
    // If all leaves are under the minimum, keep them rather than lose data or
    // leave no destination for reassignment.
    std::vector<bool> retiring(leaves_.size(), false);
    for (const std::size_t leaf_id : delete_ids_by_leaf) retiring[leaf_id] = true;
    std::size_t surviving_leaf_count = leaves_.size() - delete_ids_by_leaf.size();
    if (surviving_leaf_count == 0) {
        delete_ids_by_leaf.clear();
        std::fill(retiring.begin(), retiring.end(), false);
    }
    struct SmallPartitionMove {
        std::size_t destination;
        std::uint32_t id;
        std::vector<float> vector;
    };
    std::vector<SmallPartitionMove> small_partition_moves;
    if (!delete_ids_by_leaf.empty()) {
        std::vector<float> decoded(config_.dimension);
        std::vector<std::size_t> touched;
        std::vector<bool> touched_mask(leaves_.size(), false);
        std::size_t short_vector_count = 0;
        for (const std::size_t source : delete_ids_by_leaf)
            short_vector_count += leaves_[source].ids.size();
        small_partition_moves.reserve(short_vector_count);
        for (const std::size_t source : delete_ids_by_leaf) {
            const Leaf& leaf = leaves_[source];
            for (std::size_t position = 0; position < leaf.ids.size(); ++position) {
                decode_vector(source, position, decoded.data());
                float best_distance = std::numeric_limits<float>::max();
                std::size_t best_destination = leaves_.size();
                for (std::size_t target = 0; target < leaves_.size(); ++target) {
                    if (retiring[target]) continue;
                    const float distance = retrieval_distance(
                        decoded.data(), leaf_centroids_.data() + target * config_.dimension);
                    if (distance < best_distance) {
                        best_distance = distance;
                        best_destination = target;
                    }
                }
                if (best_destination == leaves_.size())
                    throw std::logic_error("Quake LIRE short-partition reassignment has no recipient");
                SmallPartitionMove move;
                move.destination = best_destination;
                move.id = leaf.ids[position];
                move.vector.assign(decoded.begin(), decoded.end());
                small_partition_moves.push_back(std::move(move));
                if (!touched_mask[best_destination]) {
                    touched_mask[best_destination] = true;
                    touched.push_back(best_destination);
                }
            }
        }
        for (const std::size_t source : delete_ids_by_leaf) {
            leaves_[source].ids.clear();
            clear_vector_payload(leaves_[source]);
        }
        for (const SmallPartitionMove& move : small_partition_moves) {
            Leaf& destination = leaves_[move.destination];
            destination.ids.push_back(move.id);
            store_vector(move.destination, destination, move.vector.data());
            apply_vector_delta(destination, move.vector.data(), 1.0);
        }
        for (const std::size_t leaf_id : touched)
            refresh_membership_state_preserve_residual_reference(leaves_[leaf_id]);
        report.reassigned_small_partition_vectors = small_partition_moves.size();
    }
    report.deleted_small_partitions = delete_ids_by_leaf.size();

    // Quake removes deleted partition objects before splitting. This index
    // stores leaves in a dense vector, so compact them and translate the
    // surviving split IDs to their new slots first.
    std::vector<std::size_t> old_to_new;
    report.removed_empty_partitions = remove_empty_leaves(&old_to_new);
    std::vector<std::size_t> compacted_split_ids;
    compacted_split_ids.reserve(split_ids.size());
    for (const std::size_t old_id : split_ids) {
        if (old_id < old_to_new.size() &&
            old_to_new[old_id] != std::numeric_limits<std::size_t>::max())
            compacted_split_ids.push_back(old_to_new[old_id]);
    }

    const std::size_t reencoded_before = lvq8_reencoded_vectors_;
    if (!compacted_split_ids.empty()) {
        // Split every selected partition before refining, as Quake does.
        // Refinement is performed below because Quake seeds it with both new
        // child centroids; maintain_selected()'s generic path seeds only the
        // appended child.
        MaintenanceReport split_report = maintain_selected(
                compacted_split_ids, SelectedMaintenanceAction::Split, 0, 0);
        if (refinement_radius > 0 && split_report.split_partitions > 0) {
            std::vector<std::size_t> refinement_seeds = split_report.source_split_leaf_ids;
            const std::size_t first_appended_child = leaves_.size() - split_report.split_partitions;
            for (std::size_t child = first_appended_child; child < leaves_.size(); ++child)
                refinement_seeds.push_back(child);

            // Quake's radius is top-k including each seed, whereas
            // locally_reassign() takes a count of additional neighbors.
            // Quake's refinement_iterations=0 means one assignment pass.
            locally_reassign(refinement_seeds, refinement_radius - 1,
                    std::max<std::size_t>(1, refinement_iterations), &split_report);
        }
        report.split_partitions = split_report.split_partitions;
        report.total_split_partitions = split_report.total_split_partitions;
        report.recentered_partitions = split_report.recentered_partitions;
        report.local_refinement_partitions = split_report.local_refinement_partitions;
        report.local_refinement_vectors = split_report.local_refinement_vectors;
        report.local_refinement_reassigned_vectors =
                split_report.local_refinement_reassigned_vectors;
        report.source_split_leaf_ids = split_report.source_split_leaf_ids;
    } else {
        for (Leaf& leaf : leaves_) reset_window(leaf);
        changes_since_maintenance_ = 0;
        next_maintenance_budget_ = 0;
    }

    report.removed_empty_partitions += remove_empty_leaves();
    if (active_count_ != vector_count_before)
        throw std::logic_error("Quake LIRE maintenance changed active vector count");
    report.leaf_nlist_after = leaves_.size();
    report.vector_count = active_count_;
    report.lvq8_reencoded_vectors = lvq8_reencoded_vectors_ - reencoded_before;
    report.next_change_budget = 0;
    return report;
}

MaintenanceReport AdaptiveHierarchicalIvf::maintain_impl(
        bool enable_prd_splits, bool enable_load_splits,
        bool enable_capacity_splits, bool enable_load_merges) {
    if (!trained_) throw std::logic_error("train must be called before maintain");
    const std::size_t reencoded_before = lvq8_reencoded_vectors_;
    MaintenanceReport report;
    report.leaf_nlist_before = leaves_.size();
    std::unordered_set<std::uint32_t> split_or_merged_vector_ids;
    std::vector<std::size_t> split_refinement_seeds;
    const auto capture_leaf_ids = [this, &split_or_merged_vector_ids](std::size_t leaf_id) {
        if (leaf_id >= leaves_.size()) return;
        split_or_merged_vector_ids.insert(leaves_[leaf_id].ids.begin(),
                                          leaves_[leaf_id].ids.end());
    };
    const auto capture_split_children = [&capture_leaf_ids, &split_refinement_seeds](
            std::size_t parent_id, std::size_t old_leaf_count) {
        capture_leaf_ids(parent_id);
        capture_leaf_ids(old_leaf_count);
        // Match Quake's local refinement entry point: seed from the newly
        // appended child, then include its nearest neighboring leaves.
        split_refinement_seeds.push_back(old_leaf_count);
    };
    report.removed_empty_partitions = remove_empty_leaves();
    if (leaves_.empty() && active_count_ != 0) throw std::logic_error("active vectors lost all leaves");

    const auto compute_target = [this]() {
        if (leaves_.empty()) return 0.0f;
        return
        (config_.sqrt_load_target ? config_.sqrt_load_target_multiplier *
             static_cast<float>(std::sqrt(static_cast<double>(active_count_))) :
         (config_.load_target_size > 0.0f ? config_.load_target_size :
          dynamic_load_target_size(active_count_, leaves_.size())));
    };

    // Load maintenance includes both underloaded-list merging and
    // overloaded-list splitting. The pre-merge target is used to select short
    // lists; the split target is recomputed after merges and frozen this round.
    if (enable_load_merges) {
        const float merge_target = compute_target();
        const float short_cutoff = config_.short_list_n_fraction > 0.0f
            ? config_.short_list_n_fraction * static_cast<float>(active_count_)
            : (config_.short_list_sqrt_n_fraction > 0.0f
            ? config_.short_list_sqrt_n_fraction *
                  static_cast<float>(std::sqrt(static_cast<double>(active_count_)))
            : config_.short_list_fraction * merge_target);
        std::vector<std::size_t> short_candidates;
        for (std::size_t i = 0; i < leaves_.size(); ++i)
            if (leaves_[i].ids.size() < short_cutoff)
                short_candidates.push_back(i);
        // Smallest leaves are merged first, so they cannot be used to overload
        // one another before their own underload status is considered.
        std::sort(short_candidates.begin(), short_candidates.end(), [this](std::size_t a, std::size_t b) {
            return leaves_[a].ids.size() < leaves_[b].ids.size();
        });
        for (const auto id : short_candidates) {
            std::vector<std::size_t> merged_into;
            const std::size_t source_vectors = id < leaves_.size() ? leaves_[id].ids.size() : 0;
            if (merge_short_leaf(id, short_cutoff, &merged_into)) {
                ++report.merged_short_partitions;
                report.merge_recipient_partitions += merged_into.size();
                report.merge_routed_vectors += source_vectors;
                for (const auto target : merged_into) capture_leaf_ids(target);
            }
        }
    }
    report.removed_empty_partitions += remove_empty_leaves();

    const float target = compute_target();
    report.load_target_T = target;
    report.checked_partitions = leaves_.size();
    std::vector<bool> handled(leaves_.size(), false);
    std::vector<std::size_t> prd_candidates;
    if (enable_prd_splits) {
        for (std::size_t id = 0; id < leaves_.size(); ++id)
            if (is_marked(id)) prd_candidates.push_back(id);
    }
    report.marked_partitions = prd_candidates.size();

    // Load balancing follows short-list merging. T is frozen after merging;
    // split oversized leaves before growth protection and PRD repair.
    const std::size_t existing_leaf_count = leaves_.size();
    if (enable_load_splits) {
        for (std::size_t id = 0; id < existing_leaf_count; ++id) {
            if (static_cast<float>(leaves_[id].ids.size()) <=
                config_.load_split_fraction * target) continue;
            const std::size_t old_leaf_count = leaves_.size();
            split(id);
            if (leaves_.size() > old_leaf_count) {
                ++report.total_split_partitions;
                ++report.load_split_partitions;
                capture_split_children(id, old_leaf_count);
            } else {
                ++report.recentered_partitions; // degenerate split falls back to recenter
            }
            handled[id] = true;
        }
    }

    // Capacity growth is independent of PRD. Check it before recentering so a
    // later geometric repair cannot reset the reference size and hide growth.
    if (enable_capacity_splits) {
        for (std::size_t id = 0; id < existing_leaf_count; ++id) {
            const Leaf& leaf = leaves_[id];
            bool capacity_split_due = false;
            if (config_.capacity_n_divisor > 0.0f) {
                const double linear_reference =
                    static_cast<double>(active_count_) /
                    static_cast<double>(config_.capacity_n_divisor);
                const bool cluster_gate_open = config_.capacity_cluster_count_gate > 0
                    ? leaves_.size() < config_.capacity_cluster_count_gate
                    : static_cast<double>(leaves_.size()) <= linear_reference;
                if (cluster_gate_open) {
                    const double capacity_threshold =
                        static_cast<double>(config_.capacity_sqrt_n_split_multiplier) *
                        linear_reference;
                    capacity_split_due =
                        static_cast<double>(leaf.ids.size()) > capacity_threshold;
                }
            } else if (config_.capacity_sqrt_n_split_multiplier > 0.0f) {
                const double sqrt_active_count =
                    std::sqrt(static_cast<double>(active_count_));
                const bool at_most_sqrt_n_clusters =
                    static_cast<double>(leaves_.size()) <= sqrt_active_count;
                if (at_most_sqrt_n_clusters) {
                    const double capacity_threshold =
                        static_cast<double>(config_.capacity_sqrt_n_split_multiplier) *
                        sqrt_active_count;
                    capacity_split_due =
                        static_cast<double>(leaf.ids.size()) > capacity_threshold;
                }
            } else {
                capacity_split_due =
                    leaf.ids.size() > target && capacity_growth_split_due(leaf);
            }
            if (handled[id] || !capacity_split_due) continue;
            const std::size_t old_leaf_count = leaves_.size();
            split(id);
            if (leaves_.size() > old_leaf_count) {
                ++report.total_split_partitions;
                ++report.capacity_split_partitions;
                capture_split_children(id, old_leaf_count);
            } else {
                ++report.recentered_partitions; // degenerate split falls back to recenter
            }
            handled[id] = true;
        }
    }

    if (enable_prd_splits) {
        for (const std::size_t id : prd_candidates) {
            if (handled[id] || leaves_[id].ids.empty()) continue;
            // PRD is now split-only: once a leaf crosses the PRD threshold,
            // attempt a split directly without a recenter-gain decision.
            const std::size_t old_leaf_count = leaves_.size();
            split(id, false);
            if (leaves_.size() > old_leaf_count) {
                ++report.split_partitions;
                ++report.prd_split_partitions;
                ++report.total_split_partitions;
                capture_split_children(id, old_leaf_count);
            } else {
                ++report.prd_split_failures;
            }
            handled[id] = true;
        }
    }
    if (config_.split_local_refinement_radius > 0 &&
        !split_refinement_seeds.empty()) {
        locally_reassign(split_refinement_seeds,
                         config_.split_local_refinement_radius,
                         std::max<std::size_t>(1,
                             config_.split_local_refinement_iterations),
                         &report);
    }
    if (config_.residual_tail_reassign_selected_only) {
        std::vector<std::size_t> selected_leaves;
        if (!split_or_merged_vector_ids.empty()) {
            for (std::size_t leaf_id = 0; leaf_id < leaves_.size(); ++leaf_id) {
                const Leaf& leaf = leaves_[leaf_id];
                if (std::any_of(leaf.ids.begin(), leaf.ids.end(),
                    [&split_or_merged_vector_ids](std::uint32_t id) {
                        return split_or_merged_vector_ids.find(id) !=
                               split_or_merged_vector_ids.end();
                    })) {
                    selected_leaves.push_back(leaf_id);
                }
            }
        }
        reassign_residual_tail(config_.residual_tail_reassign_fraction, report,
                               &selected_leaves);
    } else {
        reassign_residual_tail(config_.residual_tail_reassign_fraction, report);
    }
    // Clear per-batch insert/delete diagnostics after this maintenance round.
    // Unchanged leaves retain their reference state until actually repaired.
    for (Leaf& leaf : leaves_) reset_window(leaf);
    changes_since_maintenance_ = 0;
    report.leaf_nlist_after = leaves_.size();
    report.vector_count = active_count_;
    report.lvq8_reencoded_vectors = lvq8_reencoded_vectors_ - reencoded_before;
    next_maintenance_budget_ = 0;
    report.next_change_budget = next_maintenance_budget_;
    return report;
}

MaintenanceReport AdaptiveHierarchicalIvf::maintain_selected(
        const std::vector<std::size_t>& leaf_ids, SelectedMaintenanceAction action,
        std::size_t refinement_radius, std::size_t refinement_iterations) {
    if (!trained_) throw std::logic_error("train must be called before maintain_selected");
    const std::size_t reencoded_before = lvq8_reencoded_vectors_;
    MaintenanceReport report;
    report.leaf_nlist_before = leaves_.size();
    const std::size_t first_new_leaf = leaves_.size();
    std::vector<std::size_t> selected = leaf_ids;
    std::sort(selected.begin(), selected.end());
    selected.erase(std::unique(selected.begin(), selected.end()), selected.end());
    for (const std::size_t id : selected) {
        if (id >= leaves_.size() || leaves_[id].ids.empty()) continue;
        ++report.checked_partitions;
        ++report.marked_partitions;
        if (action == SelectedMaintenanceAction::Recenter) {
            recenter(id);
            ++report.recentered_partitions;
        } else {
            const std::size_t old_leaf_count = leaves_.size();
            split(id);
            if (leaves_.size() > old_leaf_count) {
                ++report.split_partitions;
                ++report.total_split_partitions;
                report.source_split_leaf_ids.push_back(id);
            } else {
                ++report.recentered_partitions;
            }
        }
    }
    if (action == SelectedMaintenanceAction::Split && refinement_radius > 0 &&
        leaves_.size() > first_new_leaf) {
        std::vector<std::size_t> refinement_seeds;
        refinement_seeds.reserve(leaves_.size() - first_new_leaf);
        for (std::size_t id = first_new_leaf; id < leaves_.size(); ++id)
            refinement_seeds.push_back(id);
        locally_reassign(refinement_seeds, refinement_radius,
                         std::max<std::size_t>(1, refinement_iterations), &report);
    }
    for (Leaf& leaf : leaves_) reset_window(leaf);
    changes_since_maintenance_ = 0;
    report.leaf_nlist_after = leaves_.size();
    report.vector_count = active_count_;
    report.lvq8_reencoded_vectors = lvq8_reencoded_vectors_ - reencoded_before;
    next_maintenance_budget_ = 0;
    report.next_change_budget = next_maintenance_budget_;
    return report;
}

MaintenanceReport AdaptiveHierarchicalIvf::maintain_top_fraction(
        MaintenanceSelectionPolicy policy, float fraction, std::uint32_t seed) {
    if (!trained_) throw std::logic_error("train must be called before maintain_top_fraction");
    if (!(fraction > 0.0f && fraction <= 1.0f))
        throw std::invalid_argument("maintenance fraction must be in (0, 1]");

    const std::size_t reencoded_before = lvq8_reencoded_vectors_;
    MaintenanceReport report;
    report.leaf_nlist_before = leaves_.size();
    report.removed_empty_partitions = remove_empty_leaves();
    if (leaves_.empty() && active_count_ != 0) throw std::logic_error("active vectors lost all leaves");
    report.checked_partitions = leaves_.size();
    report.vector_count = active_count_;
    report.load_target_T = leaves_.empty() ? 0.0f :
            dynamic_load_target_size(active_count_, leaves_.size());

    std::vector<std::size_t> ids(leaves_.size());
    std::iota(ids.begin(), ids.end(), 0);
    if (policy == MaintenanceSelectionPolicy::Random) {
        std::mt19937 rng(seed);
        std::shuffle(ids.begin(), ids.end(), rng);
    } else if (policy == MaintenanceSelectionPolicy::Longest) {
        std::stable_sort(ids.begin(), ids.end(), [this](std::size_t a, std::size_t b) {
            return leaves_[a].ids.size() > leaves_[b].ids.size();
        });
    } else {
        std::stable_sort(ids.begin(), ids.end(), [this](std::size_t a, std::size_t b) {
            const float* ca = leaf_centroids_.data() + a * config_.dimension;
            const float* cb = leaf_centroids_.data() + b * config_.dimension;
            const float da = drift_score(leaves_[a], ca);
            const float db = drift_score(leaves_[b], cb);
            return da > db;
        });
    }
    const std::size_t selected_count = ids.empty() ? 0 : std::min(ids.size(),
            std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(fraction * ids.size()))));
    ids.resize(selected_count);
    report.marked_partitions = selected_count;
    const float load_cutoff = config_.load_split_fraction * report.load_target_T;
    bool changed = report.removed_empty_partitions != 0;
    for (const std::size_t id : ids) {
        Leaf& leaf = leaves_[id];
        if (leaf.ids.empty()) continue;
        ++report.checked_partitions;
        report.selected_vectors += leaf.ids.size();
        const double before_sse = static_cast<double>(current_mean_residual(
                leaf, leaf_centroids_.data() + id * config_.dimension)) * leaf.ids.size();
        report.selected_sse_before += before_sse;
        const bool load_split_due = leaf.ids.size() > load_cutoff;
        const std::size_t old_leaf_count = leaves_.size();
        // This API explicitly selects leaves for maintenance; selected leaves
        // are split directly, with load-vs-PRD attribution preserved.
        split(id, load_split_due);
        if (leaves_.size() > old_leaf_count) {
            ++report.split_partitions;
            ++report.total_split_partitions;
            if (load_split_due) ++report.load_split_partitions;
            else ++report.prd_split_partitions;
            const Leaf& left = leaves_[id];
            const Leaf& right = leaves_.back();
            report.selected_sse_after += static_cast<double>(left.ref_mean) * left.ids.size() +
                    static_cast<double>(right.ref_mean) * right.ids.size();
        } else {
            if (load_split_due) {
                ++report.recentered_partitions;
                report.selected_sse_after += static_cast<double>(leaves_[id].ref_mean) * leaves_[id].ids.size();
            } else {
                ++report.prd_split_failures;
                report.selected_sse_after += static_cast<double>(current_mean_residual(
                    leaves_[id], leaf_centroids_.data() + id * config_.dimension)) * leaves_[id].ids.size();
            }
        }
        changed = true;
    }
    for (Leaf& leaf : leaves_) reset_window(leaf);
    changes_since_maintenance_ = 0;
    report.leaf_nlist_after = leaves_.size();
    report.lvq8_reencoded_vectors = lvq8_reencoded_vectors_ - reencoded_before;
    next_maintenance_budget_ = 0;
    report.next_change_budget = next_maintenance_budget_;
    return report;
}

std::size_t AdaptiveHierarchicalIvf::remove_ids(const std::vector<std::int64_t>& ids,
                                                const float* exact_removed_vectors) {
    if (!trained_) throw std::logic_error("train must be called before remove_ids");
    if (ids.empty()) return 0;
    bool dense_range = true;
    for (std::size_t i = 1; i < ids.size(); ++i) {
        if (ids[i - 1] == std::numeric_limits<std::int64_t>::max() || ids[i] != ids[i - 1] + 1) {
            dense_range = false;
            break;
        }
    }
    const std::int64_t first = dense_range ? ids.front() : 0;
    const std::int64_t last = dense_range ? ids.back() : -1;
    const std::unordered_set<std::int64_t> targets = dense_range
            ? std::unordered_set<std::int64_t>{}
            : std::unordered_set<std::int64_t>(ids.begin(), ids.end());
    std::unordered_map<std::int64_t, std::size_t> exact_order;
    if (exact_removed_vectors && !dense_range) {
        exact_order.reserve(ids.size());
        for (std::size_t i = 0; i < ids.size(); ++i) exact_order.emplace(ids[i], i);
    }
    std::size_t removed = 0;
    for (Leaf& leaf : leaves_) {
        std::size_t position = 0;
        while (position < leaf.ids.size()) {
            if (dense_range && position + 8 <= leaf.ids.size() &&
                simd_id_range_mask(leaf.ids.data() + position, 8, first, last) == 0) {
                position += 8;
                continue;
            }
            const bool selected = dense_range
                    ? leaf.ids[position] >= first && leaf.ids[position] <= last
                    : targets.count(leaf.ids[position]) != 0;
            if (!selected) { ++position; continue; }
            // Update PRD aggregates before swap-removing the vector from storage.
            std::vector<float> decoded((uses_lvq_storage(config_) || uses_u8_storage(config_))
                                           ? config_.dimension : 0);
            const float* removed_vector = nullptr;
            if (exact_removed_vectors) {
                const std::size_t exact_index = dense_range
                    ? static_cast<std::size_t>(leaf.ids[position] - first)
                    : exact_order.at(static_cast<std::int64_t>(leaf.ids[position]));
                removed_vector = exact_removed_vectors + exact_index * config_.dimension;
            } else if (uses_lvq_storage(config_) || uses_u8_storage(config_)) {
                decode_vector(static_cast<std::size_t>(&leaf - leaves_.data()), position,
                              decoded.data());
                removed_vector = decoded.data();
            } else {
                removed_vector = leaf.vectors.data() + position * config_.dimension;
            }
            apply_vector_delta(leaf, removed_vector, -1.0);
            const std::size_t last = leaf.ids.size() - 1;
            if (position != last) {
                leaf.ids[position] = leaf.ids[last];
            }
            remove_vector_payload(leaf, position, last);
            leaf.ids.pop_back();
            ++removed;
            --active_count_;
            ++changes_since_maintenance_;
            ++leaf.window_deletes;
        }
    }
    if (config_.automatic_maintenance && !update_batch_open_ && maintenance_due()) maintain();
    return removed;
}

std::vector<SearchResult> AdaptiveHierarchicalIvf::search(const float* query, std::size_t k) const {
    return search(query, k, nullptr);
}

std::vector<SearchResult> AdaptiveHierarchicalIvf::search(
        const float* query, std::size_t k,
        std::vector<std::size_t>* probed_leaf_ids) const {
    const std::size_t leaf_probe = config_.fixed_leaf_nprobe != 0
        ? config_.fixed_leaf_nprobe
        : std::max<std::size_t>(1, (leaves_.size() + config_.leaf_nprobe_ratio - 1) /
                                      config_.leaf_nprobe_ratio);
    return search(query, k, leaf_probe, probed_leaf_ids);
}

std::vector<SearchResult> AdaptiveHierarchicalIvf::search(
        const float* query, std::size_t k, std::size_t leaf_probe,
        std::vector<std::size_t>* probed_leaf_ids) const {
    return search_with_options(query, k, leaf_probe, QuerySearchOptions{}, nullptr,
                               probed_leaf_ids);
}

std::vector<SearchResult> AdaptiveHierarchicalIvf::search_with_options(
        const float* query, std::size_t k, const QuerySearchOptions& options,
        QuerySearchStats* stats, std::vector<std::size_t>* probed_leaf_ids) const {
    const std::size_t leaf_probe = config_.fixed_leaf_nprobe != 0
        ? config_.fixed_leaf_nprobe
        : std::max<std::size_t>(1, (leaves_.size() + config_.leaf_nprobe_ratio - 1) /
                                      config_.leaf_nprobe_ratio);
    return search_with_options(query, k, leaf_probe, options, stats, probed_leaf_ids);
}

std::vector<SearchResult> AdaptiveHierarchicalIvf::search_with_options(
        const float* query, std::size_t k, std::size_t leaf_probe,
        const QuerySearchOptions& options, QuerySearchStats* stats,
        std::vector<std::size_t>* probed_leaf_ids) const {
    if (!trained_) throw std::logic_error("train must be called before search");
    if (stats) *stats = {};
    if (probed_leaf_ids) probed_leaf_ids->clear();
    if (query == nullptr || k == 0) return {};
    const bool has_total_budget = options.max_total_distance_computations != 0;
    if (!has_total_budget && leaf_probe == 0)
        throw std::invalid_argument("nprobe must be positive");
    if (has_total_budget && options.max_total_distance_computations < leaves_.size())
        throw std::invalid_argument(
            "total distance budget must cover query-to-centroid comparisons");
    if (options.enable_early_stop &&
        (!std::isfinite(options.early_stop_fraction) || options.early_stop_fraction <= 0.0f ||
         options.early_stop_fraction > 1.0f))
        throw std::invalid_argument("early-stop fraction must be in (0, 1]");

    std::vector<std::pair<float, std::size_t>> coarse;
    coarse.reserve(leaves_.size());
    for (std::size_t id = 0; id < leaves_.size(); ++id) {
        const float* center = leaf_centroids_.data() + id * config_.dimension;
        coarse.emplace_back(options.maximum_inner_product
            ? simd_negative_dot(query, center, config_.dimension)
            : retrieval_distance(query, center), id);
    }
    const std::size_t wanted = has_total_budget
        ? coarse.size() : std::min(leaf_probe, coarse.size());
    auto centroid_order = [](const auto& a, const auto& b) {
        return a.first < b.first || (a.first == b.first && a.second < b.second);
    };
    if (has_total_budget)
        std::sort(coarse.begin(), coarse.end(), centroid_order);
    else
        std::partial_sort(coarse.begin(), coarse.begin() + wanted, coarse.end(), centroid_order);

    std::size_t available_vectors = 0;
    for (std::size_t c = 0; c < wanted; ++c)
        available_vectors += leaves_[coarse[c].second].ids.size();
    const std::size_t center_distance_computations = coarse.size();
    const std::size_t vector_scan_budget = has_total_budget
        ? std::min(options.max_total_distance_computations - center_distance_computations,
                   available_vectors)
        : available_vectors;
    const std::size_t scan_budget = has_total_budget
        ? center_distance_computations + vector_scan_budget
        : center_distance_computations + available_vectors;
    const std::size_t no_improvement_threshold = options.enable_early_stop
        ? std::max<std::size_t>(1, static_cast<std::size_t>(
              static_cast<double>(scan_budget) * options.early_stop_fraction))
        : 0;
    if (stats) {
        stats->scan_budget = scan_budget;
        stats->no_improvement_threshold = no_improvement_threshold;
        stats->center_distance_computations = center_distance_computations;
    }

    auto better = [](const SearchResult& a, const SearchResult& b) { return a.squared_l2 < b.squared_l2 || (a.squared_l2 == b.squared_l2 && a.id < b.id); };
    std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(better)> heap(better);
    std::vector<float> decoded_lvq4(config_.lvq4_storage ? config_.dimension : 0);
    std::size_t scanned_vectors = 0;
    std::size_t scanned_partitions = 0;
    std::size_t no_improvement = 0;
    bool early_stopped = false;
    for (std::size_t c = 0; c < wanted; ++c) {
        if (scanned_vectors >= vector_scan_budget) break;
        const std::size_t leaf_id = coarse[c].second;
        const Leaf& leaf = leaves_[leaf_id];
        if (probed_leaf_ids) probed_leaf_ids->push_back(leaf_id);
        ++scanned_partitions;
        const std::size_t take = std::min(leaf.ids.size(), vector_scan_budget - scanned_vectors);
        for (std::size_t i = 0; i < take; ++i) {
            float distance = 0.0f;
            if (config_.lvq8_storage) {
                distance = config_.angular_metric
                    ? simd_cosine_distance_lvq8(query,
                        leaf_centroids_.data() + leaf_id * config_.dimension,
                        leaf.lvq_codes.data() + i * config_.dimension,
                        leaf.lvq_lower[i], leaf.lvq_delta[i], config_.dimension)
                    : simd_squared_l2_lvq8(query,
                        leaf_centroids_.data() + leaf_id * config_.dimension,
                        leaf.lvq_codes.data() + i * config_.dimension,
                        leaf.lvq_lower[i], leaf.lvq_delta[i], config_.dimension);
            } else if (config_.lvq4_storage) {
                decode_vector(leaf_id, i, decoded_lvq4.data());
                distance = options.maximum_inner_product
                    ? simd_negative_dot(query, decoded_lvq4.data(), config_.dimension)
                    : retrieval_distance(query, decoded_lvq4.data());
            } else if (uses_u8_storage(config_)) {
                distance = options.maximum_inner_product
                    ? simd_negative_dot_f32_u8(query,
                        leaf.uint8_vectors.data() + i * config_.dimension,
                        config_.dimension)
                    : simd_squared_l2_f32_u8(query,
                        leaf.uint8_vectors.data() + i * config_.dimension,
                        config_.dimension);
            } else {
                const float* vector = leaf.vectors.data() + i * config_.dimension;
                distance = options.maximum_inner_product
                    ? simd_negative_dot(query, vector, config_.dimension)
                    : retrieval_distance(query, vector);
            }
            SearchResult hit{static_cast<std::int64_t>(leaf.ids[i]), distance};
            bool improved = false;
            if (heap.size() < k) { heap.push(hit); improved = true; }
            else if (better(hit, heap.top())) { heap.pop(); heap.push(hit); improved = true; }
            ++scanned_vectors;

            if (options.enable_early_stop && heap.size() == k) {
                no_improvement = improved ? 0 : no_improvement + 1;
                if (no_improvement >= no_improvement_threshold) {
                    early_stopped = true;
                    break;
                }
            }
        }
        if (early_stopped) break;
    }
    std::vector<SearchResult> result;
    while (!heap.empty()) { result.push_back(heap.top()); heap.pop(); }
    std::sort(result.begin(), result.end(), better);
    if (stats) {
        stats->scanned_vectors = scanned_vectors;
        stats->scanned_partitions = scanned_partitions;
        stats->total_distance_computations = center_distance_computations + scanned_vectors;
        stats->early_stopped = early_stopped;
    }
    return result;
}

std::vector<SearchResult> AdaptiveHierarchicalIvf::search_aps(
        const float* query, std::size_t k, float candidate_fraction,
        float recall_target, float recompute_threshold,
        std::vector<std::size_t>* probed_leaf_ids, ApsSearchStats* stats) const {
    if (!trained_) throw std::logic_error("train must be called before search");
    if (probed_leaf_ids) probed_leaf_ids->clear();
    if (stats) *stats = {};
    if (query == nullptr || k == 0 || leaves_.empty()) return {};
    if (config_.angular_metric)
        throw std::invalid_argument("Quake APS port currently supports Euclidean SIFT search only");
    candidate_fraction = std::clamp(candidate_fraction, 0.0f, 1.0f);
    recall_target = std::clamp(recall_target, 0.0f, 1.0f);

    const std::size_t candidate_count = std::min(leaves_.size(), std::max<std::size_t>(
        1, static_cast<std::size_t>(static_cast<double>(leaves_.size()) * candidate_fraction)));
    std::vector<std::pair<float, std::size_t>> coarse;
    coarse.reserve(leaves_.size());
    for (std::size_t id = 0; id < leaves_.size(); ++id)
        coarse.emplace_back(retrieval_distance(query, leaf_centroids_.data() + id * config_.dimension), id);
    std::partial_sort(coarse.begin(), coarse.begin() + candidate_count, coarse.end(),
        [](const auto& a, const auto& b) { return a.first < b.first; });
    coarse.resize(candidate_count);

    // Port of Quake's compute_boundary_distances: measure the query-to-
    // bisector plane distance between the nearest centroid and each candidate.
    std::vector<float> boundaries(candidate_count, 0.0f);
    const float* c0 = leaf_centroids_.data() + coarse[0].second * config_.dimension;
    const double norm0 = std::inner_product(c0, c0 + config_.dimension, c0, 0.0);
    std::vector<float> diff(config_.dimension);
    for (std::size_t p = 1; p < candidate_count; ++p) {
        const float* cj = leaf_centroids_.data() + coarse[p].second * config_.dimension;
        double normj = 0.0, dot_qv = 0.0, normv = 0.0;
        for (std::size_t d = 0; d < config_.dimension; ++d) {
            diff[d] = cj[d] - c0[d];
            normj += static_cast<double>(cj[d]) * cj[d];
            normv += static_cast<double>(diff[d]) * diff[d];
            dot_qv += static_cast<double>(query[d]) * diff[d];
        }
        const double b = 0.5 * (normj - norm0);
        boundaries[p] = normv > 0.0
            ? static_cast<float>(std::fabs(dot_qv - b) / std::sqrt(normv))
            : std::numeric_limits<float>::infinity();
    }

    auto better = [](const SearchResult& a, const SearchResult& b) {
        return a.squared_l2 < b.squared_l2 ||
               (a.squared_l2 == b.squared_l2 && a.id < b.id);
    };
    std::priority_queue<SearchResult, std::vector<SearchResult>, decltype(better)> heap(better);
    std::vector<float> decoded_lvq4(config_.lvq4_storage ? config_.dimension : 0);
    std::vector<float> partition_probs;
    float query_radius = std::numeric_limits<float>::max();
    float estimated_recall = 0.0f;
    std::size_t scanned = 0;
    for (std::size_t p = 0; p < candidate_count; ++p) {
        const std::size_t leaf_id = coarse[p].second;
        const Leaf& leaf = leaves_[leaf_id];
        for (std::size_t i = 0; i < leaf.ids.size(); ++i) {
            float distance = 0.0f;
            if (config_.lvq8_storage) {
                distance = simd_squared_l2_lvq8(query,
                    leaf_centroids_.data() + leaf_id * config_.dimension,
                    leaf.lvq_codes.data() + i * config_.dimension,
                    leaf.lvq_lower[i], leaf.lvq_delta[i], config_.dimension);
            } else if (config_.lvq4_storage) {
                decode_vector(leaf_id, i, decoded_lvq4.data());
                distance = retrieval_distance(query, decoded_lvq4.data());
            } else if (uses_u8_storage(config_)) {
                distance = simd_squared_l2_f32_u8(
                    query, leaf.uint8_vectors.data() + i * config_.dimension,
                    config_.dimension);
            } else {
                distance = retrieval_distance(query, leaf.vectors.data() + i * config_.dimension);
            }
            const SearchResult hit{static_cast<std::int64_t>(leaf.ids[i]), distance};
            if (heap.size() < k) heap.push(hit);
            else if (better(hit, heap.top())) { heap.pop(); heap.push(hit); }
        }
        ++scanned;
        if (probed_leaf_ids) probed_leaf_ids->push_back(leaf_id);

        // Quake's scan_list returns Euclidean (not squared) L2 distances.
        // The storage kernel returns squared L2, so convert its kth value here.
        const float kth_sq = heap.size() < k
            ? std::numeric_limits<float>::infinity() : heap.top().squared_l2;
        const float current_radius = std::sqrt(std::max(0.0f, kth_sq));
        const float percent_change = std::fabs(current_radius - query_radius) /
            std::max(current_radius, std::numeric_limits<float>::min());
        if (current_radius != 0.0f && (p == 0 || percent_change > recompute_threshold)) {
            query_radius = current_radius;
            partition_probs = aps_recall_profile(boundaries, query_radius, config_.dimension);
        }
        estimated_recall = 0.0f;
        for (std::size_t i = 0; i <= p && i < partition_probs.size(); ++i)
            estimated_recall += partition_probs[i];
        if (estimated_recall >= recall_target) break;
    }

    if (stats) {
        stats->candidate_partitions = candidate_count;
        stats->scanned_partitions = scanned;
        stats->estimated_recall = estimated_recall;
        stats->kth_distance = heap.size() < k
            ? std::numeric_limits<float>::infinity()
            : std::sqrt(std::max(0.0f, heap.top().squared_l2));
    }
    std::vector<SearchResult> result;
    while (!heap.empty()) { result.push_back(heap.top()); heap.pop(); }
    std::sort(result.begin(), result.end(), better);
    return result;
}

std::vector<PartitionHealth> AdaptiveHierarchicalIvf::partition_health() const {
    std::vector<PartitionHealth> result;
    result.reserve(leaves_.size());
    for (std::size_t i = 0; i < leaves_.size(); ++i) {
        const Leaf& leaf = leaves_[i];
        const float recent = current_mean_residual(leaf, leaf_centroids_.data() + i * config_.dimension);
        const float insert_drift = insert_residual_drift(leaf);
        const float global_fit = leaf.ref_mean <= 0.0f ? 0.0f :
            std::max(0.0f, (recent - leaf.ref_mean) / std::max(leaf.ref_mean, 1e-12f));
        const float shift = centroid_shift(leaf, leaf_centroids_.data() + i * config_.dimension);
        const float prd = partition_representation_drift(leaf, leaf_centroids_.data() + i * config_.dimension);
        const float tail = leaf.window_inserts == 0 ? 0.0f : static_cast<float>(leaf.window_tail_count) / leaf.window_inserts;
        result.push_back({i, leaf.ids.size(), leaf.window_inserts, leaf.window_deletes, leaf.ref_mean,
                          recent, insert_drift, global_fit, shift, prd, tail, is_marked(i)});
    }
    return result;
}

void AdaptiveHierarchicalIvf::reset_prd_references_for_diagnostic() {
    if (!trained_ || update_batch_open_)
        throw std::logic_error("diagnostic reference reset requires a trained index outside an update batch");
    for (std::size_t i = 0; i < leaves_.size(); ++i)
        reset_reference(leaves_[i], leaf_centroids_.data() + i * config_.dimension);
}

bool AdaptiveHierarchicalIvf::split_leaf_for_diagnostic(
        std::size_t leaf_id, std::uint32_t split_seed,
        std::size_t* created_leaf_id) {
    if (!trained_ || update_batch_open_)
        throw std::logic_error("diagnostic split requires a trained index outside an update batch");
    if (leaf_id >= leaves_.size())
        throw std::out_of_range("diagnostic split leaf id is out of range");
    if (leaves_[leaf_id].ids.size() < 2) return false;

    const std::size_t old_leaf_count = leaves_.size();
    const std::uint32_t old_seed = config_.seed;
    config_.seed = split_seed;
    try {
        // Use the production two-means split as-is, but report a degenerate
        // split as a failure instead of silently recentering the source leaf.
        split(leaf_id, false);
    } catch (...) {
        config_.seed = old_seed;
        throw;
    }
    config_.seed = old_seed;
    if (leaves_.size() == old_leaf_count) return false;
    if (created_leaf_id != nullptr) *created_leaf_id = old_leaf_count;
    return true;
}

MaintenanceReport AdaptiveHierarchicalIvf::locally_reassign_for_diagnostic(
        const std::vector<std::size_t>& seed_leaf_ids,
        std::size_t neighbor_count, std::size_t iterations) {
    if (!trained_ || update_batch_open_)
        throw std::logic_error("diagnostic local reassignment requires a trained index outside an update batch");
    MaintenanceReport report;
    report.leaf_nlist_before = leaves_.size();
    if (iterations > 0 && !seed_leaf_ids.empty())
        locally_reassign(seed_leaf_ids, neighbor_count, iterations, &report);
    report.leaf_nlist_after = leaves_.size();
    report.vector_count = active_count_;
    return report;
}

std::vector<std::pair<std::int64_t, std::size_t>> AdaptiveHierarchicalIvf::vector_leaf_assignments() const {
    std::vector<std::pair<std::int64_t, std::size_t>> result;
    result.reserve(active_count_);
    for (std::size_t leaf_id = 0; leaf_id < leaves_.size(); ++leaf_id) {
        for (const std::uint32_t id : leaves_[leaf_id].ids)
            result.emplace_back(static_cast<std::int64_t>(id), leaf_id);
    }
    return result;
}

std::vector<float> AdaptiveHierarchicalIvf::leaf_centroids() const { return leaf_centroids_; }

std::vector<std::vector<std::uint32_t>> AdaptiveHierarchicalIvf::leaf_vector_ids() const {
    std::vector<std::vector<std::uint32_t>> result;
    result.reserve(leaves_.size());
    for (const Leaf& leaf : leaves_) result.push_back(leaf.ids);
    return result;
}

std::vector<std::uint32_t> AdaptiveHierarchicalIvf::leaf_vector_ids(
        std::size_t leaf_id) const {
    if (leaf_id >= leaves_.size()) throw std::out_of_range("leaf id out of range");
    return leaves_[leaf_id].ids;
}

std::vector<std::size_t> AdaptiveHierarchicalIvf::quake_delete_partitions(
        const std::vector<std::size_t>& leaf_ids) {
    if (!trained_) throw std::logic_error("train must be called before quake_delete_partitions");
    const std::size_t old_count = leaves_.size();
    std::vector<std::size_t> selected = leaf_ids;
    std::sort(selected.begin(), selected.end());
    selected.erase(std::unique(selected.begin(), selected.end()), selected.end());

    std::vector<bool> selected_mask(old_count, false);
    for (const std::size_t leaf_id : selected)
        if (leaf_id < old_count && !leaves_[leaf_id].ids.empty()) selected_mask[leaf_id] = true;
    std::size_t surviving_nonempty = 0;
    for (std::size_t leaf_id = 0; leaf_id < old_count; ++leaf_id)
        if (!leaves_[leaf_id].ids.empty() && !selected_mask[leaf_id]) ++surviving_nonempty;
    // A malformed/degenerate decision must not empty the index: there would
    // be no surviving centroid to receive the vectors.
    if (surviving_nonempty == 0) std::fill(selected_mask.begin(), selected_mask.end(), false);

    struct ReassignedVector {
        std::uint32_t id;
        std::size_t destination;
        std::vector<float> values;
    };
    std::size_t moved_count = 0;
    for (std::size_t source = 0; source < old_count; ++source)
        if (selected_mask[source]) moved_count += leaves_[source].ids.size();
    std::vector<ReassignedVector> moved;
    moved.reserve(moved_count);
    std::vector<float> decoded(config_.dimension);
    for (std::size_t source = 0; source < old_count; ++source) {
        if (!selected_mask[source]) continue;
        const Leaf& leaf = leaves_[source];
        for (std::size_t position = 0; position < leaf.ids.size(); ++position) {
            decode_vector(source, position, decoded.data());
            std::size_t destination = old_count;
            float best_distance = std::numeric_limits<float>::max();
            for (std::size_t candidate = 0; candidate < old_count; ++candidate) {
                if (selected_mask[candidate]) continue;
                const float distance = retrieval_distance(
                    decoded.data(), leaf_centroids_.data() + candidate * config_.dimension);
                if (distance < best_distance) {
                    best_distance = distance;
                    destination = candidate;
                }
            }
            if (destination == old_count)
                throw std::logic_error("Quake partition deletion has no surviving destination");
            moved.push_back({leaf.ids[position], destination, decoded});
        }
    }

    // Upstream deletes the old lists, then re-adds their vectors through the
    // centroid index. Preserve IDs and total count, updating destination data
    // and the S/Q aggregates rather than treating a maintenance move as data loss.
    for (std::size_t source = 0; source < old_count; ++source) {
        if (!selected_mask[source]) continue;
        Leaf& leaf = leaves_[source];
        leaf.ids.clear();
        clear_vector_payload(leaf);
        leaf.vector_sum.assign(config_.dimension, 0.0);
        leaf.squared_norm_sum = 0.0;
        reset_window(leaf);
    }
    for (const ReassignedVector& item : moved) {
        Leaf& destination = leaves_[item.destination];
        destination.ids.push_back(item.id);
        store_vector(item.destination, destination, item.values.data());
        apply_vector_delta(destination, item.values.data(), 1.0);
    }

    std::vector<std::size_t> old_to_new;
    remove_empty_leaves(&old_to_new);
    if (old_to_new.size() != old_count)
        throw std::logic_error("Quake delete compaction map has an unexpected size");
    return old_to_new;
}

GlobalCenterRefreshReport AdaptiveHierarchicalIvf::refresh_all_centroids_to_means() {
    if (!trained_) throw std::logic_error("train must be called before refreshing centroids");
    GlobalCenterRefreshReport report;
    report.total_sse_before = total_sse();
    for (std::size_t leaf_id = 0; leaf_id < leaves_.size(); ++leaf_id) {
        Leaf& leaf = leaves_[leaf_id];
        if (leaf.ids.empty()) continue;
        if (leaf.vector_sum.size() != config_.dimension)
            throw std::logic_error("centroid refresh requires vector aggregates");
        float* centroid = leaf_centroids_.data() + leaf_id * config_.dimension;
        const double inverse = 1.0 / static_cast<double>(leaf.ids.size());
        for (std::size_t d = 0; d < config_.dimension; ++d)
            centroid[d] = static_cast<float>(leaf.vector_sum[d] * inverse);
        ++report.centers_updated;
        report.vectors_covered += leaf.ids.size();
    }
    report.total_sse_after = total_sse();
    return report;
}

void AdaptiveHierarchicalIvf::quake_refresh_centroids() {
    if (!trained_) throw std::logic_error("train must be called before quake_refresh_centroids");
    for (std::size_t leaf_id = 0; leaf_id < leaves_.size(); ++leaf_id) {
        const Leaf& leaf = leaves_[leaf_id];
        if (leaf.ids.empty()) continue;
        if (leaf.vector_sum.size() != config_.dimension)
            throw std::logic_error("Quake centroid refresh requires vector aggregates");
        float* centroid = leaf_centroids_.data() + leaf_id * config_.dimension;
        const double inverse = 1.0 / static_cast<double>(leaf.ids.size());
        for (std::size_t d = 0; d < config_.dimension; ++d)
            centroid[d] = static_cast<float>(leaf.vector_sum[d] * inverse);
    }
}

std::vector<std::size_t> AdaptiveHierarchicalIvf::leaf_sizes() const {
    std::vector<std::size_t> sizes;
    sizes.reserve(leaves_.size());
    for (const Leaf& leaf : leaves_) sizes.push_back(leaf.ids.size());
    return sizes;
}

double AdaptiveHierarchicalIvf::total_sse() const {
    double total = 0.0;
    for (std::size_t leaf_id = 0; leaf_id < leaves_.size(); ++leaf_id) {
        const Leaf& leaf = leaves_[leaf_id];
        if (leaf.ids.empty()) continue;
        if (leaf.vector_sum.size() != config_.dimension)
            throw std::logic_error("leaf vector aggregates are unavailable for SSE calculation");
        const float* center = leaf_centroids_.data() + leaf_id * config_.dimension;
        double dot = 0.0;
        double center_norm = 0.0;
        for (std::size_t d = 0; d < config_.dimension; ++d) {
            dot += leaf.vector_sum[d] * center[d];
            center_norm += static_cast<double>(center[d]) * center[d];
        }
        const double count = static_cast<double>(leaf.ids.size());
        const double leaf_sse = leaf.squared_norm_sum - 2.0 * dot + count * center_norm;
        total += std::max(0.0, leaf_sse);
    }
    return total;
}

double AdaptiveHierarchicalIvf::mean_sse_per_vector() const {
    return active_count_ == 0 ? 0.0 : total_sse() / static_cast<double>(active_count_);
}

std::size_t AdaptiveHierarchicalIvf::size() const noexcept { return active_count_; }
std::size_t AdaptiveHierarchicalIvf::leaf_nlist() const noexcept { return leaves_.size(); }
std::size_t AdaptiveHierarchicalIvf::top_nlist() const noexcept { return 0; }
std::size_t AdaptiveHierarchicalIvf::lvq_code_bytes() const noexcept {
    std::size_t bytes = 0;
    for (const auto& leaf : leaves_) bytes += leaf.lvq_codes.size();
    return bytes;
}
const AdaptiveHierarchicalConfig& AdaptiveHierarchicalIvf::config() const noexcept { return config_; }
void AdaptiveHierarchicalIvf::set_split_local_refinement(
        std::size_t radius, std::size_t iterations) {
    if (update_batch_open_)
        throw std::logic_error("cannot change split refinement during an update batch");
    config_.split_local_refinement_radius = radius;
    config_.split_local_refinement_iterations = std::max<std::size_t>(1, iterations);
}
void AdaptiveHierarchicalIvf::set_prd_threshold(float threshold) {
    if (update_batch_open_)
        throw std::logic_error("cannot change PRD threshold during an update batch");
    if (!std::isfinite(threshold) || threshold < 0.0f)
        throw std::invalid_argument("PRD threshold must be finite and non-negative");
    config_.prd_threshold = threshold;
}
void AdaptiveHierarchicalIvf::set_short_list_sqrt_n_fraction(float fraction) {
    if (update_batch_open_)
        throw std::logic_error("cannot change short-list merge threshold during an update batch");
    if (!std::isfinite(fraction) || fraction < 0.0f)
        throw std::invalid_argument("short-list sqrt-N fraction must be finite and non-negative");
    config_.short_list_sqrt_n_fraction = fraction;
}
void AdaptiveHierarchicalIvf::set_short_list_n_fraction(float fraction) {
    if (update_batch_open_)
        throw std::logic_error("cannot change short-list merge threshold during an update batch");
    if (!std::isfinite(fraction) || fraction < 0.0f)
        throw std::invalid_argument("short-list N fraction must be finite and non-negative");
    config_.short_list_n_fraction = fraction;
}
void AdaptiveHierarchicalIvf::set_capacity_sqrt_n_split_multiplier(float multiplier) {
    if (!std::isfinite(multiplier) || multiplier < 0.0f)
        throw std::invalid_argument("capacity sqrt-N split multiplier must be finite and non-negative");
    config_.capacity_sqrt_n_split_multiplier = multiplier;
}
void AdaptiveHierarchicalIvf::set_capacity_n_divisor(float divisor) {
    if (update_batch_open_)
        throw std::logic_error("cannot change capacity threshold during an update batch");
    if (!std::isfinite(divisor) || divisor < 0.0f)
        throw std::invalid_argument("capacity N divisor must be finite and non-negative");
    config_.capacity_n_divisor = divisor;
}
void AdaptiveHierarchicalIvf::set_capacity_cluster_count_gate(std::size_t max_live_clusters) {
    if (update_batch_open_)
        throw std::logic_error("cannot change capacity cluster-count gate during an update batch");
    config_.capacity_cluster_count_gate = max_live_clusters;
}
void AdaptiveHierarchicalIvf::set_automatic_maintenance(bool enabled) {
    if (update_batch_open_)
        throw std::logic_error("cannot change automatic maintenance during an update batch");
    config_.automatic_maintenance = enabled;
}
bool AdaptiveHierarchicalIvf::maintenance_due() const noexcept {
    return changes_since_maintenance_ != 0;
}
std::size_t AdaptiveHierarchicalIvf::next_maintenance_budget() const noexcept { return next_maintenance_budget_; }

void AdaptiveHierarchicalIvf::begin_update_batch() {
    if (!trained_) throw std::logic_error("train must be called before begin_update_batch");
    if (update_batch_open_) throw std::logic_error("an update batch is already open");
    update_batch_open_ = true;
}

MaintenanceReport AdaptiveHierarchicalIvf::end_update_batch() {
    if (!update_batch_open_) throw std::logic_error("no update batch is open");
    update_batch_open_ = false;
    // A batch only defers *automatic* maintenance.  Callers that disable it
    // must be able to inspect the post-update, pre-maintenance state or apply
    // an explicitly selected maintenance policy (e.g. an offline oracle).
    if (config_.automatic_maintenance && maintenance_due()) return maintain();
    MaintenanceReport report;
    report.leaf_nlist_before = leaves_.size();
    report.leaf_nlist_after = leaves_.size();
    report.vector_count = active_count_;
    report.next_change_budget = next_maintenance_budget_;
    return report;
}
