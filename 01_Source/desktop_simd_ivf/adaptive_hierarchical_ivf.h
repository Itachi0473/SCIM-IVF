#pragma once

#include "ivf_flat.h"

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

// A single-routing-level IVF-Flat index designed for incremental data drift
// experiments. Leaf centroids route queries directly; maintenance tracks PRD.
// The production default uses only the current whole-partition fit relative
// to its reference fit. Composite and insert-only criteria remain available
// solely for reproducing earlier ablations.
enum class DriftCriterion { CompositePrd, GlobalFitOnly, InsertOnly };

struct AdaptiveHierarchicalConfig {
    std::size_t dimension = 128;
    std::size_t initial_leaf_nlist = 1000;
    // Legacy parameters retained for source/config compatibility. Direct leaf
    // routing does not use these two-level routing settings.
    std::size_t top_nlist = 100;
    std::size_t top_nprobe = 5;
    std::size_t leaf_nprobe_ratio = 50;  // effective leaf nprobe = ceil(K / ratio)
    // Optional fixed probe count. Zero retains the adaptive ratio behavior.
    std::size_t fixed_leaf_nprobe = 0;
    std::size_t kmeans_iterations = 10;
    // Runtime-only worker count for uint8 build/assignment paths. One keeps
    // historical single-thread behavior; this is intentionally not serialized.
    std::size_t worker_threads = 1;
    // Legacy scheduling field retained for saved-index compatibility. Automatic
    // maintenance now runs after each completed update batch with changes.
    float maintenance_fraction = 0.0f;
    bool automatic_maintenance = false;
    // Legacy no-op retained for source/saved-index compatibility. PRD evaluates
    // every non-empty partition at each maintenance round.
    float min_partition_change_fraction = 0.0f;
    std::size_t min_list_size_for_split = 128;
    std::size_t local_neighbor_count = 8;
    float prd_threshold = 0.05f;
    DriftCriterion drift_criterion = DriftCriterion::GlobalFitOnly;
    float tail_fraction_threshold = 0.20f;
    // Deprecated compatibility field. Automatic PRD repairs now split marked
    // partitions directly and do not choose recentering by estimated gain.
    float recenter_gain_threshold = 0.10f;
    // Short leaves below this fraction may merge. After merging, PRD
    // maintenance first splits any leaf larger than load_split_fraction*T
    // for load balancing; the same factor limits merge destinations.
    float short_list_fraction = 0.20f;
    // Optional experimental repair: inspect the largest residual tail in each
    // leaf after ordinary maintenance and move only vectors whose closest
    // current centroid is no longer their assigned leaf. Disabled by default.
    float residual_tail_reassign_fraction = 0.0f;
    // Restrict residual-tail repair to leaves involved in a successful split
    // or merge during the current maintenance round. Disabled by default.
    bool residual_tail_reassign_selected_only = false;
    // Optional Quake-style local reassignment after successful splits.
    // The radius is the number of neighboring leaves in addition to each
    // newly-created child seed; zero disables this experiment.
    std::size_t split_local_refinement_radius = 0;
    std::size_t split_local_refinement_iterations = 1;
    float load_split_fraction = 2.20f;
    // PRD maintenance also splits a leaf once it grows beyond this multiple
    // of its capacity baseline. A merge adds the two source baselines; a
    // build/recenter/split establishes a fresh baseline. 3.0 means strictly
    // more than 300% of the reference size.
    float capacity_growth_split_factor = 3.0f;
    // Optional absolute growth-capacity trigger. When positive, capacity
    // splitting uses multiplier*sqrt(active_count) while live leaf count is
    // at most sqrt(active_count), unless capacity_n_divisor selects the
    // corresponding N/divisor reference. The size comparison is strict (>).
    float capacity_sqrt_n_split_multiplier = 1.05f;
    // Optional linear-N replacement for the two sqrt-N capacity thresholds.
    // When positive, capacity splitting uses multiplier*(N/divisor), gated
    // while live leaf count is at most N/divisor unless an explicit cluster
    // count gate is configured. Zero preserves sqrt-N.
    float capacity_n_divisor = 0.0f;
    // Optional strict live-cluster-count gate for capacity splitting: active
    // only while K < this value. Zero preserves the N/divisor or sqrt(N) gate.
    std::size_t capacity_cluster_count_gate = 0;
    // Optional fixed load target T. Zero keeps the dynamic target N/K_live,
    // using the actual number of live partitions without a denominator cap.
    float load_target_size = 0.0f;
    // If enabled, recompute T=multiplier*sqrt(active vector count) each round.
    bool sqrt_load_target = false;
    float sqrt_load_target_multiplier = 1.0f;
    // Store leaf vectors as per-vector 8-bit LVQ residuals around that leaf's
    // centroid. Disabled by default to preserve the FP32 PRD-IVF baseline.
    bool lvq8_storage = false;
    // Store leaf vectors as packed 4-bit LVQ residuals around the leaf centroid.
    bool lvq4_storage = false;
    // Preserve exact unsigned 8-bit source vectors in the inverted lists.
    // Centroids and sufficient statistics remain floating point; this mode is
    // for uint8 datasets and is distinct from signed int8 or LVQ8.
    bool uint8_storage = false;
    // Optional short-list merge cutoff as a fraction of sqrt(active N).
    // Zero retains short_list_fraction * (N / K_live).
    float short_list_sqrt_n_fraction = 0.0f;
    // Optional short-list cutoff as a fraction of active N. Takes precedence
    // over short_list_sqrt_n_fraction when positive.
    float short_list_n_fraction = 0.0f;
    // Use cosine distance for centroid routing and candidate ranking while
    // retaining Euclidean residual statistics for PRD/LVQ maintenance.
    bool angular_metric = false;
    std::uint32_t seed = 1234;
};

struct PartitionHealth {
    std::size_t list_id = 0;
    std::size_t vector_count = 0;
    std::size_t window_inserts = 0;
    std::size_t window_deletes = 0;
    float reference_mean_residual = 0.0f;
    float recent_mean_residual = 0.0f;
    float insert_residual_drift = 0.0f;
    float global_fit_drift = 0.0f;
    float centroid_shift = 0.0f;
    float partition_representation_drift = 0.0f;
    float tail_fraction = 0.0f;
    bool marked = false;
};

struct MaintenanceReport {
    std::size_t checked_partitions = 0;
    std::size_t marked_partitions = 0;
    std::size_t recentered_partitions = 0;
    // Legacy count of successful splits issued while processing PRD-marked
    // partitions. Use the reason-specific counters below for attribution.
    std::size_t split_partitions = 0;
    // Exclusive split-reason counts from maintain()/maintain_load_only().
    // If multiple triggers apply, attribution priority is load, capacity, PRD.
    std::size_t prd_split_partitions = 0;
    // PRD-marked partitions that could not be split into two non-empty leaves.
    std::size_t prd_split_failures = 0;
    std::size_t total_split_partitions = 0;
    std::size_t load_split_partitions = 0;
    std::size_t capacity_split_partitions = 0;
    std::size_t merged_short_partitions = 0;
    std::size_t merge_recipient_partitions = 0;
    std::size_t merge_routed_vectors = 0;
    std::size_t removed_empty_partitions = 0;
    // Quake LIRE drops short partition objects but reassigns their vectors to
    // the nearest surviving partition; this count is therefore expected zero.
    std::size_t deleted_small_partitions = 0;
    std::size_t deleted_small_vectors = 0;
    std::size_t reassigned_small_partition_vectors = 0;
    std::size_t lvq8_reencoded_vectors = 0;
    std::size_t residual_tail_candidates = 0;
    std::size_t residual_tail_reassigned_vectors = 0;
    std::size_t residual_tail_selected_partitions = 0;
    std::size_t local_refinement_partitions = 0;
    std::size_t local_refinement_vectors = 0;
    std::size_t local_refinement_reassigned_vectors = 0;
    // Original leaf slots that successfully split in maintain_selected().
    // Query-aware ports use this to invalidate hit-window records for a
    // partition ID whose source implementation would retire and replace.
    std::vector<std::size_t> source_split_leaf_ids;
    std::size_t leaf_nlist_before = 0;
    std::size_t leaf_nlist_after = 0;
    std::size_t vector_count = 0;
    float load_target_T = 0.0f;
    // Legacy compatibility field; zero because maintenance is batch-triggered.
    std::size_t next_change_budget = 0;
    std::size_t selected_vectors = 0;
    double selected_sse_before = 0.0;
    double selected_sse_after = 0.0;
};

struct GlobalCenterRefreshReport {
    std::size_t centers_updated = 0;
    std::size_t vectors_covered = 0;
    double total_sse_before = 0.0;
    double total_sse_after = 0.0;
};

struct ApsSearchStats {
    std::size_t candidate_partitions = 0;
    std::size_t scanned_partitions = 0;
    float estimated_recall = 0.0f;
    float kth_distance = 0.0f;
};

struct QuerySearchOptions {
    // Query-time early stopping is opt-in; existing IVF/PRD search remains unchanged.
    bool enable_early_stop = false;
    // Stop after this fraction of the total distance-computation budget has
    // produced no Top-k improvement. Only vector comparisons count toward the
    // consecutive no-improvement run.
    // The default stops after 2/5 of the total distance budget's worth of
    // consecutive vector comparisons without a Top-k improvement.
    float early_stop_fraction = 0.40f;
    // Optional cap on the total number of distance computations, including the
    // query-to-centroid calculations. Zero preserves the regular nprobe behavior.
    // With a cap, all centroids are ranked, then nearest lists are scanned in
    // order until the remaining vector-comparison budget is exhausted.
    std::size_t max_total_distance_computations = 0;
    // Rank results by maximum query-vector inner product. Partition training
    // and maintenance remain governed by the configured IVF distance metric.
    bool maximum_inner_product = false;
};

struct QuerySearchStats {
    // Requested/effective total budget, including centroid and vector distances.
    std::size_t scan_budget = 0;
    std::size_t no_improvement_threshold = 0;
    std::size_t center_distance_computations = 0;
    std::size_t scanned_vectors = 0;
    std::size_t scanned_partitions = 0;
    std::size_t total_distance_computations = 0;
    bool early_stopped = false;
};

// Offline diagnostic actions.  They deliberately bypass PRD/load selection
// and operate only on caller-selected leaves, e.g. an oracle e_j ranking.
enum class SelectedMaintenanceAction { Recenter, Split };
enum class MaintenanceSelectionPolicy { Random, Longest, GlobalDrift };

class AdaptiveHierarchicalIvf {
public:
    explicit AdaptiveHierarchicalIvf(AdaptiveHierarchicalConfig config);

    // Persist the complete trained index, including vector data and PRD/load
    // maintenance state. The file can be loaded to continue querying/updates.
    void save(const std::string& path) const;
    static AdaptiveHierarchicalIvf load(const std::string& path);

    void train(const float* vectors, std::size_t count);
    void train_u8(const std::uint8_t* vectors, std::size_t count);
    void add(const float* vectors, std::size_t count);
    void add_u8(const std::uint8_t* vectors, std::size_t count);
    // Offline-only diagnostic primitive: appends vectors to one chosen leaf
    // without coarse routing.  It is for controlled drift validation, not for
    // the online index algorithm.
    void append_to_leaf_for_diagnostic(std::size_t leaf_id, const float* vectors, std::size_t count);
    // When exact_removed_vectors is supplied, it must contain one FP32 vector
    // per ID in the same order; this keeps PRD S/Q updates exact in LVQ mode.
    std::size_t remove_ids(const std::vector<std::int64_t>& ids,
                           const float* exact_removed_vectors = nullptr);
    MaintenanceReport maintain();
    // Load-only policy used for ablations: merge short lists, remove empty
    // lists, and split oversized lists without consulting the drift score.
    MaintenanceReport maintain_load_only();
    // Configurable Triad policy. Empty-leaf cleanup remains common; load
    // maintenance (short-list merge and oversized-list split), PRD splitting,
    // and capacity splitting can be enabled independently for ablations.
    MaintenanceReport maintain_with_policy(bool enable_prd_splits,
                                           bool enable_load_splits,
                                           bool enable_capacity_splits,
                                           bool enable_load_merges = true);
    // Reproduces Quake's size-threshold LIRE branch: delete partitions smaller
    // than min_partition_size and reassign their vectors to the nearest
    // surviving partition, split partitions larger than max_partition_size,
    // then locally reassign vectors among the nearest
    // refinement_radius partitions around each new child center. A refinement
    // iteration count of zero still performs one assignment pass, as in Quake.
    MaintenanceReport maintain_quake_lire_policy(
            std::size_t max_partition_size,
            std::size_t min_partition_size = 32,
            std::size_t refinement_radius = 50,
            std::size_t refinement_iterations = 0);
    MaintenanceReport maintain_selected(const std::vector<std::size_t>& leaf_ids,
                                        SelectedMaintenanceAction action,
                                        std::size_t refinement_radius = 0,
                                        std::size_t refinement_iterations = 3);
    // Selects a fixed fraction of non-empty leaves by policy, then applies the
    // same load-aware split/recenter action to every selected leaf.
    MaintenanceReport maintain_top_fraction(MaintenanceSelectionPolicy policy,
                                            float fraction, std::uint32_t seed);
    // Legacy compatibility accessor; always returns zero.
    std::size_t next_maintenance_budget() const noexcept;
    // Compatibility indicator: true when data changed since the last
    // maintenance; there is no global change-volume threshold anymore.
    bool maintenance_due() const noexcept;
    // Defers automatic maintenance until a complete update transaction (for
    // example, a sliding-window delete+add) has been applied. Without an
    // explicit batch, each public add/remove call is its own transaction.
    void begin_update_batch();
    MaintenanceReport end_update_batch();

    std::vector<SearchResult> search(const float* query, std::size_t k) const;
    // Optional output exposes the exact partitions visited by this search.
    // It is used by query-aware policies (e.g. Quake) to count hits without
    // recomputing query-to-centroid distances or enabling adaptive probing.
    std::vector<SearchResult> search(const float* query, std::size_t k,
                                     std::vector<std::size_t>* probed_leaf_ids) const;
    // Query-time probe override for benchmark sweeps; does not mutate the saved index.
    std::vector<SearchResult> search(const float* query, std::size_t k,
                                     std::size_t leaf_nprobe,
                                     std::vector<std::size_t>* probed_leaf_ids = nullptr) const;
    // Optional query-time early-stop path. By default it scans the same nprobe
    // lists as search(); a total-distance cap includes centroid comparisons and
    // expands probing in nearest-centroid order to use the remaining budget.
    std::vector<SearchResult> search_with_options(
        const float* query, std::size_t k, const QuerySearchOptions& options,
        QuerySearchStats* stats = nullptr,
        std::vector<std::size_t>* probed_leaf_ids = nullptr) const;
    std::vector<SearchResult> search_with_options(
        const float* query, std::size_t k, std::size_t leaf_nprobe,
        const QuerySearchOptions& options, QuerySearchStats* stats = nullptr,
        std::vector<std::size_t>* probed_leaf_ids = nullptr) const;
    // Quake-style Adaptive Partition Scanning (APS) using the official
    // hyperspherical-cap recall model and a nearest-centroid candidate prefix.
    std::vector<SearchResult> search_aps(const float* query, std::size_t k,
                                         float candidate_fraction,
                                         float recall_target,
                                         float recompute_threshold,
                                         std::vector<std::size_t>* probed_leaf_ids,
                                         ApsSearchStats* stats = nullptr) const;
    // Diagnostic accessor: decode one active vector from its stored payload.
    bool reconstruct_vector(std::int64_t id, float* output) const;
    std::vector<PartitionHealth> partition_health() const;
    // Returns the current physical ownership of every active vector.  This is
    // intended for offline diagnostics such as attributing a missed exact
    // neighbour to its unique inverted list; it is not used by search.
    std::vector<std::pair<std::int64_t, std::size_t>> vector_leaf_assignments() const;
    std::vector<float> leaf_centroids() const;
    std::vector<std::size_t> leaf_sizes() const;
    std::vector<std::vector<std::uint32_t>> leaf_vector_ids() const;
    std::vector<std::uint32_t> leaf_vector_ids(std::size_t leaf_id) const;
    // Quake's source maintenance may delete whole partitions and reassign
    // their vectors to the nearest surviving centroids. Empty leaves are
    // compacted; removed leaves map to std::numeric_limits<std::size_t>::max().
    std::vector<std::size_t> quake_delete_partitions(
        const std::vector<std::size_t>& leaf_ids);
    // Move every non-empty leaf centroid to the mean represented by its
    // maintained per-leaf aggregates. This does not scan/reassign vectors and
    // intentionally leaves PRD residual and capacity reference baselines intact.
    GlobalCenterRefreshReport refresh_all_centroids_to_means();
    // Apply Quake's lazy centroid refresh before maintenance decisions: each
    // centroid becomes the mean of its current assigned vectors.
    void quake_refresh_centroids();
    // Sum of squared L2 residuals from vectors to their currently assigned
    // leaf centroids, computed from the maintained per-leaf S and Q aggregates.
    double total_sse() const;
    double mean_sse_per_vector() const;

    std::size_t size() const noexcept;
    std::size_t leaf_nlist() const noexcept;
    std::size_t top_nlist() const noexcept;
    std::size_t lvq_code_bytes() const noexcept;
    const AdaptiveHierarchicalConfig& config() const noexcept;
    // Override split-local refinement between update batches. This supports
    // reusing a saved initial index while testing different refinement radii.
    void set_split_local_refinement(std::size_t radius, std::size_t iterations);
    // Override incremental-maintenance thresholds on a reused initial snapshot.
    void set_prd_threshold(float threshold);
    void set_short_list_sqrt_n_fraction(float fraction);
    void set_short_list_n_fraction(float fraction);
    void set_capacity_sqrt_n_split_multiplier(float multiplier);
    void set_capacity_n_divisor(float divisor);
    void set_capacity_cluster_count_gate(std::size_t max_live_clusters);
    void set_automatic_maintenance(bool enabled);
    // Offline control experiments only: establish a fresh PRD baseline after
    // structural preprocessing, without changing membership or centers.
    void reset_prd_references_for_diagnostic();
    // Offline selection-policy controls. These invoke the existing binary
    // split and local reassignment implementations without running automatic
    // maintenance. The split seed override is scoped to this one split.
    bool split_leaf_for_diagnostic(std::size_t leaf_id, std::uint32_t split_seed,
                                   std::size_t* created_leaf_id = nullptr);
    MaintenanceReport locally_reassign_for_diagnostic(
            const std::vector<std::size_t>& seed_leaf_ids,
            std::size_t neighbor_count, std::size_t iterations);

private:
    struct Leaf {
        // Experiment IDs fit in 32 bits; the public API remains int64_t.
        std::vector<std::uint32_t> ids;
        // Exactly one representation is active: FP32 vectors, exact uint8
        // vectors, or packed LVQ codes plus per-vector residual metadata.
        std::vector<float> vectors;
        std::vector<std::uint8_t> uint8_vectors;
        std::vector<std::uint8_t> lvq_codes;
        std::vector<float> lvq_lower;
        std::vector<float> lvq_delta;
        // S = sum(x), Q = sum(||x||^2).  Updated on every insert/delete so
        // whole-partition residuals and means need no full list scan.
        std::vector<double> vector_sum;
        double squared_norm_sum = 0.0;
        float ref_mean = 0.0f;
        float ref_p95 = 0.0f;
        std::size_t window_inserts = 0;
        std::size_t window_deletes = 0;
        std::size_t capacity_reference_size = 0;
        // Dinsert accumulator; GlobalFitOnly intentionally leaves it unused
        // and omits it from the compact v3 leaf record.
        double insert_residual_sum = 0.0;
        std::size_t window_tail_count = 0;
    };

    float squared_l2(const float* a, const float* b) const;
    float retrieval_distance(const float* a, const float* b) const;
    std::vector<std::size_t> nearest(const float* query, const std::vector<float>& centers,
                                     std::size_t center_count, std::size_t wanted) const;
    void train_kmeans(const float* vectors, std::size_t count, std::size_t clusters,
                      std::size_t iterations, std::vector<float>* centers,
                      const std::vector<float>* weights = nullptr) const;
    void train_kmeans_u8(const std::uint8_t* vectors, std::size_t count,
                         std::size_t clusters, std::size_t iterations,
                         std::vector<float>* centers) const;
    void assign_initial(const float* vectors, std::size_t count);
    void assign_initial_u8(const std::uint8_t* vectors, std::size_t count);
    void decode_vector(std::size_t leaf_id, std::size_t position, float* output) const;
    void store_vector(std::size_t leaf_id, Leaf& leaf, const float* vector);
    void store_vector_u8(Leaf& leaf, const std::uint8_t* vector);
    void clear_vector_payload(Leaf& leaf);
    void remove_vector_payload(Leaf& leaf, std::size_t position, std::size_t last);
    void reset_reference(Leaf& leaf, const float* centroid);
    bool capacity_growth_split_due(const Leaf& leaf) const noexcept;
    void reset_window(Leaf& leaf);
    void rebuild_aggregates(Leaf& leaf);
    void refresh_membership_state_preserve_residual_reference(Leaf& leaf);
    void apply_vector_delta(Leaf& leaf, const float* vector, double sign);
    float current_mean_residual(const Leaf& leaf, const float* centroid) const;
    float centroid_shift(const Leaf& leaf, const float* centroid) const;
    float insert_residual_drift(const Leaf& leaf) const;
    float partition_representation_drift(const Leaf& leaf, const float* centroid) const;
    float drift_score(const Leaf& leaf, const float* centroid) const;
    bool is_marked(std::size_t leaf_id) const;
    void recenter(std::size_t leaf_id);
    void split(std::size_t leaf_id, bool recenter_on_degenerate = true);
    void locally_reassign(const std::vector<std::size_t>& seeds,
                          std::size_t neighbor_count, std::size_t iterations,
                          MaintenanceReport* report = nullptr);
    void reassign_residual_tail(float fraction, MaintenanceReport& report,
                                const std::vector<std::size_t>* selected_leaves = nullptr);
    std::size_t remove_empty_leaves(std::vector<std::size_t>* old_to_new = nullptr);
    bool merge_short_leaf(std::size_t leaf_id, float short_cutoff,
                          std::vector<std::size_t>* merged_into = nullptr);
    MaintenanceReport maintain_impl(bool enable_prd_splits,
                                    bool enable_load_splits,
                                    bool enable_capacity_splits,
                                    bool enable_load_merges);
    std::vector<std::size_t> local_neighbours(std::size_t leaf_id,
                                              std::size_t neighbor_count) const;
    void record_insert(Leaf& leaf, const float* vector, const float* centroid);

    AdaptiveHierarchicalConfig config_;
    bool trained_ = false;
    std::int64_t next_id_ = 0;
    std::size_t active_count_ = 0;
    // Pending successful inserts/deletes. The threshold budget is a legacy
    // serialization slot and remains zero.
    std::size_t changes_since_maintenance_ = 0;
    std::size_t next_maintenance_budget_ = 0;
    std::size_t lvq8_reencoded_vectors_ = 0;
    bool update_batch_open_ = false;
    std::vector<float> leaf_centroids_;
    std::vector<Leaf> leaves_;
    // No upper routing layer: queries probe leaf_centroids_ directly.
};
