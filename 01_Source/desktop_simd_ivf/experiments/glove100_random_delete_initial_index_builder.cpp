#include "adaptive_hierarchical_ivf.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::size_t kRows = 1'183'514;
constexpr std::size_t kDim = 100;
constexpr std::size_t kNlist = 1'100;
constexpr std::uint32_t kSeed = 1234;
}

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "Usage: glove100_random_delete_initial_index_builder BASE.f32 SHARED_INDEX.ivfidx\n";
            return 2;
        }
        const std::filesystem::path base_path(argv[1]);
        const std::filesystem::path snapshot(argv[2]);
        if (std::filesystem::exists(snapshot))
            throw std::runtime_error("refusing to overwrite shared initial index: " + snapshot.string());
        if (std::filesystem::file_size(base_path) != kRows * kDim * sizeof(float))
            throw std::runtime_error("expected exactly 1,183,514 x 100 FP32 base vectors");

        std::ifstream in(base_path, std::ios::binary);
        if (!in) throw std::runtime_error("cannot open GloVe base: " + base_path.string());
        std::vector<float> base(kRows * kDim);
        in.read(reinterpret_cast<char*>(base.data()), static_cast<std::streamsize>(base.size() * sizeof(float)));
        if (!in) throw std::runtime_error("truncated GloVe base payload");

        AdaptiveHierarchicalConfig config;
        config.dimension = kDim;
        config.initial_leaf_nlist = kNlist;
        config.kmeans_iterations = 10;
        config.automatic_maintenance = false;
        config.uint8_storage = false;
        config.angular_metric = false; // normalized vectors give the same ranking under squared L2.
        config.prd_threshold = 0.075f;
        config.drift_criterion = DriftCriterion::GlobalFitOnly;
        config.short_list_sqrt_n_fraction = 0.20f;
        config.load_split_fraction = 2.20f;
        config.capacity_sqrt_n_split_multiplier = 1.05f;
        config.split_local_refinement_radius = 50;
        config.split_local_refinement_iterations = 1;
        config.seed = kSeed;

        AdaptiveHierarchicalIvf index(config);
        std::cout << "BUILD_BEGIN vectors=" << kRows << " dim=" << kDim << " nlist=" << kNlist
                  << " iterations=10 seed=" << kSeed << "\n" << std::flush;
        const auto start = std::chrono::steady_clock::now();
        index.train(base.data(), kRows);
        const double build_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
        if (index.size() != kRows || index.leaf_nlist() != kNlist)
            throw std::runtime_error("shared initial index validation failed");
        index.save(snapshot.string());

        const auto sizes = index.leaf_sizes();
        const auto [min_size, max_size] = std::minmax_element(sizes.begin(), sizes.end());
        std::ofstream manifest(snapshot.string() + ".manifest.txt", std::ios::trunc);
        if (!manifest) throw std::runtime_error("cannot write initial-index manifest");
        manifest << "dataset=GloVe100-angular ANN-Benchmarks\n"
                 << "base=" << std::filesystem::absolute(base_path).string() << "\n"
                 << "vectors=" << kRows << "\ndimension=" << kDim << "\n"
                 << "metric=squared L2 over independently normalized FP32 vectors\n"
                 << "initial_clusters=" << kNlist << "\nkmeans_iterations=10\nseed=" << kSeed << "\n"
                 << "build_ms=" << std::setprecision(12) << build_ms << "\n"
                 << "initial_min_posting=" << *min_size << "\ninitial_max_posting=" << *max_size << "\n"
                 << "storage=FP32 exact vectors\n";
        if (!manifest) throw std::runtime_error("failed writing initial-index manifest");
        std::cout << "BUILD_COMPLETE build_ms=" << std::fixed << std::setprecision(1) << build_ms
                  << " min_posting=" << *min_size << " max_posting=" << *max_size
                  << " snapshot=" << std::filesystem::absolute(snapshot).string() << "\n" << std::flush;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
