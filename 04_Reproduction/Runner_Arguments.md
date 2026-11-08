# Runner arguments extracted from the supplied source

## deep10m_insert_query_nlist_sweep.cpp
```text
Line 543:
            throw std::runtime_error(
                "usage: runner BASE.fbin QUERIES.fbin GROUNDTRUTH.ibin OUTPUT_DIR [THREADS [all|query]]");
        const std::size_t workers = argc >= 6
            ? static_cast<std::size_t>(std::stoul(argv[5]))
            : std::min<std::size_t>(12, std::max(1u, std::thread::hardware_concurrency()));
        const std::string mode = argc == 7 ? argv[6] : "all";
        if (workers == 0) throw std::runtime_error("THREADS must be positive");
        if (mode != "all" && mode != "query")
            throw std::runtime_error("MODE must be 'all' or 'query'");

```

## deep1m_current_lire_freewayml_runner.cpp
```text
Line 390:
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
```

## deep1m_dedrift_freewayml_runner.cpp
```text
Line 278:
    if(argc==2&&std::string(argv[1])=="--self-test"){dedrift_self_test();return 0;}
    if(argc<5||argc>11){std::cerr<<"Usage: deep1m_dedrift_freewayml_runner DATA_ROOT INITIAL_SNAPSHOT TRUTH_ROOT OUTPUT [k=4] [threads=16] [budget_divisor=100] [case=all] [stages=100] [queries=1000]\n";return 2;}
    std::filesystem::path data_root=argv[1],snapshot_path=argv[2],truth_root=argv[3],output=argv[4];
    std::size_t k=argc>5?std::stoull(argv[5]):4,threads=argc>6?std::stoull(argv[6]):16,divisor=argc>7?std::stoull(argv[7]):100,max_stage=argc>9?std::stoull(argv[9]):100,nq=argc>10?std::stoull(argv[10]):1000;std::string selected=argc>8?argv[8]:"all";
    if(!k||k>=kInitialNlist||!threads||!divisor||max_stage>kStages||!nq||nq>kQueries)throw std::invalid_argument("invalid experiment parameters");
    std::vector<std::string> cases;for(auto name:kCases)if(selected=="all"||selected==name)cases.emplace_back(name);if(cases.empty())throw std::invalid_argument("unknown scenario");
    if(std::filesystem::is_directory(truth_root/"_prepared_truth"))truth_root/="_prepared_truth";
    if(std::filesystem::exists(output)&&(!std::filesystem::is_directory(output)||!std::filesystem::is_empty(output)))throw std::runtime_error("output must be new or empty");std::filesystem::create_directories(output);
    auto snapshot=AdaptiveHierarchicalIvf::load(snapshot_path.string());const auto& cfg=snapshot.config();if(snapshot.size()!=kInitialRows||snapshot.leaf_nlist()!=kInitialNlist||cfg.dimension!=kDimension||cfg.uint8_storage||cfg.lvq4_storage||cfg.lvq8_storage||cfg.angular_metric)throw std::runtime_error("invalid shared initial DEEP-1M snapshot");
    auto centers=snapshot.leaf_centroids();auto members=snapshot.leaf_vector_ids();MappedFbin reference(data_root/"G/base.fbin");
```

## deep1m_ivf_flat_freewayml_runner.cpp
```text
Line 296:
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
```

## deep1m_quake_freewayml_runner.cpp
```text
Line 420:
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
```

## deep1m_size_preserving_prd_control.cpp
```text
Line 431:
        if (argc < 4 || argc > 7) {
            std::cerr << "Usage: deep1m_size_preserving_prd_control BASE_10M.fbin QUERY_10K.fbin OUTPUT_DIR [COMMON_START_INDEX] [TARGET_COUNTS_CSV] [BUDGET_DIVISORS_CSV]\n";
            return 2;
        }
        const std::string base_path = argv[1];
        const std::string query_path = argv[2];
        const std::string output_dir = argv[3];
        const std::string common_start_path = argc >= 5 ? argv[4] : "";
        const std::vector<std::size_t> target_counts = argc >= 6
            ? parse_positive_list(argv[5], "target count") : std::vector<std::size_t>{32, 128};
```

## deep1m_spfresh_lire_freewayml_runner.cpp
```text
Line 633:
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
```

## glove100_random_delete_initial_index_builder.cpp
```text
Line 24:
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
```

## ivf_flat_ih_staged_runner.cpp
```text
Line 572:
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
Line 584:
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
```

## msturing_ih_runner.cpp
```text
Line 6252:
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
Line 6260:
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
Line 6272:
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
Line 6273:
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
Line 6274:
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
Line 6388:
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
Line 6413:
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
Line 6425:
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
Line 6426:
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
Line 6432:
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
Line 6435:
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
Line 6448:
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
Line 6449:
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
Line 6450:
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
Line 6466:
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
Line 6538:
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
Line 6614:
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
Line 6622:
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
Line 6644:
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
Line 6654:
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
Line 6665:
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
Line 6684:
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
Line 6704:
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
Line 6750:
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
Line 6875:
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
Line 6926:
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
Line 6980:
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
Line 7021:
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
Line 7030:
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
Line 7041:
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
Line 7061:
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
Line 7102:
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
Line 7156:
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
Line 7206:
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
Line 7267:
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
Line 7452:
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
Line 7457:
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
Line 7520:
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
Line 7561:
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
Line 7602:
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
Line 7659:
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
Line 7712:
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
Line 7776:
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
Line 7838:
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
Line 7839:
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
Line 7841:
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
Line 7849:
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
Line 7856:
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
Line 7900:
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
Line 7942:
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
Line 7957:
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
Line 7962:
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
Line 7969:
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
Line 7976:
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
Line 7983:
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
Line 7994:
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
Line 8001:
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
Line 8010:
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
Line 8019:
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
Line 8026:
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
Line 8030:
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
Line 8040:
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
Line 8049:
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
Line 8054:
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
Line 8059:
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
Line 8067:
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
Line 8073:
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
Line 8078:
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
Line 8231:
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
Line 8312:
        }
        if (argc == 5 && std::string(argv[1]) == "--deep1m-freewayml-component-ablation-final-aps") {
            const std::filesystem::path data_root(argv[2]);
            const std::filesystem::path shared_initial_index(argv[3]);
            const std::filesystem::path root(argv[4]);
            ensure_empty_or_create(root);
            if (!std::filesystem::is_regular_file(shared_initial_index))
                throw std::runtime_error("shared initial DEEP-1M index does not exist: " + shared_initial_index.string());

            constexpr std::array<const char*, 9> case_names{
Line 8480:
        }
        if (argc == 5 && std::string(argv[1]) == "--deep1m-freewayml-component-ablation-checkpoints-aps") {
            const std::filesystem::path data_root(argv[2]);
            const std::filesystem::path shared_initial_index(argv[3]);
            const std::filesystem::path root(argv[4]);
            ensure_empty_or_create(root);
            if (!std::filesystem::is_regular_file(shared_initial_index))
                throw std::runtime_error("shared initial DEEP-1M index does not exist: " + shared_initial_index.string());

            constexpr std::array<const char*, 4> case_names{"G", "B1", "B3", "C2"};
Line 8593:
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
Line 8718:
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
Line 8818:
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
```

## sift1m_fixed_nprobe4_full_ivf_vs_prd_runner.cpp
```text
Line 258:
        if (argc != 5) {
            std::cerr << "Usage: sift1m_fixed_nprobe4_full_ivf_vs_prd_runner BASE.u8bin QUERIES.fvecs TRUTH.ivecs OUTPUT_DIR\n";
            return 2;
        }
        const std::filesystem::path base_path(argv[1]);
        const std::filesystem::path query_path(argv[2]);
        const std::filesystem::path truth_path(argv[3]);
        const std::filesystem::path output_dir(argv[4]);
        if (std::filesystem::exists(output_dir) &&
            std::filesystem::directory_iterator(output_dir) != std::filesystem::directory_iterator())
```

## sift1m_insert_cost_recall_sweep.cpp
```text
Line 501:
            throw std::runtime_error(
                "usage: runner BASE.f32 QUERIES.fvecs GROUNDTRUTH.ivecs OUTPUT_DIR [THREADS [all|query]]");
        const std::size_t workers = argc >= 6
            ? static_cast<std::size_t>(std::stoul(argv[5]))
            : std::min<std::size_t>(12, std::max(1u, std::thread::hardware_concurrency()));
        const std::string mode = argc == 7 ? argv[6] : "all";
        if (workers == 0) throw std::runtime_error("THREADS must be positive");
        if (mode != "all" && mode != "query")
            throw std::runtime_error("MODE must be 'all' or 'query'");

```

## sift1m_original_order_capacity_ablation_runner.cpp
```text
Line 254:
        if (argc < 5 || argc > 9) {
            std::cerr << "Usage: sift1m_original_order_capacity_ablation_runner BASE.u8bin QUERIES.fvecs "
                         "TRUTH.ivecs OUTPUT_DIR [PRD_THRESHOLD] "
                         "[all|full|prd-only|no-capacity|no-prd|no-load-balance] [DATASET_LABEL] "
                         "[INITIAL_NLIST]\n";
            return 2;
        }
        const float prd_threshold = argc >= 6 ? std::stof(argv[5]) : 0.05f;
        const std::string run_mode = argc >= 7 ? argv[6] : "all";
        const std::string dataset_label = argc >= 8 ? argv[7] : "SIFT1M in original row order";
```

## sift1m_pca_module_ablation_sqrt_merge_runner.cpp
```text
Line 225:
        if (argc != 5) {
            std::cerr << "Usage: sift1m_pca_module_ablation_sqrt_merge_runner BASE.f32 QUERIES.fvecs "
                         "FINAL_TOP10.ivecs OUTPUT_DIR\n";
            return 2;
        }
        const std::string output_dir = argv[4];
        if (std::filesystem::exists(output_dir) &&
            std::filesystem::directory_iterator(output_dir) != std::filesystem::directory_iterator())
            throw std::runtime_error("output directory is not empty: " + output_dir);
        std::filesystem::create_directories(output_dir);
```

## sift1m_random_delete_full_runner.cpp
```text
Line 222:
        if (argc != 5 && argc != 6) {
            std::cerr << "Usage: sift1m_random_delete_full_runner BASE.u8bin QUERIES.fvecs "
                         "PREPARED_DATA_DIR OUTPUT_DIR [SHARED_INITIAL_INDEX.ivfidx]\n";
            return 2;
        }
        const std::filesystem::path base_path(argv[1]);
        const std::filesystem::path query_path(argv[2]);
        const std::filesystem::path prepared_dir(argv[3]);
        const std::filesystem::path output(argv[4]);
        const std::filesystem::path shared_index_path = argc == 6 ? std::filesystem::path(argv[5]) : std::filesystem::path();
```

## sift1m_random_delete_lire_quake_runner.cpp
```text
Line 502:
        if (argc < 5 || argc > 7) {
            std::cerr << "Usage: sift1m_random_delete_lire_quake_runner BASE.u8bin QUERIES.fvecs PREPARED_DIR OUTPUT_DIR [SHARED_INITIAL_INDEX] [both|lire|quake]\n";
            return 2;
        }
        const std::filesystem::path base_path(argv[1]), query_path(argv[2]), prepared(argv[3]), output(argv[4]);
        const std::filesystem::path supplied_snapshot = argc >= 6 ? std::filesystem::path(argv[5]) : std::filesystem::path();
        const std::string methods = argc >= 7 ? argv[6] : "both";
        if (methods != "both" && methods != "lire" && methods != "quake")
            throw std::invalid_argument("method selector must be both, lire, or quake");
        if (std::filesystem::exists(output) && std::filesystem::directory_iterator(output) != std::filesystem::directory_iterator())
```

## sift1m_random_delete_spfresh_lire_runner.cpp
```text
Line 410:
        if (argc != 6) {
            std::cerr << "Usage: sift1m_random_delete_spfresh_lire_runner BASE.u8bin QUERIES.fvecs PREPARED_DIR SHARED_INDEX.ivfidx OUTPUT_DIR\n";
            return 2;
        }
        const std::filesystem::path base_path(argv[1]), query_path(argv[2]), prepared(argv[3]), snapshot(argv[4]), output(argv[5]);
        if (!std::filesystem::exists(snapshot)) throw std::runtime_error("shared initial snapshot does not exist");
        if (std::filesystem::exists(output) && std::filesystem::directory_iterator(output) != std::filesystem::directory_iterator())
            throw std::runtime_error("output must be absent or empty: " + output.string());
        std::filesystem::create_directories(output);
        auto base = read_base(base_path);
```