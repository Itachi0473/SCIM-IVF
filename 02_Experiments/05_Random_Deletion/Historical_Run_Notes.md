# 05_Random_Deletion

Paper: Section V-E; Table VII.

SIFT1M and GloVe100 pure random deletion, metrics, deletion IDs and ground-truth CSV records.

## Protocol

40 x 20K deletes; 1000 queries; live N/20 budget; 32 warm-up queries excluded; Quake 50 ns. No monotonic global cluster-shrink claim.

## Program entry points

Paths below are relative to `01_Source`; filenames are preserved. Targets and compile definitions are in `01_Source/CMakeLists.txt`.

desktop_simd_ivf/experiments/sift1m_random_delete_full_runner.cpp; sift1m_random_delete_lire_quake_runner.cpp; sift1m_random_delete_spfresh_lire_runner.cpp; prepare_sift1m_random_delete_truth.cpp; prepare_glove100_random_delete_truth.cpp

Original run configurations and structured batch/query logs are preserved in this folder. `.gz` files contain the original text without modification. Data paths in old manifests are provenance, not portable defaults; adapt paths before launching a new run.
