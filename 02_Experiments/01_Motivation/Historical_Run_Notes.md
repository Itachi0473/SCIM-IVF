# 01_Motivation

Paper: Section III; Figures 1-2.

Residual degradation, geometry-only pure growth, and capacity/routing tradeoff.

## Protocol

Geometry-only scale control uses 10% (an explicit control, not the default). SIFT: 50K/64 to 1M; capacity tradeoff retains all measured cluster counts.

## Program entry points

Paths below are relative to `01_Source`; filenames are preserved. Targets and compile definitions are in `01_Source/CMakeLists.txt`.

desktop_simd_ivf/experiments/sift1m_fixed_nprobe4_full_ivf_vs_prd_runner.cpp; sift1m_insert_cost_recall_sweep.cpp; deep10m_insert_query_nlist_sweep.cpp

Original run configurations and structured batch/query logs are preserved in this folder. `.gz` files contain the original text without modification. Data paths in old manifests are provenance, not portable defaults; adapt paths before launching a new run.
