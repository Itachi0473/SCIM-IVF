# 08_Parameter_Sensitivity

Paper: Section V-H; Figure 7; Table XI.

Geometry-only threshold sensitivity and configurable capacity targets.

## Protocol

Default full Triad threshold=7.5%. Geometry-only sweep spans 5%-20%; no 10% default recommendation. SIFT sweep starts at 50K/128 and reaches 1M. MSTuring fixed targets 1000/5000 scale capacity and removal thresholds with N/K_target; matched-K rebuild comparisons.

## Program entry points

Paths below are relative to `01_Source`; filenames are preserved. Targets and compile definitions are in `01_Source/CMakeLists.txt`.

desktop_simd_ivf/experiments/sift1m_original_order_capacity_ablation_runner.cpp; msturing_ih_runner.cpp; msturing_faiss_static_ivf.py

Original run configurations and structured batch/query logs are preserved in this folder. `.gz` files contain the original text without modification. Data paths in old manifests are provenance, not portable defaults; adapt paths before launching a new run.
