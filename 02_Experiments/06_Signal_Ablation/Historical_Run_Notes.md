# 06_Signal_Ablation

Paper: Section V-F; Tables VIII-IX; Figure 6.

Three-signal ablation under two SIFT orders, plus table-only geometry ablation in B1-B4/C1-C2.

## Protocol

SIFT: 50K/64 plus 19 x 50K; first 1000 queries; N/20. Disabling load disables both overload splitting and short-partition removal. Geometry-drift table: mean R@10 of stages 1-100, all settings matched except geometry. Six database-changing cases selected at author request. All nine raw ablation cases are retained for audit; G/A1/A2 are not presented in the focused table. Existing SIFT trajectory figure retained; no additional geometry-ablation figure.

## Program entry points

Paths below are relative to `01_Source`; filenames are preserved. Targets and compile definitions are in `01_Source/CMakeLists.txt`.

desktop_simd_ivf/experiments/sift1m_pca_module_ablation_sqrt_merge_runner.cpp; msturing_ih_runner.cpp

Original run configurations and structured batch/query logs are preserved in this folder. `.gz` files contain the original text without modification. Data paths in old manifests are provenance, not portable defaults; adapt paths before launching a new run.
