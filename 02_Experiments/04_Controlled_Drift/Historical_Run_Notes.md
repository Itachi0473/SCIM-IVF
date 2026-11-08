# 04_Controlled_Drift

Paper: Section V-D; Tables V-VI; Figure 5.

Nine scenarios, four online methods; the manuscript figure has no geometry ablation curve.

## Protocol

500K/512 plus 100 x 5K inserts, 1000 stage-matched queries, N/100 total budget, 16 query workers. Curves show stages 0-100; table means use 1-100. Triad geometry=7.5%, radius=50; LIRE radius=64/max posting=1000.

## Program entry points

Paths below are relative to `01_Source`; filenames are preserved. Targets and compile definitions are in `01_Source/CMakeLists.txt`.

tools/candor_prd/make_deep1m_freewayml_shift.py; compute_deep1m_freewayml_truth.py; desktop_simd_ivf/experiments/msturing_ih_runner.cpp; deep1m_quake_freewayml_runner.cpp; deep1m_spfresh_lire_freewayml_runner.cpp; deep1m_ivf_flat_freewayml_runner.cpp

Original run configurations and structured batch/query logs are preserved in this folder. `.gz` files contain the original text without modification. Data paths in old manifests are provenance, not portable defaults; adapt paths before launching a new run.
