# 02_Continuous_Growth

Paper: Section V-C; Table III; Figure 4.

Four online methods on YFCC10M, Deep10M, MSTuring10M, paired update repeats and matched-K rebuild references.

## Protocol

1M/1000 to 10M; 900 x 10K insertions; 100 checkpoints; 10K queries. YFCC/Deep N/100, MSTuring N/20. LIRE max posting=3200; radius=50. Triad/YFCC has 4 cost runs, every other configuration 3. Update=insert+maintenance; means and sample SD from paired run totals. Query latency uses individual search-call durations under 16 workers, not batch wall time/query count.

## Program entry points

Paths below are relative to `01_Source`; filenames are preserved. Targets and compile definitions are in `01_Source/CMakeLists.txt`.

desktop_simd_ivf/experiments/msturing_ih_runner.cpp; ivf_flat_ih_staged_runner.cpp; rebuild_reference_sample10pct_all.py

Original run configurations and structured batch/query logs are preserved in this folder. `.gz` files contain the original text without modification. Data paths in old manifests are provenance, not portable defaults; adapt paths before launching a new run.
