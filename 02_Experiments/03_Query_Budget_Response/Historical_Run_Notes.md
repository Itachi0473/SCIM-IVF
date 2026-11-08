# 03_Query_Budget_Response

Paper: Section V-C; Table IV.

Deep10M maintenance response at two budgets.

## Protocol

N/100 records are the accepted growth runs; N/20 records here. The Quake scan-cost calibrations also differ, as stated in the paper.

## Program entry points

Paths below are relative to `01_Source`; filenames are preserved. Targets and compile definitions are in `01_Source/CMakeLists.txt`.

desktop_simd_ivf/experiments/msturing_ih_runner.cpp

Original run configurations and structured batch/query logs are preserved in this folder. `.gz` files contain the original text without modification. Data paths in old manifests are provenance, not portable defaults; adapt paths before launching a new run.
