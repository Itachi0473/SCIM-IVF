# 07_Local_Reassignment

Paper: Section V-G; Table X.

Full versus no-local-reassignment at the same final K.

## Protocol

Independent N/20 ablation setting, 500K final distance budget; full/no reassignment differ only in that switch. MSTuring and Deep results are one run per configuration. Keep split partitioning, short-partition routing and center refresh.

## Program entry points

Paths below are relative to `01_Source`; filenames are preserved. Targets and compile definitions are in `01_Source/CMakeLists.txt`.

desktop_simd_ivf/experiments/msturing_ih_runner.cpp

Original run configurations and structured batch/query logs are preserved in this folder. `.gz` files contain the original text without modification. Data paths in old manifests are provenance, not portable defaults; adapt paths before launching a new run.
