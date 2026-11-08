# Reproduction status

The supplied source, recorded experiment settings, input preparation scripts and result records provide the materials needed to rerun the experiments reported in the manuscript. Raw public datasets are external inputs; acquisition and preprocessing requirements are documented in `Datasets.md` and per-experiment settings.

The archive includes the source snapshot, workload settings, stage/batch records, preparation helpers, and current manuscript package. `Build.ps1` builds the supplied CMake targets. Runner arguments are indexed in `Runner_Arguments.md`; controlled-drift preparation is described by its helper scripts and settings.

Reruns regenerate machine-dependent timing and result files. Archived CSV files are reference outputs. Raw vectors, per-query neighbor dumps, saved indexes, and executables are not included.
