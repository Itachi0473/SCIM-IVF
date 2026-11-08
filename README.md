# SCIM paper reproduction materials

The archive is organized by the experiments in the current local manuscript. Historical code/data may use the name Triad. Original files were copied, not moved.

- `01_Source/`: the two SCIM core files, required IVF/SIMD dependencies, paper experiment runners, preparation helpers and calibration inputs. CMake includes the current LIRE, DeDrift and size-preserving-control entries.
- `02_Experiments/01_Motivation`: residual/growth and capacity tradeoff evidence.
- `02_Experiments/02_Continuous_Growth`: stage curves, query aggregates, 37 accepted timing repetitions, batch records, matched-K rebuilds and recorded run settings.
- `02_Experiments/03_Query_Budget_Response`: Deep N/20 versus N/100, shared calibration and Quake batch records.
- `02_Experiments/04_Controlled_Drift`: canonical five-method stages, current LIRE and DeDrift source summaries/settings. Mean recall excludes stage0; all_cases_summary is a final-stage summary, not a stage mean.
- `02_Experiments/05_Random_Deletion`: final and stage metrics, recorded settings.
- `02_Experiments/06_Signal_Ablation`: original/PC1 stage data, geometry drift ablation. All curves still appear in the manuscript trajectory plot; therefore their data are retained even though the final table now has four rows.
- `02_Experiments/07_Local_Reassignment`: paper table and timing evidence. See the unresolved final-recall provenance gap in Reproduction_Status.md.
- `02_Experiments/08_Parameter_Sensitivity`: residual threshold sweep and two capacity targets with rebuild comparisons.
- `02_Experiments/09_Size_Preserving_Control`: paired results, preprocessing records and exact experimental manifests.
- `04_Reproduction/`: build script, size-control command, extracted runner arguments, dataset instructions and explicit remaining reproduction gaps.

Start with `04_Reproduction/Reproduction_Status.md`, then `Datasets.md`. Windows/C++17/CMake/Ninja/GCC with AVX2/FMA are required for the recorded environment; Python preparation helpers require NumPy, and Faiss rebuild helpers require faiss-cpu. Build with `./04_Reproduction/Build.ps1`. Numeric reporting conventions: recall source fractions versus manuscript percent; update=insert+delete+maintenance; sample standard deviation across accepted timing runs.

No raw neighbor outputs, per-query logs, vectors, index binaries, executables, trial runs, unrelated experiments or paper PDFs are included. FILE_MANIFEST.csv contains the archive file sizes and SHA256 hashes. Machine-local provenance is kept outside the distributable archive. The archive has not been published online.
