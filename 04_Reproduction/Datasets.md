# Required inputs
Raw vectors, query vectors, truth arrays and saved indexes are not distributed in this archive. Obtain the dataset versions cited by the supplied manuscript. Preserve official vector order and native storage type.

|Experiment|Inputs and processing|
|---|---|
|Growth|MSTuring10M: 100-D FP32, shared random order; YFCC10M: 192-D uint8, original order; Deep10M: 96-D FP32, original order without additional normalization. First 1M initial vectors, 900 batches of 10K, 10K official queries. MSTuring N/20; YFCC/Deep N/100.|
|Nine-case drift|Deep1M base/query fvecs; make_deep1m_freewayml_shift.py generates G,A1,A2,B1-B4,C1-C2 with seed 1234 and individually normalized vectors. compute_deep1m_freewayml_truth.py recomputes exact stage truths; prepare_deep1m_quake_truth.py converts them for DLQTRTH1 consumers. Read each script's --help for its paths.|
|SIFT motivation/ablation|SIFT1M native uint8; 50K initial, K=64, 19 x 50K batches. Original order and PC1 order are separate workloads. Final queries: first 1000, distance budget 50000.|
|Deletion|SIFT1M uint8 and normalized GloVe100 FP32 following ANN-Benchmarks processing. Same deleted IDs across methods, 40 x 20K deletions, 1000 queries, current N/20.|
|Size-preserving control|Deep10M native 96-D FP32; first 1M base vectors, first 1000 official queries, no normalization. Size_Control.ps1 builds and preprocesses its common start when no snapshot is supplied; 32/128/256/512 targets and N/100.|

Dataset landing pages are recorded in the manuscript bibliography: Big ANN benchmark pages for MSTuring/Deep/YFCC and ANN-Benchmarks dataset preparation for SIFT/GloVe. Exact filenames and historical local paths are preserved in Run_Settings for identification; replace local paths when replaying. Local machine paths are not download URLs.
