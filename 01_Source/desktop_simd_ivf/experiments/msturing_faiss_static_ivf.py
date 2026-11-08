#!/usr/bin/env python3
"""Build full-data FAISS IVF-Flat baselines for MSTuring-10M and score them."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


FAISS_PYTHON = Path(__file__).resolve().parents[2] / "build" / "faiss_python"
if FAISS_PYTHON.exists():
    sys.path.insert(0, str(FAISS_PYTHON))

import faiss  # noqa: E402


DATA_PATH_DEFAULT = Path(r"C:\Users\29188\Downloads\msturing-10M-clustered.fbin")
QUERY_PATH_DEFAULT = Path(r"C:\Users\29188\Downloads\testQuery10K.fbin")
TRUTH_PATH_DEFAULT = Path(
    r"E:\MSTuring10M\experiment_results\msturing10m_ih_prd0_no_reassignment_quake_split560_delete3740_lire_prd_localref25_i1_all10k_n20_seed1234_20261021\groundtruth_stage_top100.bin"
)
OUTPUT_DEFAULT = Path(
    r"E:\MSTuring10M\experiment_results\msturing10m_static_faiss_ivf_full10m_sample10pct_budget500k_seed1234_20261023"
)

ROWS = 10_000_000
DIM = 100
QUERY_COUNT = 10_000
TOPK = 100
TRAIN_COUNT = ROWS // 10
SEED = 1234
BUDGET = 500_000
NLISTS = (3163, 1698, 13625)
METHOD_FOR_NLIST = {3163: "PRD final nlist", 1698: "Quake final nlist", 13625: "LIRE final nlist"}


def read_fbin_shape(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = np.fromfile(f, dtype="<u4", count=2)
    if len(header) != 2:
        raise RuntimeError(f"invalid fbin header: {path}")
    rows, dim = int(header[0]), int(header[1])
    expected = 8 + rows * dim * 4
    if path.stat().st_size != expected:
        raise RuntimeError(f"fbin size/shape mismatch for {path}: {rows}x{dim}")
    return rows, dim


def read_truth(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        magic = f.read(8)
        shape = np.fromfile(f, dtype="<u4", count=3)
    if magic != b"MSTGTH01" or tuple(map(int, shape)) != (100, QUERY_COUNT, TOPK):
        raise RuntimeError(f"expected MSTGTH01 [100,10000,100], got {magic!r} {shape}")
    expected_bytes = 8 + 12 + 100 * QUERY_COUNT * TOPK * 4
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"ground-truth size does not match header: {path}")
    truth = np.memmap(path, dtype="<u4", mode="r", offset=20,
                      shape=(100, QUERY_COUNT, TOPK))
    return np.asarray(truth[99])


def write_csv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def build_one(
    nlist: int,
    data_path: Path,
    query_path: Path,
    truth_path: Path,
    vectors: np.ndarray,
    training_vectors: np.ndarray,
    queries: np.ndarray,
    truth: np.ndarray,
    out_dir: Path,
    threads: int,
    query_threads: int,
    recall_only: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=False)
    index_path = out_dir / "index.faiss"
    manifest_path = out_dir / "manifest.txt"
    event_path = out_dir / "build_events.csv"
    events: list[list[object]] = []
    t0 = time.perf_counter()
    faiss.omp_set_num_threads(threads)
    quantizer = faiss.IndexFlatL2(DIM)
    index = faiss.IndexIVFFlat(quantizer, DIM, nlist, faiss.METRIC_L2)
    index.cp.seed = SEED
    index.cp.niter = 25
    # The sample has 1M points. This prevents FAISS's default 256 points/centroid
    # cap from silently subsampling the training sample for the smaller nlists.
    index.cp.max_points_per_centroid = 1024
    index.cp.verbose = True

    print(f"[{nlist}] training_started sample={len(training_vectors)} threads={threads}", flush=True)
    train_start = time.perf_counter()
    index.train(training_vectors)
    train_seconds = time.perf_counter() - train_start
    events.append(["train", len(training_vectors), train_seconds])
    if not index.is_trained:
        raise RuntimeError(f"FAISS training failed for nlist={nlist}")
    print(f"[{nlist}] training_done seconds={train_seconds:.3f}; adding_all={ROWS}", flush=True)

    add_start = time.perf_counter()
    chunk_rows = 100_000
    for start in range(0, ROWS, chunk_rows):
        end = min(start + chunk_rows, ROWS)
        index.add(vectors[start:end])
        if end % 1_000_000 == 0 or end == ROWS:
            elapsed = time.perf_counter() - add_start
            print(f"[{nlist}] add_progress={end}/{ROWS} elapsed_s={elapsed:.1f}", flush=True)
    add_seconds = time.perf_counter() - add_start
    events.append(["add_all_10m", ROWS, add_seconds])
    if index.ntotal != ROWS:
        raise RuntimeError(f"nlist={nlist} indexed {index.ntotal} of {ROWS} rows")

    if recall_only:
        params = faiss.SearchParametersIVF()
        params.nprobe = nlist
        params.max_codes = BUDGET - nlist
        if params.max_codes <= 0:
            raise RuntimeError("budget must exceed nlist")
        faiss.omp_set_num_threads(query_threads)
        print(f"[{nlist}] recall_query_started queries={QUERY_COUNT} threads={query_threads}", flush=True)
        query_start = time.perf_counter()
        _, found_ids = index.search(queries, TOPK, params=params)
        query_seconds = time.perf_counter() - query_start
        hits10 = 0
        hits100 = 0
        for qid in range(QUERY_COUNT):
            got = found_ids[qid]
            got = got[got >= 0]
            hits10 += int(np.isin(truth[qid, :10], got[:10]).sum())
            hits100 += int(np.isin(truth[qid], got).sum())
        recall10 = hits10 / (QUERY_COUNT * 10.0)
        recall100 = hits100 / (QUERY_COUNT * TOPK)
        write_csv(
            out_dir / "recall_only_metrics.csv",
            ["indexed_vectors", "nlist", "training_rows", "training_fraction",
             "query_count", "budget_per_query", "recall_at_10", "recall_at_100",
             "build_seconds", "query_seconds", "query_threads"],
            [[ROWS, nlist, len(training_vectors), len(training_vectors) / ROWS,
              QUERY_COUNT, BUDGET, recall10, recall100,
              train_seconds + add_seconds, query_seconds, query_threads]],
        )
        write_csv(event_path, ["event", "vectors", "seconds"], events)
        label = METHOD_FOR_NLIST.get(nlist, "matched Triad final nlist")
        manifest_path.write_text(
            "experiment=MSTuring-10M full-data FAISS IndexIVFFlat recall-only reference\n"
            f"faiss_version={faiss.__version__}\n"
            f"dataset={data_path}\nqueries={query_path}\ntruth={truth_path}\n"
            "training_sample_fraction=0.10\n"
            f"training_count={len(training_vectors)}\ntraining_sample_seed={SEED}\n"
            f"nlist={nlist}\nmatched_reference={label}\n"
            f"metric=squared_L2\nindexed_vectors={index.ntotal}\n"
            f"query_count={QUERY_COUNT}\ntop_k={TOPK}\n"
            f"budget_per_query={BUDGET} total distances, including {nlist} centroid distances\n"
            f"max_vector_codes_per_query={params.max_codes}\n"
            f"build_threads={threads}\nquery_threads={query_threads}\n"
            "query_mode=one batched FAISS search call; no warmup; no latency collection\n"
            "index_saved=false\n"
            f"recall_at_10={recall10:.10f}\nrecall_at_100={recall100:.10f}\n",
            encoding="utf-8",
        )
        (out_dir / "COMPLETE.txt").write_text(
            f"status=complete\nnlist={nlist}\nrecall_at_10={recall10:.10f}\n"
            f"recall_at_100={recall100:.10f}\nindex_saved=false\n",
            encoding="utf-8",
        )
        total_seconds = time.perf_counter() - t0
        print(
            f"[{nlist}] recall_only_complete recall10={recall10:.6f} "
            f"recall100={recall100:.6f} build_s={train_seconds + add_seconds:.1f} "
            f"query_s={query_seconds:.1f} total_s={total_seconds:.1f}",
            flush=True,
        )
        del index, quantizer
        gc.collect()
        return

    print(f"[{nlist}] save_started path={index_path}", flush=True)
    save_start = time.perf_counter()
    faiss.write_index(index, str(index_path))
    save_seconds = time.perf_counter() - save_start
    events.append(["save_index", index.ntotal, save_seconds])
    index_bytes = index_path.stat().st_size

    # Warm up 100 queries; use one search thread to match the single-query
    # latency measurement path used by the project's CPU runners.
    faiss.omp_set_num_threads(1)
    params = faiss.SearchParametersIVF()
    params.nprobe = nlist
    params.max_codes = BUDGET - nlist
    if params.max_codes <= 0:
        raise RuntimeError("budget must exceed nlist")
    for qid in range(100):
        index.search(queries[qid:qid + 1], TOPK, params=params)

    faiss.cvar.indexIVF_stats.reset()
    latencies_ms = np.empty(QUERY_COUNT, dtype=np.float64)
    hits10 = np.empty(QUERY_COUNT, dtype=np.uint8)
    hits100 = np.empty(QUERY_COUNT, dtype=np.uint8)
    per_query_rows: list[list[object]] = []
    query_start = time.perf_counter()
    for qid in range(QUERY_COUNT):
        one_query = queries[qid:qid + 1]
        start_ns = time.perf_counter_ns()
        _, ids = index.search(one_query, TOPK, params=params)
        elapsed_ns = time.perf_counter_ns() - start_ns
        latencies_ms[qid] = elapsed_ns / 1_000_000.0
        got = ids[0]
        got = got[got >= 0].astype(np.uint32, copy=False)
        expected10 = truth[qid, :10]
        expected100 = truth[qid, :100]
        h10 = int(np.isin(expected10, got).sum())
        h100 = int(np.isin(expected100, got).sum())
        hits10[qid] = h10
        hits100[qid] = h100
        per_query_rows.append([
            qid, h10, h10 / 10.0, h100, h100 / 100.0,
            nlist, BUDGET - nlist, BUDGET, latencies_ms[qid],
        ])
        if (qid + 1) % 1000 == 0:
            print(f"[{nlist}] query_progress={qid + 1}/{QUERY_COUNT}", flush=True)
    query_seconds = time.perf_counter() - query_start

    vector_distances = int(faiss.cvar.indexIVF_stats.ndis)
    expected_vector_distances = QUERY_COUNT * (BUDGET - nlist)
    if vector_distances != expected_vector_distances:
        raise RuntimeError(
            f"FAISS scan mismatch nlist={nlist}: expected {expected_vector_distances} vector distances, "
            f"observed {vector_distances}"
        )

    mean_ms = float(latencies_ms.mean())
    p95_ms = float(np.percentile(latencies_ms, 95))
    r10 = float(hits10.sum() / (QUERY_COUNT * 10))
    r100 = float(hits100.sum() / (QUERY_COUNT * 100))
    total_query_distances = QUERY_COUNT * BUDGET
    query_rows = [[
        ROWS, nlist, QUERY_COUNT, BUDGET, nlist, BUDGET - nlist,
        total_query_distances, r10, r100, mean_ms, p95_ms, query_seconds,
        train_seconds * 1000, add_seconds * 1000, save_seconds * 1000,
        index_bytes,
    ]]
    write_csv(
        out_dir / "fixed_budget_metrics.csv",
        ["indexed_vectors", "nlist", "query_count", "budget_per_query",
         "center_distances_per_query", "vector_distances_per_query",
         "total_distance_computations", "recall_at_10", "recall_at_100",
         "mean_query_ms", "p95_query_ms", "total_query_seconds",
         "training_ms", "add_all_ms", "save_index_ms", "index_file_bytes"],
        query_rows,
    )
    write_csv(
        out_dir / "per_query.csv",
        ["query_id", "hits_at_10", "recall_at_10", "hits_at_100", "recall_at_100",
         "center_distances", "vector_distances", "total_distances", "query_ms"],
        per_query_rows,
    )
    write_csv(event_path, ["event", "vectors", "seconds"], events)

    manifest_path.write_text(
        "experiment=MSTuring-10M full-data FAISS IndexIVFFlat fixed-budget baseline\n"
        f"faiss_version={faiss.__version__}\n"
        f"dataset={DATA_PATH_DEFAULT}\nqueries={QUERY_PATH_DEFAULT}\n"
        f"groundtruth_final_stage={TRUTH_PATH_DEFAULT}\n"
        f"indexed_vectors={index.ntotal}\ndimension={DIM}\nmetric=squared_L2\n"
        f"nlist={nlist}\nmethod_reference={METHOD_FOR_NLIST[nlist]}\n"
        f"training_count={TRAIN_COUNT}\ntraining_sample_fraction=0.10\n"
        f"training_sample_seed={SEED}\ntraining_sample_rng=numpy.default_rng(PCG64)\n"
        f"faiss_kmeans_niter={index.cp.niter}\n"
        f"faiss_kmeans_max_points_per_centroid={index.cp.max_points_per_centroid}\n"
        f"faiss_kmeans_seed={index.cp.seed}\n"
        f"training_seconds={train_seconds:.6f}\nadd_all_10m_seconds={add_seconds:.6f}\n"
        f"index_save_seconds={save_seconds:.6f}\nindex_file_bytes={index_bytes}\n"
        f"query_count={QUERY_COUNT}\ntop_k={TOPK}\nwarmup_queries=100\n"
        f"search_threads=1\nscan_order=all coarse centroids ranked by squared L2, then IVF lists in that order\n"
        f"budget_per_query={BUDGET}\ncenter_distance_comparisons_per_query={nlist}\n"
        f"max_vector_distance_comparisons_per_query={BUDGET - nlist}\n"
        f"observed_vector_distance_comparisons={vector_distances}\n"
        f"observed_total_distance_comparisons={total_query_distances}\n"
        f"recall_at_10={r10:.10f}\nrecall_at_100={r100:.10f}\n"
        f"mean_query_ms={mean_ms:.8f}\np95_query_ms={p95_ms:.8f}\n"
        f"total_query_seconds={query_seconds:.6f}\n"
        f"index={index_path}\n",
        encoding="utf-8",
    )
    total_seconds = time.perf_counter() - t0
    print(
        f"[{nlist}] complete recall10={r10:.6f} recall100={r100:.6f} "
        f"mean_ms={mean_ms:.3f} p95_ms={p95_ms:.3f} index_bytes={index_bytes} "
        f"elapsed_s={total_seconds:.1f}",
        flush=True,
    )
    del index, quantizer
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=DATA_PATH_DEFAULT)
    parser.add_argument("--queries", type=Path, default=QUERY_PATH_DEFAULT)
    parser.add_argument("--truth", type=Path, default=TRUTH_PATH_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--query-threads", type=int, default=None)
    parser.add_argument("--nlists", type=int, nargs="+", default=list(NLISTS))
    parser.add_argument("--recall-only", action="store_true",
                        help="skip index saving and latency/per-query output; batch-search for recall only")
    args = parser.parse_args()
    sample_file = args.output / "training_sample_ids_seed1234.u32bin"
    if args.output.exists():
        existing = list(args.output.iterdir())
        if (len(existing) != 1 or existing[0] != sample_file or
                not sample_file.is_file() or sample_file.stat().st_size != TRAIN_COUNT * 4):
            raise SystemExit(f"output directory exists and is not a resumable partial run: {args.output}")
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    query_threads = args.threads if args.query_threads is None else args.query_threads
    if query_threads < 1:
        raise SystemExit("--query-threads must be positive")
    if not args.nlists or any(nlist < 1 or nlist >= BUDGET for nlist in args.nlists):
        raise SystemExit("--nlists must contain positive values smaller than the query budget")
    if not hasattr(faiss, "SearchParametersIVF"):
        raise RuntimeError("installed FAISS lacks SearchParametersIVF")

    rows, dim = read_fbin_shape(args.data)
    query_rows, query_dim = read_fbin_shape(args.queries)
    if (rows, dim) != (ROWS, DIM) or query_rows < QUERY_COUNT or query_dim != DIM:
        raise RuntimeError(f"unexpected dataset/query shapes: {(rows, dim)}, {(query_rows, query_dim)}")
    vectors = np.memmap(args.data, dtype="<f4", mode="r", offset=8, shape=(ROWS, DIM))
    queries = np.memmap(args.queries, dtype="<f4", mode="r", offset=8,
                        shape=(query_rows, query_dim))[:QUERY_COUNT]
    truth = read_truth(args.truth)
    rng = np.random.default_rng(SEED)
    sample_ids = rng.choice(ROWS, size=TRAIN_COUNT, replace=False).astype(np.uint32)
    sample_ids.sort()
    if sample_file.exists():
        old_sample_ids = np.fromfile(sample_file, dtype="<u4")
        if old_sample_ids.shape != sample_ids.shape or not np.array_equal(old_sample_ids, sample_ids):
            raise RuntimeError("existing partial output has a different training sample")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        sample_ids.tofile(sample_file)
    print("training_sample_ids_ready; gathering sampled vectors", flush=True)
    training_vectors = np.ascontiguousarray(vectors[sample_ids])
    print("training_sample_vectors_ready", flush=True)
    sample_hash = hashlib.sha256(sample_ids.tobytes()).hexdigest()
    (args.output / "manifest.txt").write_text(
        "experiment=MSTuring-10M static IVF baselines via FAISS\n"
        f"faiss_version={faiss.__version__}\n"
        f"dataset={args.data}\nqueries={args.queries}\ntruth_final_stage={args.truth}\n"
        f"indexed_vectors={ROWS}\ndimension={DIM}\nmetric=squared_L2\n"
        f"nlist_values={','.join(map(str, args.nlists))}\ntraining_count={TRAIN_COUNT}\n"
        f"training_fraction=0.10\ntraining_seed={SEED}\n"
        "training_rng=numpy.default_rng(PCG64)\n"
        f"training_sample_id_file={sample_file}\ntraining_sample_id_sha256={sample_hash}\n"
        f"query_count={QUERY_COUNT}\ntop_k={TOPK}\nquery_budget={BUDGET} total distance computations/query\n"
        "budget includes all nlist centroid distances plus vector comparisons\n"
        "each IVF index uses all 10M vectors; only coarse quantizer training uses the 1M-row sample\n",
        encoding="utf-8",
    )
    print(
        f"FAISS {faiss.__version__}; data={rows}x{dim}; sample={TRAIN_COUNT}; "
        f"queries={QUERY_COUNT}; budget={BUDGET}; output={args.output}",
        flush=True,
    )
    for nlist in args.nlists:
        build_one(nlist, args.data, args.queries, args.truth,
                  vectors, training_vectors, queries, truth,
                  args.output / f"query_nlist{nlist}", args.threads,
                  query_threads, args.recall_only)
    (args.output / "COMPLETE.txt").write_text(
        "All requested full-data FAISS IVF-Flat baselines completed and were scored.\n",
        encoding="utf-8",
    )
    print(f"ALL_COMPLETE output={args.output}", flush=True)


if __name__ == "__main__":
    main()
