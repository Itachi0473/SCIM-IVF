#!/usr/bin/env python3
"""Recall-only, full-data IVF references using 10% deterministic k-means samples.

The full inverted lists are stored temporarily as memory-mapped files, so the
uint8 YFCC dataset is never expanded into a 7.7 GB in-memory FAISS index.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import gc
import hashlib
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FAISS_PYTHON = ROOT / "build" / "faiss_python"
if FAISS_PYTHON.exists():
    sys.path.insert(0, str(FAISS_PYTHON))
_DLL_HANDLES = []
for _dll_dir in (FAISS_PYTHON / "faiss_cpu.libs", FAISS_PYTHON / "faiss"):
    if _dll_dir.exists() and hasattr(os, "add_dll_directory"):
        _DLL_HANDLES.append(os.add_dll_directory(str(_dll_dir)))
import faiss  # noqa: E402


SEED = 1234
TOPK = 100
BUDGET = 500_000
TRAIN_FRACTION = 0.10
CHUNK_ROWS = 1024
SCAN_CHUNK_ROWS = 8192
METHODS = {
    "MSTuring-10M": {"Triad": 3163, "Quake": 3120, "LIRE": 13328},
    "YFCC-10M": {"Triad": 3179, "Quake": 5803, "LIRE": 12437},
    "Deep10M": {"Triad": 3163, "Quake": 1885, "LIRE": 12224},
}


def path_config() -> dict[str, dict[str, object]]:
    return {
        "MSTuring-10M": {
            "data": Path(r"E:\MSTuring10M\shuffled_input_seed1234_20261025\msturing-10M-shuffled-seed1234.fbin"),
            "queries": Path(r"C:\Users\29188\Downloads\testQuery10K.fbin"),
            "truth": Path(r"E:\MSTuring10M\experiment_results\msturing10m_triad_quake_lire_onepass_full_ih_seed1234_20261026\groundtruth_stage_top100.bin"),
            "dtype": np.dtype("<f4"), "header": "fbin", "query_count": 10_000,
        },
        "YFCC-10M": {
            "data": Path(r"C:\Users\29188\Downloads\YFCC10m\base.10M.u8bin"),
            "queries": Path(r"E:\MSTuring10M\experiment_results\yfcc10m_ih_quake_tau250_sample1k_n20_seed1234_20261023\prepared\query.sample1K.seed1234.u8bin"),
            "truth": Path(r"E:\MSTuring10M\experiment_results\yfcc10m_ih_quake_tau250_sample1k_n20_seed1234_20261023\prepared\groundtruth_stage_top100.bin"),
            "dtype": np.dtype("u1"), "header": "u8bin", "query_count": 1_000,
        },
        "Deep10M": {
            "data": Path(r"C:\Users\29188\Downloads\Deep10M_2\base.10M.fbin"),
            "queries": Path(r"C:\Users\29188\Downloads\Deep10M_2\query.public.10K.fbin"),
            "truth": Path(r"E:\Deep10M\deep10m_triad_quake_lire_100stages_n20_lire1000_seed1234_20261028\groundtruth_stage_top100.bin"),
            "dtype": np.dtype("<f4"), "header": "fbin", "query_count": 10_000,
        },
    }


def read_header(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = np.fromfile(f, dtype="<u4", count=2)
    if len(header) != 2:
        raise ValueError(f"invalid vector file header: {path}")
    return int(header[0]), int(header[1])


def vector_memmap(path: Path, dtype: np.dtype, expected_header: str) -> tuple[np.memmap, int, int]:
    rows, dim = read_header(path)
    itemsize = dtype.itemsize
    expected_size = 8 + rows * dim * itemsize
    if path.stat().st_size != expected_size:
        raise ValueError(f"vector file size/header mismatch: {path}, {rows}x{dim}, {path.stat().st_size}")
    if expected_header not in ("fbin", "u8bin"):
        raise ValueError(expected_header)
    return np.memmap(path, dtype=dtype, mode="r", offset=8, shape=(rows, dim)), rows, dim


def truth_memmap(path: Path, query_count: int) -> np.ndarray:
    with path.open("rb") as f:
        magic = f.read(8)
        shape = tuple(map(int, np.fromfile(f, dtype="<u4", count=3)))
    if magic != b"MSTGTH01" or shape != (100, query_count, TOPK):
        raise ValueError(f"unexpected stage ground truth: {path}: {magic!r}, {shape}")
    expected = 20 + int(np.prod(shape)) * 4
    if path.stat().st_size != expected:
        raise ValueError(f"ground-truth size mismatch: {path}")
    arr = np.memmap(path, dtype="<u4", mode="r", offset=20, shape=shape)
    return arr[99]


def write_rows(path: Path, header: list[str], rows: list[list[object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def train_centers(training: np.ndarray, dim: int, nlist: int, threads: int) -> np.ndarray:
    faiss.omp_set_num_threads(threads)
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    index.cp.seed = SEED
    index.cp.niter = 25
    index.cp.max_points_per_centroid = 1024
    index.cp.verbose = False
    tick = time.perf_counter()
    print(f"  training start: rows={len(training):,}, K={nlist:,}, threads={threads}", flush=True)
    index.train(training)
    seconds = time.perf_counter() - tick
    if not index.is_trained:
        raise RuntimeError(f"FAISS failed to train K={nlist}")
    centers = np.empty((nlist, dim), dtype=np.float32)
    index.quantizer.reconstruct_n(0, nlist, centers)
    print(f"  training done: {seconds:.1f}s", flush=True)
    del index, quantizer
    gc.collect()
    return centers


def build_postings(
    data: np.ndarray,
    centers: np.ndarray,
    temp_dir: Path,
    threads: int,
) -> tuple[np.memmap, np.memmap, np.ndarray, np.ndarray, float]:
    rows, dim = data.shape
    nlist = len(centers)
    temp_dir.mkdir(parents=True, exist_ok=False)
    labels_path = temp_dir / "labels.i32"
    labels = np.memmap(labels_path, dtype="<i4", mode="w+", shape=(rows,))
    coarse = faiss.IndexFlatL2(dim)
    coarse.add(np.ascontiguousarray(centers))
    faiss.omp_set_num_threads(threads)
    assign_start = time.perf_counter()
    for begin in range(0, rows, CHUNK_ROWS):
        end = min(begin + CHUNK_ROWS, rows)
        batch = np.asarray(data[begin:end], dtype=np.float32, order="C")
        _, ids = coarse.search(batch, 1)
        labels[begin:end] = ids[:, 0]
        if end % 1_000_000 == 0 or end == rows:
            print(f"  assignment: {end:,}/{rows:,}", flush=True)
    labels.flush()
    assign_seconds = time.perf_counter() - assign_start

    counts = np.bincount(np.asarray(labels), minlength=nlist).astype(np.int64)
    offsets = np.zeros(nlist + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    order = np.argsort(labels, kind="stable").astype(np.uint32, copy=False)
    ids_path = temp_dir / "posting_ids.u32"
    sorted_ids = np.memmap(ids_path, dtype="<u4", mode="w+", shape=(rows,))
    sorted_ids[:] = order
    sorted_ids.flush()
    del order

    posting_path = temp_dir / "posting_vectors.bin"
    postings = np.memmap(posting_path, dtype=data.dtype, mode="w+", shape=(rows, dim))
    copy_start = time.perf_counter()
    for begin in range(0, rows, 50_000):
        end = min(begin + 50_000, rows)
        idx = np.asarray(sorted_ids[begin:end], dtype=np.int64)
        postings[begin:end] = data[idx]
        if end % 1_000_000 == 0 or end == rows:
            print(f"  full-data postings: {end:,}/{rows:,}", flush=True)
    postings.flush()
    build_seconds = assign_seconds + time.perf_counter() - copy_start
    del labels, coarse
    gc.collect()
    print(f"  full-data assignment/posting build done: {build_seconds:.1f}s", flush=True)
    return postings, sorted_ids, offsets, counts, build_seconds


def distance_block(block: np.ndarray, query: np.ndarray) -> np.ndarray:
    if block.dtype == np.uint8:
        diff = block.astype(np.int16) - query.astype(np.int16, copy=False)
        return np.einsum("ij,ij->i", diff, diff, dtype=np.int64, optimize=False)
    diff = np.subtract(block, query, dtype=np.float32)
    return np.einsum("ij,ij->i", diff, diff, dtype=np.float32, optimize=False)


def query_one(
    qid: int,
    query: np.ndarray,
    truth: np.ndarray,
    coarse: object,
    postings: np.ndarray,
    posting_ids: np.ndarray,
    offsets: np.ndarray,
    counts: np.ndarray,
    nlist: int,
    vector_budget: int,
) -> tuple[int, int]:
    _, ordered = coarse.search(np.asarray(query, dtype=np.float32).reshape(1, -1), nlist)
    best_dist = np.empty(0, dtype=np.float32 if query.dtype != np.uint8 else np.int64)
    best_ids = np.empty(0, dtype=np.int64)
    remaining = vector_budget
    for cluster in ordered[0]:
        c = int(cluster)
        take = min(int(counts[c]), remaining)
        if take <= 0:
            continue
        pos = int(offsets[c])
        end = pos + take
        while pos < end:
            stop = min(pos + SCAN_CHUNK_ROWS, end)
            block = np.asarray(postings[pos:stop])
            dist = distance_block(block, query)
            ids = np.asarray(posting_ids[pos:stop], dtype=np.int64)
            if len(dist) > TOPK:
                keep = np.argpartition(dist, TOPK - 1)[:TOPK]
                dist = dist[keep]
                ids = ids[keep]
            comb_dist = np.concatenate((best_dist, dist))
            comb_ids = np.concatenate((best_ids, ids))
            if len(comb_dist) > TOPK:
                keep = np.argpartition(comb_dist, TOPK - 1)[:TOPK]
                best_dist = comb_dist[keep]
                best_ids = comb_ids[keep]
            else:
                best_dist, best_ids = comb_dist, comb_ids
            pos = stop
        remaining -= take
        if remaining == 0:
            break
    if remaining != 0:
        raise RuntimeError(f"query {qid} scanned fewer than the fixed vector budget")
    sorted_best = best_ids[np.argsort(best_dist)]
    gt = np.asarray(truth[qid], dtype=np.uint32)
    hits10 = int(np.isin(gt[:10], sorted_best[:10]).sum())
    hits100 = int(np.isin(gt, sorted_best[:TOPK]).sum())
    return hits10, hits100


def run_method(
    dataset_name: str,
    method: str,
    nlist: int,
    data: np.ndarray,
    queries: np.ndarray,
    truth: np.ndarray,
    sample_training: np.ndarray,
    temp_root: Path,
    output_dir: Path,
    threads: int,
) -> dict[str, object]:
    rows, dim = data.shape
    qcount = len(queries)
    if BUDGET <= nlist:
        raise ValueError(f"distance budget {BUDGET} <= K {nlist}")
    print(f"{dataset_name} / {method}: K={nlist:,}, N={rows:,}, Q={qcount:,}", flush=True)
    total_start = time.perf_counter()
    centers = train_centers(sample_training, dim, nlist, threads)
    with tempfile.TemporaryDirectory(prefix=f"{dataset_name}_{method}_", dir=temp_root) as td:
        postings, posting_ids, offsets, counts, assignment_seconds = build_postings(
            data, centers, Path(td) / "postings", threads
        )
        coarse = faiss.IndexFlatL2(dim)
        coarse.add(np.ascontiguousarray(centers))
        faiss.omp_set_num_threads(1)
        vector_budget = BUDGET - nlist
        hit10 = hit100 = 0
        print(f"  recall query start: threads={threads}, vector_budget={vector_budget:,}/query", flush=True)
        query_start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = (
                executor.submit(query_one, qid, queries[qid], truth, coarse,
                                postings, posting_ids, offsets, counts, nlist, vector_budget)
                for qid in range(qcount)
            )
            for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                h10, h100 = future.result()
                hit10 += h10
                hit100 += h100
                if done % 1000 == 0 or done == qcount:
                    print(f"  recall progress: {done:,}/{qcount:,}", flush=True)
        query_seconds = time.perf_counter() - query_start
        del postings, posting_ids, coarse
        gc.collect()

    r10 = hit10 / (qcount * 10)
    r100 = hit100 / (qcount * TOPK)
    result = {
        "dataset": dataset_name,
        "method_reference": method,
        "nlist": nlist,
        "indexed_vectors": rows,
        "dimension": dim,
        "training_rows": len(sample_training),
        "training_fraction": len(sample_training) / rows,
        "training_seed": SEED,
        "query_count": qcount,
        "budget_per_query": BUDGET,
        "recall_at_10": r10,
        "recall_at_100": r100,
        "assignment_and_posting_seconds": assignment_seconds,
        "query_seconds": query_seconds,
        "total_seconds": time.perf_counter() - total_start,
        "query_threads": threads,
        "index_saved": False,
    }
    row_path = output_dir / "summary.csv"
    fields = list(result)
    needs_header = not row_path.exists()
    with row_path.open("a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if needs_header:
            writer.writeheader()
        writer.writerow(result)
    method_dir = output_dir / dataset_name.replace("-", "") / method
    method_dir.mkdir(parents=True, exist_ok=True)
    with (method_dir / "recall_only_metrics.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(result)
    print(f"COMPLETE {dataset_name}/{method}: R@10={r10:.6f}, R@100={r100:.6f}, "
          f"query={query_seconds:.1f}s, total={result['total_seconds']:.1f}s", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=ROOT / "outputs" / "rebuild_reference_sample10pct_20261030")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--datasets", nargs="*", default=list(METHODS))
    args = parser.parse_args()
    if args.threads < 1:
        raise SystemExit("--threads must be positive")
    configs = path_config()
    args.output.mkdir(parents=True, exist_ok=True)
    tmp_root = args.output / "_temporary_index_data"
    tmp_root.mkdir(exist_ok=True)
    summary = args.output / "summary.csv"
    done = set()
    if summary.exists():
        with summary.open("r", newline="", encoding="utf-8-sig") as f:
            done = {(r["dataset"], r["method_reference"]) for r in csv.DictReader(f)}
    config_lines = [
        "experiment=full-data IVF-Flat rebuild references, recall-only",
        f"faiss_version={faiss.__version__}", f"training_fraction={TRAIN_FRACTION}",
        f"training_seed={SEED}", "training_sample_rng=numpy.default_rng(PCG64), without replacement",
        "faiss_kmeans_iterations=25", "faiss_kmeans_max_points_per_centroid=1024",
        f"build_threads={args.threads}", f"query_threads={args.threads}",
        f"query_budget={BUDGET} total distances/query, including K center distances",
        "metric=squared_L2", "search=coarse-centroid distance order then exact vector distances",
        "index_snapshots=not retained; disk-backed temporary postings removed after each method",
    ]
    (args.output / "manifest.txt").write_text("\n".join(config_lines) + "\n", encoding="utf-8")
    print(f"output={args.output}; FAISS={faiss.__version__}; threads={args.threads}", flush=True)

    for dataset_name in args.datasets:
        cfg = configs[dataset_name]
        data, rows, dim = vector_memmap(cfg["data"], cfg["dtype"], cfg["header"])
        queries, query_rows, query_dim = vector_memmap(cfg["queries"], cfg["dtype"], cfg["header"])
        qcount = int(cfg["query_count"])
        if query_dim != dim or query_rows < qcount:
            raise ValueError(f"query shape does not match {dataset_name}: {query_rows}x{query_dim}")
        queries = queries[:qcount]
        truth = truth_memmap(cfg["truth"], qcount)
        train_rows = int(rows * TRAIN_FRACTION)
        rng = np.random.default_rng(SEED)
        sample_ids = rng.choice(rows, size=train_rows, replace=False).astype(np.uint32)
        sample_ids.sort()
        sample_dir = args.output / dataset_name.replace("-", "")
        sample_dir.mkdir(parents=True, exist_ok=True)
        ids_path = sample_dir / f"training_sample_ids_seed{SEED}.u32bin"
        if not ids_path.exists():
            sample_ids.tofile(ids_path)
        sample_hash = hashlib.sha256(sample_ids.tobytes()).hexdigest()
        sample_vectors = np.asarray(data[sample_ids], dtype=np.float32, order="C")
        print(f"\nDATASET {dataset_name}: shape={rows:,}x{dim}, Q={qcount:,}, "
              f"training={train_rows:,} (10%), sample_sha256={sample_hash}", flush=True)
        for method, nlist in METHODS[dataset_name].items():
            if (dataset_name, method) in done:
                print(f"SKIP already complete: {dataset_name}/{method}", flush=True)
                continue
            run_method(dataset_name, method, nlist, data, queries, truth,
                       sample_vectors, tmp_root, args.output, args.threads)
            done.add((dataset_name, method))
        del data, queries, truth, sample_vectors
        gc.collect()

    shutil.rmtree(tmp_root, ignore_errors=True)
    (args.output / "COMPLETE.txt").write_text(
        "status=complete\nAll requested 10%-trained, full-data IVF rebuild recall references completed.\n",
        encoding="utf-8",
    )
    print(f"ALL_COMPLETE summary={summary}", flush=True)


if __name__ == "__main__":
    main()
