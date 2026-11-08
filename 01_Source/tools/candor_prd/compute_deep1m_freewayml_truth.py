#!/usr/bin/env python3
"""Compute exact stage-wise top-10 ground truth for DEEP-1M drift streams.

IDs in neighbor_ids are row IDs in each scenario's emitted base.fbin stream
(initial rows first, then insertion batches). source_row_ids map those IDs back
to the original DEEP-1M fvecs file.
"""

import argparse
import csv
import json
import math
import struct
import time
from pathlib import Path

import numpy as np


CASES = ("G", "A1", "A2", "B1", "B2", "B3", "B4", "C1", "C2")
TOPK = 10


def open_fbin(path: Path):
    with path.open("rb") as stream:
        header = stream.read(8)
    if len(header) != 8:
        raise ValueError(f"truncated fbin header: {path}")
    rows, dimension = struct.unpack("<II", header)
    expected = 8 + rows * dimension * 4
    if path.stat().st_size != expected:
        raise ValueError(f"fbin size/header mismatch: {path}")
    vectors = np.memmap(path, dtype="<f4", mode="r", offset=8,
                        shape=(rows, dimension))
    return vectors, rows, dimension


def read_queries(path: Path, expected_count: int, dimension: int) -> np.ndarray:
    vectors, rows, dims = open_fbin(path)
    if rows != expected_count or dims != dimension:
        raise ValueError(f"unexpected query shape in {path}: {(rows, dims)}")
    return np.asarray(vectors, dtype=np.float32).copy()


def update_topk(queries: np.ndarray, vectors: np.ndarray, first_id: int,
                best_scores: np.ndarray, best_ids: np.ndarray):
    """Merge a vector block into per-query top-k by exact squared-L2 ranking.

    All scenario vectors are unit normalized. The dot-product identity is used
    with each row's measured norm to retain squared-L2 ranking while BLAS
    reuses each vector block across all queries.
    """
    block = np.asarray(vectors, dtype=np.float32)
    dots = queries @ block.T
    qnorm = np.einsum("ij,ij->i", queries, queries, dtype=np.float32)
    xnorm = np.einsum("ij,ij->i", block, block, dtype=np.float32)
    dots *= -2.0
    dots += qnorm[:, None]
    dots += xnorm[None, :]
    dots *= -1.0  # largest score == smallest squared L2 distance

    take = min(TOPK, block.shape[0])
    local = np.argpartition(dots, dots.shape[1] - take, axis=1)[:, -take:]
    local_scores = np.take_along_axis(dots, local, axis=1)
    local_ids = (local.astype(np.uint32) + np.uint32(first_id))
    candidate_scores = np.concatenate((best_scores, local_scores), axis=1)
    candidate_ids = np.concatenate((best_ids, local_ids), axis=1)
    order = np.lexsort((candidate_ids, -candidate_scores), axis=1)[:, :TOPK]
    rows = np.arange(queries.shape[0])[:, None]
    return (np.take_along_axis(candidate_scores, order, axis=1),
            np.take_along_axis(candidate_ids, order, axis=1))


def fresh_topk(query_count: int):
    return (np.full((query_count, TOPK), -np.inf, dtype=np.float32),
            np.full((query_count, TOPK), np.iinfo(np.uint32).max, dtype=np.uint32))


def scan_prefix(queries: np.ndarray, vectors: np.ndarray, active: int,
                chunk_rows: int):
    best_scores, best_ids = fresh_topk(queries.shape[0])
    for begin in range(0, active, chunk_rows):
        end = min(begin + chunk_rows, active)
        best_scores, best_ids = update_topk(
            queries, vectors[begin:end], begin, best_scores, best_ids)
    return best_ids, np.maximum(-best_scores, 0.0)


def save_stage(neighbors, distances, stage, ids, d2):
    neighbors[stage] = ids
    distances[stage] = d2


def compute_incremental_case(case: str, directory: Path, vectors: np.ndarray,
                             source_rows: np.ndarray, initial: int, batch: int,
                             steps: int, query_count: int, chunk_rows: int,
                             output_dir: Path):
    start_time = time.perf_counter()
    query0 = read_queries(directory / "query_000.fbin", query_count, vectors.shape[1])
    dual = case in ("C1", "C2")
    if dual:
        query1 = read_queries(directory / "query_001.fbin", query_count, vectors.shape[1])
        active_states = [query0, query1]
    else:
        active_states = [query0]
        for stage in (1, steps, max(1, steps // 2)):
            check = read_queries(directory / f"query_{stage:03d}.fbin", query_count, vectors.shape[1])
            if not np.array_equal(check, query0):
                raise ValueError(f"{case} should use one fixed query set; stage {stage} differs")

    neighbors = np.empty((steps + 1, query_count, TOPK), dtype=np.uint32)
    distances = np.empty((steps + 1, query_count, TOPK), dtype=np.float32)
    active_counts = initial + np.arange(steps + 1, dtype=np.uint32) * batch
    stage_query_state = np.zeros(steps + 1, dtype=np.uint8)
    states = [fresh_topk(query_count) for _ in active_states]

    # Build exact top-k for the initial stage, then only inspect each newly
    # inserted batch. This is exact for fixed query vectors.
    for begin in range(0, initial, chunk_rows):
        end = min(begin + chunk_rows, initial)
        for i, query in enumerate(active_states):
            states[i] = update_topk(query, vectors[begin:end], begin, *states[i])
    for i in range(steps + 1):
        state = 0
        if dual and i > 0:
            current_query = read_queries(directory / f"query_{i:03d}.fbin", query_count, vectors.shape[1])
            if np.array_equal(current_query, query1):
                state = 1
            elif not np.array_equal(current_query, query0):
                raise ValueError(f"{case} query stage {i} is neither alternating state")
        stage_query_state[i] = state
        save_stage(neighbors, distances, i, states[state][1], np.maximum(-states[state][0], 0.0))
        if i == steps:
            break
        begin = initial + i * batch
        end = begin + batch
        for j, query in enumerate(active_states):
            states[j] = update_topk(query, vectors[begin:end], begin, *states[j])
        if (i + 1) % 10 == 0 or i + 1 == steps:
            print(f"{case}: exact incremental stage {i + 1}/{steps}", flush=True)

    return neighbors, distances, active_counts, stage_query_state, time.perf_counter() - start_time


def compute_changing_query_case(case: str, directory: Path, vectors: np.ndarray,
                                initial: int, batch: int, steps: int,
                                query_count: int, chunk_rows: int,
                                output_dir: Path):
    stage_count = steps + 1
    neighbors = np.empty((stage_count, query_count, TOPK), dtype=np.uint32)
    distances = np.empty((stage_count, query_count, TOPK), dtype=np.float32)
    active_counts = initial + np.arange(stage_count, dtype=np.uint32) * batch
    stage_query_state = np.arange(stage_count, dtype=np.uint8)
    timings = []
    for stage, active in enumerate(active_counts):
        start_time = time.perf_counter()
        query = read_queries(directory / f"query_{stage:03d}.fbin", query_count, vectors.shape[1])
        ids, d2 = scan_prefix(query, vectors, int(active), chunk_rows)
        save_stage(neighbors, distances, stage, ids, d2)
        elapsed = time.perf_counter() - start_time
        timings.append(elapsed)
        if stage % 5 == 0 or stage == steps:
            print(f"{case}: exact full-prefix stage {stage}/{steps} active={active} seconds={elapsed:.1f}", flush=True)
    return neighbors, distances, active_counts, stage_query_state, sum(timings), timings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=8192)
    args = parser.parse_args()
    root = args.data_root.resolve()
    if args.chunk_rows <= 0:
        parser.error("chunk-rows must be positive")
    data_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    initial = int(data_manifest["initial_count"])
    batch = int(data_manifest["batch_size"])
    steps = int(data_manifest["steps"])
    query_count = int(data_manifest["query_count"])
    source_rows = np.load(root / "base_source_rows.npy", mmap_mode="r")
    truth_root = root / "truth"
    if truth_root.exists() and any(truth_root.iterdir()):
        parser.error(f"truth output directory must be absent or empty: {truth_root}")
    truth_root.mkdir(parents=True, exist_ok=True)
    summaries = []

    for case in CASES:
        case_dir = root / case
        vectors, rows, dimension = open_fbin(case_dir / "base.fbin")
        if rows != int(data_manifest["base_count"]) or dimension != int(data_manifest["dimension"]):
            raise ValueError(f"unexpected base shape for {case}: {(rows, dimension)}")
        print(f"starting exact truth: {case}; rows={rows}; queries={query_count}; stages={steps + 1}", flush=True)
        if case in ("A1", "A2"):
            result = compute_changing_query_case(
                case, case_dir, vectors, initial, batch, steps, query_count,
                args.chunk_rows, truth_root)
            neighbors, distances, active_counts, stage_query_state, elapsed, timings = result
            per_stage_seconds = timings
        else:
            result = compute_incremental_case(
                case, case_dir, vectors, source_rows, initial, batch, steps,
                query_count, args.chunk_rows, truth_root)
            neighbors, distances, active_counts, stage_query_state, elapsed = result
            per_stage_seconds = [0.0] * (steps + 1)

        if np.any(neighbors >= active_counts[:, None, None]):
            raise RuntimeError(f"out-of-range neighbor IDs in {case}")
        if not np.all(np.diff(distances, axis=2) >= -1e-6):
            raise RuntimeError(f"top-10 distances are not sorted for {case}")
        source_neighbor_rows = np.asarray(source_rows[neighbors], dtype=np.uint32)
        case_out = truth_root / case
        case_out.mkdir()
        np.savez_compressed(
            case_out / "groundtruth.npz",
            neighbor_ids=neighbors,
            squared_l2_distances=distances,
            source_row_ids=source_neighbor_rows,
            active_counts=active_counts,
            stage_query_state=stage_query_state,
        )
        with (case_out / "stage_times.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("stage", "active_vectors", "queries", "exact_scan_seconds"))
            for stage, active in enumerate(active_counts):
                writer.writerow((stage, int(active), query_count, f"{per_stage_seconds[stage]:.6f}"))
        case_manifest = {
            "case": case,
            "metric": "squared_l2",
            "ranking_method": "exact exhaustive matrix products with actual row norms",
            "queries_per_stage": query_count,
            "stages": steps + 1,
            "neighbor_ids_space": "0-based row IDs in this case's emitted base.fbin stream",
            "source_row_ids_space": "0-based row IDs in deep1b_base_1m_96d.fvecs",
            "array_shapes": {
                "neighbor_ids": list(neighbors.shape),
                "squared_l2_distances": list(distances.shape),
                "source_row_ids": list(source_neighbor_rows.shape),
            },
            "exact_computation_seconds": elapsed,
        }
        (case_out / "manifest.json").write_text(
            json.dumps(case_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries.append({"case": case, "seconds": elapsed, "truth_file": str(case_out / "groundtruth.npz")})
        print(f"completed exact truth: {case}; seconds={elapsed:.1f}; file={case_out / 'groundtruth.npz'}", flush=True)

    overall = {
        "data_root": str(root),
        "method": "exact stage-wise top-10 ground truth",
        "metric": "squared_l2",
        "all_cases": list(CASES),
        "queries_per_stage": query_count,
        "stages_per_case": steps + 1,
        "truth_ids_are_stream_row_ids": True,
        "source_row_ids_also_saved": True,
        "cases": summaries,
    }
    (truth_root / "manifest.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    data_manifest["ground_truth_computed"] = True
    data_manifest["ground_truth_path"] = "truth"
    data_manifest["ground_truth_metric"] = "squared_l2"
    (root / "manifest.json").write_text(
        json.dumps(data_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"ALL GROUND TRUTH COMPLETE: {truth_root}", flush=True)


if __name__ == "__main__":
    main()
