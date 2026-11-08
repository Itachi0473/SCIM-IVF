#!/usr/bin/env python3
"""Build nine FreewayML-style controlled drift streams from DEEP-1M fvecs.

The source base rows are all used exactly once, with a seeded permutation. Each
scenario stores a 1M-row, unit-normalized stream and 101 stage-specific query
files. No ground truth or index is computed here.
"""

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


CASES = ("G", "A1", "A2", "B1", "B2", "B3", "B4", "C1", "C2")
DIMENSION = 96


def fvecs_mmap(path: Path, dimension: int) -> np.ndarray:
    row_bytes = (dimension + 1) * 4
    if path.stat().st_size == 0 or path.stat().st_size % row_bytes:
        raise ValueError(f"invalid fvecs length for {dimension}D: {path}")
    rows = path.stat().st_size // row_bytes
    packed = np.memmap(path, dtype="<i4", mode="r", shape=(rows, dimension + 1))
    if not np.all(packed[:, 0] == dimension):
        raise ValueError(f"unexpected vector dimension in {path}")
    return packed[:, 1:].view("<f4")


def coordinate_sigma(vectors: np.ndarray, block_rows: int) -> float:
    sums = np.zeros(vectors.shape[1], dtype=np.float64)
    sums_sq = np.zeros(vectors.shape[1], dtype=np.float64)
    for begin in range(0, vectors.shape[0], block_rows):
        block = np.asarray(vectors[begin:begin + block_rows], dtype=np.float64)
        sums += block.sum(axis=0)
        sums_sq += np.square(block).sum(axis=0)
    mean = sums / vectors.shape[0]
    variance = np.maximum(sums_sq / vectors.shape[0] - mean * mean, 0.0)
    return float(np.sqrt(variance).mean())


def unit_direction(seed: int, dimension: int) -> np.ndarray:
    direction = np.random.default_rng(seed).normal(size=dimension).astype(np.float32)
    direction /= np.linalg.norm(direction)
    return direction


def write_fbin(path: Path, vectors: np.ndarray) -> None:
    vectors = np.ascontiguousarray(vectors, dtype="<f4")
    with path.open("xb") as stream:
        stream.write(struct.pack("<II", *vectors.shape))
        vectors.tofile(stream)


def normalize_rows(vectors: np.ndarray, where: str) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or not np.isfinite(norms).all():
        raise ValueError(f"non-finite values in {where}")
    if np.any(norms == 0):
        raise ValueError(f"zero vector in {where}")
    vectors /= norms
    return vectors


def shift_tables(case: str, insert_rows: int, steps: int, batch_size: int,
                 dimension: int, small: float, large: float, seed: int):
    direction = unit_direction(seed + 1, dimension)
    blocks_5k = math.ceil(insert_rows / 5000)
    blocks_30k = math.ceil(insert_rows / 30000)
    zero = np.zeros(dimension, dtype=np.float32)

    if case == "B1":
        base = np.stack([(i + 1) * small * direction for i in range(blocks_5k)])
    elif case == "B2":
        increments = np.stack([
            small * unit_direction(seed + 100 + i, dimension)
            for i in range(blocks_5k)
        ])
        base = np.cumsum(increments, axis=0, dtype=np.float32)
    elif case in ("B3", "B4"):
        values = []
        accumulated = zero.copy()
        for i in range(blocks_30k):
            step = large * unit_direction(seed + 200 + i, dimension)
            accumulated = step if i == 0 else accumulated + step
            values.append(accumulated.copy())
        base = np.stack(values)
    elif case in ("C1", "C2"):
        magnitude = small if case == "C1" else large
        base = np.stack([
            magnitude * direction if i % 2 == 0 else zero
            for i in range(blocks_5k)
        ])
    else:
        base = np.zeros((blocks_5k, dimension), dtype=np.float32)

    if case in ("B1", "B2", "B3"):
        query = np.repeat(base[-1:, :], steps + 1, axis=0)
    elif case == "A1":
        query = np.zeros((steps + 1, dimension), dtype=np.float32)
        query[:, 0] = small * np.arange(steps + 1, dtype=np.float32) / steps
    elif case in ("C1", "C2"):
        magnitude = small if case == "C1" else large
        query = np.zeros((steps + 1, dimension), dtype=np.float32)
        for stage in range(1, steps + 1):
            if ((stage * batch_size - 1) // 5000) % 2 == 0:
                query[stage] = magnitude * direction
    else:
        query = np.zeros((steps + 1, dimension), dtype=np.float32)
    return base, query


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-fvecs", type=Path, required=True)
    parser.add_argument("--query-fvecs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--initial", type=int, default=500_000)
    parser.add_argument("--batch-size", type=int, default=5_000)
    parser.add_argument("--queries", type=int, default=1_000)
    parser.add_argument("--small-sigma", type=float, default=2.0)
    parser.add_argument("--large-sigma", type=float, default=8.0)
    parser.add_argument("--block-rows", type=int, default=8192)
    args = parser.parse_args()

    if args.initial <= 0 or args.batch_size <= 0 or args.queries <= 0 or args.block_rows <= 0:
        parser.error("initial, batch-size, queries, and block-rows must be positive")
    if args.initial % args.batch_size != 0:
        parser.error("initial must be divisible by batch-size for equal-size stages")
    if args.small_sigma <= 0 or args.large_sigma <= args.small_sigma:
        parser.error("require 0 < small-sigma < large-sigma")
    if args.batch_size != 5000:
        parser.error("this construction keeps the defined 5K drift period; use batch-size=5000")

    base_source = fvecs_mmap(args.base_fvecs, DIMENSION)
    query_source = fvecs_mmap(args.query_fvecs, DIMENSION)
    if args.initial >= base_source.shape[0]:
        parser.error("initial must leave at least one insertion batch")
    if args.queries > query_source.shape[0]:
        parser.error("requested query count exceeds query file rows")
    total_rows = base_source.shape[0]
    insert_rows = total_rows - args.initial
    steps = insert_rows // args.batch_size
    if insert_rows % args.batch_size:
        parser.error("remaining DEEP-1M vectors must divide evenly into 5K batches")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error(f"output directory must be absent or empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    sigma = coordinate_sigma(base_source, args.block_rows)
    small = sigma * args.small_sigma
    large = sigma * args.large_sigma
    order = np.random.default_rng(args.seed).permutation(total_rows).astype("<i4")
    np.save(args.output / "base_source_rows.npy", order)
    query_ids = np.arange(args.queries, dtype="<i4")
    np.save(args.output / "query_file_row_ids.npy", query_ids)
    raw_queries = np.asarray(query_source[:args.queries], dtype=np.float32).copy()
    if not np.isfinite(raw_queries).all() or np.any(np.linalg.norm(raw_queries, axis=1) == 0):
        parser.error("query source contains non-finite or zero vectors")

    for case in CASES:
        case_dir = args.output / case
        case_dir.mkdir()
        per_block, query_offsets = shift_tables(
            case, insert_rows, steps, args.batch_size, DIMENSION, small, large, args.seed)
        base_path = case_dir / "base.fbin"
        with base_path.open("xb") as output:
            output.write(struct.pack("<II", total_rows, DIMENSION))
            for begin in range(0, total_rows, args.block_rows):
                end = min(begin + args.block_rows, total_rows)
                positions = np.arange(begin, end)
                source_ids = order[begin:end]
                block = np.asarray(base_source[source_ids], dtype=np.float32).copy()
                insert_mask = positions >= args.initial
                if np.any(insert_mask):
                    relative = positions[insert_mask] - args.initial
                    period = 30000 if case in ("B3", "B4") else 5000
                    phase = np.minimum(relative // period, len(per_block) - 1)
                    block[insert_mask] += per_block[phase]
                normalize_rows(block, f"{case} base rows {begin}:{end}")
                np.asarray(block, dtype="<f4").tofile(output)

        query_shift_norms = []
        for stage in range(steps + 1):
            queries = raw_queries + query_offsets[stage]
            if case == "A2":
                noise_std = small * stage / steps / math.sqrt(DIMENSION)
                noise = np.random.default_rng(args.seed + 5000 + stage).normal(
                    0, noise_std, queries.shape).astype(np.float32)
                queries += noise
            normalize_rows(queries, f"{case} query stage {stage}")
            write_fbin(case_dir / f"query_{stage:03d}.fbin", queries)
            query_shift_norms.append(float(np.linalg.norm(query_offsets[stage])))
            if stage % 10 == 0 or stage == steps:
                print(f"{case}: wrote query stage {stage}/{steps}", flush=True)

        metadata = {
            "case": case,
            "dataset": "DEEP-1M prefix from DEEP1B ANN-Benchmarks subset",
            "base_source": str(args.base_fvecs.resolve()),
            "query_source": str(args.query_fvecs.resolve()),
            "dimension": DIMENSION,
            "base_count": total_rows,
            "query_count": args.queries,
            "initial_count": args.initial,
            "inserted_count": insert_rows,
            "batch_size": args.batch_size,
            "steps": steps,
            "seed": args.seed,
            "base_source_rows_file": "../base_source_rows.npy",
            "query_file_row_ids_file": "../query_file_row_ids.npy",
            "base_rows_used_exactly_once": True,
            "metric": "euclidean on independently unit-normalized shifted vectors; ranking matches cosine on shifted vectors",
            "source_vectors_unit_norm": True,
            "post_shift_vectors_unit_norm": True,
            "coordinate_sigma_mean": sigma,
            "small_shift_l2": small,
            "large_shift_l2": large,
            "small_sigma": args.small_sigma,
            "large_sigma": args.large_sigma,
            "base_shift_l2_by_drift_block": [float(np.linalg.norm(x)) for x in per_block],
            "query_shift_l2_by_stage": query_shift_norms,
            "interpretation": "controlled FreewayML-style scenario; not official FreewayML source data or exact reproduction",
        }
        (case_dir / "manifest.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"prepared {case}: {case_dir} ({total_rows} base rows, {steps} batches)", flush=True)

    root_manifest = {
        "experiment": "DEEP-1M nine controlled drift scenarios",
        "cases": list(CASES),
        "seed": args.seed,
        "base_count": total_rows,
        "initial_count": args.initial,
        "inserted_count": insert_rows,
        "batch_size": args.batch_size,
        "steps": steps,
        "query_count": args.queries,
        "dimension": DIMENSION,
        "metric": "squared L2 on unit-normalized post-shift vectors",
        "ground_truth_computed": False,
        "scenario_notes": {
            "G": "no drift",
            "A1": "query-only gradual translation in coordinate 0",
            "A2": "query-only gradually increasing Gaussian noise",
            "B1": "insert-only cumulative shift along a fixed direction every 5K rows; queries use final shifted distribution",
            "B2": "insert-only cumulative random-direction shift every 5K rows; queries use final shifted distribution",
            "B3": "insert-only cumulative random-direction shift every 30K rows; queries use final shifted distribution",
            "B4": "same inserted vectors as B3, but queries stay in the original distribution",
            "C1": "small insert/query shift alternates on/off every 5K rows",
            "C2": "large insert/query shift alternates on/off every 5K rows",
        },
        "interpretation": "paper-inspired controlled scenarios, not official FreewayML data",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(root_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"COMPLETE: {args.output}", flush=True)


if __name__ == "__main__":
    main()
