"""Export the prepared DEEP-1M stage truths to a compact C++-readable format."""

import argparse
import json
import struct
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "build" / "experiment_data" / "deep1m_freewayml_9scenarios_seed1234"
CASES = ("G", "A1", "A2", "B1", "B2", "B3", "B4", "C1", "C2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    (output_dir / "_prepared_truth").mkdir(parents=True, exist_ok=True)
    root_manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8"))
    if tuple(root_manifest["cases"]) != CASES or not root_manifest.get("ground_truth_computed"):
        raise ValueError("expected the prepared nine-case DEEP-1M ground-truth set")
    for case in CASES:
        case_manifest = json.loads((data_root / case / "manifest.json").read_text(encoding="utf-8"))
        path = data_root / "truth" / case / "groundtruth.npz"
        with np.load(path) as f:
            ids = np.asarray(f["neighbor_ids"], dtype="<u4")
            active = np.asarray(f["active_counts"], dtype=np.uint32)
        expected = (int(case_manifest["steps"]) + 1, int(case_manifest["query_count"]), 10)
        if ids.shape != expected or not np.array_equal(
                active, int(case_manifest["initial_count"]) + np.arange(expected[0]) * int(case_manifest["batch_size"])):
            raise ValueError(f"unexpected truth shape or active counts for {case}")
        for stage, count in enumerate(active):
            if np.any(ids[stage] >= count):
                raise ValueError(f"truth contains an out-of-prefix ID for {case} stage {stage}")
        target = output_dir / "_prepared_truth" / f"{case}.u32bin"
        with target.open("xb") as stream:
            stream.write(b"DLQTRTH1")
            stream.write(struct.pack("<III", *ids.shape))
            stream.write(ids.tobytes(order="C"))
        print(f"prepared {case}: {ids.shape} -> {target}")


if __name__ == "__main__":
    main()
