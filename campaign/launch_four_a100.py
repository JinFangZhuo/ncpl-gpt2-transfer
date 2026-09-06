#!/usr/bin/env python3
"""Supervise two 2-GPU DDP workers for the frozen 20-config campaign."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "candidate_manifest.json"
RESULTS = HERE / "results.tsv"
STOP = HERE / "STOP"
CHILDREN: dict[int, subprocess.Popen] = {}
STOPPING = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu-pairs",
        default="0,1;2,3",
        help="semicolon-separated 2-GPU worker assignments",
    )
    parser.add_argument(
        "--acknowledge-exclusive-campaign",
        action="store_true",
        help="confirm no other server can claim the same unfinished candidates",
    )
    return parser.parse_args()


def parse_pairs(raw: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for encoded in raw.split(";"):
        values = tuple(int(value.strip()) for value in encoded.split(","))
        if len(values) != 2:
            raise ValueError(f"each worker must receive exactly two GPUs: {encoded!r}")
        pairs.append(values)
    flattened = [gpu for pair in pairs for gpu in pair]
    if not pairs or len(flattened) != len(set(flattened)):
        raise ValueError("GPU pairs must be non-empty and disjoint")
    return pairs


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def result_ids() -> set[str]:
    if not RESULTS.exists():
        return set()
    with RESULTS.open(newline="", encoding="utf-8") as handle:
        return {row["candidate_id"] for row in csv.DictReader(handle, delimiter="\t")}


def spawn_worker(index: int, pair: tuple[int, int], count: int) -> subprocess.Popen:
    environment = os.environ.copy()
    environment.update(
        {
            "GPT2_CAMPAIGN_GPUS": ",".join(map(str, pair)),
            "GPT2_CAMPAIGN_SHARD_INDEX": str(index),
            "GPT2_CAMPAIGN_SHARD_COUNT": str(count),
            "PYTHONUNBUFFERED": "1",
        }
    )
    child = subprocess.Popen(
        [sys.executable, str(HERE / "controller.py")],
        cwd=HERE,
        env=environment,
        start_new_session=True,
    )
    CHILDREN[index] = child
    return child


def stop_children() -> None:
    for child in CHILDREN.values():
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def signal_handler(_signum: int, _frame: object) -> None:
    global STOPPING
    STOPPING = True
    stop_children()


def main() -> int:
    args = parse_args()
    if not args.acknowledge_exclusive_campaign:
        raise SystemExit(
            "Refusing to launch: first stop/partition every other controller, then "
            "pass --acknowledge-exclusive-campaign"
        )
    pairs = parse_pairs(args.gpu_pairs)
    payload = manifest()
    if payload.get("candidate_count") != 20:
        raise RuntimeError("expected the frozen 20-candidate manifest")
    if STOP.exists():
        raise SystemExit(f"remove {STOP} before launching")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    baseline = payload["baseline_candidate_id"]

    spawn_worker(0, pairs[0], len(pairs))
    while baseline not in result_ids() and not STOPPING and not STOP.exists():
        child = CHILDREN[0]
        if child.poll() is not None:
            if child.returncode == 0:
                return 0
            spawn_worker(0, pairs[0], len(pairs))
        time.sleep(15)

    if baseline not in result_ids():
        stop_children()
        return 1
    for index, pair in enumerate(pairs[1:], start=1):
        spawn_worker(index, pair, len(pairs))

    expected = int(payload["candidate_count"])
    while not STOPPING and not STOP.exists():
        if len(result_ids()) == expected:
            stop_children()
            return 0
        for index, pair in enumerate(pairs):
            child = CHILDREN[index]
            if child.poll() is not None and child.returncode != 0:
                spawn_worker(index, pair, len(pairs))
        time.sleep(30)
    stop_children()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
