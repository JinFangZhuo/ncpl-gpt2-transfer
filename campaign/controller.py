#!/usr/bin/env python3
"""Finite restart-safe controller for the 20-config GPT-2 campaign."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE_CONTROLLER = HERE / "base_controller.py"
spec = importlib.util.spec_from_file_location("gpt2_new20_base_controller", BASE_CONTROLLER)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot import base controller: {BASE_CONTROLLER}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

base.HERE = HERE
base.BUILDER = HERE / "build_candidates.py"
base.MANIFEST = HERE / "candidate_manifest.json"
base.PREDICTIONS = HERE / "ncpl_predictions.json"
base.RESULTS = HERE / "results.tsv"
base.RUNS = HERE / "runs"
base.DECISIONS = HERE / "decisions"
base.PREDICTION_LOGS = HERE / "prediction_logs"
base.STOP_FILE = HERE / "STOP"
base.RESULTS_LOCK = HERE / "results.lock"

gpu_list = os.environ.get("GPT2_CAMPAIGN_GPUS")
gpus = (
    tuple(int(value.strip()) for value in gpu_list.split(",") if value.strip())
    if gpu_list
    else (
        (int(os.environ["GPT2_CAMPAIGN_GPU"]),)
        if "GPT2_CAMPAIGN_GPU" in os.environ
        else (0, 1)
    )
)
if not gpus or len(set(gpus)) != len(gpus):
    raise ValueError(f"invalid GPU assignment: {gpus}")
shard_count = int(os.environ.get("GPT2_CAMPAIGN_SHARD_COUNT", "1"))
shard_index = int(os.environ.get("GPT2_CAMPAIGN_SHARD_INDEX", "0"))
if shard_count < 1 or not 0 <= shard_index < shard_count:
    raise ValueError(
        f"invalid shard assignment: index={shard_index}, count={shard_count}"
    )
suffix = f"_shard{shard_index}" if shard_count > 1 else ""
base.FAILURES = HERE / f"failures{suffix}.jsonl"
base.STATE = HERE / f"controller_state{suffix}.json"
base.SUMMARY = HERE / f"summary{suffix}.json"
base.CONTROLLER_LOCK = HERE / f"controller{suffix}.lock"
base.TARGET_GPUS = gpus
base.CANDIDATE_SHARD_COUNT = shard_count
base.CANDIDATE_SHARD_INDEX = shard_index
base.DISTRIBUTED_BACKEND = "nccl"
base.NCPL_PREDICTION_DEVICE = "cpu"
base.UPDATES = 5120
base.TRAINING_SEED = 0
base.CANDIDATE_ORDER_BY_NCPL_PREDICTION = False
base.CAMPAIGN_MAINTENANCE_LEADER = shard_index == 0
base.EXIT_WHEN_QUEUE_EMPTY = True
base.RESULT_FIELDS = base.RESULT_FIELDS[:-1] + ("training_seed", "config_json")


if __name__ == "__main__":
    try:
        raise SystemExit(base.main())
    except Exception as error:
        base.atomic_json(
            base.STATE,
            {"status": "controller_crashed", "updated_at": base.now(), "error": repr(error)},
        )
        raise
