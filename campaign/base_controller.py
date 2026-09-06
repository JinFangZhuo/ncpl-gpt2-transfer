#!/usr/bin/env python3
"""Persistent, restart-safe controller for the frozen NCPL-to-GPT-2 campaign."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import signal
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parent
RMNP_REPO = Path(
    os.environ.get("RMNP_REPO", str(PACKAGE_ROOT / "RMNP"))
).expanduser().resolve()
GPT2_ROOT = RMNP_REPO / "GPT-2"
FARMS_ROOT = Path(
    os.environ.get("FARMS_ROOT", str(PACKAGE_ROOT.parent))
).expanduser().resolve()
TRAINER = GPT2_ROOT / "RMNP/train_adamw_streaming.py"
MODEL = GPT2_ROOT / "RMNP/model.py"
TORCHRUN = Path(
    os.environ.get(
        "TORCHRUN",
        shutil.which("torchrun") or str(Path(sys.executable).with_name("torchrun")),
    )
)
TRAIN_PYTHON = Path(os.environ.get("TRAIN_PYTHON", sys.executable))
NCPL_PYTHON = Path(os.environ.get("NCPL_PYTHON", sys.executable))
NCPL_PREDICTOR = FARMS_ROOT / "ncpl_marin_aligned/predict_candidates.py"
NCPL_CHECKPOINT = Path(
    os.environ.get("NCPL_CHECKPOINT", str(PACKAGE_ROOT / "NCPL-final"))
).expanduser().resolve()
OFFICIAL_NCPL_REPO = Path(
    os.environ.get("OFFICIAL_NCPL_REPO", str(PACKAGE_ROOT / "NCPL"))
).expanduser().resolve()
BUILDER = HERE / "build_candidates.py"
MANIFEST = HERE / "candidate_manifest.json"
PREDICTIONS = HERE / "ncpl_predictions.json"
RESULTS = HERE / "results.tsv"
FAILURES = HERE / "failures.jsonl"
STATE = HERE / "controller_state.json"
SUMMARY = HERE / "summary.json"
RUNS = HERE / "runs"
DECISIONS = HERE / "decisions"
PREDICTION_LOGS = HERE / "prediction_logs"
STOP_FILE = HERE / "STOP"
TASK_DEPENDENCIES = Path(
    os.environ.get("GPT2_EXTRA_PYTHONPATH", str(RMNP_REPO / ".local_deps"))
).expanduser().resolve()

TARGET_GPUS = (6, 7)
DISTRIBUTED_BACKEND = "gloo"
NCPL_PREDICTION_DEVICE = "cuda:0"
UPDATES = 5120
TRAINING_SEED = 0
GLOBAL_BATCH = 128
SEQUENCE_LEN = 4096
STALL_SECONDS = 4 * 60 * 60
EXPANSION_SIZE = 28
BASELINE_RETRY_SECONDS = 300
NONBASELINE_RETRY_SECONDS = 60
PREDICTION_RETRY_SECONDS = 300

# Optional queue sharding lets multiple GPU workers consume disjoint candidates
# from one scientific campaign.  The defaults preserve the original single-
# controller behavior.  Campaign wrappers may override these values after
# importing this module.
CANDIDATE_SHARD_COUNT = 1
CANDIDATE_SHARD_INDEX = 0
CANDIDATE_REQUIRED_ORIGIN: str | None = None
CANDIDATE_ORDER_BY_NCPL_PREDICTION = False
CAMPAIGN_MAINTENANCE_LEADER = True
EXIT_WHEN_QUEUE_EMPTY = False
CONTROLLER_LOCK: Path | None = None
RESULTS_LOCK: Path | None = None

RESULT_FIELDS = (
    "candidate_index",
    "candidate_id",
    "label",
    "origin",
    "varied_field",
    "predicted_final_loss",
    "predicted_delta_vs_baseline",
    "validation_loss",
    "actual_delta_vs_baseline",
    "predicted_direction_correct",
    "train_loss",
    "optimizer_updates",
    "training_elapsed_seconds",
    "peak_memory_allocated_mb_rank0",
    "attempt",
    "launcher_exit_code",
    "completed_at",
    "manifest_sha256",
    "predictions_sha256",
    "config_json",
)

ACTIVE_CHILD: subprocess.Popen[str] | None = None
INTERRUPTED = False


def world_size() -> int:
    return len(TARGET_GPUS)


def gradient_accumulation_steps() -> int:
    if GLOBAL_BATCH % world_size() != 0:
        raise RuntimeError(
            f"global batch {GLOBAL_BATCH} is not divisible by world size {world_size()}"
        )
    return GLOBAL_BATCH // world_size()


def visible_devices() -> str:
    return ",".join(str(index) for index in TARGET_GPUS)


def candidate_assigned_to_this_worker(
    index: int, candidate: dict[str, Any] | None = None
) -> bool:
    if CANDIDATE_SHARD_COUNT < 1:
        raise RuntimeError("CANDIDATE_SHARD_COUNT must be positive")
    if not 0 <= CANDIDATE_SHARD_INDEX < CANDIDATE_SHARD_COUNT:
        raise RuntimeError(
            "CANDIDATE_SHARD_INDEX must be in "
            f"[0, {CANDIDATE_SHARD_COUNT}), got {CANDIDATE_SHARD_INDEX}"
        )
    if index % CANDIDATE_SHARD_COUNT != CANDIDATE_SHARD_INDEX:
        return False
    if CANDIDATE_REQUIRED_ORIGIN is not None:
        if candidate is None or candidate.get("origin") != CANDIDATE_REQUIRED_ORIGIN:
            return False
    return True


def controller_lock_path() -> Path:
    return CONTROLLER_LOCK if CONTROLLER_LOCK is not None else HERE / "controller.lock"


def results_lock_path() -> Path:
    return RESULTS_LOCK if RESULTS_LOCK is not None else RESULTS.with_suffix(".lock")


def now() -> str:
    return datetime.now().astimezone().isoformat()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, value: Any) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def stop_requested() -> bool:
    return INTERRUPTED or STOP_FILE.exists()


def interruptible_sleep(seconds: int, status: dict[str, Any]) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not stop_requested():
        remaining = max(0, int(deadline - time.monotonic()))
        atomic_json(
            STATE,
            {
                **status,
                "updated_at": now(),
                "retry_in_seconds": remaining,
            },
        )
        time.sleep(min(30, max(1, remaining)))


def result_rows() -> list[dict[str, str]]:
    if not RESULTS.exists():
        return []
    with RESULTS.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def append_result(row: dict[str, Any]) -> None:
    lock_path = results_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # A second worker may have completed the same recovered attempt while
        # this worker was waiting for the lock.  Keep one scientific result row
        # per candidate even in that recovery race.
        candidate_id = str(row.get("candidate_id", ""))
        if candidate_id and any(
            existing.get("candidate_id") == candidate_id for existing in result_rows()
        ):
            return
        first = not RESULTS.exists()
        with RESULTS.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, delimiter="\t")
            if first:
                writer.writeheader()
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})
            handle.flush()
            os.fsync(handle.fileno())


def prediction_binding_is_current() -> bool:
    if not MANIFEST.is_file() or not PREDICTIONS.is_file():
        return False
    try:
        manifest_raw = MANIFEST.read_bytes()
        manifest = json.loads(manifest_raw)
        predictions = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if predictions.get("candidate_manifest_sha256") != hashlib.sha256(manifest_raw).hexdigest():
        return False
    if predictions.get("candidate_count") != manifest.get("candidate_count"):
        return False
    manifest_ids = {row["candidate_id"] for row in manifest["candidates"]}
    predicted_ids = {row["candidate_id"] for row in predictions.get("predictions", [])}
    return manifest_ids == predicted_ids


def load_inputs() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not prediction_binding_is_current():
        raise RuntimeError("NCPL predictions are absent or not bound to the current manifest")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    predictions = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    by_id = {row["candidate_id"]: row for row in predictions["predictions"]}
    return manifest, by_id


def gpu_snapshot() -> dict[str, Any]:
    gpu_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    gpus = {}
    for line in gpu_query.stdout.splitlines():
        index, uuid, memory, utilization = [part.strip() for part in line.split(",")]
        gpus[int(index)] = {
            "uuid": uuid,
            "memory_used_mib": int(memory),
            "utilization_percent": int(utilization),
            "compute_pids": [],
        }
    app_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    by_uuid = {row["uuid"]: row for row in gpus.values()}
    for line in app_query.stdout.splitlines():
        if not line.strip():
            continue
        uuid, pid, process_name, memory = [part.strip() for part in line.split(",", 3)]
        if uuid in by_uuid:
            by_uuid[uuid]["compute_pids"].append(
                {"pid": int(pid), "process_name": process_name, "memory_used_mib": int(memory)}
            )
    return {str(index): gpus[index] for index in TARGET_GPUS}


def target_gpus_free() -> tuple[bool, dict[str, Any]]:
    try:
        snapshot = gpu_snapshot()
    except Exception as error:
        return False, {"error": repr(error)}
    free = all(
        not row["compute_pids"] and row["memory_used_mib"] <= 512
        for row in snapshot.values()
    )
    return free, snapshot


def wait_for_free_gpus(reason: str) -> bool:
    while not stop_requested():
        free, snapshot = target_gpus_free()
        if free:
            return True
        atomic_json(
            STATE,
            {
                "status": "waiting_for_target_gpus",
                "updated_at": now(),
                "reason": reason,
                "gpu_snapshot": snapshot,
            },
        )
        time.sleep(30)
    return False


def training_command(config: dict[str, Any], attempt_dir: Path) -> list[str]:
    final_metrics = attempt_dir / "final_metrics.json"
    out_dir = attempt_dir / "no_checkpoints"
    return [
        str(TORCHRUN),
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={world_size()}",
        str(TRAINER),
        "--dataset=openwebtext",
        "--use_streaming=True",
        "--streaming_dataset=Skylion007/openwebtext",
        "--streaming_timeout=7200",
        "--streaming_max_retries=10",
        "--init_from=scratch",
        "--n_layer=32",
        "--n_head=8",
        "--n_embd=512",
        "--n_inner=3072",
        "--block_size=4096",
        "--bias=False",
        "--split_qkv=False",
        "--scale_attn_by_inverse_layer_idx=False",
        "--dropout=0.0",
        "--batch_size=1",
        f"--gradient_accumulation_steps={gradient_accumulation_steps()}",
        f"--max_iters={UPDATES}",
        f"--lr_decay_iters={UPDATES - 1}",
        f"--warmup_iters={config['warmup_steps']}",
        "--optimizer_name=adamw",
        f"--learning_rate={config['learning_rate']}",
        f"--weight_decay={config['weight_decay']}",
        f"--beta1={config['beta1']}",
        f"--beta2={config['beta2']}",
        f"--epsilon={10 ** -config['epsilon_exponent']}",
        f"--grad_clip={config['max_grad_norm']}",
        f"--min_lr={config['min_lr']}",
        "--schedule=cosine",
        "--decay_lr=True",
        "--eval_interval=1000000",
        "--eval_iters=1",
        "--eval_at_start=False",
        "--log_interval=20",
        "--wandb_log=False",
        "--always_save_checkpoint=False",
        "--save_checkpoints=False",
        "--exact_num_steps=True",
        "--final_eval_iters=20",
        f"--final_metrics_path={final_metrics}",
        f"--out_dir={out_dir}",
        "--attention_backend=sdpa",
        f"--backend={DISTRIBUTED_BACKEND}",
        "--device=cuda",
        "--dtype=bfloat16",
        "--compile=True",
        f"--seed={TRAINING_SEED}",
    ]


def training_environment() -> dict[str, str]:
    env = os.environ.copy()
    cache_root = Path(
        env.get(
            "GPT2_CACHE_ROOT",
            str(Path(tempfile.gettempdir()) / "gpt2_ncpl_transfer_cache"),
        )
    ).expanduser()
    env.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": visible_devices(),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": str(cache_root / "huggingface"),
            "HF_DATASETS_CACHE": str(cache_root / "huggingface/datasets"),
            "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "torchinductor"),
            "OMP_NUM_THREADS": "8",
        }
    )
    if TASK_DEPENDENCIES.is_dir():
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(TASK_DEPENDENCIES) + (f":{existing}" if existing else "")
    return env


def valid_metrics(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        metrics = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if metrics.get("optimizer_updates") != UPDATES or metrics.get("final_step_index") != UPDATES - 1:
        return None
    for key in ("validation_loss", "train_loss", "training_elapsed_seconds"):
        if not isinstance(metrics.get(key), (int, float)) or not math.isfinite(metrics[key]):
            return None
    config = metrics.get("config", {})
    required = {
        "n_layer": 32,
        "n_head": 8,
        "n_embd": 512,
        "n_inner": 3072,
        "block_size": SEQUENCE_LEN,
        "batch_size": 1,
        "gradient_accumulation_steps": gradient_accumulation_steps(),
        "max_iters": UPDATES,
        "seed": TRAINING_SEED,
    }
    if any(config.get(field) != expected for field, expected in required.items()):
        return None
    return metrics


def recovered_attempt(candidate_id: str) -> tuple[int, dict[str, Any]] | None:
    run_dir = RUNS / candidate_id
    for attempt_dir in sorted(run_dir.glob("attempt_*"), reverse=True):
        metrics = valid_metrics(attempt_dir / "final_metrics.json")
        if metrics is not None:
            return int(attempt_dir.name.rsplit("_", 1)[-1]), metrics
    return None


def next_attempt(candidate_id: str) -> tuple[int, Path]:
    run_dir = RUNS / candidate_id
    prior = [int(path.name.rsplit("_", 1)[-1]) for path in run_dir.glob("attempt_*")]
    attempt = max(prior, default=0) + 1
    attempt_dir = run_dir / f"attempt_{attempt:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    return attempt, attempt_dir


def terminate_process_group(child: subprocess.Popen[str], grace_seconds: int = 60) -> None:
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=grace_seconds)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def signal_handler(signum: int, _frame: Any) -> None:
    global INTERRUPTED, ACTIVE_CHILD
    INTERRUPTED = True
    child = ACTIVE_CHILD
    if child is not None:
        terminate_process_group(child)
    atomic_json(
        STATE,
        {
            "status": "interrupted",
            "updated_at": now(),
            "signal": signum,
            "will_restart_unless_stop_file_exists": not STOP_FILE.exists(),
        },
    )


def run_child_with_heartbeat(
    command: list[str], log_path: Path, state: dict[str, Any], env: dict[str, str]
) -> tuple[int, bool]:
    global ACTIVE_CHILD
    started = time.time()
    last_state = 0.0
    stalled = False
    with log_path.open("w", encoding="utf-8") as log:
        ACTIVE_CHILD = subprocess.Popen(
            command,
            cwd=GPT2_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        while ACTIVE_CHILD.poll() is None and not stop_requested():
            time.sleep(30)
            current = time.time()
            if current - last_state >= 300:
                mtime = log_path.stat().st_mtime if log_path.exists() else started
                atomic_json(
                    STATE,
                    {
                        **state,
                        "status": "training",
                        "updated_at": now(),
                        "launcher_pid": ACTIVE_CHILD.pid,
                        "elapsed_seconds": current - started,
                        "seconds_since_log_output": current - mtime,
                        "train_log": str(log_path),
                    },
                )
                last_state = current
            mtime = log_path.stat().st_mtime if log_path.exists() else started
            if current - mtime > STALL_SECONDS:
                stalled = True
                terminate_process_group(ACTIVE_CHILD)
                break
        if stop_requested() and ACTIVE_CHILD.poll() is None:
            terminate_process_group(ACTIVE_CHILD)
        exit_code = ACTIVE_CHILD.wait()
    ACTIVE_CHILD = None
    return exit_code, stalled


def record_result(
    index: int,
    candidate: dict[str, Any],
    prediction: dict[str, Any],
    baseline_prediction: float,
    metrics: dict[str, Any],
    attempt: int,
    exit_code: int | str,
) -> None:
    prior = result_rows()
    baseline_id = json.loads(MANIFEST.read_text(encoding="utf-8"))["baseline_candidate_id"]
    predicted_loss = float(prediction["predicted_final_loss"])
    predicted_delta = predicted_loss - baseline_prediction
    if candidate["candidate_id"] == baseline_id:
        actual_delta = 0.0
        direction_correct: bool | str = ""
    else:
        baseline = next((row for row in prior if row["candidate_id"] == baseline_id), None)
        if baseline is None:
            raise RuntimeError("baseline must complete before other candidates are recorded")
        actual_delta = float(metrics["validation_loss"]) - float(baseline["validation_loss"])
        direction_correct = (predicted_delta < 0) == (actual_delta < 0)
    append_result(
        {
            "candidate_index": index,
            "candidate_id": candidate["candidate_id"],
            "label": candidate["label"],
            "origin": candidate["origin"],
            "varied_field": candidate["varied_field"],
            "predicted_final_loss": predicted_loss,
            "predicted_delta_vs_baseline": predicted_delta,
            "validation_loss": metrics["validation_loss"],
            "actual_delta_vs_baseline": actual_delta,
            "predicted_direction_correct": direction_correct,
            "train_loss": metrics["train_loss"],
            "optimizer_updates": metrics["optimizer_updates"],
            "training_elapsed_seconds": metrics["training_elapsed_seconds"],
            "peak_memory_allocated_mb_rank0": metrics.get("peak_memory_allocated_mb", ""),
            "attempt": attempt,
            "launcher_exit_code": exit_code,
            "completed_at": now(),
            "manifest_sha256": sha256(MANIFEST),
            "predictions_sha256": sha256(PREDICTIONS),
            "training_seed": TRAINING_SEED,
            "config_json": json.dumps(candidate["config"], sort_keys=True),
        }
    )


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mean_x, mean_y = statistics.mean(xs), statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs)
        * sum((y - mean_y) ** 2 for y in ys)
    )
    return numerator / denominator if denominator else None


def update_summary(manifest: dict[str, Any]) -> None:
    rows = result_rows()
    nonbaseline = [row for row in rows if float(row["actual_delta_vs_baseline"]) != 0.0]
    correct = sum(row["predicted_direction_correct"] == "True" for row in nonbaseline)
    payload: dict[str, Any] = {
        "updated_at": now(),
        "scope": "NCPL Marin-fixed-slice to matched-size GPT-2/OpenWebText generalization",
        "candidate_count": manifest["candidate_count"],
        "completed_count": len(rows),
        "pending_count": manifest["candidate_count"] - len(rows),
        "direction_audited_count": len(nonbaseline),
        "direction_correct_count": correct,
        "direction_accuracy": correct / len(nonbaseline) if nonbaseline else None,
        "continuous_expansion_enabled": not EXIT_WHEN_QUEUE_EMPTY,
        "physical_gpus": list(TARGET_GPUS),
        "model_alignment": manifest["model_alignment"],
        "runtime_contract": manifest["runtime_contract"],
    }
    if rows:
        payload["best_observed"] = min(
            (
                {
                    "candidate_id": row["candidate_id"],
                    "validation_loss": float(row["validation_loss"]),
                }
                for row in rows
            ),
            key=lambda row: row["validation_loss"],
        )
        payload["predicted_actual_loss_pearson"] = pearson(
            [float(row["predicted_final_loss"]) for row in rows],
            [float(row["validation_loss"]) for row in rows],
        )
        payload["predicted_actual_delta_pearson"] = pearson(
            [float(row["predicted_delta_vs_baseline"]) for row in rows],
            [float(row["actual_delta_vs_baseline"]) for row in rows],
        )
    atomic_json(SUMMARY, payload)


def refresh_predictions() -> bool:
    if not wait_for_free_gpus("NCPL prediction refresh"):
        return False
    PREDICTION_LOGS.mkdir(parents=True, exist_ok=True)
    sequence = len(list(PREDICTION_LOGS.glob("refresh_*.log"))) + 1
    log_path = PREDICTION_LOGS / f"refresh_{sequence:04d}.log"
    next_output = HERE / "ncpl_predictions.next.json"
    next_output.unlink(missing_ok=True)
    command = [
        str(NCPL_PYTHON),
        str(NCPL_PREDICTOR),
        str(MANIFEST),
        "--checkpoint",
        str(NCPL_CHECKPOINT),
        "--official-ncpl-repo",
        str(OFFICIAL_NCPL_REPO),
        "--output",
        str(next_output),
        "--device",
        NCPL_PREDICTION_DEVICE,
        "--dtype",
        "float32",
        "--batch-size",
        "4",
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": str(TARGET_GPUS[0]),
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    atomic_json(
        STATE,
        {
            "status": "refreshing_ncpl_predictions",
            "updated_at": now(),
            "candidate_count": json.loads(MANIFEST.read_text())["candidate_count"],
            "log": str(log_path),
        },
    )
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=FARMS_ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0 or not next_output.is_file():
        next_output.unlink(missing_ok=True)
        return False
    try:
        predicted = json.loads(next_output.read_text(encoding="utf-8"))
        expected_hash = sha256(MANIFEST)
        manifest_count = json.loads(MANIFEST.read_text(encoding="utf-8"))["candidate_count"]
        if predicted["candidate_manifest_sha256"] != expected_hash:
            raise ValueError("new predictions do not bind to manifest")
        if predicted["candidate_count"] != manifest_count:
            raise ValueError("new prediction count mismatch")
        if len(predicted["predictions"]) != manifest_count:
            raise ValueError("new prediction rows are incomplete")
    except Exception as error:
        append_jsonl(
            FAILURES,
            {"at": now(), "stage": "prediction_validation", "error": repr(error)},
        )
        next_output.unlink(missing_ok=True)
        return False
    os.replace(next_output, PREDICTIONS)
    return prediction_binding_is_current()


def ensure_predictions() -> bool:
    # This published campaign is deliberately frozen.  Refusing to regenerate
    # predictions prevents a new machine or dependency version from silently
    # changing the pre-training decision record.
    return prediction_binding_is_current()


def extend_manifest() -> bool:
    if not wait_for_free_gpus("deterministic candidate expansion"):
        return False
    command = [str(TRAIN_PYTHON), str(BUILDER), "--append", str(EXPANSION_SIZE)]
    completed = subprocess.run(command, cwd=HERE, capture_output=True, text=True)
    if completed.returncode != 0:
        append_jsonl(
            FAILURES,
            {
                "at": now(),
                "stage": "candidate_expansion",
                "returncode": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-4000:],
            },
        )
        return False
    return ensure_predictions()


def run_candidate(
    index: int,
    candidate: dict[str, Any],
    prediction: dict[str, Any],
    baseline_prediction: float,
) -> bool:
    recovered = recovered_attempt(candidate["candidate_id"])
    if recovered is not None:
        attempt, metrics = recovered
        record_result(index, candidate, prediction, baseline_prediction, metrics, attempt, "recovered")
        return True
    if not wait_for_free_gpus(candidate["candidate_id"]):
        return False
    attempt, attempt_dir = next_attempt(candidate["candidate_id"])
    command = training_command(candidate["config"], attempt_dir)
    decision = {
        "created_at": now(),
        "candidate_index": index,
        "candidate": candidate,
        "prediction": prediction,
        "baseline_prediction": baseline_prediction,
        "attempt": attempt,
        "manifest_sha256": sha256(MANIFEST),
        "predictions_sha256": sha256(PREDICTIONS),
        "trainer_sha256": sha256(TRAINER),
        "model_sha256": sha256(MODEL),
        "command": command,
        "environment_contract": {
            "CUDA_VISIBLE_DEVICES": visible_devices(),
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    }
    DECISIONS.mkdir(exist_ok=True)
    atomic_json(DECISIONS / f"{index:06d}_{candidate['candidate_id']}.json", decision)
    atomic_json(attempt_dir / "effective_decision.json", decision)
    log_path = attempt_dir / "train.log"
    state = {
        "candidate_index": index,
        "candidate_id": candidate["candidate_id"],
        "label": candidate["label"],
        "attempt": attempt,
    }
    exit_code, stalled = run_child_with_heartbeat(
        command, log_path, state, training_environment()
    )
    metrics = valid_metrics(attempt_dir / "final_metrics.json")
    # Some local dataset/Arrow cleanup threads have aborted during interpreter
    # finalization after atomically writing valid metrics.  The metrics contract,
    # not the launcher exit status, determines scientific completion.
    if metrics is not None:
        record_result(
            index, candidate, prediction, baseline_prediction, metrics, attempt, exit_code
        )
        return True
    failure = {
        "at": now(),
        "stage": "training",
        "candidate_index": index,
        "candidate_id": candidate["candidate_id"],
        "attempt": attempt,
        "exit_code": exit_code,
        "stalled": stalled,
        "valid_final_metrics": False,
        "train_log": str(log_path),
    }
    append_jsonl(FAILURES, failure)
    atomic_json(STATE, {**failure, "status": "training_failed_will_retry"})
    return False


def static_check() -> dict[str, Any]:
    for required_path in (TORCHRUN, TRAINER, MODEL, MANIFEST, PREDICTIONS):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("candidate_count") != 20:
        raise RuntimeError("expected the frozen 20-candidate manifest")
    if not prediction_binding_is_current():
        raise RuntimeError("frozen NCPL predictions do not match the manifest")
    for candidate in manifest["candidates"]:
        command = training_command(candidate["config"], Path("/tmp/ncpl_gpt2_check"))
        required = {
            "--n_layer=32",
            "--n_head=8",
            "--n_embd=512",
            "--n_inner=3072",
            "--block_size=4096",
            "--batch_size=1",
            f"--gradient_accumulation_steps={gradient_accumulation_steps()}",
            f"--max_iters={UPDATES}",
            "--exact_num_steps=True",
            f"--seed={TRAINING_SEED}",
        }
        if not required.issubset(command):
            raise RuntimeError("training command lost the Marin alignment contract")
    return {
        "status": "ok",
        "candidate_count": manifest["candidate_count"],
        "prediction_binding_current": prediction_binding_is_current(),
        "manifest_sha256": sha256(MANIFEST),
        "trainer_sha256": sha256(TRAINER),
        "model_sha256": sha256(MODEL),
        "physical_gpus": list(TARGET_GPUS),
        "global_batch_sequences": world_size() * 1 * gradient_accumulation_steps(),
        "total_tokens": UPDATES * GLOBAL_BATCH * SEQUENCE_LEN,
        "model_alignment": manifest["model_alignment"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(static_check(), indent=2))
        return 0

    HERE.mkdir(parents=True, exist_ok=True)
    lock = controller_lock_path().open("w", encoding="utf-8")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Marin-matched GPT-2 controller is already running")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    if STOP_FILE.exists():
        atomic_json(STATE, {"status": "stopped_by_request", "updated_at": now()})
        return 0

    while not stop_requested():
        if not MANIFEST.exists():
            raise FileNotFoundError(
                "candidate_manifest.json is frozen and must not be regenerated"
            )
        if not ensure_predictions():
            break
        manifest, predictions = load_inputs()
        update_summary(manifest)
        completed = {row["candidate_id"] for row in result_rows()}
        baseline_id = manifest["baseline_candidate_id"]
        baseline_prediction = float(predictions[baseline_id]["predicted_final_loss"])
        globally_missing = [
            (index, candidate)
            for index, candidate in enumerate(manifest["candidates"])
            if candidate["candidate_id"] not in completed
        ]
        missing = [
            (index, candidate)
            for index, candidate in globally_missing
            if candidate_assigned_to_this_worker(index, candidate)
        ]
        if CANDIDATE_ORDER_BY_NCPL_PREDICTION:
            missing.sort(
                key=lambda row: (
                    float(predictions[row[1]["candidate_id"]]["predicted_final_loss"]),
                    row[0],
                )
            )
        if not globally_missing:
            if EXIT_WHEN_QUEUE_EMPTY:
                update_summary(manifest)
                atomic_json(
                    STATE,
                    {
                        "status": "complete",
                        "updated_at": now(),
                        "completed_count": len(completed),
                        "candidate_count": manifest["candidate_count"],
                    },
                )
                return 0
            if not CAMPAIGN_MAINTENANCE_LEADER:
                interruptible_sleep(
                    NONBASELINE_RETRY_SECONDS,
                    {
                        "status": "waiting_for_campaign_expansion",
                        "candidate_shard_count": CANDIDATE_SHARD_COUNT,
                        "candidate_shard_index": CANDIDATE_SHARD_INDEX,
                    },
                )
                continue
            atomic_json(
                STATE,
                {
                    "status": "expanding_candidate_queue",
                    "updated_at": now(),
                    "completed_count": len(completed),
                    "append_count": EXPANSION_SIZE,
                },
            )
            if not extend_manifest():
                interruptible_sleep(
                    PREDICTION_RETRY_SECONDS,
                    {"status": "candidate_expansion_failed_will_retry"},
                )
            continue

        if not missing:
            interruptible_sleep(
                NONBASELINE_RETRY_SECONDS,
                {
                    "status": "waiting_for_other_candidate_shard",
                    "global_pending_count": len(globally_missing),
                    "candidate_shard_count": CANDIDATE_SHARD_COUNT,
                    "candidate_shard_index": CANDIDATE_SHARD_INDEX,
                },
            )
            continue

        baseline_done = baseline_id in completed
        for index, candidate in missing:
            if stop_requested():
                break
            if candidate["candidate_id"] != baseline_id and not baseline_done:
                break
            success = run_candidate(
                index,
                candidate,
                predictions[candidate["candidate_id"]],
                baseline_prediction,
            )
            if success:
                completed.add(candidate["candidate_id"])
                if candidate["candidate_id"] == baseline_id:
                    baseline_done = True
                update_summary(manifest)
                print(
                    f"{now()} completed {candidate['candidate_id']} "
                    f"({len(completed)}/{manifest['candidate_count']})",
                    flush=True,
                )
            else:
                delay = (
                    BASELINE_RETRY_SECONDS
                    if candidate["candidate_id"] == baseline_id
                    else NONBASELINE_RETRY_SECONDS
                )
                interruptible_sleep(
                    delay,
                    {
                        "status": "training_failed_will_retry",
                        "candidate_id": candidate["candidate_id"],
                    },
                )
                if candidate["candidate_id"] == baseline_id:
                    break

    atomic_json(
        STATE,
        {
            "status": "stopped_by_request" if STOP_FILE.exists() else "interrupted",
            "updated_at": now(),
            "stop_file": str(STOP_FILE),
        },
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        atomic_json(
            STATE,
            {
                "status": "controller_crashed_launcher_will_restart",
                "updated_at": now(),
                "error": repr(error),
            },
        )
        raise
