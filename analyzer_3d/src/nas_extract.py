"""Nightly Instagram-clip 3D extraction for the Synology NAS.

This CPU-side runner is intended for a ``python:3.11-slim`` container with
only numpy and the Modal 1.3.4 client installed.  It reads the ingest queue,
submits clips to the deployed GVHMR service when a reusable raw pickle is not
already present, then runs the repository's numpy-only cleanup and geometric
contact extraction locally.

Example (paths and Modal credentials are deployment-specific)::

    docker run --rm \
      -v /volume1/docker/breakdance-coach:/app:ro \
      -v /volume1/data:/data \
      -e MODAL_TOKEN_ID -e MODAL_TOKEN_SECRET \
      -w /app python:3.11-slim sh -lc \
      'pip install --no-cache-dir numpy modal==1.3.4 && \
       python -m analyzer_3d.src.nas_extract'
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from analyzer_3d.src import contact_geometric
from analyzer_3d.src.contact_signatures import build_vertex_parts
from analyzer_3d.src.motion_cleanup import clean_pkl


# IG clips are not stored under the reference library's aistpp/finedance
# source directories.  This runner is the only consumer of this patched module
# instance, so label its generated signatures directly as Instagram sources.
contact_geometric._source_from_path = lambda path: "ig"


DEFAULT_QUEUE = Path("/data/ig-extract/extract_queue.json")
DEFAULT_CLIPS_DIR = Path("/data/ig-ingest/clips")
DEFAULT_RESULTS_DIR = Path("/data/ig-extract/results")
DEFAULT_LOG = Path("/data/logs/ig-extract.log")

APP_NAME = "gvhmr-pose-estimation"
CLASS_NAME = "GVHMREstimator"
POLL_TIMEOUT_S = 20.0
HEARTBEAT_S = 60.0


class _ClipFailure(Exception):
    """A known per-clip failure with a metrics status."""

    def __init__(self, status: str, message: str) -> None:
        super().__init__(message)
        self.status = status


class _ModalWallTimeout(Exception):
    """Raised after the client-side per-clip Modal deadline expires."""

    def __init__(self, elapsed_s: float) -> None:
        super().__init__(f"GVHMR exceeded the {elapsed_s:.1f}s wall timeout")
        self.elapsed_s = elapsed_s


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _utc_timestamp() -> str:
    """Return a seconds-precision ISO-8601 UTC timestamp."""
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _one_line_error(error: BaseException | str) -> str:
    """Turn an exception or message into compact, traceback-free text."""
    if isinstance(error, BaseException):
        text = f"{type(error).__name__}: {error}"
    else:
        text = str(error)
    return " ".join(text.split()) or "unknown error"


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    """Atomically replace ``path`` with bytes via a same-directory temp file."""
    temp_path = path.with_name(path.name + ".tmp")
    try:
        with temp_path.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    """Atomically write indented UTF-8 JSON with a trailing newline."""
    temp_path = path.with_name(path.name + ".tmp")
    try:
        with temp_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, indent=1, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass


def _append_log(path: Path, line: str) -> None:
    """Print a batch line and append it to the configured log."""
    print(line, flush=True)
    try:
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
    except Exception as exc:
        print(
            f"warning: could not append to {path}: {_one_line_error(exc)}",
            file=sys.stderr,
            flush=True,
        )


def _load_queue(path: Path) -> list[dict[str, Any]]:
    """Load and minimally validate the queue document."""
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict) or not isinstance(document.get("clips"), list):
        raise ValueError("queue root must be an object containing a 'clips' list")
    clips = document["clips"]
    for index, entry in enumerate(clips):
        if not isinstance(entry, dict):
            raise ValueError(f"queue clip {index} must be an object")
        if not isinstance(entry.get("clip"), str) or not entry["clip"]:
            raise ValueError(f"queue clip {index} must contain a non-empty 'clip'")
    return clips


def _paths_for(results_dir: Path, stem: str) -> dict[str, Path]:
    """Return all result paths for one queue stem."""
    clip_dir = results_dir / stem
    return {
        "dir": clip_dir,
        "raw": clip_dir / f"{stem}_smpl.pkl",
        "bad": clip_dir / f"{stem}_smpl.pkl.bad",
        "clean": clip_dir / f"{stem}_smpl_clean.pkl",
        "contacts": clip_dir / f"{stem}_contacts.json",
        "metrics": clip_dir / f"{stem}_metrics.json",
    }


def _read_previous_attempts(path: Path) -> int:
    """Read an existing attempts count, treating malformed metrics as absent."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        if not isinstance(document, dict):
            return 0
        attempts = document.get("attempts", 0)
        if (
            isinstance(attempts, int)
            and not isinstance(attempts, bool)
            and attempts >= 0
        ):
            return attempts
    except (OSError, TypeError, ValueError):
        pass
    return 0


def _result_state(paths: dict[str, Path], max_attempts: int) -> str:
    """Classify a queue item solely from its artifacts on disk."""
    if paths["clean"].is_file() and paths["contacts"].is_file():
        return "done"
    try:
        with paths["metrics"].open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)
        if not isinstance(metrics, dict):
            return "pending"
        attempts = metrics.get("attempts", 0)
        if (
            isinstance(attempts, int)
            and not isinstance(attempts, bool)
            and attempts >= max_attempts
            and metrics.get("status") != "ok"
        ):
            return "gave_up"
    except (OSError, TypeError, ValueError):
        pass
    return "pending"


def _validate_raw_result(data: Any) -> dict[str, Any]:
    """Validate the keys required by cleanup and contact extraction."""
    if not isinstance(data, dict):
        raise ValueError("pickle root is not a dict")
    missing = [
        key
        for key in ("smpl_params_global", "fps", "num_frames", "mesh")
        if key not in data
    ]
    if missing:
        raise ValueError(f"missing required result keys: {', '.join(missing)}")
    params = data["smpl_params_global"]
    if not isinstance(params, dict):
        raise ValueError("smpl_params_global is not a dict")
    missing_params = [
        key
        for key in ("body_pose", "global_orient", "transl", "betas")
        if key not in params
    ]
    if missing_params:
        raise ValueError(
            "smpl_params_global is missing: " + ", ".join(missing_params)
        )
    if not isinstance(data["mesh"], dict):
        raise ValueError("mesh is not a dict")
    return data


def _load_valid_raw(path: Path) -> dict[str, Any]:
    """Unpickle and validate a reusable raw GVHMR artifact."""
    with path.open("rb") as handle:
        return _validate_raw_result(pickle.load(handle))


def _modal_timeout_types(modal_module: Any) -> tuple[type[BaseException], ...]:
    """Build the timeout exception tuple across Modal 1.x variants."""
    timeout_types: list[type[BaseException]] = [TimeoutError]
    function_timeout: Any = None
    try:
        function_timeout = modal_module.exception.FunctionTimeoutError
    except AttributeError:
        try:
            function_timeout = importlib.import_module(
                "modal.exception"
            ).FunctionTimeoutError
        except (AttributeError, ImportError):
            pass
    if (
        isinstance(function_timeout, type)
        and issubclass(function_timeout, BaseException)
        and function_timeout not in timeout_types
    ):
        timeout_types.append(function_timeout)
    return tuple(timeout_types)


def _request_gvhmr(
    estimator: Any,
    video_bytes: bytes,
    stem: str,
    timeout_s: float,
    timeout_types: tuple[type[BaseException], ...],
) -> tuple[Any, float]:
    """Spawn GVHMR and actively poll until completion or the wall deadline."""
    started = time.monotonic()
    function_call = estimator.estimate_pose.spawn(video_bytes, static_cam=True)
    deadline = started + timeout_s
    next_heartbeat = HEARTBEAT_S

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            elapsed = time.monotonic() - started
            try:
                function_call.cancel()
            except Exception:
                pass
            raise _ModalWallTimeout(elapsed)
        try:
            result = function_call.get(timeout=min(POLL_TIMEOUT_S, remaining))
            return result, time.monotonic() - started
        except timeout_types:
            elapsed = time.monotonic() - started
            if elapsed >= next_heartbeat:
                print(
                    f"[{stem}] waiting on GVHMR ... {int(elapsed)}s",
                    flush=True,
                )
                while next_heartbeat <= elapsed:
                    next_heartbeat += HEARTBEAT_S


def _available_video_fields(data: dict[str, Any] | None) -> tuple[Any, Any]:
    """Extract JSON-friendly fps/frame values when a result supplies them."""
    if data is None:
        return None, None
    try:
        fps = float(data["fps"])
    except (KeyError, TypeError, ValueError):
        fps = None
    try:
        num_frames = int(data["num_frames"])
    except (KeyError, TypeError, ValueError):
        num_frames = None
    return fps, num_frames


def _process_clip(
    entry: dict[str, Any],
    clips_dir: Path,
    results_dir: Path,
    timeout_s: float,
    gpu_rate: float,
    get_estimator: Callable[[], Any],
    timeout_types: tuple[type[BaseException], ...],
) -> dict[str, Any]:
    """Run one clip end to end, converting every ordinary error to metrics."""
    clip_started = time.monotonic()
    clip_name = entry["clip"]
    stem = Path(clip_name).stem
    paths = _paths_for(results_dir, stem)
    attempts = _read_previous_attempts(paths["metrics"]) + 1
    status = "unexpected_error"
    error: str | None = None
    gpu_wall_s = 0.0
    reused_raw = False
    result_data: dict[str, Any] | None = None
    cleanup_metrics: dict[str, Any] | None = None
    contact_windows: int | None = None
    contact_summary: str | None = None

    try:
        paths["dir"].mkdir(parents=True, exist_ok=True)
        clip_path = clips_dir / clip_name
        if not clip_path.is_file():
            raise _ClipFailure("missing_clip", f"clip not found: {clip_path}")

        if paths["raw"].is_file():
            try:
                result_data = _load_valid_raw(paths["raw"])
                reused_raw = True
            except Exception:
                result_data = None

        if result_data is None:
            try:
                video_bytes = clip_path.read_bytes()
            except Exception as exc:
                raise _ClipFailure("video_read_failed", _one_line_error(exc)) from exc

            try:
                estimator = get_estimator()
            except Exception as exc:
                raise _ClipFailure("gvhmr_failed", _one_line_error(exc)) from exc

            try:
                modal_started = time.monotonic()
                output, gpu_wall_s = _request_gvhmr(
                    estimator,
                    video_bytes,
                    stem,
                    timeout_s,
                    timeout_types,
                )
            except _ModalWallTimeout as exc:
                gpu_wall_s = exc.elapsed_s
                raise _ClipFailure("gvhmr_timeout", str(exc)) from exc
            except Exception as exc:
                gpu_wall_s = time.monotonic() - modal_started
                raise _ClipFailure("gvhmr_failed", _one_line_error(exc)) from exc

            output_bytes: bytes | None = None
            try:
                if not isinstance(output, (bytes, bytearray, memoryview)):
                    raise TypeError(
                        f"GVHMR returned {type(output).__name__}, expected bytes"
                    )
                output_bytes = bytes(output)
                result_data = _validate_raw_result(pickle.loads(output_bytes))
            except Exception as exc:
                forensic_error: str | None = None
                if output_bytes is not None:
                    try:
                        _atomic_write_bytes(paths["bad"], output_bytes)
                    except Exception as write_exc:
                        forensic_error = _one_line_error(write_exc)
                message = _one_line_error(exc)
                if forensic_error is not None:
                    message += f"; could not save forensic payload: {forensic_error}"
                raise _ClipFailure("bad_result", message) from exc

            try:
                _atomic_write_bytes(paths["raw"], output_bytes)
            except Exception as exc:
                raise _ClipFailure("raw_write_failed", _one_line_error(exc)) from exc

        try:
            clean_pkl(paths["raw"], paths["clean"])
            with paths["clean"].open("rb") as handle:
                cleaned = pickle.load(handle)
            cleanup = cleaned["cleanup"]
            cleanup_metrics = {
                "before": cleanup["metrics_before"],
                "after": cleanup["metrics_after"],
            }
        except Exception as exc:
            raise _ClipFailure("cleanup_failed", _one_line_error(exc)) from exc

        try:
            data = contact_geometric._load_pkl(paths["clean"])
            vertex_parts, part_names = build_vertex_parts(str(paths["clean"]))
            geometry = contact_geometric.compute_geometry(
                data,
                vertex_parts,
                part_names,
                label=stem,
            )
            signature = contact_geometric.build_clip_signature(
                paths["clean"], data, geometry, part_names
            )
            document = contact_geometric._new_document(part_names)
            document["clips"][stem] = signature
            contact_windows = len(signature["windows"])
            contact_summary = signature["summary"]
            _atomic_write_json(paths["contacts"], document)
        except Exception as exc:
            raise _ClipFailure("contacts_failed", _one_line_error(exc)) from exc

        status = "ok"
    except _ClipFailure as exc:
        status = exc.status
        error = _one_line_error(str(exc))
    except Exception as exc:
        status = "unexpected_error"
        error = _one_line_error(exc)

    fps, num_frames = _available_video_fields(result_data)
    metrics: dict[str, Any] = {
        "shortcode": entry.get("shortcode"),
        "status": status,
        "attempts": attempts,
        "ts": _utc_timestamp(),
        "video": {
            "duration_s": entry.get("duration_s"),
            "fps": fps,
            "num_frames": num_frames,
        },
        "gpu_wall_s": round(gpu_wall_s, 1),
        "gpu_cost_usd": round(gpu_wall_s * gpu_rate, 4),
        "reused_raw": reused_raw,
        "review": bool(entry.get("review", False)),
        "floor_plane_only": bool(entry.get("floor_plane_only", False)),
        "error": error,
        "elapsed_s": round(time.monotonic() - clip_started, 1),
    }
    if cleanup_metrics is not None:
        metrics["cleanup"] = cleanup_metrics
    if contact_windows is not None:
        metrics["contact_windows"] = contact_windows
    if contact_summary is not None:
        metrics["contact_summary"] = contact_summary

    try:
        _atomic_write_json(paths["metrics"], metrics)
    except Exception as exc:
        metrics_error = _one_line_error(exc)
        if metrics["status"] == "ok":
            metrics["status"] = "metrics_write_failed"
            metrics["error"] = metrics_error
        else:
            prior_error = metrics.get("error")
            metrics["error"] = (
                f"{prior_error}; metrics write failed: {metrics_error}"
                if prior_error
                else f"metrics write failed: {metrics_error}"
            )
    return metrics


def _clip_log_line(entry: dict[str, Any], metrics: dict[str, Any]) -> str:
    """Format the required one-line per-clip log record."""
    identifier = entry.get("shortcode") or Path(entry["clip"]).stem
    line = (
        f"{_utc_timestamp()} [ig-extract] {identifier} {metrics['status']} "
        f"gpu={metrics['gpu_wall_s']:.1f}s "
        f"cost=${metrics['gpu_cost_usd']:.4f}"
    )
    if "contact_windows" in metrics:
        line += f" windows={metrics['contact_windows']}"
    line += (
        f" review={int(bool(metrics['review']))}"
        f" floor_only={int(bool(metrics['floor_plane_only']))}"
    )
    if metrics.get("error"):
        line += f" error={_one_line_error(str(metrics['error']))}"
    return line


def build_parser() -> argparse.ArgumentParser:
    """Build the nightly batch command-line parser."""
    parser = argparse.ArgumentParser(
        description="Run the nightly NAS Instagram-clip GVHMR extraction batch"
    )
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--clips-dir", type=Path, default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument(
        "--limit",
        type=_nonnegative_int,
        default=20,
        help="maximum pending clips to process (skips do not count)",
    )
    parser.add_argument("--timeout-s", type=_positive_float, default=900.0)
    parser.add_argument("--max-attempts", type=_positive_int, default=3)
    parser.add_argument("--gpu-rate", type=_nonnegative_float, default=0.000164)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the batch; return 2 only for setup failures."""
    args = build_parser().parse_args(argv)

    try:
        modal_module = importlib.import_module("modal")
    except Exception as exc:
        print(
            f"setup error: unable to import the modal client: {_one_line_error(exc)}",
            file=sys.stderr,
        )
        return 2

    try:
        queue = _load_queue(args.queue)
    except Exception as exc:
        print(
            f"setup error: queue unreadable ({args.queue}): {_one_line_error(exc)}",
            file=sys.stderr,
        )
        return 2
    if not args.clips_dir.is_dir():
        print(
            f"setup error: clips directory does not exist: {args.clips_dir}",
            file=sys.stderr,
        )
        return 2
    try:
        args.results_dir.mkdir(parents=True, exist_ok=True)
        if not args.results_dir.is_dir():
            raise NotADirectoryError(args.results_dir)
    except Exception as exc:
        print(
            f"setup error: results directory is not usable ({args.results_dir}): "
            f"{_one_line_error(exc)}",
            file=sys.stderr,
        )
        return 2
    try:
        args.log.parent.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(
            f"setup error: log directory is not usable ({args.log.parent}): "
            f"{_one_line_error(exc)}",
            file=sys.stderr,
        )
        return 2

    pending: list[dict[str, Any]] = []
    skipped_done = 0
    gave_up = 0
    for entry in queue:
        stem = Path(entry["clip"]).stem
        state = _result_state(_paths_for(args.results_dir, stem), args.max_attempts)
        if state == "done":
            skipped_done += 1
        elif state == "gave_up":
            gave_up += 1
        else:
            pending.append(entry)

    estimator: Any = None
    estimator_error: Exception | None = None
    estimator_initialized = False

    def get_estimator() -> Any:
        nonlocal estimator, estimator_error, estimator_initialized
        if not estimator_initialized:
            estimator_initialized = True
            try:
                estimator = modal_module.Cls.from_name(APP_NAME, CLASS_NAME)()
            except Exception as exc:
                estimator_error = exc
        if estimator_error is not None:
            raise estimator_error
        return estimator

    timeout_types = _modal_timeout_types(modal_module)
    batch_started = time.monotonic()
    attempted = 0
    ok = 0
    gpu_total_s = 0.0

    for entry in pending[: args.limit]:
        attempted += 1
        try:
            metrics = _process_clip(
                entry,
                args.clips_dir,
                args.results_dir,
                args.timeout_s,
                args.gpu_rate,
                get_estimator,
                timeout_types,
            )
        except Exception as exc:
            # _process_clip is defensive, but the batch still contains a final
            # boundary so an unforeseen per-clip bug never stops later clips.
            metrics = {
                "status": "unexpected_error",
                "gpu_wall_s": 0.0,
                "gpu_cost_usd": 0.0,
                "review": bool(entry.get("review", False)),
                "floor_plane_only": bool(entry.get("floor_plane_only", False)),
                "error": _one_line_error(exc),
            }
        if metrics["status"] == "ok":
            ok += 1
        gpu_total_s += float(metrics.get("gpu_wall_s", 0.0))
        _append_log(args.log, _clip_log_line(entry, metrics))

    remaining_pending = sum(
        _result_state(
            _paths_for(args.results_dir, Path(entry["clip"]).stem),
            args.max_attempts,
        )
        == "pending"
        for entry in queue
    )
    failed = attempted - ok
    wall_min = (time.monotonic() - batch_started) / 60.0
    summary = (
        f"{_utc_timestamp()} [ig-extract] SUMMARY attempted={attempted} "
        f"ok={ok} failed={failed} skipped_done={skipped_done} gave_up={gave_up} "
        f"remaining_pending={remaining_pending} gpu_total_s={gpu_total_s:.1f} "
        f"cost_usd={gpu_total_s * args.gpu_rate:.4f} "
        f"(gpu_s x {args.gpu_rate:g}) wall_min={wall_min:.1f}"
    )
    _append_log(args.log, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
