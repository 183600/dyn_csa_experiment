#!/usr/bin/env python3
"""Task-level parallel executor: one GPU per independent experiment run.

Environment:
    CSA_PAR=auto       One worker per visible GPU.
    CSA_PAR=N          At most N GPU workers, or N CPU workers.
    CSA_PAR=off        Serial execution.
    CSA_PAR_GPUS=0,2   Parent-logical GPU indices to use.

GPU indices in CSA_PAR_GPUS are relative to the parent's visible devices.
Children receive the corresponding original CUDA_VISIBLE_DEVICES tokens.

Completed experiment cells remain the responsibility of the existing
experiment-library resume logic. This module manages dispatch, process
lifecycle, logs, and child result files.
"""

import importlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import traceback


REPO = os.path.dirname(os.path.abspath(__file__))
JOBS_DIR = os.path.join(REPO, "par_jobs")
_KILL = {"on": False}
_OFF = ("", "0", "no", "off", "false", "none", "serial")


def _num_or_none(value):
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError, OverflowError):
        return None
    return number if number > 0 else None


def _torch():
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def gpu_slots(want_gpus=None):
    """Return child CUDA visibility tokens and whether CUDA is available.

    CSA_PAR_GPUS uses parent-logical indices. When the parent has
    CUDA_VISIBLE_DEVICES=2,3, logical indices 0,1 map to tokens 2,3.
    UUID and MIG visibility tokens are preserved.
    """
    torch = _torch()
    count = 0
    if torch is not None:
        try:
            if torch.cuda.is_available():
                count = int(torch.cuda.device_count())
        except Exception as exc:
            raise RuntimeError(
                "Could not query parent CUDA devices."
            ) from exc

    if count <= 0:
        if want_gpus is not None and str(want_gpus).strip():
            raise ValueError(
                "CSA_PAR_GPUS was specified, but no CUDA device is visible."
            )
        return [], False

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        tokens = [str(index) for index in range(count)]
    else:
        tokens = [item.strip() for item in visible.split(",")]
        if (
            len(tokens) != count
            or any(not item or item == "-1" for item in tokens)
            or len(set(tokens)) != len(tokens)
        ):
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES cannot be mapped safely: "
                f"value={visible!r}, torch device_count={count}. "
                "Use a valid explicit device list."
            )

    if want_gpus is None or not str(want_gpus).strip():
        return tokens, True

    selected = []
    for item in str(want_gpus).split(","):
        item = item.strip()
        if not re.fullmatch(r"[0-9]+", item):
            raise ValueError(
                f"Invalid CSA_PAR_GPUS entry {item!r}; "
                "use parent-logical integer GPU indices."
            )
        index = int(item)
        if index >= count:
            raise ValueError(
                f"CSA_PAR_GPUS index {index} is out of range "
                f"for {count} visible device(s)."
            )
        token = tokens[index]
        if token not in selected:
            selected.append(token)

    if not selected:
        raise ValueError("CSA_PAR_GPUS selected no devices.")
    return selected, True


def resolve_ctx(par=None, label=""):
    """Resolve a driver context or CSA_PAR environment configuration."""
    if isinstance(par, dict):
        return par or None
    if par is None:
        par = os.environ.get("CSA_PAR")
    if par is None or par is False:
        return None

    value = "auto" if par is True else str(par).strip().lower()
    if value in _OFF:
        return None

    if value == "auto":
        wanted = None
    elif re.fullmatch(r"[0-9]+", value):
        wanted = int(value)
        if wanted <= 0:
            return None
    else:
        raise ValueError(
            f"Invalid CSA_PAR value {par!r}; use auto, N, or off."
        )

    slots, on_cuda = gpu_slots(os.environ.get("CSA_PAR_GPUS"))
    if on_cuda:
        workers = len(slots) if wanted is None else min(wanted, len(slots))
        if wanted is not None and wanted > len(slots):
            print(
                f"[par] Requested {wanted} workers; "
                f"capping at {len(slots)} GPU slot(s).",
                flush=True,
            )
        slots = slots[:workers]
    else:
        workers = 1 if wanted is None else wanted
        slots = [""] * workers
        print(
            f"[par] No visible CUDA device: using {workers} CPU worker(s).",
            flush=True,
        )

    context = {
        "workers": workers,
        "slots": slots,
        "on_cuda": on_cuda,
        "label": label,
    }
    location = "GPUs " + ",".join(slots) if on_cuda else "CPU"
    print(
        f"[par] Parallel execution enabled: {workers} worker(s), {location}.",
        flush=True,
    )
    return context


def ctx_active(ctx, n_cells=1):
    return (
        bool(ctx)
        and int(ctx.get("workers", 0)) > 1
        and n_cells > 1
    )


def _norm(obj):
    """Convert driver payloads to JSON-compatible values."""
    if isinstance(obj, dict):
        return {str(key): _norm(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_norm(value) for value in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted(
            (_norm(value) for value in obj),
            key=lambda value: json.dumps(
                value, sort_keys=True, default=str
            ),
        )
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _denorm(obj):
    if isinstance(obj, dict):
        return {str(key): _denorm(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_denorm(value) for value in obj]
    return obj


def _slug(value):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))[:80]


def _child_env(slot):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(slot)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _atomic_json(path, obj):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix="." + os.path.basename(path) + ".",
        suffix=".tmp",
        dir=directory,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(_norm(obj), stream, indent=1, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_result(path):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as stream:
            result = json.load(stream)
        if not isinstance(result, dict):
            raise ValueError("Child result must be a JSON object.")
        return result
    except Exception as exc:
        return {
            "status": "unreadable_result",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _dump_log_tail(entry, n=40):
    try:
        with open(
            entry["log_path"], encoding="utf-8", errors="replace"
        ) as stream:
            tail = stream.readlines()[-n:]
        print(f"[par] ---- {entry['log_path']} ----", flush=True)
        sys.stdout.writelines("  " + line for line in tail)
        print("[par] ------------------------------", flush=True)
    except OSError:
        pass


def _terminate_entries(entries):
    """Terminate children and their process groups, then reap them."""
    active = [
        entry for entry in entries
        if entry.get("proc") is not None
        and entry["proc"].poll() is None
    ]
    for entry in active:
        proc = entry["proc"]
        try:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGTERM)
            else:
                proc.terminate()
        except ProcessLookupError:
            pass
        except OSError:
            try:
                proc.terminate()
            except OSError:
                pass

    deadline = time.monotonic() + 30.0
    for entry in active:
        proc = entry["proc"]
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGKILL)
                else:
                    proc.kill()
            except OSError:
                pass
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                print(
                    f"[par] WARNING: child PID {proc.pid} "
                    "could not be reaped promptly.",
                    flush=True,
                )


def run_jobs(jobs, ctx, phase_label="phase"):
    """Dispatch jobs and return result objects in input order.

    A normal experiment failure can return an error status with rc=0.
    Nonzero exit codes or invalid/missing result files are infrastructure
    failures and abort dispatch.
    """
    jobs = list(jobs)
    if not jobs:
        return []
    if not isinstance(ctx, dict):
        raise ValueError("run_jobs requires a resolved context dict.")

    slots = [str(slot) for slot in ctx.get("slots", [])]
    requested = int(ctx.get("workers", 1))
    if requested <= 0 or not slots:
        raise ValueError("Parallel context has no usable worker slots.")
    workers = min(requested, len(slots))
    slots = slots[:workers]

    if ctx.get("on_cuda"):
        if any(not slot for slot in slots):
            raise ValueError("GPU context contains an empty device token.")
        if len(set(slots)) != len(slots):
            raise ValueError("GPU worker slots must use distinct devices.")
    elif any(slots):
        raise ValueError("CPU context must use empty CUDA visibility.")

    os.makedirs(JOBS_DIR, exist_ok=True)
    prefix = (
        _slug(phase_label)
        + "_"
        + time.strftime("%Y%m%d_%H%M%S")
        + "_"
    )
    run_dir = tempfile.mkdtemp(prefix=prefix, dir=JOBS_DIR)
    print(
        f"[par] Dispatching {len(jobs)} cell(s) to {workers} worker(s): "
        f"{os.path.relpath(run_dir, REPO)}",
        flush=True,
    )

    entries = []
    for index, job in enumerate(jobs):
        if not isinstance(job, dict):
            raise TypeError(f"Job {index} must be a dict.")
        name = (
            f"{index:03d}_{_slug(job.get('kind'))}_"
            f"{_slug(job.get('v'))}_s{_slug(job.get('seed'))}"
        )
        extra = job.get("extra")
        if isinstance(extra, dict) and extra.get("warm") is not None:
            name += f"_w{_slug(extra['warm'])}"
        directory = os.path.join(run_dir, name)
        os.mkdir(directory)
        entry = {
            "idx": index,
            "name": name,
            "job": job,
            "job_path": os.path.join(directory, "job.json"),
            "result_path": os.path.join(directory, "result.json"),
            "log_path": os.path.join(directory, "log.txt"),
            "proc": None,
            "log_f": None,
            "worker": None,
            "done": False,
            "result": None,
        }
        _atomic_json(entry["job_path"], job)
        entries.append(entry)

    pending = list(entries)
    running = []
    results = [None] * len(entries)
    started = time.monotonic()
    next_heartbeat = started + 60.0
    _KILL["on"] = False

    try:
        while pending or running:
            busy = {entry["worker"] for entry in running}
            free_workers = [
                index for index in range(workers) if index not in busy
            ]

            for worker in free_workers:
                if not pending:
                    break
                entry = pending.pop(0)
                entry["worker"] = worker
                entry["log_f"] = open(
                    entry["log_path"], "w", encoding="utf-8"
                )
                entry["proc"] = subprocess.Popen(
                    [
                        sys.executable,
                        os.path.abspath(__file__),
                        "__child__",
                        entry["job_path"],
                        entry["result_path"],
                    ],
                    cwd=REPO,
                    env=_child_env(slots[worker]),
                    stdout=entry["log_f"],
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                running.append(entry)
                location = (
                    f"GPU {slots[worker]}"
                    if ctx.get("on_cuda")
                    else f"CPU worker {worker}"
                )
                print(
                    f"[par] Start {entry['name']} on {location}.",
                    flush=True,
                )

            still_running = []
            for entry in running:
                rc = entry["proc"].poll()
                if rc is None:
                    still_running.append(entry)
                    continue

                entry["log_f"].close()
                entry["done"] = True
                result = _read_result(entry["result_path"])
                entry["result"] = result
                results[entry["idx"]] = result
                status = (result or {}).get("status", "missing_result")
                print(
                    f"[par] Done {entry['name']}: rc={rc}, "
                    f"status={status}, "
                    f"{(time.monotonic() - started) / 60:.1f} min "
                    "since dispatch.",
                    flush=True,
                )

                valid_result = (
                    isinstance(result, dict)
                    and isinstance(status, str)
                    and bool(status.strip())
                    and status != "unreadable_result"
                )
                if rc != 0 or not valid_result:
                    _dump_log_tail(entry)
                    raise RuntimeError(
                        f"Child {entry['name']} failed: "
                        f"rc={rc}, status={status}. "
                        f"See {entry['log_path']}. "
                        "Completed experiment records are preserved."
                    )

            running = still_running
            now = time.monotonic()
            if running and now >= next_heartbeat:
                completed = sum(entry["done"] for entry in entries)
                names = ", ".join(entry["name"] for entry in running)
                print(
                    f"[par] {len(running)} running / {completed} done / "
                    f"{len(pending)} pending: {names}",
                    flush=True,
                )
                next_heartbeat = now + 60.0

            if running:
                time.sleep(0.2)

    except BaseException:
        _KILL["on"] = True
        _terminate_entries(entries)
        print(
            "[par] Dispatch aborted; remaining children were terminated. "
            "Completed records are preserved. Inspect time reservations "
            "before resuming if a child was killed.",
            flush=True,
        )
        raise
    finally:
        for entry in entries:
            stream = entry.get("log_f")
            if stream is not None and not stream.closed:
                stream.close()

    return results


def run_child(job_path, result_path):
    result = {"status": "fatal"}
    try:
        with open(job_path, encoding="utf-8") as stream:
            job = _denorm(json.load(stream))
        if not isinstance(job, dict):
            raise ValueError("Job payload must be a JSON object.")

        result.update({
            "kind": job.get("kind"),
            "v": job.get("v"),
            "seed": job.get("seed"),
        })

        module_name = job.get("module") or "exp_lib"
        mod = importlib.import_module(module_name)

        # Preserve the existing driver-specific checkpoint patch chain.
        if job.get("ckpt_install") and hasattr(mod, "V9"):
            mod.V9._install_ckpt_patch()
        if job.get("ckpt_flag") and hasattr(mod, "V9"):
            mod.V9._V9_CKPT["on"] = True

        kind = job["kind"]
        if kind == "run":
            response = getattr(mod, "L", mod).run_cell_job(job)
        elif kind == "lenphase":
            response = importlib.import_module(
                "v7_supp"
            ).run_lenphase_cell_job(job)
        elif kind == "probe":
            response = importlib.import_module(
                "v10_supp"
            ).run_probe_cell_job(job)
        elif kind == "warm":
            response = importlib.import_module(
                "v7_supp"
            ).run_warmup_cell_job(job)
        else:
            raise ValueError(f"Unknown job kind {kind!r}.")

        if response is not None and not isinstance(response, dict):
            raise TypeError("Cell handler must return a dict or None.")

        result["status"] = "ok"
        if response is not None:
            result.update(response)

        status = result.get("status")
        if not isinstance(status, str) or not status.strip():
            raise ValueError("Cell result must contain a nonempty status.")

    except SystemExit as exc:
        result["status"] = f"exit_{exc.code}"
        result["error"] = traceback.format_exc()
        raise
    except BaseException:
        result["status"] = "fatal"
        result["error"] = traceback.format_exc()
        traceback.print_exc()
        raise
    finally:
        # A write failure must propagate, not masquerade as child success.
        _atomic_json(result_path, result)


def main(argv):
    if len(argv) == 4 and argv[1] == "__child__":
        run_child(argv[2], argv[3])
        return 0
    if len(argv) > 1 and argv[1] == "__child__":
        raise SystemExit(
            "Usage: par.py __child__ JOB_JSON RESULT_JSON"
        )
    print(__doc__)
    print(
        "[par] Imported by experiment drivers; "
        "set CSA_PAR=auto|N to enable parallel execution."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
