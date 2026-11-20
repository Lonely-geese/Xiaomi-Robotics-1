#!/usr/bin/env python3
"""Official XR1 runtime plus a thin, full-chunk transport for the robot PC."""
from __future__ import annotations

import argparse
import ast
import json
import logging
import os
from pathlib import Path
import queue
import signal
import socket
import socketserver
import subprocess
import sys
import threading
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def _find_bundle():
    """Locate the checked-in rollout runtime."""
    for candidate in ROOT.parents:
        if (candidate / "server/mibot").is_dir():
            return candidate
    raise RuntimeError(f"cannot locate Xiaomi checkpoint bundle above {ROOT}")


BUNDLE = _find_bundle()
sys.path.insert(0, str(BUNDLE / "server"))
sys.path.insert(0, str(ROOT))
from conversion import CartesianLimits, make_cartesian_chunk, robot_state
from wire import WIRE_FORMAT, recv_arrays, send_arrays

LOG = logging.getLogger("xr1.cartesian.server")
PREFIX_RAW_ATOL = 1e-3  # official normalize -> bfloat16 -> denormalize round-trip


class PredictionWriter:
    """Persist audit artifacts outside the inference response critical path."""
    def __init__(self, output, capacity=256):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "rejected").mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue(maxsize=capacity)
        self.thread = threading.Thread(
            target=self._run, name="xr1-prediction-writer", daemon=True)
        self.thread.start()

    def _submit(self, path, payload):
        payload = {key: np.asarray(value).copy() for key, value in payload.items()}
        if any(value.dtype.hasobject for value in payload.values()):
            raise ValueError("prediction audit payload cannot contain object arrays")
        try:
            self.queue.put_nowait((path, payload))
        except queue.Full:
            LOG.error("prediction writer queue full; audit artifact dropped: %s", path)
            return None
        return path

    def submit(self, serial, prefix, chunk):
        path = self.output / f"{time.time_ns()}_{serial:06d}.npz"
        payload = {
            "status": np.asarray("accepted"),
            "serial": np.asarray(serial, dtype=np.int64),
            "action_prefix": np.empty((0, 60), np.float32) if prefix is None
                else np.asarray(prefix).copy(),
            **{key: np.asarray(value).copy() for key, value in chunk.items()},
        }
        return self._submit(path, payload)

    def submit_rejected(self, serial, prefix, raw, result, error):
        """Queue enough input/output evidence to reproduce a rejected prediction."""
        path = self.output / "rejected" / (
            f"{time.time_ns()}_{serial:06d}_rejected.npz")
        payload = {
            "status": np.asarray("rejected"),
            "serial": np.asarray(serial, dtype=np.int64),
            "action_prefix": np.empty((0, 60), np.float32) if prefix is None
                else np.asarray(prefix),
            "error_type": np.asarray(type(error).__name__),
            "error_code": np.asarray(getattr(error, "error_code", "validation_error")),
            "error_message": np.asarray(str(error)),
            "task": np.asarray(raw.get("task", "")),
        }
        for key, value in raw.get("state", {}).items():
            payload[f"state__{key}"] = value
        # Rejected predictions are rare. Preserve the three triggering images so
        # the model output can be audited against the exact visual observation.
        for key, value in raw.get("images", {}).items():
            payload[f"image__{key}"] = value
        if isinstance(result, dict):
            if result.get("raw_action") is not None:
                payload["raw_action"] = result["raw_action"]
            targets = result.get("action_targets")
            if isinstance(targets, dict):
                for key, value in targets.items():
                    if value is not None:
                        payload[f"target__{key}"] = value
        return self._submit(path, payload)

    def _run(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                path, payload = item
                with path.open("xb") as stream:
                    np.savez(stream, **payload)
            except Exception:
                LOG.exception("failed to save prediction audit artifact")
            finally:
                self.queue.task_done()

    def close(self):
        self.queue.join()
        self.queue.put(None)
        self.thread.join(5)


def _resolve_model_dir(override=None):
    configured = override or os.environ.get("XR1_MODEL_DIR")
    if configured:
        candidate = Path(configured).expanduser()
    elif (ROOT / "active_model").exists():
        candidate = ROOT / "active_model"
    else:
        candidate = BUNDLE / "model"
    candidate = candidate.resolve(strict=True)
    required = [
        candidate / "config.py",
        candidate / "last.ckpt/checkpoint/mp_rank_00_model_states.pt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"selected Xiaomi model is incomplete: {missing}")
    return candidate


def _read_model_manifest(model_dir):
    profile_manifest = model_dir / "model_manifest.json"
    path = profile_manifest if profile_manifest.is_file() else BUNDLE / "manifest.json"
    manifest = json.loads(path.read_text())
    checkpoint = model_dir / "last.ckpt/checkpoint/mp_rank_00_model_states.pt"
    expected_bytes = manifest.get("checkpoint_bytes")
    if expected_bytes is not None and checkpoint.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"checkpoint size mismatch: {checkpoint.stat().st_size} != {expected_bytes}")
    return manifest


class NativeRuntime:
    """Run the bundle's unmodified official Client/Server inference path."""
    def __init__(self, gpu="0", model_dir=None):
        self.process = self.client = self.log = None
        self.gpu = gpu
        self.model_dir = _resolve_model_dir(model_dir)
        self.failed = False

    def start(self):
        os.environ["XR1_VLM_PATH"] = str(BUNDLE / "models/Qwen3-VL-4B-Instruct")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(BUNDLE / "xr1")
        environment["CUDA_VISIBLE_DEVICES"] = self.gpu
        with socket.socket() as temporary:
            temporary.bind(("127.0.0.1", 0))
            port = temporary.getsockname()[1]
        (ROOT / "runs").mkdir(exist_ok=True)
        self.log = (ROOT / "runs" / f"native_{time.time_ns()}.log").open("w")
        self.process = subprocess.Popen(
            [sys.executable, "-u", str(BUNDLE / "xr1/mibot/server/deploy.py"),
             "--model", str(self.model_dir), "--host", "127.0.0.1", "--port", str(port)],
            env=environment, stdout=self.log, stderr=subprocess.STDOUT,
        )
        LOG.info("loading owned official XR1 server, model=%s pid=%d log=%s",
                 self.model_dir, self.process.pid, self.log.name)
        from mibot.server.runtime.client import Client
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"official server startup failed; see {self.log.name}")
            try:
                self.client = Client("127.0.0.1", port)
                self.client.socket.settimeout(30)
                return
            except ConnectionRefusedError:
                time.sleep(.5)
        raise TimeoutError(f"official server loading timed out; see {self.log.name}")

    def predict(self, raw, action_prefix=None):
        if self.failed:
            raise RuntimeError("official inference connection failed; restart the adapter")
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("official model process is not running")
        images = []
        for name in ("head_rgb", "left_wrist_rgb", "right_wrist_rgb"):
            image = np.asarray(raw["images"][name])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(f"{name} must be uint8 CHW RGB")
            images.append(Image.fromarray(image.transpose(1, 2, 0)))
        try:
            result = self.client(
                robot_state(raw), *images, raw["task"], action_prefix=action_prefix)
        except Exception:
            # Never let a response arriving after timeout become the next
            # observation's result on this stream.
            self.failed = True
            self.client.close()
            raise
        return result

    def close(self):
        if self.client is not None:
            try:
                self.client.close()
            finally:
                self.client = None
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.log is not None:
            self.log.close()


def decode_request(arrays, metadata):
    if metadata.get("frequency") != 30:
        raise ValueError("this checkpoint requires 30 Hz actions")
    raw = {
        "images": {key[7:]: value for key, value in arrays.items() if key.startswith("image__")},
        "state": {key[7:]: value for key, value in arrays.items() if key.startswith("state__")},
        "task": str(metadata["task"]),
        "frequency": 30,
    }
    prefix = arrays.get("action_prefix")
    if prefix is not None:
        prefix = np.asarray(prefix, dtype=np.float32)
        if prefix.ndim != 2 or prefix.shape[1] != 60 or not 1 <= len(prefix) <= 6:
            raise ValueError("official asynchronous prefix must have shape (1..6, 60)")
        if not np.isfinite(prefix).all():
            raise ValueError("action prefix contains non-finite values")
    return raw, prefix


class Policy:
    def __init__(self, runtime, limits, prediction_writer=None):
        self.runtime = runtime
        self.limits = limits
        self.lock = threading.Lock()
        self.prediction_writer = prediction_writer or PredictionWriter(
            ROOT / "runs/predictions")
        self.warmed = False
        model_dir = getattr(runtime, "model_dir", None) or _resolve_model_dir()
        manifest = _read_model_manifest(model_dir)
        if manifest["active_action_dims"] != list(range(7)) or manifest["action_horizon"] != 30:
            raise ValueError("bundle is not the expected left-arm 30-frame checkpoint")
        tree = ast.parse((model_dir / "config.py").read_text(), filename=str(model_dir / "config.py"))
        assignments = {
            target.id: ast.literal_eval(node.value)
            for node in tree.body if isinstance(node, ast.Assign)
            for target in node.targets if isinstance(target, ast.Name)
        }
        model_config = assignments.get("model", {}).get("params", {}).get("model", {})
        if model_config.get("async_train") is not True:
            raise ValueError("checkpoint config does not enable asynchronous training")
        if model_config.get("active_action_dims") != list(range(7)):
            raise ValueError("checkpoint config is not the expected left-arm action layout")
        self.handshake = {
            "status": "ready", "adapter": "xr1_official_prefix_cartesian_v1",
            "checkpoint_step": manifest["checkpoint_step"], "checkpoint_sha256": manifest["checkpoint_sha256"],
            "checkpoint_epoch": manifest.get("checkpoint_epoch"),
            "completed_epochs": manifest.get("completed_epochs"),
            "model_profile": manifest.get("profile", manifest.get("bundle")),
            "action_horizon": 30, "frequency_hz": 30, "active_arm": "left",
            "action_space": "world_cartesian_xyz_xyzw", "async_prefix": True,
            "max_prefix_length": 6, "wire_format": WIRE_FORMAT,
            "trajectory_rejection_protocol": "structured_retry_v1",
        }

    def handshake_metadata(self):
        return {**self.handshake, "model_warmed": self.warmed}

    def infer(self, raw, prefix, serial):
        started = time.monotonic()
        with self.lock:
            result = self.runtime.predict(raw, action_prefix=prefix)
            self.warmed = True
        inferred = time.monotonic()
        if not isinstance(result, dict) or "raw_action" not in result or "action_targets" not in result:
            raise ValueError("official runtime did not return raw_action and action_targets")
        official_gripper = np.asarray(
            result["action_targets"].get("left_gripper_pos"), dtype=np.float64)
        if official_gripper.shape == (30, 1) and np.isfinite(official_gripper).all():
            clipped = (official_gripper < 0) | (official_gripper > .09)
            if np.any(clipped):
                LOG.warning(
                    "request %d: saturating %d/30 gripper targets to 0..0.09 m "
                    "(predicted min=%.5f max=%.5f)",
                    serial, int(np.count_nonzero(clipped)),
                    float(official_gripper.min()), float(official_gripper.max()))
        try:
            chunk = make_cartesian_chunk(raw, result["raw_action"], self.limits,
                                         official_targets=result["action_targets"])
        except Exception as error:
            path = self.prediction_writer.submit_rejected(
                serial, prefix, raw, result, error)
            if path is not None:
                error.rejected_prediction_path = str(path)
            raise
        if prefix is not None:
            if not np.allclose(chunk["raw_action"][:len(prefix), :7], prefix[:, :7],
                               atol=PREFIX_RAW_ATOL, rtol=0):
                raise ValueError("model did not preserve the conditioned action prefix")
        converted = time.monotonic()
        path = self.prediction_writer.submit(serial, prefix, chunk)
        latency = time.monotonic() - started
        metrics = {"latency_s": latency,
                   "model_latency_s": inferred-started,
                   "postprocess_latency_s": converted-inferred,
                   "prediction_log_queued": path is not None}
        LOG.info("request %d: prefix=%d model=%.3fs postprocess=%.3fs "
                 "response_ready=%.3fs output=%s",
                 serial, 0 if prefix is None else len(prefix),
                 metrics["model_latency_s"], metrics["postprocess_latency_s"],
                 latency, path)
        return chunk, metrics

    def close(self):
        self.prediction_writer.close()


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        send_arrays(self.request, {}, self.server.policy.handshake_metadata())
        while True:
            try:
                arrays, metadata = recv_arrays(self.request)
            except ConnectionError:
                return
            serial = int(metadata.get("serial", 0))
            try:
                raw, prefix = decode_request(arrays, metadata)
                chunk, metrics = self.server.policy.infer(raw, prefix, serial)
                send_arrays(self.request, chunk, {"status": "ok", "serial": serial,
                                                  "prefix_length": 0 if prefix is None else len(prefix),
                                                  **metrics})
            except Exception as error:
                LOG.exception("request %d rejected", serial)
                error_metadata = getattr(error, "metadata", None)
                details = error_metadata() if callable(error_metadata) else {}
                rejected_path = getattr(error, "rejected_prediction_path", None)
                if rejected_path is not None:
                    details["rejected_prediction_path"] = rejected_path
                send_arrays(self.request, {}, {
                    "status": "error", "serial": serial,
                    "error": f"{type(error).__name__}: {error}", **details})


class ThreadedServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10087)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model-dir", default=None,
                        help="Xiaomi model directory; defaults to XR1_MODEL_DIR, "
                             "then this experiment's active_model")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runtime = NativeRuntime(args.gpu, args.model_dir)
    server = None
    try:
        runtime.start()
        limits = CartesianLimits()
        server = ThreadedServer((args.host, args.port), Handler)
        server.policy = Policy(runtime, limits)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        LOG.info("XR1 official-prefix Cartesian adapter ready at %s:%d", args.host, args.port)
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
            server.policy.close()
        runtime.close()


if __name__ == "__main__":
    main()
