import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from . import recap, steam

ROOT = Path(__file__).resolve().parents[1]
STAGES = ("value", "label", "policy")


def main():
    parser = argparse.ArgumentParser(description="Run an offline XR1 RL stage or pipeline")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    backend = {"steam": steam, "recap": recap}.get(cfg["method"])
    if backend is None:
        parser.error("method must be steam or recap")
    run = (ROOT / cfg["run_dir"]).resolve()
    ranks = cfg["gpus"]
    if type(ranks) is not int or ranks < 1 or (backend is recap and ranks != 8):
        parser.error("gpus must be positive; the native RECAP policy requires 8")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "xr1"), env.get("PYTHONPATH")]))
    stages = STAGES if args.stage == "all" else (args.stage,)
    launcher = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
                "--master-addr=127.0.0.1", "--master-port=" + str(cfg.get("port", 29557)),
                "--nproc-per-node=" + str(ranks)]
    if not args.dry_run:
        import fcntl
        run.mkdir(parents=True, exist_ok=True)
        lock = (run / "pipeline.lock").open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot = run / "pipeline.json"
        if snapshot.exists() and json.loads(snapshot.read_text()) != cfg:
            parser.error("configuration changed: select a new run_dir")
        tmp = snapshot.with_suffix(".tmp")
        tmp.write_text(json.dumps(cfg, indent=2))
        tmp.replace(snapshot)
    for stage in stages:
        done = run / f"{stage}.done"
        if not args.dry_run and done.exists():
            backend.verify(cfg, run, stage, ROOT)
            print(f"{stage}: already completed", flush=True)
            continue
        command, extra = backend.command(ROOT, cfg, run, stage, launcher, args.dry_run)
        command = list(map(str, command))
        print(json.dumps(dict(stage=stage, cwd=str(ROOT / "xr1"), command=command, environment=extra)), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT / "xr1", env={**env, **extra}, check=True)
            backend.verify(cfg, run, stage, ROOT)
            done.touch()


if __name__ == "__main__":
    main()
