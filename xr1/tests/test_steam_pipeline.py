"""Stage transitions must honor the configured devices, paths and rank count."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("complete", [False, True])
def test_pipeline_keeps_selected_devices_and_pause_boundary(tmp_path, complete):
    mock = tmp_path / "python"
    mock.write_text(
        f"#!{sys.executable}\n"
        "import json, os, subprocess, sys\n"
        "if sys.argv[1] == '-c':\n"
        "    raise SystemExit(subprocess.call([sys.executable, *sys.argv[1:]]))\n"
        "if sys.argv[1] == '-':\n"
        "    sys.stdin.read()\n"
        "    raise SystemExit(0 if os.environ['CRITIC_COMPLETE'] == '1' else 1)\n"
        "with open(os.environ['CALLS'], 'a') as f:\n"
        "    f.write(json.dumps({'args': sys.argv[1:], 'devices': os.environ['CUDA_VISIBLE_DEVICES']}) + '\\n')\n"
    )
    mock.chmod(0o755)
    critic_out = tmp_path / "critic"
    critic_out.mkdir()
    (critic_out / "latest.json").write_text("{}")
    policy_out = tmp_path / "policy"
    critic_cfg, policy_cfg = tmp_path / "critic.json", tmp_path / "policy.json"
    critic_cfg.write_text(
        json.dumps({"output": str(critic_out), "manifest": "manifest"})
    )
    policy_cfg.write_text(
        json.dumps({"output": str(policy_out), "advantages": "adv.parquet"})
    )
    log_dir, calls = tmp_path / "logs", tmp_path / "calls.jsonl"
    env = {
        **os.environ,
        "STEAM_PYTHON": str(mock),
        "STEAM_LOG_DIR": str(log_dir),
        "STEAM_CRITIC_CONFIG": str(critic_cfg),
        "STEAM_POLICY_CONFIG": str(policy_cfg),
        "STEAM_NPROC": "7",
        "CUDA_VISIBLE_DEVICES": "1,2,3,4,5,6,7",
        "CRITIC_COMPLETE": str(int(complete)),
        "CALLS": str(calls),
    }
    script = Path(__file__).resolve().parents[1] / "scripts/train_steam_pipeline.sh"
    subprocess.run(["bash", str(script)], env=env, check=True)
    events = [json.loads(line) for line in calls.read_text().splitlines()]
    assert len(events) == (3 if complete else 1)
    first = events[0]
    assert first["devices"] == "1,2,3,4,5,6,7"
    assert "--nproc_per_node=7" in first["args"]
    assert first["args"][first["args"].index("--resume") + 1] == str(critic_out)
    if complete:
        assert events[1]["devices"] == "1"
        assert (
            events[1]["args"][events[1]["args"].index("--output") + 1] == "adv.parquet"
        )
        assert events[2]["devices"] == "1,2,3,4,5,6,7"
        assert "--nproc_per_node=7" in events[2]["args"]
    assert (log_dir / "status").read_text().strip() == (
        "complete" if complete else "paused_critic"
    )
