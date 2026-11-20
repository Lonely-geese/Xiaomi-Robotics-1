"""CPU-only contracts for method dispatch and incomplete training runs."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from rl import recap, steam
from rl.__main__ import ROOT


class PipelineTests(unittest.TestCase):
    def test_steam_requires_completed_critic_before_labeling(self):
        cfg = json.loads((ROOT / "rl/configs/steam.json").read_text())
        with TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "value").mkdir()
            checkpoint = run / "value/model.pt"
            checkpoint.touch()
            pointer = run / "value/latest.json"
            for step in (512, 16000):
                pointer.write_text(json.dumps(dict(step=step, checkpoint=str(checkpoint))))
                if step == 512:
                    with self.assertRaises(RuntimeError):
                        steam.command(ROOT, cfg, run, "label", [], False)
                else:
                    cmd, _ = steam.command(ROOT, cfg, run, "label", [], False)
                    self.assertIn(run / "value", cmd)
                    self.assertIn(run / "advantages.parquet", cmd)
            params = steam.settings(ROOT, cfg, run, "policy")
            self.assertEqual(params["advantages"], str(run / "advantages.parquet"))
            self.assertTrue(params["async_train"])

    def test_recap_rejects_labels_from_other_experiment(self):
        cfg = json.loads((ROOT / "rl/configs/recap.json").read_text())
        with TemporaryDirectory() as directory:
            run = Path(directory).resolve()
            cfg["prepared_data"] = str(run)
            marker = run / "advantages_ready.json"
            marker.write_text(json.dumps(dict(ready=True, value_checkpoint="other/best.pt")))
            with self.assertRaises(RuntimeError):
                recap.command(ROOT, cfg, run, "policy", [], False)
            marker.write_text(json.dumps(dict(ready=True, value_checkpoint=str(run / "value/best.pt"))))
            _, env = recap.command(ROOT, cfg, run, "policy", [], False)
            self.assertEqual(env["RECAP_CONFIG"], "recap_rlinf")
            self.assertEqual(env["RECAP_DATA_DIR"], str(run))
