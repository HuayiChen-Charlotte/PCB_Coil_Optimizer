import json
import unittest
from pathlib import Path

from coil_optimizer import load_config, run


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SmokeRunTests(unittest.TestCase):
    def test_smoke_config_produces_ranked_outputs(self):
        config = load_config(PROJECT_ROOT / "examples" / "smoke_test.json")
        output_dir = PROJECT_ROOT / "outputs" / "test_smoke"
        config["output_dir"] = str(output_dir)
        config["search"]["show_progress"] = False
        summary = run(config)

        self.assertGreater(summary["counts"]["valid_loaded_candidate_count"], 0)
        self.assertIsNotNone(summary["top_loaded_efficiency_candidate"])
        self.assertIsNotNone(summary["top_eta_max_candidate"])
        self.assertTrue((output_dir / "candidates_loaded_efficiency.csv").is_file())
        self.assertTrue((output_dir / "candidates_eta_max.csv").is_file())
        self.assertTrue((output_dir / "summary.json").is_file())

        saved_summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        self.assertEqual(saved_summary["case_name"], "smoke_test")


if __name__ == "__main__":
    unittest.main()
