"""CPU-only regression; no model inference or load test."""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
import crossmetric_app as app


def main():
    expected = json.loads((ROOT / "evaluation/expected_results.json").read_text())
    frame = app.load_frame()
    checks = 0
    for task, values in expected.items():
        _, _, _, decision = app.compute_evidence(frame, task)
        for key, value in values.items():
            actual = decision[key]
            if isinstance(value, (float, int)):
                assert math.isclose(actual, value, rel_tol=1e-8, abs_tol=1e-6), (task, key, actual, value)
            else:
                assert actual == value, (task, key, actual, value)
            checks += 1
        for language in ("English", "中文"):
            narrative = app.safe_narrative(decision, language)
            valid, reason = app.validate_model_response(narrative, decision, language)
            assert valid, (task, language, reason)
            accepted, _ = app.validate_model_response(narrative + "\nInvented metric: £999999", decision, language)
            assert not accepted, (task, language, "fabricated metric accepted")
            checks += 2
        print(f"PASS: {task}: calculation, bilingual templates, fabricated-number rejection")
    print(f"PASS: {checks} assertions; no GPU inference or load test.")


if __name__ == "__main__":
    main()
