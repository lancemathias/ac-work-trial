"""Integration checks against the actual downloaded evaluation artifacts."""
import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class BenchmarkArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((ROOT / "data/manifests/benchmark-preprocessing.json").read_text())
        cls.rows = {}
        for suite, report in cls.manifest["suites"].items():
            path = ROOT / "data/processed" / f"{suite}.jsonl"
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != report["sha256"]:
                raise AssertionError(f"Output checksum mismatch: {suite}")
            cls.rows[suite] = [json.loads(line) for line in payload.splitlines()]

    def test_complete_unique_evaluation_schema(self):
        self.assertEqual(len(self.rows), 11)
        ids = set()
        for suite, rows in self.rows.items():
            self.assertEqual(len(rows), self.manifest["suites"][suite]["rows"])
            for row in rows:
                self.assertNotIn(row["id"], ids)
                ids.add(row["id"])
                self.assertTrue(row["evaluation_only"])
                self.assertTrue(row["group"])
                q, label = row["question"], row["label"]
                if q["type"] == "choice":
                    self.assertIn(label, q["criteria"])
                elif q["type"] == "noul":
                    self.assertIs(type(label), bool)
                elif q["type"] == "score":
                    self.assertIs(type(label), int)
                    self.assertTrue(0 <= label < len(q["criteria"]))
                else:
                    self.fail(f"Unknown question type: {q['type']}")

    def test_annotation_fields_excluded_from_model_inputs(self):
        for row in self.rows["RAGTruth"]:
            self.assertEqual(set(row["state"]), {"source", "response"})
        for row in self.rows["ContractNLI"]:
            self.assertEqual(set(row["state"]), {"contract", "hypothesis"})
        for row in self.rows["JevBench_public_hard"]:
            self.assertNotIn("provenance", row)
            self.assertNotIn("rationale", row)
            self.assertNotIn("surface_answer", row)

    def test_grouping_and_missing_labels(self):
        rows = self.rows["TruthfulQA_binary"]
        self.assertEqual(len({r["group"] for r in rows}), 817)
        self.assertGreater(sum(r["group"] == "0" for r in rows), 1)
        self.assertNotIn("NA", {r["label"] for r in self.rows["Circa"]})
        self.assertEqual(sum(r["question"]["type"] == "score" for r in self.rows["JevBench_public_hard"]), 6)


if __name__ == "__main__":
    unittest.main()
