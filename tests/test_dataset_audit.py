from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from model_router.dataset_audit import (
    BALANCED_DATASET,
    MULTITURN_FAMILY,
    PRIMARY_DATASET,
    audit_datagen_directory,
    write_report,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


class DatasetAuditTests(unittest.TestCase):
    def test_audit_reports_overlap_without_emitting_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary_rows = [
                {
                    "prompt": "private alpha task",
                    "category": "Facts",
                    "complexity_level": 1,
                    "complexity_audit_status": "appropriate",
                    "complexity_audit_confidence": 0.9,
                    "complexity_audit_suggested_level": 1,
                },
                {
                    "prompt": "private beta task",
                    "category": "Logic",
                    "complexity_level": 2,
                    "complexity_audit_status": "borderline",
                    "complexity_audit_confidence": 0.8,
                    "complexity_audit_suggested_level": 3,
                },
            ]
            _write_jsonl(root / PRIMARY_DATASET, primary_rows)
            _write_jsonl(root / BALANCED_DATASET, primary_rows[:1])
            family_rows = [
                primary_rows[0],
                {
                    "prompt": "private novel task",
                    "category": "Logic",
                    "complexity_level": 3,
                },
            ]
            for name in MULTITURN_FAMILY:
                _write_jsonl(root / name, family_rows)

            report = audit_datagen_directory(root)
            self.assertEqual(report["multiturn_family"]["novel_union_vs_primary"], 1)
            self.assertEqual(
                report["primary_audit_annotations"][
                    "borderline_with_adjacent_suggestion"
                ],
                1,
            )
            encoded = json.dumps(report)
            self.assertNotIn("private alpha task", encoded)
            self.assertNotIn("private novel task", encoded)

            json_path = root / "audit.json"
            markdown_path = root / "audit.md"
            write_report(report, json_path, markdown_path)
            self.assertTrue(json_path.is_file())
            self.assertIn("aggregate counts", markdown_path.read_text())


if __name__ == "__main__":
    unittest.main()
