from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from test import TestWorkflow


class AutomationWorkflowTests(unittest.TestCase):
    def test_failed_stage_creates_bug_report_with_reproduction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workflow = TestWorkflow("smoke", Path(directory))

            def fail_deliberately() -> dict:
                raise RuntimeError("controlled test failure")

            stage = workflow.run_stage(
                "controlled",
                "Controlled failure",
                fail_deliberately,
                "python test.py smoke",
            )
            json_report, markdown_report = workflow.write_report()
            report = json.loads(json_report.read_text(encoding="utf-8"))

            self.assertEqual(stage.status, "failed")
            self.assertFalse(workflow.passed)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["bugs"][0]["bug_id"], "BUG-001")
            self.assertEqual(report["bugs"][0]["reproduce"], "python test.py smoke")
            self.assertIn("controlled test failure", report["bugs"][0]["traceback"])
            self.assertIn("BUG-001", markdown_report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
