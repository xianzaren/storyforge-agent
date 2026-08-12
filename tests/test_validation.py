from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from storyforge.validation import audit_run


class ValidationTests(unittest.TestCase):
    def test_completed_with_warnings_is_a_successful_terminal_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "state.json").write_text(
                json.dumps({"status": "completed_with_warnings", "artifacts": {}, "warnings": ["expected"]}),
                encoding="utf-8",
            )

            audit = audit_run(run_dir)

            self.assertFalse(any(issue.code == "workflow_status" for issue in audit.issues))

    def test_incomplete_run_produces_actionable_issues(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            audit = audit_run(Path(directory))
            self.assertFalse(audit.passed)
            codes = {issue.code for issue in audit.issues}
            self.assertIn("missing_artifact", codes)
            self.assertIn("workflow_status", codes)
            self.assertIn("asset_assignments", codes)


if __name__ == "__main__":
    unittest.main()
