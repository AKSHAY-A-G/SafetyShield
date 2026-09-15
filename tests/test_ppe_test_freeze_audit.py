from __future__ import annotations

import unittest

from scripts.audit_ppe_test_freeze import manual_screenshot_record, original_name


class PPETestFreezeAuditTests(unittest.TestCase):
    def test_roboflow_filename_recovers_manual_screenshot_name(self) -> None:
        self.assertEqual(
            original_name("cam2_ppe_test_2026-09-15_manual_101-jpg_png.rf.abc123.jpg"),
            "cam2_ppe_test_2026-09-15_manual_101-jpg.png",
        )

    def test_manual_vlc_provenance_does_not_infer_cam2(self) -> None:
        record = manual_screenshot_record("export.jpg", "screenshot.png")
        self.assertIsNone(record["camera_id"])
        self.assertEqual(record["source_type"], "live_stream")
        self.assertEqual(record["capture_method"], "manual_vlc_screenshot")
        self.assertIsNone(record["source_frame_number"])


if __name__ == "__main__":
    unittest.main()
