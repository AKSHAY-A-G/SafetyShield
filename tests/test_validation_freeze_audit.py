from __future__ import annotations

import unittest

from scripts.audit_ppe_validation_freeze import original_name, resolve_manual_provenance


class ValidationFreezeAuditTests(unittest.TestCase):
    def test_roboflow_filename_recovers_original_jpg_and_png_names(self) -> None:
        self.assertEqual(
            original_name("cam1_clip_t0002000_p00_jpg.rf.abc123.jpg"),
            "cam1_clip_t0002000_p00.jpg",
        )
        self.assertEqual(
            original_name("Screenshot-2026-09-15-105420_png.rf.abc123.jpg"),
            "Screenshot-2026-09-15-105420.png",
        )

    def test_user_confirmed_manual_provenance_keeps_frame_metadata_unknown(self) -> None:
        resolved = resolve_manual_provenance(
            {"original_filename": "Screenshot-2026-09-15-105420.png"},
            camera_id="cam1",
            clip_id="cam1_ppe_validation_2026-09-15",
            source_video="cam1_ppe_validation_2026-09-15.mp4.mp4",
        )
        self.assertEqual(resolved["provenance_basis"], "explicit_user_confirmation")
        self.assertEqual(resolved["source_group"], "cam1_ppe_validation_2026-09-15")
        self.assertIsNone(resolved["source_frame_number"])
        self.assertIsNone(resolved["source_timestamp_seconds"])


if __name__ == "__main__":
    unittest.main()
