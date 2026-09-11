"""Non-interactive tests for the zone-definition utility."""

from pathlib import Path
import tempfile
import unittest

from scripts.draw_zone import save_zone
from src.rules.zones import load_camera_zones


class ZoneDrawingPersistenceTests(unittest.TestCase):
    def test_explicit_save_writes_loadable_camera_zone(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zones.yaml"
            save_zone(
                path=path,
                camera_id="camera-a",
                width=100,
                height=80,
                zone_id="restricted-a",
                zone_name="Restricted A",
                zone_type="restricted",
                polygon=[(10, 10), (90, 10), (90, 70), (10, 70)],
                confirm_inside_seconds=0.2,
                confirm_outside_seconds=0.2,
                state_ttl_seconds=2.0,
            )
            config = load_camera_zones(path, "camera-a", 100, 80)
            self.assertEqual(config.expected_width, 100)
            self.assertEqual(config.expected_height, 80)
            self.assertEqual(len(config.zones), 1)
            self.assertEqual(config.zones[0].zone_id, "restricted-a")

    def test_save_rejects_existing_resolution_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "zones.yaml"
            save_zone(
                path,
                "camera-a",
                100,
                80,
                "first",
                "First",
                "counting",
                [(1, 1), (90, 1), (1, 70)],
                0.2,
                0.2,
                2.0,
            )
            with self.assertRaisesRegex(ValueError, "resolution does not match"):
                save_zone(
                    path,
                    "camera-a",
                    200,
                    80,
                    "second",
                    "Second",
                    "restricted",
                    [(1, 1), (190, 1), (1, 70)],
                    0.2,
                    0.2,
                    2.0,
                )


if __name__ == "__main__":
    unittest.main()
