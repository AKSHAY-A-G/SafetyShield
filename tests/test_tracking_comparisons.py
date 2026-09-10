"""Tests for matched tracking comparison contact sheets."""

import unittest

from scripts.extract_tracking_comparisons import interval_timestamps, parse_interval


class TrackingComparisonTests(unittest.TestCase):
    def test_ten_second_interval_has_twenty_half_second_samples(self):
        timestamps = interval_timestamps(5.0, 15.0)
        self.assertEqual(len(timestamps), 20)
        self.assertEqual(timestamps[0], 5.0)
        self.assertEqual(timestamps[-1], 14.5)

    def test_invalid_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            interval_timestamps(15.0, 5.0)

    def test_cli_interval_parser(self):
        self.assertEqual(parse_interval("25:35"), (25.0, 35.0))


if __name__ == "__main__":
    unittest.main()
