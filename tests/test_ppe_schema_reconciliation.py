from __future__ import annotations

import unittest

from scripts.canonicalize_ppe_test_schema import remap_label_line


class PPESchemaReconciliationTests(unittest.TestCase):
    def test_remap_changes_only_the_verified_class_id(self) -> None:
        self.assertEqual(remap_label_line("0 0.1 0.2 0.3 0.4"), "1 0.1 0.2 0.3 0.4")
        self.assertEqual(remap_label_line("1 0.1 0.2 0.3 0.4"), "2 0.1 0.2 0.3 0.4")
        self.assertEqual(remap_label_line("2 0.1 0.2 0.3 0.4"), "3 0.1 0.2 0.3 0.4")

    def test_unexpected_source_id_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            remap_label_line("3 0.1 0.2 0.3 0.4")


if __name__ == "__main__":
    unittest.main()
