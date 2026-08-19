#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("plot.py")
SPEC = importlib.util.spec_from_file_location("pretraining_overlap_plot", MODULE_PATH)
assert SPEC and SPEC.loader
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


class PretrainingOverlapFigureTests(unittest.TestCase):
    def test_only_two_morphem_defined_exclusions_are_displayed(self) -> None:
        frame = plot.validate_inputs()
        self.assertEqual(set(frame["subset"]), set(plot.SUBSET_ORDER))
        self.assertNotIn("joint_named_jump_and_morphem_plates_excluded", set(frame["subset"]))

    def test_three_comparators_per_exclusion(self) -> None:
        frame = plot.validate_inputs()
        self.assertEqual(set(frame["comparator"]), set(plot.COMPARATOR_ORDER))
        self.assertEqual(frame.groupby("subset").size().to_dict(), {name: 3 for name in plot.SUBSET_ORDER})

    def test_all_six_frozen_intervals_are_positive_and_supported(self) -> None:
        frame = plot.validate_inputs()
        self.assertTrue((frame["ci_low"] > 0).all())
        self.assertTrue((frame["holm_p"] <= 0.05).all())

    def test_render_is_byte_deterministic(self) -> None:
        frame = plot.validate_inputs()
        with tempfile.TemporaryDirectory() as tmp_name:
            tmp = Path(tmp_name)
            plot.render(frame, tmp / "a.pdf", tmp / "a.png")
            plot.render(frame, tmp / "b.pdf", tmp / "b.png")
            self.assertEqual(plot.sha256(tmp / "a.pdf"), plot.sha256(tmp / "b.pdf"))
            self.assertEqual(plot.sha256(tmp / "a.png"), plot.sha256(tmp / "b.png"))


if __name__ == "__main__":
    unittest.main()
