import numpy as np
import unittest

from MPC_dual_model.heave_dynamics_analysis import _segments_from_phases


class HeaveDynamicsAnalysisTest(unittest.TestCase):
    def test_signed_excitations_include_following_neutral_coast(self) -> None:
        phases = np.asarray(
            ["cycle_1_positive"] * 25
            + ["cycle_1_neutral_blocked"] * 25
            + ["cycle_1_negative"] * 25
            + ["cycle_1_neutral_after_blocked"] * 25,
            dtype=object,
        )
        times = np.arange(len(phases), dtype=float) * 0.05
        depths = np.concatenate(
            (
                np.linspace(0.2, 0.25, 25),
                np.linspace(0.25, 0.2, 25),
                np.linspace(0.2, 0.15, 25),
                np.linspace(0.15, 0.2, 25),
            )
        )
        segments, summaries = _segments_from_phases(times, depths, phases)
        self.assertEqual(len(segments), 2)
        self.assertEqual(summaries[0]["phase"], "cycle_1_positive")
        self.assertGreater(summaries[0]["sample_count_with_neutral_coast"], 25)
        self.assertEqual(summaries[1]["phase"], "cycle_1_negative")

    def test_positive_only_buoyancy_coast_is_supported(self) -> None:
        phases = np.asarray(
            ["cycle_1_positive"] * 25
            + ["cycle_1_neutral_after_blocked"] * 25,
            dtype=object,
        )
        times = np.arange(len(phases), dtype=float) * 0.05
        depths = np.concatenate(
            (np.linspace(0.2, 0.25, 25), np.linspace(0.25, 0.2, 25))
        )
        segments, summaries = _segments_from_phases(times, depths, phases)
        self.assertEqual(len(segments), 1)
        self.assertEqual(summaries[0]["phase"], "cycle_1_positive")

    def test_an_excitation_is_required(self) -> None:
        phases = np.asarray(["cycle_1_neutral_before_blocked"] * 50, dtype=object)
        times = np.arange(len(phases), dtype=float) * 0.05
        depths = np.zeros(len(phases))
        with self.assertRaisesRegex(RuntimeError, "no down excitation"):
            _segments_from_phases(times, depths, phases)


if __name__ == "__main__":
    unittest.main()
