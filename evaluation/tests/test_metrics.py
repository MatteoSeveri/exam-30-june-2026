from __future__ import annotations

import math
import unittest

from evaluation.metrics import compute_metrics, standard_error


class TestEvaluationMetrics(unittest.TestCase):
    def test_compute_metrics_usa_una_partita_come_unita_statistica(self):
        # Win/draw/loss and margin are computed from per-match margins.
        metrics = compute_metrics([10, -4, 0, 14])

        self.assertEqual(metrics.games, 4)
        self.assertEqual(metrics.win_rate, 0.5)
        self.assertEqual(metrics.draw_rate, 0.25)
        self.assertEqual(metrics.loss_rate, 0.25)
        self.assertEqual(metrics.mean_point_difference, 5.0)

    def test_standard_error_usa_deviazione_campionaria(self):
        # With few matches, we use sample stdev, not population deviation.
        values = [2, 4, 6]

        self.assertAlmostEqual(standard_error(values), 2.0 / math.sqrt(3))

    def test_confidence_interval_95_centrato_sulla_media(self):
        # The CI is symmetric around the estimated mean margin.
        metrics = compute_metrics([2, 4, 6])
        expected_stderr = 2.0 / math.sqrt(3)

        self.assertAlmostEqual(metrics.standard_error, expected_stderr)
        self.assertAlmostEqual(
            metrics.confidence_interval_95[0],
            4.0 - 1.96 * expected_stderr,
        )
        self.assertAlmostEqual(
            metrics.confidence_interval_95[1],
            4.0 + 1.96 * expected_stderr,
        )

    def test_partita_singola_ha_standard_error_zero(self):
        # With a single statistical unit, we do not estimate between-match variance.
        metrics = compute_metrics([12])

        self.assertEqual(metrics.games, 1)
        self.assertEqual(metrics.mean_point_difference, 12.0)
        self.assertEqual(metrics.standard_error, 0.0)
        self.assertEqual(metrics.confidence_interval_95, (12.0, 12.0))

    def test_input_vuoto_solleva_value_error(self):
        # Metrics without matches would be numerically arbitrary.
        with self.assertRaises(ValueError):
            compute_metrics([])

        with self.assertRaises(ValueError):
            standard_error([])


if __name__ == "__main__":
    unittest.main()
