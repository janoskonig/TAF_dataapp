import unittest

import numpy as np
import pandas as pd

from longitudinal_analysis import PREDICTORS, fit_continuous_model, fit_ordinal_model, simulate_followup_frame


class LongitudinalModelTests(unittest.TestCase):
    def test_models_recover_expected_direction(self):
        rng = np.random.default_rng(20260819)
        n = 60
        predictor = rng.integers(0, 2, n).astype(float)
        baseline = rng.normal(8, 2, n)
        followup = 0.65 * baseline + 1.8 * predictor + rng.normal(0, 1, n)
        latent_anchor = -0.9 * predictor + rng.logistic(size=n)
        anchor = pd.qcut(latent_anchor, 5, labels=False, duplicates="drop") + 1
        frame = pd.DataFrame(
            {
                "outcome": followup,
                "baseline": baseline,
                "predictor": predictor,
                "anchor": anchor,
            }
        )

        linear = fit_continuous_model(
            frame,
            outcome="outcome",
            baseline="baseline",
            outcome_label="teszt",
            scale_note="teszt",
            predictor="predictor",
            predictor_label="teszt",
            predictor_scale="teszt",
        )
        ordinal = fit_ordinal_model(
            frame,
            anchor="anchor",
            anchor_label="teszt",
            anchor_role="teszt",
            predictor="predictor",
            predictor_label="teszt",
            predictor_scale="teszt",
        )

        self.assertEqual(linear["status"], "ok")
        self.assertGreater(linear["beta"], 0)
        self.assertEqual(ordinal["status"], "ok")
        self.assertLess(ordinal["odds_ratio"], 1)

    def test_small_sample_does_not_report_estimate(self):
        frame = pd.DataFrame(
            {
                "outcome": [1, 2, 3, 4, 5],
                "baseline": [1, 1, 2, 2, 3],
                "predictor": [0, 1, 0, 1, 0],
            }
        )
        result = fit_continuous_model(
            frame,
            outcome="outcome",
            baseline="baseline",
            outcome_label="teszt",
            scale_note="teszt",
            predictor="predictor",
            predictor_label="teszt",
            predictor_scale="teszt",
        )
        self.assertEqual(result["status"], "insufficient")
        self.assertNotIn("beta", result)

    def test_simulation_is_deterministic_and_kept_in_memory(self):
        source_values = {
            key: [0.0, 0.25, 0.5, 0.75, 1.0]
            for key, _, _ in PREDICTORS
        }
        source_values.update(
            ohip_baseline=[4, 6, 8, 10, 12],
            gohai_baseline=[56, 52, 48, 44, 40],
            mai_baseline=[30, 35, 40, 45, 50],
        )
        source = pd.DataFrame(source_values)
        first = simulate_followup_frame(source, n=100, seed=123)
        second = simulate_followup_frame(source, n=100, seed=123)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(len(first), 100)
        self.assertTrue(first["ohip_followup"].between(0, 20).all())
        self.assertTrue(first["gohai_followup"].between(12, 60).all())


if __name__ == "__main__":
    unittest.main()
