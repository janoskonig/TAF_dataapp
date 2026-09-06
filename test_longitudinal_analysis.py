import unittest

import numpy as np
import pandas as pd

from longitudinal_analysis import (
    PREDICTORS,
    PREDICTOR_GROUPS,
    SUCCESS_INDEX_OUTCOME,
    add_success_index,
    fit_continuous_model,
    fit_ordinal_model,
    mai_reference_delta,
    prepare_analysis_frame,
    simulate_followup_frame,
    success_index_summary,
)


class LongitudinalModelTests(unittest.TestCase):
    def test_predictors_are_the_four_lower_and_four_upper_categories(self):
        self.assertEqual(len(PREDICTORS), 8)
        self.assertEqual([len(group) for _, _, group in PREDICTOR_GROUPS], [4, 4])

        values = {
            "f1": [20, 10],
            "f5": [1, 2],
            "f7": [1, 3],
            "f9": [1, 3],
            "a1_kaan": [1, 5],
            "a4_jobb": [1, 2],
            "a4_bal": [1, 1],
            "a11": [2, 3],
            "mai_baseline": [30, 40],
            "mai_followup": [25, 45],
            "oral_anchor_text": ["Kicsit javult", "Változatlan maradt"],
            "chewing_anchor_text": ["Kicsit javult", "Változatlan maradt"],
        }
        for number in range(6, 10):
            for side in ("jobb", "bal"):
                values[f"a{number}_{side}"] = [1, 2 if number == 8 else 3]
        for prefix, count in (("ohip", 5), ("gohai", 12)):
            for number in range(1, count + 1):
                values[f"{prefix}_{number}"] = [1, 2]
                values[f"{prefix}_{number}_recall"] = [1, 2]

        prepared = prepare_analysis_frame(pd.DataFrame(values))
        predictor_keys = [key for key, _, _ in PREDICTORS]
        self.assertTrue((prepared.loc[0, predictor_keys] == 0).all())
        self.assertTrue((prepared.loc[1, predictor_keys] == 1).all())
        # OHIP and GOHAI unchanged; MAI 30→25 (+5/80) and 40→45 (−5/80), mean of three
        self.assertAlmostEqual(prepared.loc[0, "success_index"], 100 * (5 / 80) / 3)
        self.assertAlmostEqual(prepared.loc[1, "success_index"], -100 * (5 / 80) / 3)

    def test_success_index_uses_fixed_denominators(self):
        frame = pd.DataFrame(
            {
                "ohip_baseline": [10, 4, 2, 6],
                "ohip_followup": [5, 6, 2, np.nan],
                "gohai_baseline": [40, 50, 30, 45],
                "gohai_followup": [52, 47, 30, 55],
                "mai_baseline": [60, 30, 50, 70],
                "mai_followup": [40, 40, 50, 20],
            }
        )
        indexed = add_success_index(frame, mai_reference=80)
        # Row 0: every component is 25% of its maximum conceivable improvement.
        self.assertAlmostEqual(indexed.loc[0, "success_index_ohip"], 25.0)
        self.assertAlmostEqual(indexed.loc[0, "success_index_gohai"], 25.0)
        self.assertAlmostEqual(indexed.loc[0, "success_index_mai"], 25.0)
        self.assertAlmostEqual(indexed.loc[0, "success_index"], 25.0)
        # Row 1: net deterioration gives a negative index.
        self.assertLess(indexed.loc[1, "success_index"], 0)
        # Row 2: no change at all gives exactly zero.
        self.assertEqual(indexed.loc[2, "success_index"], 0.0)
        # Row 3: a missing pair leaves the index missing (no imputation).
        self.assertTrue(np.isnan(indexed.loc[3, "success_index"]))
        # The denominators are fixed: an extra extreme patient does not rescale the others.
        extended = pd.concat([frame, pd.DataFrame({
            "ohip_baseline": [20], "ohip_followup": [0], "gohai_baseline": [12],
            "gohai_followup": [60], "mai_baseline": [80], "mai_followup": [0],
        })], ignore_index=True)
        rescaled = add_success_index(extended, mai_reference=80)
        self.assertAlmostEqual(rescaled.loc[0, "success_index"], 25.0)
        self.assertAlmostEqual(rescaled.loc[4, "success_index"], 100.0)

        summary = success_index_summary(indexed, mai_reference=80)
        self.assertEqual(summary["n"], 3)
        self.assertEqual([component["denominator"] for component in summary["components"]], [20.0, 48.0, 80.0])
        self.assertFalse(summary["mai_reference_exceeded"])
        self.assertTrue(success_index_summary(indexed, mai_reference=40)["mai_reference_exceeded"])

    def test_mai_reference_must_be_positive(self):
        self.assertEqual(mai_reference_delta(65), 65.0)
        with self.assertRaises(ValueError):
            mai_reference_delta(0)
        with self.assertRaises(ValueError):
            mai_reference_delta("nem szám")

    def test_index_model_is_unadjusted(self):
        rng = np.random.default_rng(20260906)
        n = 60
        predictor = rng.integers(0, 2, n).astype(float)
        index = 20 - 8 * predictor + rng.normal(0, 4, n)
        frame = pd.DataFrame({"success_index": index, "predictor": predictor})
        result = fit_continuous_model(
            frame,
            outcome=SUCCESS_INDEX_OUTCOME[0],
            baseline=SUCCESS_INDEX_OUTCOME[1],
            outcome_label=SUCCESS_INDEX_OUTCOME[2],
            scale_note=SUCCESS_INDEX_OUTCOME[3],
            predictor="predictor",
            predictor_label="teszt",
            predictor_scale="teszt",
        )
        self.assertEqual(result["status"], "ok")
        self.assertLess(result["beta"], 0)
        self.assertTrue(result["adjustment"].startswith("nincs"))

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
        self.assertTrue(np.isfinite(first["success_index"]).all())
        self.assertTrue(first["success_index"].between(-100, 100).all())


if __name__ == "__main__":
    unittest.main()
