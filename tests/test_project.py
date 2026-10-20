"""Run: python -m unittest discover -s tests -v

Database checks additionally require BIKE_TEST_DB=1 and a seeded, disposable DB.
"""
import os
import unittest

import numpy as np
import pandas as pd
import psycopg

from project_lib import ROOT, connect, load_training_data, prepare_features, validate_raw


class DataSafetyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(ROOT / "data" / "hour.csv")

    def test_source_row_count_and_unique_hours(self):
        timestamps = validate_raw(self.raw)
        self.assertEqual(len(timestamps), 17379)
        self.assertEqual(timestamps.nunique(), len(timestamps))

    def test_duplicate_hour_is_rejected(self):
        raw = pd.concat([self.raw, self.raw.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_raw(raw)

    def test_missing_temperature_is_rejected(self):
        raw = self.raw.copy()
        raw.loc[0, "temp"] = np.nan
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_raw(raw)

    def test_fractional_hour_is_rejected(self):
        raw = self.raw.copy()
        raw["hr"] = raw["hr"].astype(float)
        raw.loc[0, "hr"] = 1.5
        with self.assertRaisesRegex(ValueError, "integer"):
            validate_raw(raw)

    def test_count_components_are_checked(self):
        raw = self.raw.copy()
        raw.loc[0, "cnt"] += 1
        with self.assertRaisesRegex(ValueError, "registered"):
            validate_raw(raw)


@unittest.skipUnless(os.environ.get("BIKE_TEST_DB") == "1", "Disposable PostgreSQL DB not requested")
class DatabaseAndModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = connect()

    @classmethod
    def tearDownClass(cls):
        cls.conn.close()

    def test_time_split_and_join_cardinality(self):
        data = load_training_data(self.conn)
        self.assertEqual(len(data), 17379)
        self.assertEqual(set(data["split"]), {"train", "validation", "test"})

    def test_target_and_components_cannot_change_features(self):
        data = load_training_data(self.conn).head(20)
        original, names = prepare_features(data)
        changed = data.copy()
        changed["rental_count"] = -999
        changed["casual"] = 100000
        changed["registered"] = 200000
        modified, _ = prepare_features(changed)
        np.testing.assert_array_equal(original, modified)
        self.assertEqual(len(names), 53)

    def test_database_rejects_negative_target(self):
        with self.assertRaises(psycopg.errors.CheckViolation):
            with self.conn.transaction():
                self.conn.execute("UPDATE bike_project.hourly_demand SET rental_count=-1 "
                                  "WHERE observed_at=TIMESTAMP '2011-01-01'")

    def test_sql_and_exported_prediction_metrics_match(self):
        import json
        report = json.loads((ROOT / "artifacts" / "metrics.json").read_text(encoding="utf-8"))
        count, sql_mae = self.conn.execute(
            "SELECT COUNT(*), AVG(ABS(p.predicted_count-d.rental_count)) "
            "FROM bike_project.predictions p JOIN bike_project.hourly_demand d "
            "ON p.observed_at=d.observed_at WHERE p.run_id=%s", (report["run_id"],)).fetchone()
        self.assertEqual(count, report["test_rows"])
        self.assertAlmostEqual(sql_mae, report["test_scores"]["mlp"]["mae"], places=4)

    def test_checkpoint_reload_reproduces_saved_prediction(self):
        from project_lib import predict_one
        from datetime import datetime
        data = pd.read_csv(ROOT / "artifacts" / "test_predictions.csv")
        selected = data[data.observed_at == "2012-12-30 17:00:00"].iloc[0]
        result = predict_one(self.conn, datetime(2012, 12, 30, 17))
        self.assertAlmostEqual(result, selected.predicted_count, places=3)


if __name__ == "__main__":
    unittest.main()
