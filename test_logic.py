import unittest
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from data_loader import parse_gpx_file, generate_synthetic_runs, load_gpx_folder
from feature_engineering import compute_point_features, compute_run_summary, build_feature_matrix
from models import train_pace_regression, train_run_classifier, predict_pace, classify_run
from gpx_exporter import dataframe_to_gpx_bytes, runs_to_zip_bytes

class TestGPXAnalyticsLogic(unittest.TestCase):
    
    def setUp(self):
        # Generate small set of synthetic runs for testing
        self.runs = generate_synthetic_runs(n_runs=5, seed=42)
        self.run_names = list(self.runs.keys())
        
    def test_synthetic_runs_generation(self):
        self.assertEqual(len(self.runs), 5)
        for name, df in self.runs.items():
            self.assertIsInstance(df, pd.DataFrame)
            self.assertTrue(all(col in df.columns for col in ["lat", "lon", "elevation", "timestamp"]))
            self.assertGreater(len(df), 0)

    def test_point_features_computation(self):
        first_run_df = self.runs[self.run_names[0]]
        enriched = compute_point_features(first_run_df)
        
        expected_cols = [
            "segment_dist_m", "cumulative_dist_m", "dist_km", 
            "elapsed_s", "speed_kmh", "pace_min_km", 
            "elev_diff", "elev_gain", "elev_loss", 
            "rolling_pace", "pace_variability"
        ]
        for col in expected_cols:
            self.assertIn(col, enriched.columns)
            
        # Cumulative distance should be non-decreasing
        self.assertTrue((enriched["cumulative_dist_m"].diff().dropna() >= -1e-5).all())

    def test_run_summary_computation(self):
        first_run_df = self.runs[self.run_names[0]]
        summary = compute_run_summary(first_run_df, run_name=self.run_names[0])
        
        self.assertIsInstance(summary, dict)
        self.assertEqual(summary["run_name"], self.run_names[0])
        self.assertIn("total_dist_km", summary)
        self.assertIn("avg_pace_min_km", summary)
        self.assertIn("fatigue_index", summary)
        self.assertIn("elev_gain_per_km", summary)

    def test_feature_matrix_building(self):
        feat_df = build_feature_matrix(self.runs)
        self.assertIsInstance(feat_df, pd.DataFrame)
        self.assertEqual(len(feat_df), 5)
        self.assertIn("avg_pace_min_km", feat_df.columns)
        self.assertIn("run_type_code", feat_df.columns)

    def test_model_training_and_prediction(self):
        feat_df = build_feature_matrix(self.runs)
        
        # Test regression
        reg_result = train_pace_regression(feat_df, model_name="linear_regression", test_size=0.2)
        self.assertIn("mae", reg_result)
        self.assertIn("rmse", reg_result)
        self.assertIn("pipeline", reg_result)
        
        hyp_features = {
            "total_dist_km": 10.0,
            "total_elevation_gain_m": 100.0,
            "elev_gain_per_km": 10.0,
            "run_type_code": 0,
            "avg_pace_variability": 0.3,
            "fatigue_index": 0.1,
        }
        pred_pace = predict_pace(reg_result["pipeline"], hyp_features, reg_result["features"])
        self.assertIsInstance(pred_pace, float)
        self.assertGreater(pred_pace, 0.0)

        # Test classification
        clf_result = train_run_classifier(feat_df, model_name="logistic_regression", test_size=0.2)
        self.assertIn("accuracy", clf_result)
        self.assertIn("pipeline", clf_result)
        
        hyp_features_clf = {
            "total_dist_km": 10.0,
            "total_elevation_gain_m": 100.0,
            "avg_pace_min_km": 6.0,
            "std_pace_min_km": 0.4,
            "avg_pace_variability": 0.3,
            "fatigue_index": 0.1,
            "total_duration_min": 60.0,
            "elev_gain_per_km": 10.0,
        }
        pred_label, proba = classify_run(clf_result["pipeline"], hyp_features_clf, clf_result["features"])
        self.assertIsInstance(pred_label, str)
        self.assertIn(pred_label, ["easy", "tempo", "interval", "long"])

    def test_gpx_export(self):
        first_run_df = self.runs[self.run_names[0]]
        gpx_bytes = dataframe_to_gpx_bytes(first_run_df)
        self.assertIsInstance(gpx_bytes, bytes)
        self.assertTrue(gpx_bytes.startswith(b"<?xml"))
        
        zip_bytes = runs_to_zip_bytes(self.runs)
        self.assertIsInstance(zip_bytes, bytes)

if __name__ == "__main__":
    unittest.main()
