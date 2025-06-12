#!/usr/bin/env python3
import unittest
import os
import subprocess
import pandas as pd
import sys
from pathlib import Path
import tempfile

# --- Define project structure relative to this test file ---
TEST_DIR = Path(__file__).parent.resolve()
MOCK_DATA_DIR = TEST_DIR / "mock_data"
PROJECT_ROOT = TEST_DIR.parent  # Assumes test is in metrics_calculator/test/
SCRIPT_PATH = PROJECT_ROOT / "calculate_metrics.py"
# --- End Structure Definition ---


class TestCalculateMetricsScript(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mock_data_dir = MOCK_DATA_DIR
        os.makedirs(cls.mock_data_dir, exist_ok=True)

        cls.script_path = SCRIPT_PATH
        if not cls.script_path.exists():
            raise FileNotFoundError(f"Script not found at {cls.script_path}")

        # Input paths using pathlib
        cls.agreement_map_path = cls.mock_data_dir / "agreement_output.tif"
        
        # Output paths using pathlib - use temp file for output
        cls.metrics_output_path = cls.mock_data_dir / "test_metrics_output.csv"

        # Check for input files
        required_files = [cls.agreement_map_path]
        missing_files = [p for p in required_files if not p.exists()]
        if missing_files:
            print(
                f"\nWARNING: The following test files are missing: {missing_files}"
            )
            print(
                "Ensure the agreement map file exists from the agreement_maker tests."
            )
            print("Tests requiring these files may fail or be skipped.")
            cls.input_files_exist = False
        else:
            cls.input_files_exist = True

    def test_metrics_calculation(self):
        """Tests metrics calculation from agreement map."""
        if not self.input_files_exist:
            self.skipTest("Skipping test because input files are missing")

        # Remove output file if it exists
        if self.metrics_output_path.exists():
            self.metrics_output_path.unlink()

        cmd = [
            sys.executable,  # Use the current Python interpreter
            str(self.script_path),
            "--agreement_map_path",
            str(self.agreement_map_path),
            "--metrics_path",
            str(self.metrics_output_path),
            "--chunk_size",
            "512"
        ]

        # --- Set Environment Variables for subprocess ---
        test_env = os.environ.copy()
        test_env["GDAL_CACHEMAX"] = "1024"

        print(f"\nRunning command: {' '.join(cmd)}")
        print(f"With Environment: GDAL_CACHEMAX={test_env['GDAL_CACHEMAX']}")

        # Run with full output capture
        result = subprocess.run(cmd, capture_output=True, text=True, env=test_env)

        # Print full output for debugging - stderr contains JSON logs
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")

        # Check return code first
        self.assertEqual(
            result.returncode,
            0,
            f"Script failed with return code {result.returncode}. Check STDERR logs above.",
        )

        # Check if the output file was created
        self.assertTrue(
            self.metrics_output_path.exists(),
            f"Metrics output file was not created at {self.metrics_output_path}",
        )

        # Verify the output CSV properties
        try:
            # Read the CSV file
            metrics_df = pd.read_csv(self.metrics_output_path, index_col=0)
            
            # Check that the DataFrame is not empty
            self.assertFalse(
                metrics_df.empty,
                "Metrics CSV should contain data"
            )
            
            # Check for expected metrics columns (based on gval categorical metrics)
            expected_metrics = [
                'critical_success_index',
                'f_score',
                'matthews_correlation_coefficient', 
                'cohens_kappa',
                'overall_accuracy',
                'producers_accuracy',
                'users_accuracy'
            ]
            
            # Check that at least some expected metrics are present
            found_metrics = [col for col in expected_metrics if col in metrics_df.columns]
            self.assertGreater(
                len(found_metrics),
                0,
                f"Expected at least some metrics from {expected_metrics}, but found columns: {list(metrics_df.columns)}"
            )
            
            # Check that metric values are numeric
            for col in metrics_df.columns:
                values = metrics_df[col]
                self.assertTrue(
                    pd.api.types.is_numeric_dtype(values),
                    f"Metric {col} should have numeric values"
                )
            
            print(f"Successfully calculated metrics:")
            print(metrics_df.to_string())

        except Exception as e:
            self.fail(
                f"Failed to read or validate the metrics CSV file: {self.metrics_output_path}. Error: {e}"
            )

    def test_metrics_with_custom_chunk_size(self):
        """Tests metrics calculation with custom chunk size."""
        if not self.input_files_exist:
            self.skipTest("Skipping test because input files are missing")

        # Use different output path for this test
        metrics_output_path_chunked = self.mock_data_dir / "test_metrics_output_chunked.csv"
        
        # Remove output file if it exists
        if metrics_output_path_chunked.exists():
            metrics_output_path_chunked.unlink()

        cmd = [
            sys.executable,
            str(self.script_path),
            "--agreement_map_path",
            str(self.agreement_map_path),
            "--metrics_path",
            str(metrics_output_path_chunked),
            "--chunk_size",
            "256"  # Different chunk size
        ]

        # --- Set Environment Variables for subprocess ---
        test_env = os.environ.copy()
        test_env["GDAL_CACHEMAX"] = "512"

        print(f"\nRunning command (custom chunk): {' '.join(cmd)}")

        # Run with full output capture
        result = subprocess.run(cmd, capture_output=True, text=True, env=test_env)

        # Print output for debugging
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")

        # Check return code
        self.assertEqual(
            result.returncode,
            0,
            f"Script failed with return code {result.returncode}. Check STDERR logs above.",
        )

        # Check if the output file was created
        self.assertTrue(
            metrics_output_path_chunked.exists(),
            f"Metrics output file was not created at {metrics_output_path_chunked}",
        )

        # Verify that results are consistent (should be same as default chunk size)
        if self.metrics_output_path.exists():
            try:
                metrics_df_default = pd.read_csv(self.metrics_output_path, index_col=0)
                metrics_df_chunked = pd.read_csv(metrics_output_path_chunked, index_col=0)
                
                # Check that both DataFrames have the same shape
                self.assertEqual(
                    metrics_df_default.shape,
                    metrics_df_chunked.shape,
                    "Metrics from different chunk sizes should have same shape"
                )
                
                # Check that values are approximately equal (allowing for small numerical differences)
                for col in metrics_df_default.columns:
                    if col in metrics_df_chunked.columns:
                        pd.testing.assert_series_equal(
                            metrics_df_default[col],
                            metrics_df_chunked[col],
                            check_names=False,
                            rtol=1e-10,  # Allow for small numerical differences
                            atol=1e-10
                        )

            except Exception as e:
                self.fail(
                    f"Failed to compare metrics from different chunk sizes. Error: {e}"
                )


if __name__ == "__main__":
    unittest.main()