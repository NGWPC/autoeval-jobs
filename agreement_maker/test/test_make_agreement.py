#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import rasterio

# --- Define project structure relative to this test file ---
TEST_DIR = Path(__file__).parent.resolve()
MOCK_DATA_DIR = TEST_DIR / "mock_data"
PROJECT_ROOT = TEST_DIR.parent  # Assumes test is in agreement_maker/test/
SCRIPT_PATH = PROJECT_ROOT / "make_agreement.py"
# --- End Structure Definition ---


class TestAgreementScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock_data_dir = MOCK_DATA_DIR
        os.makedirs(cls.mock_data_dir, exist_ok=True)

        cls.script_path = SCRIPT_PATH
        if not cls.script_path.exists():
            raise FileNotFoundError(f"Script not found at {cls.script_path}")

        # Input paths using pathlib
        cls.candidate_path = cls.mock_data_dir / "candidate_raster.tif"
        cls.benchmark_path = cls.mock_data_dir / "benchmark_raster.tif"
        cls.mask_dict_path = cls.mock_data_dir / "mask_dict.json"
        cls.clip_gpkg_path = cls.mock_data_dir / "clip_square.gpkg"

        # Output paths using pathlib
        cls.output_path = cls.mock_data_dir / "agreement_output.tif"
        cls.metrics_path = cls.mock_data_dir / "metrics_output.csv"

        # Check for input files
        required_files = [cls.candidate_path, cls.benchmark_path, cls.mask_dict_path]
        missing_files = [p for p in required_files if not p.exists()]
        if missing_files:
            print(f"\nWARNING: The following test files are missing: {missing_files}")
            print("Run 'make_test_agreement_data.py' in the mock_data directory or ensure they exist.")
            print("Tests requiring these files may fail or be skipped.")
            cls.input_files_exist = False
        else:
            cls.input_files_exist = True

    def test_agreement_creation_extent(self):
        """Tests agreement creation with fim_type='extent'."""
        if not self.input_files_exist:
            self.skipTest("Skipping test because input files are missing")

        # Remove output files if they exist
        if self.output_path.exists():
            self.output_path.unlink()
        if self.metrics_path.exists():
            self.metrics_path.unlink()

        cmd = [
            sys.executable,  # Use the current Python interpreter
            str(self.script_path),
            "--fim_type",
            "extent",
            "--candidate_path",
            str(self.candidate_path),
            "--benchmark_path",
            str(self.benchmark_path),
            "--output_path",
            str(self.output_path),
            "--metrics_path",
            str(self.metrics_path),
            "--mask_dict",
            str(self.mask_dict_path),
            "--block_size",
            "512",
        ]

        # --- Set Environment Variables for subprocess ---
        test_env = os.environ.copy()
        test_env["GDAL_CACHEMAX"] = "1024"
        test_env["DASK_CLUST_MAX_MEM"] = "2GB"

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

        # Check if the output files were created
        self.assertTrue(
            self.output_path.exists(),
            f"Agreement output file was not created at {self.output_path}",
        )

        self.assertTrue(
            self.metrics_path.exists(),
            f"Metrics output file was not created at {self.metrics_path}",
        )

        # Verify the output raster properties
        try:
            with rasterio.open(self.output_path) as src:
                # Check data type (should be numeric for extent)
                self.assertIn(
                    src.dtypes[0],
                    ["uint8", "float32", "int32"],
                    f"Expected numeric data type for extent, got {src.dtypes[0]}",
                )

                # Check nodata value (255 for agreement maps)
                self.assertEqual(
                    src.nodata,
                    255,
                    f"Expected 255 nodata value for agreement, got {src.nodata}",
                )

                # Check that data exists
                data = src.read(1)
                self.assertTrue(
                    np.any(data != src.nodata),
                    "Raster contains no valid data (all nodata values)",
                )

                # Check agreement map values are within expected range
                # Agreement map encoding: 0=TN, 1=FN, 2=FP, 3=TP, 4=Masked, 10=NoData
                valid_data_mask = data != src.nodata
                if np.any(valid_data_mask):
                    valid_data = data[valid_data_mask]
                    unique_values = np.unique(valid_data)
                    expected_values = {0, 1, 2, 3, 4}  # TN, FN, FP, TP, Masked

                    # Check that all values are in expected set
                    unexpected_values = set(unique_values) - expected_values
                    self.assertEqual(
                        len(unexpected_values),
                        0,
                        f"Agreement raster contains unexpected values: {unexpected_values}. Expected: {expected_values}",
                    )

                    # Check that we have different agreement types (should have TN, FN, FP, TP based on test data)
                    self.assertGreater(
                        len(unique_values),
                        1,
                        "Agreement raster should contain multiple agreement types based on test data stripes",
                    )

                    # Check for True Positives (both rasters have wet areas that overlap)
                    self.assertIn(
                        3,
                        unique_values,
                        "Agreement raster should contain True Positives (value 3) based on overlapping wet stripes",
                    )

        except rasterio.RasterioIOError as e:
            self.fail(f"Failed to open or read the output raster file: {self.output_path}. Error: {e}")

    def test_agreement_creation_no_clip(self):
        """Tests agreement creation without clipping."""
        if not self.input_files_exist:
            self.skipTest("Skipping test because input files are missing")

        # Use different output path
        output_path_no_clip = self.mock_data_dir / "agreement_output_no_clip.tif"

        # Remove output file if it exists
        if output_path_no_clip.exists():
            output_path_no_clip.unlink()

        cmd = [
            sys.executable,
            str(self.script_path),
            "--fim_type",
            "extent",
            "--candidate_path",
            str(self.candidate_path),
            "--benchmark_path",
            str(self.benchmark_path),
            "--output_path",
            str(output_path_no_clip),
            "--block_size",
            "512",
        ]

        # --- Set Environment Variables for subprocess ---
        test_env = os.environ.copy()
        test_env["GDAL_CACHEMAX"] = "1024"
        test_env["DASK_CLUST_MAX_MEM"] = "2GB"

        print(f"\nRunning command (no clip): {' '.join(cmd)}")

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
            output_path_no_clip.exists(),
            f"Agreement output file was not created at {output_path_no_clip}",
        )

        # Verify that no clip was applied (should not have masked values)
        try:
            with rasterio.open(output_path_no_clip) as src:
                data = src.read(1)
                valid_data_mask = data != src.nodata
                if np.any(valid_data_mask):
                    valid_data = data[valid_data_mask]
                    unique_values = np.unique(valid_data)

                    # Should not contain masked values (4) when no clipping
                    self.assertNotIn(
                        4,
                        unique_values,
                        "Agreement raster should not contain masked values (4) when no clipping is applied",
                    )

        except rasterio.RasterioIOError as e:
            self.fail(f"Failed to open or read the output raster file: {output_path_no_clip}. Error: {e}")


if __name__ == "__main__":
    unittest.main()
