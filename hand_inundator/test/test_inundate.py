#!/usr/bin/env python3
import unittest
import os
import tempfile
import subprocess
import numpy as np
import rasterio


class TestInundateScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get the directory containing test data
        cls.test_dir = os.path.dirname(os.path.abspath(__file__))
        cls.mock_data_dir = os.path.join(cls.test_dir, "mock_data")

        # Path to the script being tested
        cls.script_path = os.path.join(os.path.dirname(cls.test_dir), "inundate.py")

        # Input paths
        cls.catchment_parquet = os.path.join(
            cls.mock_data_dir, "test_hydrotable.parquet"
        )
        cls.forecast_path = "s3://fimc-data/benchmark/ripple_fim_30/nwm_return_period_flows_10_yr_cms.csv"

    def test_extent_inundation_mapping(self):
        """Test binary extent output"""
        # Create a temporary file for the output
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_output:
            tmp_output_path = tmp_output.name

        try:
            # Run the inundation script for extent
            cmd = [
                "python3",
                self.script_path,
                "--catchment_data_path",
                self.catchment_parquet,
                "--forecast_path",
                self.forecast_path,
                "--fim_output_path",
                tmp_output_path,
                "--fim_type",
                "extent",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                result.returncode, 0, f"Script failed with error: {result.stderr}"
            )

            # Compare with expected extent output
            expected_extent = os.path.join(self.mock_data_dir, "expected_extent_output.tif")
            with rasterio.open(tmp_output_path) as generated_raster, rasterio.open(
                expected_extent
            ) as expected_raster:
                # Check profile
                gen_profile = generated_raster.profile
                exp_profile = expected_raster.profile
                self.assertEqual(gen_profile["dtype"], "uint8")
                self.assertEqual(gen_profile["nodata"], 255)
                
                # Compare data
                generated_data = generated_raster.read(1)
                expected_data = expected_raster.read(1)
                np.testing.assert_array_equal(
                    generated_data,
                    expected_data,
                    "Generated extent raster does not match expected output",
                )

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_output_path):
                os.unlink(tmp_output_path)

    def test_depth_inundation_mapping(self):
        """Test depth output"""
        # Create a temporary file for the output
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_output:
            tmp_output_path = tmp_output.name

        try:
            # Run the inundation script for depth
            cmd = [
                "python3",
                self.script_path,
                "--catchment_data_path",
                self.catchment_parquet,
                "--forecast_path",
                self.forecast_path,
                "--fim_output_path",
                tmp_output_path,
                "--fim_type",
                "depth",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                result.returncode, 0, f"Script failed with error: {result.stderr}"
            )

            # Compare with expected depth output
            expected_depth = os.path.join(self.mock_data_dir, "expected_depth_output.tif")
            with rasterio.open(tmp_output_path) as generated_raster, rasterio.open(
                expected_depth
            ) as expected_raster:
                # Check profile
                gen_profile = generated_raster.profile
                exp_profile = expected_raster.profile
                self.assertEqual(gen_profile["dtype"], "float32")
                self.assertEqual(gen_profile["nodata"], -9999.0)
                
                # Compare data
                generated_data = generated_raster.read(1)
                expected_data = expected_raster.read(1)
                np.testing.assert_array_equal(
                    generated_data,
                    expected_data,
                    "Generated depth raster does not match expected output",
                )

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_output_path):
                os.unlink(tmp_output_path)


if __name__ == "__main__":
    unittest.main()
