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
        cls.catchment_json = os.path.join(cls.mock_data_dir, "test_catchment.json")
        cls.forecast_path = "s3://fimc-data/benchmark/ripple_fim_30/nwm_return_period_flows_10_yr_cms.csv"
        cls.expected_output = os.path.join(
            cls.mock_data_dir, "inundate_test_extent_output.tif"
        )

    def test_inundation_mapping(self):
        # Create a temporary file for the output
        with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp_output:
            tmp_output_path = tmp_output.name

        try:
            # Run the inundation script
            cmd = [
                "python3",
                self.script_path,
                "--catchment-data",
                self.catchment_json,
                "--forecast-path",
                self.forecast_path,
                "--output-path",
                tmp_output_path,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                result.returncode, 0, f"Script failed with error: {result.stderr}"
            )

            # Compare the output with expected result
            with rasterio.open(tmp_output_path) as generated_raster, rasterio.open(
                self.expected_output
            ) as expected_raster:

                # Check if the raster profiles match
                self.assertEqual(
                    generated_raster.profile,
                    expected_raster.profile,
                    "Raster profiles do not match",
                )

                # Read and compare the raster data
                generated_data = generated_raster.read(1)
                expected_data = expected_raster.read(1)

                np.testing.assert_array_equal(
                    generated_data,
                    expected_data,
                    "Generated raster data does not match expected output",
                )

        finally:
            # Clean up temporary file
            if os.path.exists(tmp_output_path):
                os.unlink(tmp_output_path)


if __name__ == "__main__":
    unittest.main()
