"""
Monkey patches for gval to optimize memory usage with large rasters.
"""

import logging
import os
import numpy as np
import pandas as pd
import xarray as xr
import gval.comparison.tabulation as gval_tabulation
from collections.abc import Iterable
from flox.xarray import xarray_reduce
from numbers import Number
from typing import Union


def optimized_crosstab_2d_DataArrays(
    agreement_map: xr.DataArray,
    band_name: str = "band",
    band_value: Union[str, Number] = 1,
) -> pd.DataFrame:
    """
    Optimized version of gval's _crosstab_2d_DataArrays that uses known pairing
    dictionary values instead of computing unique values from the entire array.
    
    This avoids the memory-intensive dask.array.unique() call that causes
    memory issues with large rasters.
    """
    # Check if we're dealing with a dask array
    is_dsk = False
    if hasattr(agreement_map, 'chunks') and agreement_map.chunks is not None:
        if 'spatial_ref' in agreement_map.coords:
            agreement_map = agreement_map.drop_vars("spatial_ref")
        is_dsk = True
    
    agreement_map.name = "group"
    ag_dtype = agreement_map.dtype
    
    # Get pairing dictionary from attributes
    if "pairing_dictionary" not in agreement_map.attrs:
        raise ValueError("Agreement map must have 'pairing_dictionary' in attrs")
    
    pairing_dict = agreement_map.attrs["pairing_dictionary"]
    
    # KEY OPTIMIZATION: Use known agreement values from pairing dictionary
    # instead of computing unique values from the entire array
    # Get UNIQUE agreement values only (avoid duplicates)
    unique_agreement_values = set(v for v in pairing_dict.values() if not np.isnan(v))
    expected_agreement_values = np.array(sorted(unique_agreement_values), dtype=ag_dtype)
    
    logging.info(f"Using {len(expected_agreement_values)} known agreement values from pairing dictionary")
    
    # Rechunk if needed for better performance
    if is_dsk and agreement_map.chunks is not None:
        chunk_size = int(os.getenv("CROSSTAB_CHUNK_SIZE", "4096"))
        rechunk_dict = {}
        for dim in agreement_map.dims:
            if dim in ['x', 'y', 'X', 'Y']:
                rechunk_dict[dim] = chunk_size
        
        if rechunk_dict:
            logging.info(f"Rechunking agreement map to {chunk_size}x{chunk_size} for crosstab")
            # Use chunk() method for xarray DataArrays (not rechunk)
            agreement_map = agreement_map.chunk(rechunk_dict)
            agreement_map = agreement_map.persist()
    
    # Use xarray_reduce with known expected groups
    if is_dsk:
        # Use the known expected values instead of dask.array.unique()
        agreement_counts = xarray_reduce(
            agreement_map,
            agreement_map,
            engine="numba",
            expected_groups=expected_agreement_values,  # KEY CHANGE: Use known values
            func="count",
        )
    else:
        agreement_counts = xarray_reduce(
            agreement_map, 
            agreement_map, 
            engine="numba", 
            func="count",
        )
    
    def not_nan(number):
        return not np.isnan(number)
    
    # Create reverse dictionary for looking up candidate/benchmark pairs
    rev_dict = {}
    for k, v in pairing_dict.items():
        if np.isnan(v):
            continue
        if v in rev_dict:
            rev_dict[v].append(list(k))
        else:
            rev_dict[v] = [list(k)]
    
    # Build crosstab dataframe
    crosstab_df = pd.DataFrame({
        "candidate_values": [
            [y[0] for y in rev_dict[x]]
            for x in filter(not_nan, agreement_counts.coords["group"].values)
        ],
        "benchmark_values": [
            [y[1] for y in rev_dict[x]]
            for x in filter(not_nan, agreement_counts.coords["group"].values)
        ],
        "agreement_values": list(
            filter(
                not_nan, agreement_counts.coords["group"].values.astype(ag_dtype)
            )
        ),
        "counts": [
            x
            for x, y in zip(
                agreement_counts.values.astype(np.int64),
                agreement_counts.coords["group"].values.astype(ag_dtype),
            )
            if not np.isnan(y)
        ],
    })
    
    # Add all entries that don't exist in crosstab that exist in pairing dictionary with 0 count
    # Only add one representative entry per unique agreement value (not all pairs)
    existing_agreement_values = set(crosstab_df["agreement_values"].values)
    added_values = set()
    for k, v in pairing_dict.items():
        if v not in existing_agreement_values and v not in added_values and not np.isnan(v):
            # Add a new row with the same format as above
            new_row = pd.DataFrame({
                "candidate_values": [[k[0]]],
                "benchmark_values": [[k[1]]],
                "agreement_values": [v],
                "counts": [0]
            })
            crosstab_df = pd.concat([crosstab_df, new_row], ignore_index=True)
            added_values.add(v)
    
    # Sort and reindex
    crosstab_df.sort_values(["agreement_values"], inplace=True)
    crosstab_df.reset_index(drop=True, inplace=True)
    
    def is_iterable(x):
        return x[0] if isinstance(x, Iterable) and len(x) > 0 else x
    
    # Handle multiple candidate/benchmark pairs being mapped to the same agreement value
    crosstab_df.loc[:, "candidate_values"] = crosstab_df["candidate_values"].apply(is_iterable)
    crosstab_df.loc[:, "benchmark_values"] = crosstab_df["benchmark_values"].apply(is_iterable)
    
    # Add band information
    crosstab_df.insert(0, band_name, band_value)
    
    return crosstab_df


def apply_gval_optimizations():
    """
    Apply monkey patches to gval for better memory efficiency with large rasters.
    
    This function should be called before using gval's compute_crosstab() method.
    """
    # Store the original function for potential restoration
    if not hasattr(gval_tabulation, '_original_crosstab_2d_DataArrays'):
        gval_tabulation._original_crosstab_2d_DataArrays = gval_tabulation._crosstab_2d_DataArrays
    
    # Apply the monkey patch
    gval_tabulation._crosstab_2d_DataArrays = optimized_crosstab_2d_DataArrays
    
    logging.info("Applied gval optimizations for memory-efficient crosstab computation")
    

def restore_gval_defaults():
    """
    Restore original gval functions if needed.
    """
    if hasattr(gval_tabulation, '_original_crosstab_2d_DataArrays'):
        gval_tabulation._crosstab_2d_DataArrays = gval_tabulation._original_crosstab_2d_DataArrays
        delattr(gval_tabulation, '_original_crosstab_2d_DataArrays')
        logging.info("Restored original gval functions")