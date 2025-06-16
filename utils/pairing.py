#!/usr/bin/env python3
"""
Shared pairing dictionary for agreement map encoding.

Agreement map encoding:
- 0: True Negative (TN) - both dry
- 1: False Negative (FN) - candidate dry, benchmark wet
- 2: False Positive (FP) - candidate wet, benchmark dry
- 3: True Positive (TP) - both wet
- 4: Masked - excluded from analysis
- 255: NoData
"""

# Pairing dictionary for binary rasters (0=dry, 1=wet, 255=nodata)
AGREEMENT_PAIRING_DICT = {
    (0, 0): 0,  # True Negative: both dry
    (0, 1): 1,  # False Negative: candidate dry, benchmark wet
    (0, 255): 255,  # NoData
    (1, 0): 2,  # False Positive: candidate wet, benchmark dry
    (1, 1): 3,  # True Positive: both wet
    (1, 255): 255,  # NoData
    (4, 0): 4,  # Masked
    (4, 1): 4,  # Masked
    (4, 255): 255,  # NoData
    (255, 0): 255,  # NoData
    (255, 1): 255,  # NoData
    (255, 255): 255,  # NoData
}