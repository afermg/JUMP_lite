#!/usr/bin/env python3
"""Quick comparison of all available preset outputs."""
import sys
sys.path.insert(0, 'src')
from pathlib import Path
import polars as pl
import numpy as np

def compare_outputs(old_path, new_path, tolerance=1e-6):
    old = pl.read_parquet(old_path)
    new = pl.read_parquet(new_path)

    result = {'match': True, 'errors': []}

    if old.shape != new.shape:
        result['match'] = False
        result['errors'].append(f'Shape mismatch: {old.shape} vs {new.shape}')
        return result

    if old.columns != new.columns:
        result['match'] = False
        result['errors'].append('Column mismatch')
        return result

    # Sort both dataframes
    sort_cols = ['Metadata_Plate', 'Metadata_Well']
    if all(c in old.columns for c in sort_cols):
        old = old.sort(sort_cols)
        new = new.sort(sort_cols)

    feature_cols = [c for c in old.columns if not c.startswith('Metadata_')]
    max_diff = 0.0
    worst_col = None

    for col in feature_cols:
        if old[col].dtype in (pl.Float32, pl.Float64):
            old_vals = old[col].to_numpy()
            new_vals = new[col].to_numpy()

            old_nan = np.isnan(old_vals)
            new_nan = np.isnan(new_vals)

            if not np.array_equal(old_nan, new_nan):
                result['errors'].append(f'NaN pattern differs in {col}')
                continue

            mask = ~old_nan
            if mask.any():
                diff = np.max(np.abs(old_vals[mask] - new_vals[mask]))
                if diff > max_diff:
                    max_diff = diff
                    worst_col = col

                if diff > tolerance:
                    result['match'] = False
                    result['errors'].append(f'{col}: diff={diff:.2e}')

    result['max_diff'] = max_diff
    result['worst_col'] = worst_col
    return result

# Test all available presets
presets = ['cp_measure', 'dinov2_490', 'dinov2_random', 'dinov2_tilesize_224', 'morphem', 'openphenom_8bit', 'subcell']
test_dir = Path('test_comparison')

print('PRESET COMPARISON RESULTS (with sorting fix)')
print('=' * 60)

passed = 0
failed = 0
skipped = 0

for preset in presets:
    old_path = test_dir / 'old' / preset / 'processed.parquet'
    new_path = test_dir / 'new' / preset / 'processed.parquet'

    if not old_path.exists() or not new_path.exists():
        print(f'{preset}: SKIPPED (output not available)')
        skipped += 1
        continue

    result = compare_outputs(old_path, new_path)
    status = 'PASS' if result['match'] else 'FAIL'
    print(f'{preset}: {status} (max diff: {result["max_diff"]:.2e})')

    if result['match']:
        passed += 1
    else:
        failed += 1
        for e in result['errors'][:3]:
            print(f'    {e}')

print('=' * 60)
print(f'Total: {passed} passed, {failed} failed, {skipped} skipped')
