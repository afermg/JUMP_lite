#!/usr/bin/env python3
"""Check if Spherize produces different results with different row orders."""
import sys
sys.path.insert(0, 'src')
import numpy as np

print("Spherize Row Order Test")
print("=" * 40)

# Create test data
np.random.seed(42)
X = np.random.randn(200, 50)

# Create shuffled version (same data, different row order)
perm = np.random.permutation(200)
X_shuffled = X[perm]

print(f"Original shape: {X.shape}")
print(f"Shuffled shape: {X_shuffled.shape}")
print(f"Same data: {np.allclose(np.sort(X, axis=0), np.sort(X_shuffled, axis=0))}")

# Import Spherize
from norm.operations.normalize import Spherize as OldSpherize
from norm_2.core import Spherize as NewSpherize

# Test 1: Old Spherize with both orders
print("\n--- Old Spherize ---")
sph1 = OldSpherize(method="ZCA", epsilon=1e-6)
X1 = sph1.fit_transform(X)

sph2 = OldSpherize(method="ZCA", epsilon=1e-6)
X2_temp = sph2.fit_transform(X_shuffled)
# Unshuffle to compare
inv_perm = np.argsort(perm)
X2 = X2_temp[inv_perm]

diff_old = np.max(np.abs(X1 - X2))
print(f"Max diff (different row order): {diff_old:.2e}")

# Test 2: Compare old vs new on same data
print("\n--- Old vs New Spherize ---")
old_sph = OldSpherize(method="ZCA", epsilon=1e-6)
X_old = old_sph.fit_transform(X)

new_sph = NewSpherize(method="ZCA", epsilon=1e-6)
X_new = new_sph.fit_transform(X)

diff_impl = np.max(np.abs(X_old - X_new))
print(f"Max diff (old vs new): {diff_impl:.2e}")

if diff_old < 1e-10 and diff_impl < 1e-10:
    print("\nPASS: Spherize is order-independent and implementations match")
else:
    print(f"\nResults: row_order_diff={diff_old:.2e}, impl_diff={diff_impl:.2e}")
