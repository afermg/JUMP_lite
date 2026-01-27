#!/usr/bin/env python3
"""Simple TVN test with small data."""
import sys
sys.path.insert(0, 'src')
import numpy as np

print("Simple TVN Test")
print("=" * 40)

# Create test data (100 samples, 50 features)
np.random.seed(42)
X = np.random.randn(100, 50)

# Import both TVN classes
from norm.operations.normalize import TVN as OldTVN
from norm_2.core import TVN as NewTVN

# Apply both
old_tvn = OldTVN(alpha=0.3, epsilon=1.0)
X_old = old_tvn.fit_transform(X)

new_tvn = NewTVN(alpha=0.3, epsilon=1.0)
X_new = new_tvn.fit_transform(X)

# Compare
max_diff = np.max(np.abs(X_old - X_new))
print(f"Max difference: {max_diff:.2e}")

if max_diff < 1e-10:
    print("PASS: TVN outputs are identical")
else:
    print("FAIL: TVN outputs differ")
