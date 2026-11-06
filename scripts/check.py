"""Compare C++ float logits to the PyTorch reference saved during export."""
import sys
import numpy as np
reference = np.load('artifacts/pytorch.npy')
actual = np.loadtxt(sys.argv[1], delimiter=',')[:, 1:]
if actual.shape != reference.shape:
    raise SystemExit(f'shape mismatch: {actual.shape} vs {reference.shape}')
difference = np.max(np.abs(actual - reference))
np.testing.assert_allclose(actual, reference, rtol=1e-4, atol=1e-5)
print(f'PyTorch vs C++ passed: max absolute difference {difference:.8g} (atol=1e-5, rtol=1e-4)')
