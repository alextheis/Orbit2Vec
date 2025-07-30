# file: testing2.py
# file: testing2.py
import os, sys

# Make the project root (parent of this tests folder) importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # adjust if needed
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    
import unittest
import torch
from math import pi, cos, sin
from group import from_matrices
from group import circular


# ----------------------------- helpers ---------------------------------

def rotation_matrix(theta):
    """2×2 rotation matrix for angle theta."""
    return torch.tensor([
        [cos(theta), -sin(theta)],
        [sin(theta),  cos(theta)]
    ], dtype=torch.float32)

def _slow_circular_curve(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Reference: c[a] = <v1, roll(v2, a)>, a=0..N-1
    Vector-only (1-D).
    """
    if v1.dim() != 1 or v2.dim() != 1:
        raise ValueError("v1 and v2 must be 1-D")
    if v1.numel() != v2.numel():
        raise ValueError("length mismatch")
    N = v1.numel()
    out = []
    for a in range(N):
        out.append(torch.dot(v1, torch.roll(v2, shifts=a, dims=0)))
    return torch.stack(out, dim=0)  # (N,)


# ----------------------------- tests for from_matrices -----------------

class TestFromMatrices(unittest.TestCase):

    def setUp(self):
        # Define group elements (isometries)
        self.identity = torch.eye(2)
        self.rotation_90 = rotation_matrix(pi / 2)
        self.reflection_x = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
        self.group_elements = [self.identity, self.rotation_90, self.reflection_x]

        # Define the filter bank object
        self.G = from_matrices(self.group_elements)

        # Define templates (3 vectors in R^2)
        self.templates = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ], dtype=torch.float32)

        # Define input x ∈ R^2
        self.x = torch.tensor([1.0, 2.0], dtype=torch.float32)

    def test_max_filter_output_shape(self):
        max_filter_fn = self.G.max_filter(self.templates)
        result = max_filter_fn(self.x)
        self.assertEqual(result.shape, (3,))

    def test_max_filter_values(self):
        max_filter_fn = self.G.max_filter(self.templates)
        result = max_filter_fn(self.x)

        # Manually compute expected max inner products
        expected_values = []
        for t in self.templates:
            inner_products = [torch.dot(self.x, g @ t) for g in self.group_elements]
            expected_values.append(torch.max(torch.stack(inner_products)))

        expected_tensor = torch.stack(expected_values)
        self.assertTrue(torch.allclose(result, expected_tensor, atol=1e-6))


# ----------------------------- tests for circular ----------------------

class TestCircular(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(1234)

    @staticmethod
    def _unpack_output(out):
        """
        Accept either (max_val, argmax) or (max_val, argmax, curve).
        Return (max_val, argmax, curve_or_None).
        """
        if isinstance(out, tuple):
            if len(out) == 2:
                return out[0], out[1], None
            elif len(out) >= 3:
                return out[0], out[1], out[2]
        raise AssertionError("Unexpected return value from circular.max_filter()")

    def test_values_match_reference(self):
        N = 97
        v1 = torch.randn(N)
        v2 = torch.randn(N)

        C = circular()            # no stored vectors
        compute = C.max_filter()  # get the callable
        out = compute(v1, v2)     # call with plain vectors
        max_val, argmax_shift, curve = self._unpack_output(out)

        ref = _slow_circular_curve(v1, v2)
        rmax, rarg = ref.max(dim=0)
        self.assertTrue(torch.allclose(max_val, rmax, atol=1e-5, rtol=1e-5))
        self.assertEqual(int(argmax_shift), int(rarg))
        if curve is not None:
            self.assertTrue(torch.allclose(curve, ref, atol=1e-5, rtol=1e-5))

    def test_stored_vectors(self):
        N = 64
        v1 = torch.randn(N)
        v2 = torch.randn(N)

        C = circular(v1, v2)      # store vectors in the instance
        compute = C.max_filter()
        out = compute()           # no args -> use stored v1, v2
        max_val, argmax_shift, curve = self._unpack_output(out)

        ref = _slow_circular_curve(v1, v2)
        rmax, rarg = ref.max(dim=0)
        self.assertTrue(torch.allclose(max_val, rmax, atol=1e-5, rtol=1e-5))
        self.assertEqual(int(argmax_shift), int(rarg))
        if curve is not None:
            self.assertTrue(torch.allclose(curve, ref, atol=1e-5, rtol=1e-5))

    def test_length_mismatch_raises(self):
        v1 = torch.randn(10)
        v2 = torch.randn(9)
        C = circular()
        compute = C.max_filter()
        with self.assertRaises(AssertionError):
            _ = compute(v1, v2)


if __name__ == '__main__':
    unittest.main()
