import unittest
import torch
import math
import sys
import os

# Add parent directory to path to import orbit2vec
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orbit2vec import orbit2vec

class TestMaxInnerProduct(unittest.TestCase):
    def setUp(self):
        self.model = orbit2vec()

    def test_rotation_group(self):
        # Define test vectors
        x = torch.tensor([1.0, 0.0], dtype=torch.float32)
        y = torch.tensor([0.0, 1.0], dtype=torch.float32)

        # Define rotation matrices for 0°, 90°, 180°, 270°
        def rotation_matrix(degrees):
            theta = math.radians(degrees)
            return torch.tensor([
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta),  math.cos(theta)]
            ], dtype=torch.float32)

        group = [rotation_matrix(a) for a in [0, 90, 180, 270]]

        # Call the method from orbit2vec
        result = self.model.max_inner_product(x, y, group)

        # Expect the maximum inner product to be 1
        self.assertAlmostEqual(result.item(), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
