import unittest
import torch
from math import pi, cos, sin
from group import from_matrices


# Helper to define a 2D rotation matrix
def rotation_matrix(theta):
    return torch.tensor([
        [cos(theta), -sin(theta)],
        [sin(theta),  cos(theta)]
    ], dtype=torch.float32)

# Helper to define reflection matrices
def reflection_matrix(axis: str):
    if axis == 'x':
        return torch.tensor([[1, 0], [0, -1]], dtype=torch.float32)
    elif axis == 'y':
        return torch.tensor([[-1, 0], [0, 1]], dtype=torch.float32)
    elif axis == 'main_diag':  # y = x
        return torch.tensor([[0, 1], [1, 0]], dtype=torch.float32)
    elif axis == 'other_diag':  # y = -x
        return torch.tensor([[0, -1], [-1, 0]], dtype=torch.float32)
    else:
        raise ValueError(f"Unknown axis: {axis}")

class TestFromMatrices(unittest.TestCase):

    def setUp(self):
        # Define full D4 group: 4 rotations and 4 reflections
        rotations = [rotation_matrix(k * pi / 2) for k in range(4)]
        reflections = [
            reflection_matrix('x'),
            reflection_matrix('y'),
            reflection_matrix('main_diag'),
            reflection_matrix('other_diag')
        ]
        self.group_elements = rotations + reflections

        # Define the filter bank object
        self.G = from_matrices(self.group_elements)

        # Define templates (3 vectors in R^2)
        self.templates = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])

        # Define input x ∈ R^2
        self.x = torch.tensor([1.0, 2.0])

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

if __name__ == '__main__':
    unittest.main()
