# file: testing2.py
# file: testing2.py
import os, sys
from Code.group import from_matrices, circular
import kagglehub
import numpy as np

# Download latest version
path ="nswitzer/usa-2019-congressional-district-shape-files"

# Make the project root (parent of this tests folder) importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # adjust if needed
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import unittest
import torch
from math import pi, cos, sin
from Code.group import from_matrices
from Code.group import circular
from Code.shape2matrix import shape2matrix

# ----------------------------- helpers ---------------------------------

def rotation_matrix(theta):
    """2×2 rotation matrix for angle theta."""
    return torch.tensor([
        [cos(theta), -sin(theta)],
        [sin(theta),  cos(theta)]
    ], dtype=torch.float32)


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

    def test_circular_invariance(self):

        C = circular()
        circ_conv = C.max_filter()
        
        template = torch.tensor([7, 9, 2, 8, 0])
        v1 = torch.tensor([1, 2, 3, 4, 5])
        v2 = torch.tensor([3, 4, 5, 1, 2])

        result1 = circ_conv(template, v1)
        result2 = circ_conv(template, v2)

        torch.testing.assert_close(result1, result2)

class TestShape(unittest.TestCase):
    #add stuff later
    def test_import(self):
        S = shape2matrix(200)
        S.path = "nswitzer/usa-2019-congressional-district-shape-files"

        sf = S.import_shape()

        res = S.extract_shape(sf)

        # print(S.equidistant(res)[0]) 

class TestPca(unittest.TestCase):
    S = shape2matrix(200)
    S.path = "nswitzer/usa-2019-congressional-district-shape-files"

    c = circular()

    sf = S.import_shape()

    res = S.extract_shape(sf)

    shapes = S.equidistant(res)

    filter_bank = c.max_filter2D()

    num = len(shapes)
    num_points = shapes[0].shape[0]  # e.g., 199
    
    templates = np.random.normal(loc=0, scale=1, size=(num, num_points, 2))
    new_shapes = []

    for i in range(num):
        new_shapes.append(filter_bank(shapes[i], templates[i]))
    print(new_shapes)
    S.pca(new_shapes)
    # pass shapes through max filter bank (max_filter2)
    # run pca 
if __name__ == '__main__':
    unittest.main()


