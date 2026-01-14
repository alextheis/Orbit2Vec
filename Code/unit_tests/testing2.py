# file: testing2.py
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from group import from_matrices, circular
import kagglehub
import numpy as np

# Download latest version
mypath ="nswitzer/usa-2019-congressional-district-shape-files"

# Make the project root (parent of this tests folder) importable
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))  # adjust if needed
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import unittest
import torch
from math import pi, cos, sin
from shape2matrix import shape2matrix


def rotation_matrix(theta):
    """2×2 rotation matrix for angle theta."""
    return torch.tensor([
        [cos(theta), -sin(theta)],
        [sin(theta),  cos(theta)]
    ], dtype=torch.float32)




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
        # S.path = "nswitzer/usa-2019-congressional-district-shape-files"

        sf = S.import_shape()

        res = S.extract_shape(sf)

        # print(S.equidistant(res)[0]) 
class TestPca(unittest.TestCase):

    def test_tsne_with_area(self):
        S = shape2matrix(200)
        S.path = mypath

        c = circular()
        sf = S.import_shape()
        res = S.extract_shape(sf)

        boundaries = S.equidistant(res)
        shapes = S.center(boundaries)

        filter_bank = c.max_filter2D()

        templates = shapes[:2]
        new_shapes = []

        for shape in shapes:
            v = []
            for t in templates:
                v.append(filter_bank(shape, t))
            new_shapes.append(v)

        new_shapes = np.array(new_shapes, dtype=float)

        areas = [S.shoelace_formula(b) for b in boundaries]
        S.tsne(new_shapes, 42, 2, areas, "Area", outlines=boundaries, outline_scale=0.5)

        perimeters = [S.compute_perimeter(b) for b in boundaries]
        S.tsne(new_shapes, 43, 2, perimeters, "Perimeter", outlines=boundaries, outline_scale=0.5)

        roundness = [S.roundness(b) for b in boundaries]
        S.tsne(new_shapes, 44, 2, roundness, "roundness", outlines=boundaries, outline_scale=0.5)

        diameters = S.compute_diameter(boundaries)
        S.tsne(new_shapes, 45, 2, diameters, "Diameter", outlines=boundaries, outline_scale=0.5)



if __name__ == '__main__':
    unittest.main()




