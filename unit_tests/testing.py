import unittest 
from orbit2vec import *
import torch

class TestMaps(unittest.TestCase):
    def test_map1(self):

        v1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        neg_v1 = v1 * -1

        v2 = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float32)
        neg_v2 = v2 * -1

        v3 = torch.tensor([[3, 1, 2, 9], [2, 7, 0, 4]], dtype=torch.float32)
        neg_v3 = v3 * -1

        self.assertAlmostEqual(orbit2vec.map1(v1), orbit2vec.map1(neg_v1))
        self.assertAlmostEqual(orbit2vec.map1(v2), orbit2vec.map1(neg_v2))
        self.assertAlmostEqual(orbit2vec.map1(v3), orbit2vec.map1(neg_v3))

    def test_map2(self):

        v1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        neg_v1 = v1 * -1

        v2 = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float32)
        neg_v2 = v2 * -1

        v3 = torch.tensor([[3, 1, 2, 9], [2, 7, 0, 4]], dtype=torch.float32)
        neg_v3 = v3 * -1

        self.assertAlmostEqual(orbit2vec.map1(v1), orbit2vec.map1(neg_v1))
        self.assertAlmostEqual(orbit2vec.map1(v2), orbit2vec.map1(neg_v2))
        self.assertAlmostEqual(orbit2vec.map1(v3), orbit2vec.map1(neg_v3))

    def test_map3(self):

        v1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        neg_v1 = v1 * -1

        v2 = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float32)
        neg_v2 = v2 * -1

        v3 = torch.tensor([[3, 1, 2, 9], [2, 7, 0, 4]], dtype=torch.float32)
        neg_v3 = v3 * -1

        self.assertAlmostEqual(orbit2vec.map1(v1), orbit2vec.map1(neg_v1))
        self.assertAlmostEqual(orbit2vec.map1(v2), orbit2vec.map1(neg_v2))
        self.assertAlmostEqual(orbit2vec.map1(v3), orbit2vec.map1(neg_v3))

    def test_map4(self):

        v1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        neg_v1 = v1 * -1

        v2 = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float32)
        neg_v2 = v2 * -1

        v3 = torch.tensor([[3, 1, 2, 9], [2, 7, 0, 4]], dtype=torch.float32)
        neg_v3 = v3 * -1

        self.assertAlmostEqual(orbit2vec.map1(v1), orbit2vec.map1(neg_v1))
        self.assertAlmostEqual(orbit2vec.map1(v2), orbit2vec.map1(neg_v2))
        self.assertAlmostEqual(orbit2vec.map1(v3), orbit2vec.map1(neg_v3))
    
    def test_map5(self):

        v1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        neg_v1 = v1 * -1

        v2 = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float32)
        neg_v2 = v2 * -1

        v3 = torch.tensor([[3, 1, 2, 9], [2, 7, 0, 4]], dtype=torch.float32)
        neg_v3 = v3 * -1

        self.assertAlmostEqual(orbit2vec.map1(v1), orbit2vec.map1(neg_v1))
        self.assertAlmostEqual(orbit2vec.map1(v2), orbit2vec.map1(neg_v2))
        self.assertAlmostEqual(orbit2vec.map1(v3), orbit2vec.map1(neg_v3))