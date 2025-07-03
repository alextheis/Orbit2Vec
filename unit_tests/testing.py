import unittest 
import torch
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from orbit2vec import orbit2vec

class TestMaps(unittest.TestCase):
    def setUp(self):
        # Create an instance of the orbit2vec class
        self.orbit2vec_instance = orbit2vec()
    
    def test_map1(self):

        v1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        neg_v1 = v1 * -1

        v2 = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float32)
        neg_v2 = v2 * -1

        v3 = torch.tensor([[3, 1, 2, 9], [2, 7, 0, 4]], dtype=torch.float32)
        neg_v3 = v3 * -1

        torch.testing.assert_close(self.orbit2vec_instance.map1(vec=v1), self.orbit2vec_instance.map1(vec=neg_v1))
        torch.testing.assert_close(self.orbit2vec_instance.map1(vec=v2), self.orbit2vec_instance.map1(vec=neg_v2))
        torch.testing.assert_close(self.orbit2vec_instance.map1(vec=v3), self.orbit2vec_instance.map1(vec=neg_v3))

    def test_map2(self):

        v1 = [1, 1, -1]
        neg_v1 = [1,-1, 1]

        v2 = [2, 0, 1]
        neg_v2 =[1, 0, 2]

        v3 = [9,5,4]
        neg_v3 = [9,4,5]

        self.assertAlmostEqual(self.orbit2vec_instance.map2(list=v1), self.orbit2vec_instance.map2(list=neg_v1))
        self.assertAlmostEqual(self.orbit2vec_instance.map2(list=v2), self.orbit2vec_instance.map2(list=neg_v2))
        self.assertAlmostEqual(self.orbit2vec_instance.map2(list=v3), self.orbit2vec_instance.map2(list=neg_v3))

    def test_map3(self):
        
        ortho = torch.tensor([[math.cos(math.pi), math.sin(math.pi)],[math.sin(math.pi), -math.cos(math.pi)]], dtype=torch.float32)

        v1 = torch.tensor([[7, 8], [2,2]], dtype=torch.float32)
        v2 = torch.tensor([[3, 3], [4,4]], dtype=torch.float32)
        v3 = torch.tensor([[1, 0], [7,4]], dtype=torch.float32)

        torch.testing.assert_close(self.orbit2vec_instance.map3(matrix=v1), self.orbit2vec_instance.map3(matrix=(v1 @ ortho)))
        torch.testing.assert_close(self.orbit2vec_instance.map3(matrix=v2), self.orbit2vec_instance.map3(matrix=(v2 @ ortho)))
        torch.testing.assert_close(self.orbit2vec_instance.map3(matrix=v3), self.orbit2vec_instance.map3(matrix=(v3 @ ortho)))

    def test_map4(self):

        ortho = torch.tensor([[math.cos(math.pi), math.sin(math.pi)],[math.sin(math.pi), -math.cos(math.pi)]], dtype=torch.float32)

        v1 = torch.tensor([[7, 8], [2,2]], dtype=torch.float32)
        v2 = torch.tensor([[3, 3], [4,4]], dtype=torch.float32)
        v3 = torch.tensor([[1, 0], [7,4]], dtype=torch.float32)

        torch.testing.assert_close(self.orbit2vec_instance.map4(matrix=v1), self.orbit2vec_instance.map4(matrix=(v1 @ ortho)))
        torch.testing.assert_close(self.orbit2vec_instance.map4(matrix=v2), self.orbit2vec_instance.map4(matrix=(v2 @ ortho)))
        torch.testing.assert_close(self.orbit2vec_instance.map4(matrix=v3), self.orbit2vec_instance.map4(matrix=(v3 @ ortho)))
    
if __name__ == '__main__':
    unittest.main()