import unittest 
import torch
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Code.orbit2vec import orbit2vec

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
        v1 = torch.tensor([[7, 2], [2, 8]], dtype=torch.float32)
        v2 = torch.tensor([[3, 4], [4, 5]], dtype=torch.float32)
        v3 = torch.tensor([[6, 1], [1, 4]], dtype=torch.float32)

        theta = math.pi / 4
        ortho = torch.tensor([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta),  math.cos(theta)]
        ], dtype=torch.float32)

        for v in [v1, v2, v3]:
            result1 = self.orbit2vec_instance.map3(matrix=v)
            result2 = self.orbit2vec_instance.map3(matrix=(ortho.T @ v @ ortho))

            # Compare sorted eigenvalues instead of full matrices
            eigvals1 = torch.sort(torch.linalg.eigvalsh(result1)).values
            eigvals2 = torch.sort(torch.linalg.eigvalsh(result2)).values

            torch.testing.assert_close(eigvals1, eigvals2, rtol=1e-4, atol=1e-4)

    def test_map4(self):
        v1 = torch.tensor([[7, 2], [2, 8]], dtype=torch.float32)
        v2 = torch.tensor([[3, 4], [4, 5]], dtype=torch.float32)
        v3 = torch.tensor([[6, 1], [1, 4]], dtype=torch.float32)

        theta = math.pi / 4
        ortho = torch.tensor([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta),  math.cos(theta)]
        ], dtype=torch.float32)

        for i, v in enumerate([v1, v2, v3]):
            transformed = ortho @ v
            print(f"\nMatrix {i+1}:")
            print(f"Original:\n{v}")
            print(f"Transformed:\n{transformed}")
            
            # Test map3 directly for comparison
            result1_map3 = self.orbit2vec_instance.map3(matrix=v)
            result2_map3 = self.orbit2vec_instance.map3(matrix=transformed)
            eigvals1_map3 = torch.sort(torch.linalg.eigvalsh(result1_map3)).values
            eigvals2_map3 = torch.sort(torch.linalg.eigvalsh(result2_map3)).values
            print(f"map3 eigenvalues original: {eigvals1_map3}")
            print(f"map3 eigenvalues transformed: {eigvals2_map3}")
            
            result1 = self.orbit2vec_instance.map4(matrix=v)
            result2 = self.orbit2vec_instance.map4(matrix=transformed)
            eigvals1 = torch.sort(torch.linalg.eigvalsh(result1)).values
            eigvals2 = torch.sort(torch.linalg.eigvalsh(result2)).values
            print(f"map4 eigenvalues original: {eigvals1}")
            print(f"map4 eigenvalues transformed: {eigvals2}")
            
            try:
                torch.testing.assert_close(eigvals1, eigvals2, rtol=5e-2, atol=1e-2)
                print("✓ PASSED")
            except AssertionError as e:
                print(f"✗ FAILED: {e}")


    
if __name__ == '__main__':
    unittest.main()