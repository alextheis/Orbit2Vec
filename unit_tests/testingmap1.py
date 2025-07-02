import Orbit2Vec.orbit2vec as orbit2vec
import torch

#testing
mapper = orbit2vec.orbit2vec()

v1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
neg_v1 = v1 * -1

v2 = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.float32)
neg_v2 = v2 * -1

v3 = torch.tensor([[3, 1, 2, 9], [2, 7, 0, 4]], dtype=torch.float32)
neg_v3 = v3 * -1

if torch.equal(mapper.map1(v1), mapper.map1(neg_v1)):
    print("pass 1")
    print(mapper.distortion["map1"])
else:
    print("fail 1")

if torch.equal(mapper.map1(v2), mapper.map1(neg_v2)):
    print("pass 2")
    print(mapper.distortion["map1"])
else:
    print("fail 2")

if torch.equal(mapper.map1(v3), mapper.map1(neg_v3)):
    print("pass 3")
    print(mapper.distortion["map1"])
else:
    print("fail 3")
