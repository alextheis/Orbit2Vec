import torch
import math

class orbit2vec:
    def __init__(self):
        #stores the distortion for each map that we have available
        self.distortion = {}

    #map1 function
    def map1(self, vec):
        #set the distortion for our map1
        self.distortion["map1"] = math.sqrt(2)

        trans_vec = vec.t()
        norm = torch.linalg.norm(vec)
        gramian = vec @ trans_vec

        return (1/norm) * gramian


