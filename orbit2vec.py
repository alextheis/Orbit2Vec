import torch
import math

class orbit2vec:
    def __init__(self):
        #stores the distortion for each map that we have available
        self.distortion = {}

    def symsqrt(self, matrix):
        """Compute the square root of a positive definite matrix."""
        # perform the decomposition
        # s, v = matrix.symeig(eigenvectors=True)
        _, s, v = matrix.svd()  # passes torch.autograd.gradcheck()
        # truncate small components
        above_cutoff = s > s.max() * s.size(-1) * torch.finfo(s.dtype).eps
        s = s[..., above_cutoff]
        v = v[..., above_cutoff]
        # compose the square root matrix
        return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)

    #map1 function
    def map1(self, vec):
        #set the distortion for our map1
        self.distortion["map1"] = math.sqrt(2)

        transposed_vec = vec.t()
        norm = torch.linalg.norm(vec)
        gramian = vec @ transposed_vec

        return (1/norm) * gramian
    
    #map2 function
    def map2(self, list):
        #sorts list
        return list.sort
    
    #map3 function
    def map3(self, matrix):
        #calculates the eigenvalues and the "S" matrix, takes the transpose of "S" (St)  
        eigenvalues, eigenvectors = torch.linalg.eig(matrix)
        transposed_eigenvectors = eigenvectors.t()

        #creates the diagonal "D" matrix, takes sqrt of said matrix
        diagonal_eigenvalues = torch.diag_embed(eigenvalues)
        sqrt_matrix = self.symsqrt(diagonal_eigenvalues)

        #returns S*D*St
        return eigenvectors @ sqrt_matrix @ transposed_eigenvectors

    #map4 function
    def map4(self, matrix):
        #centers the matrix by shifting it by the avg value
        avg = torch.mean(matrix)

        #pass through map3 after being centered 
        return self.map3(matrix + avg)

        

