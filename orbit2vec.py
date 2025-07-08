import torch
import math

class orbit2vec:
    def __init__(self):
        #stores the distortion for each map that we have available
        self.distortion = {}

    # def symsqrt(self, matrix):
    #     """Compute the square root of a positive definite matrix."""
    #     # perform the decomposition
    #     # s, v = matrix.symeig(eigenvectors=True)
    #     _, s, v = matrix.svd()  # passes torch.autograd.gradcheck()
    #     # truncate small components
    #     above_cutoff = s > s.max() * s.size(-1) * torch.finfo(s.dtype).eps
    #     s = s[..., above_cutoff]
    #     v = v[..., above_cutoff]
    #     # compose the square root matrix
    #     return (v * s.sqrt().unsqueeze(-2)) @ v.transpose(-2, -1)
    
    def matrix_sqrt_sym(self, matrix):
        """
        Computes the symmetric square root of a real, symmetric positive definite matrix.
        Returns a matrix `sqrtm` such that: sqrtm @ sqrtm.T = matrix

        Args:
            matrix (torch.Tensor): A symmetric, positive definite matrix of shape (n, n)

        Returns:
            torch.Tensor: The matrix square root of shape (n, n)
        """
        # Ensure symmetry (optional but safe)
        matrix = 0.5 * (matrix + matrix.T)

        # Eigen-decomposition
        eigvals, eigvecs = torch.linalg.eigh(matrix)  # for symmetric matrices

        # Avoid negative/near-zero eigenvalues due to numerical error
        eps = torch.finfo(matrix.dtype).eps
        eigvals_clamped = torch.clamp(eigvals, min=eps)

        # Square root the eigenvalues
        sqrt_eigvals = torch.sqrt(eigvals_clamped)

        # Handle complex eigenvectors by taking real part
        # For real symmetric matrices, eigenvectors should be real, but numerical errors can introduce small imaginary parts
        if eigvecs.dtype.is_complex:
            eigvecs = eigvecs.real
        
        # Reconstruct the square root matrix
        sqrt_matrix = eigvecs @ torch.diag(sqrt_eigvals) @ eigvecs.T

        return sqrt_matrix

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
        list.sort()
        return list
    
    #map3 function
    def map3(self, matrix):
        #calculates the eigenvalues and the "S" matrix, takes the transpose of "S" (St)  
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        transposed_eigenvectors = eigenvectors.T

        #creates the diagonal "D" matrix, takes sqrt of said matrix
        diagonal_eigenvalues = torch.diag_embed(eigenvalues)
        sqrt_matrix = self.matrix_sqrt_sym(diagonal_eigenvalues)

        #returns S*D*St
        return (eigenvectors @ sqrt_matrix) @ transposed_eigenvectors

    # #map4 function
    # def map4(self, matrix):
    #     #centers the matrix by shifting it by the avg value
    #     avg = torch.mean(matrix)

    #     #pass through map3 after being centered 
    #     return self.map3(matrix + avg)

    def map4(self, matrix):
        avg = torch.trace(matrix) / matrix.shape[0]
        shifted_matrix = matrix + avg * torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
        return self.map3(shifted_matrix)

        

