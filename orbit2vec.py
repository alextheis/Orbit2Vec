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
    
    def map3(self, matrix):
        return self.matrix_sqrt_sym(matrix.t() @ matrix)

    # def map4(self, matrix):
    #     # Instead of centering columns, make the matrix have zero trace
    #     # This is invariant under similarity transformations
    #     n = matrix.shape[0]
    #     trace_centered = matrix - (matrix.trace() / n) * torch.eye(n)
        
    #     return self.map3(trace_centered)
                    
    def map4(self, matrix):
        num_columns = matrix.shape[1]
        column_length = matrix.shape[0]
        col_sum = torch.zeros(column_length, 1)

        for i in range(num_columns):
            column = matrix[:, i]  # shape: (column_length,)
            col_sum += column.unsqueeze(1)  # shape: (column_length, 1)

        col_avg = col_sum / num_columns

        for i in range(num_columns):
            matrix[:, i] = matrix[:, i] - col_avg.squeeze(1)  # match dimensions

        return self.map3(matrix)

    def max_inner_product(self, x: torch.Tensor, y: torch.Tensor, group: list):
        """
        Compute max_{g ∈ G} ⟨x, g(y)⟩ where G is a group of isometries represented by matrices.

        Args:
            x: Tensor of shape (n,)
            y: Tensor of shape (n,)
            group: List of tensors, each of shape (n, n), representing isometries.

        Returns:
            Scalar tensor: the maximum inner product.
        """
        inner_products = [torch.dot(x, g @ y) for g in group]
        return torch.max(torch.stack(inner_products))

    