from abc import ABC, abstractmethod
from typing import List, Callable, Union
import torch

class Group(ABC):
    """Abstract base class for mathematical groups used in Orbit2Vec."""
    @abstractmethod
    def max_filter(self):
        """Return a group-specific max filter function."""
        pass


class from_matrices(Group):
    """Group generated from a collection of matrices."""
    def __init__(self, matrices: List[torch.Tensor]) -> None:
        """Initialize a group from a list of matrices.

        Args:
            matrices (List[torch.Tensor]): List of square matrices 
                representing the group elements.
        """
        self.group = matrices

    def max_filter(self, template: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        Create a function that computes the max inner product between 
        a given vector and transformed templates under the group action.

        Args:
            template (torch.Tensor): A template tensor of shape (m, n).

        Returns:
            Callable[[torch.Tensor], torch.Tensor]: A function that takes
            an input vector ``x`` and returns the max inner product over
            all group actions, for each row of the template.
        """
        def max_inner_product(x: torch.Tensor) -> torch.Tensor:
            """_summary_

            Args:
                x (torch.Tensor): _description_

            Returns:
                torch.Tensor: _description_
            """
            max_values = []
            for i in range(template.shape[0]):
                t = template[i]  # (n,)
                inner_products = [torch.dot(x, g @ t) for g in self.group]
                max_val = torch.max(torch.stack(inner_products))
                max_values.append(max_val)
            return torch.stack(max_values)  # (m,)

        return max_inner_product

class circular(Group):
    """Circular group actions (rotations via Fourier methods)."""
    def reversal(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Reverse the entries of a vector.

        Args:
            vec (torch.Tensor): Input vector.

        Returns:
            torch.Tensor: Reversed vector.
        """
        return torch.flip(vec, dims=(0,)) 

    def fourier(self, vec: torch.Tensor) -> torch.Tensor:
        """
        Compute the discrete Fourier transform (DFT) of a vector.

        Args:
            vec (torch.Tensor): Input vector.
        Returns:
            torch.Tensor: Complex Fourier coefficients.
        """
        return torch.fft.fft(vec.to(torch.complex64))

    def inv_fourier(self, vec: torch.Tensor) -> torch.Tensor:
        """ 
        Compute the inverse discrete Fourier transform (IDFT).

        Args:
            vec (torch.Tensor): Complex Fourier coefficients.

        Returns:
            torch.Tensor: Real vector after inverse transform.
        """
        return torch.fft.ifft(vec)

    def max_filter(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """        
        Create a function that computes the maximum circular convolution
        between two vectors under the group action.

        Returns:
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
            A function that takes two vectors ``f`` and ``g`` and returns
            the maximum value of their circular correlation.
        """
        # Returns a function that computes max_{a in C_n} <f, Rg(a)>
        def max_circ_conv(f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:

            return max((self.inv_fourier(self.fourier(f) * self.fourier(self.reversal(g)))).real)  
          
        return max_circ_conv
    
    # comment later
    def rotate_tensor(self, t: list)-> torch.Tensor: 

        q = dequeue(t)
        q.rotate(1) 

        return torch.Tensor(list(q))
        
    # comment later
    def max_filter2(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:

        C = circular()
        circ_conv = C.max_filter()

        def max_circ_conv2(f: torch.Tensor, g: torch.Tensor, C: torch.Tensor) -> torch.Tensor:

            max_sum = float('-inf')
            a1, a2, b1, b2 = [], [], [], []

            for i in range(len(f)):
                a1.append(f[i][0])
                b1.append(f[i][1])
                a2.append(g[i][0])
                b2.append(g[i][1])

            a1, a2, b1, b2 = torch.Tensor(a1), torch.Tensor(a2), torch.Tensor(b1), torch.Tensor(b2)

            b1, b2 = self.reversal(b1), self.reversal(b2)

            for i in range(len(f)):

                sum = (circ_conv(a1, b1) + circ_conv(a2, b2))
                max_sum = max(max_sum, sum)

                b1, b2 = self.rotate_tensor(b1), self.rotate_tensor(b2)

            return max_sum
        
        return max_circ_conv2   