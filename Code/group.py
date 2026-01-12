from abc import ABC, abstractmethod
from typing import List, Callable, Union
from collections import deque
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
    

    def rotate_tensor(self, t: list)-> torch.Tensor: 
        """
        Rotates a tensor by shifting all elements one position to the right.

        Args:
            t (list): Input list of numerical values to be rotated

        Returns:
            torch.Tensor: A new tensor with elements rotated one position clockwise
        """
        q = deque(t)
        q.rotate(1) 

        return torch.Tensor(list(q))
 
    def max_filter2D(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Creates a maximum circular convolution filter for 2D tensor pairs.

        Returns:
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: A function that computes 
            the maximum value of summed circular convolutions between corresponding tensor 
            components
        """
        C = circular()
        circ_conv = C.max_filter()


        def max_circ_conv2D(f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
            """
            Computes the maximum circular convolution between two 2D tensor pairs.

            Decomposes each input tensor into column vectors, performs circular convolution
            on corresponding pairs, and returns the maximum value of their sum.

            Args:
                f (torch.Tensor): First 2D tensor to be decomposed into column vectors
                g (torch.Tensor): Second 2D tensor to be decomposed into column vectors

            Returns:
                torch.Tensor: Maximum value from the sum of circular convolutions of 
                corresponding column pairs
            """
            
            a1, b1 = map(list, zip(*f))
            a2, b2 = map(list, zip(*g))

            a1, a2, b1, b2 = torch.Tensor(a1), torch.Tensor(a2), torch.Tensor(b1), torch.Tensor(b2)

            # a1, b1 = f[:,0], f[:,1]
            # print(torch.max(circ_conv(a1, b1) + circ_conv(a2, b2)))
            return torch.max(circ_conv(a1, a2) + circ_conv(b1, b2)) # maxk(<a1, b1C^k> + <a2, b2C^k>)
            
        
        return max_circ_conv2D   