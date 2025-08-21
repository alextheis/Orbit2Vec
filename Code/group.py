from abc import ABC, abstractmethod
from typing import List, Callable, Union
import torch

class Group(ABC):
    
    @abstractmethod
    def max_filter(self):
        pass


class from_matrices(Group):
    def __init__(self, matrices: List[torch.Tensor]) -> None:
        """_summary_

        Args:
            matrices (List[torch.Tensor]): _description_
        """
        self.group = matrices

    def max_filter(self, template: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
        """_summary_

        Args:
            template (torch.Tensor): _description_

        Returns:
            Callable[[torch.Tensor], torch.Tensor]: _description_
        """
        def max_inner_product(x: torch.Tensor) -> torch.Tensor:
            max_values = []
            for i in range(template.shape[0]):
                t = template[i]  # (n,)
                inner_products = [torch.dot(x, g @ t) for g in self.group]
                max_val = torch.max(torch.stack(inner_products))
                max_values.append(max_val)
            return torch.stack(max_values)  # (m,)

        return max_inner_product

class circular(Group):
    def reversal(self, vec: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            vec (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return torch.flip(vec, dims=(0,)) 

    def fourier(self, vec: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            vec (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return torch.fft.fft(vec.to(torch.complex64))

    def inv_fourier(self, vec: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            vec (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return torch.fft.ifft(vec)

    def max_filter(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """_summary_

        Returns:
            Callable[[torch.Tensor, torch.Tensor], torch.Tensor]: _description_
        """
        # Returns a function that computes max_{a in C_n} <f, Rg(a)>
        def max_circ_conv(f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:

            return max((self.inv_fourier(self.fourier(f) * self.fourier(self.reversal(g)))).real)  
          
        return max_circ_conv

