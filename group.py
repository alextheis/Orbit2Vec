from abc import ABC, abstractmethod
import torch

class Group(ABC):
    
    @abstractmethod
    def max_filter(self):
        pass


class from_matrices(Group):
    def __init__(self, matrices):
        self.group = matrices

    def max_filter(self, template: torch.Tensor):
        """
        Returns a function computing max inner products between x and each g(template[i])

        Args:
            template: Tensor of shape (m, n) — m template vectors of size n

        Returns:
            A function that takes x (Tensor of shape (n,)) and returns a tensor of shape (m,)
        """
        def max_inner_product(x: torch.Tensor):
            max_values = []
            for i in range(template.shape[0]):
                t = template[i]  # (n,)
                inner_products = [torch.dot(x, g @ t) for g in self.group]
                max_val = torch.max(torch.stack(inner_products))
                max_values.append(max_val)
            return torch.stack(max_values)  # (m,)

        return max_inner_product

class circular(Group):
    def reversal(self, vec):
         return torch.flip(vec, dims=(0,)) 

    def fourier(self, vec):
        return torch.fft.fft(vec.to(torch.complex64))

    def inv_fourier(self, vec):
        return torch.fft.ifft(vec)

    def max_filter(self):
        # Returns a function that computes max_{a in C_n} <f, Rg(a)>
        def max_circ_conv(f: torch.tensor, g: torch.tensor):

            return max((self.inv_fourier(self.fourier(f) * self.fourier(self.reversal(g)))).real)  
          
        return max_circ_conv

