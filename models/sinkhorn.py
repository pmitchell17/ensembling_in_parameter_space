import torch
import torch.nn as nn


class SinkhornStableNormalizer(nn.Module):
    """Implementation of Sinkhorn Algorithm
    
    The Sinkhorn algorithm converts a square matrix of positive
    values to a Doubly-Stochastic-Matrix (DSM), by performing
    alternating row and column normalizations for a given number
    of iterations.

    Args:
        iterations (int): The number of Sinkhorn iterations
        temp (float): sinkhorn_temp (float): The temperature 
            used in the sinkhorn algorithm. This controls the
            softness of the DSM. Low values lead to more mass 
            concentration on fewer matrix cells. In other words
            they are closer to hard permutation matrices, where
            each row and column has a single value equal to 1 and
            all other values 0.
        epsilon (float): A small pertubation for numerical stability

    """

    def __init__(self, iterations=100, temp=1, epsilon=0.001):
        super().__init__()
        self.eps = epsilon
        self.iterations = iterations
        self.temp = temp

    def _row_normalization(self, x):
        """ Divides each value by the sum of its row

        Args:
            x (torch.tensor): A square matrix

        Returns:
            normalized_x (torch.tensor): A row-normalized
                matrix
        
        """
        return x / torch.sum(x, dim=-1, keepdims=True)

    def _column_normalization(self, x):
        """ Divides each value by the sum of its column

        Args:
            x (torch.tensor): A square matrix

        Returns:
            normalized_x (torch.tensor): A column-normalized
                matrix
        
        """
        return x / torch.sum(x, dim=-2, keepdims=True)

    def forward(self, x):
        """Runs the Sinkhorn algorithm on the given matrix

        Args:
            x (torch.tensor): A square matrix

        Returns:
            DSM (torch.tensor): A doubly-stochastic matrix
        
        """
        x = x / self.temp
        x = torch.exp(x)
        x = x + self.eps
        for _ in range(self.iterations):
            x = self._row_normalization(x)
            x = self._column_normalization(x)
        return x