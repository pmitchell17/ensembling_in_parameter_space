import torch
import torch.nn as nn


class MLP(nn.Module):
    """A simple multi-layer-perceptron model"""

    def __init__(self, in_out_units):
        """
        Args:
            in_out_units (list[tuples]): A list of (num_in_units, num_out_units)
                tuples specifying the shape of each fully-connected layer.

        """
        super().__init__()

        self.linears = nn.ModuleList(
            self._generate_layers(in_out_units)
        )
        self.relu = nn.ReLU()

    def init_weights_as_int(self, num):
        """Initializes weights to an integer

        Args:
            num (int): The int to which all weights should
                be initialized.

        """
        with torch.no_grad():
            for linear in self.linears:
                linear.weight = nn.Parameter(
                    torch.ones_like(linear.weight) * num
                ) 
                linear.bias = nn.Parameter(
                    torch.zeros_like(linear.bias)
                )

    def _generate_layers(self, in_out_units):
        """Create all fully-connected layers based on in_out_units list"

        Args:
            in_out_units (list[tuples]): A list of (num_in_units, num_out_units)
                tuples specifying the shape of each fully-connected layer.
        
        Returns:
            layers (list(nn.linear)): A list of fully-connected linear
                layers
        
        """
        layers = []
        num_layers = len(in_out_units)
        for layer_index in range(num_layers):
            in_feat, out_feat = in_out_units[layer_index]
            layer = nn.Linear(in_feat, out_feat)
            layers.append(layer)
        return layers
    
    def forward(self, input):
        """Evaluation with relu acitivation function
        
        Args:
            input (torch.tensor): The batch of input images
        """
        input = input.reshape(input.shape[0],-1)
        out = input
        for layer_index, layer in enumerate(self.linears):
            out = layer(out)
            if layer_index != len(self.linears) - 1:
                out = self.relu(out)
        return out
