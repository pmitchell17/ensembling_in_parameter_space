import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from models.mlp import MLP


class TestMLP(unittest.TestCase):

    def setUp(self):
        in_out_units = [(6, 6), (6, 6), (6,2)]
        mlp = MLP(in_out_units=in_out_units)
        mlp.init_weights_as_int(num=1)

        self.mlp = mlp

    def test_param_shape(self):
        """Tests that mlp is created with correct number of layers
           and parameters
        """
        num_layers = len(self.mlp.linears)
        shapes = [list(linear.weight.shape) for linear in self.mlp.linears]

        self.assertEqual(num_layers, 3)
        self.assertEqual(shapes, [[6, 6], [6, 6], [2, 6]])
    
    def test_forward_flattened_img(self):
        """Tests forward loop"""
        img = torch.ones(size=(4,1,2,3))
        output = self.mlp(img)
        self.assertEqual(list(output[0]), [216, 216])
    
    def test_forward_relu_with_negative_input(self):
        """Tests that relu is called"""
        img = torch.ones(size=(1, 1, 3, 2)) * -1
        output = self.mlp(img)
        self.assertEqual(output[0].tolist(), [0, 0])

    def test_forward_relu_last_layer(self):
        """Tests that the last layer does not apply relu"""
        img = torch.ones(size=(1, 1, 3, 2))
        mlp = self.mlp
        with torch.no_grad():
            mlp.linears[2].weight *= -1
        output = mlp(img)
        self.assertEqual(output[0].tolist(), [-216, -216])

    
if __name__ == "__main__":
    unittest.main()