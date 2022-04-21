import unittest
from unittest.mock import patch

import torch
import torch.nn as nn

from models.mlp import MLP
from models.perm_avg import PermAVG


class TestPermAVGForward(unittest.TestCase):

    def setUp(self):
        model_0 = MLP(in_out_units=[(3, 3), (3, 2)])
        model_0.init_weights_as_int(num=1)
        model_1 = MLP(in_out_units=[(3, 3), (3, 2)])
        model_1.init_weights_as_int(num=2)

        self.models = [model_0, model_1]

        self.perm_avg = PermAVG(models=self.models, naive=True, sinkhorn_temp=1)

    def test_model_weights_are_frozen(self):
        """Tests that all model parameters do not require gradients"""
        for model in self.perm_avg.models:
            for param in model.parameters():
                self.assertEqual(param.requires_grad, False)

    def test_num_param_matrices(self):
        """Tests that a param matrix is initialized for ever layer of
        every model"""
        keys = []
        sizes = []
        requires_grads = []
        for key, params in self.perm_avg.param_matrices.items():
            keys.append(key)
            sizes.append(params.shape[0])

        self.assertEqual([(0, 0), (0, 1), (1, 0), (1, 1)], keys)
        self.assertEqual(sizes, [3, 2, 3, 2])

    def test_param_matrices_naive_true(self):
        """Tests that all param matrices do not require gradients
        when naive=True
        """
        self.perm_avg.naive = True
        requires_grads = []
        for key, params in self.perm_avg.param_matrices.items():
            requires_grads.append(params.requires_grad)
        
        self.assertEqual(requires_grads, [False, False, False, False])

    def test_param_matrices_naive_false(self):
        """Tests that all param matrices do not require gradients
        when naive=False
        """
        self.perm_avg.naive = False
        self.perm_avg._init_param_matrices()
        requires_grads = []
        for key, params in self.perm_avg.param_matrices.items():
            requires_grads.append(params.requires_grad)
        
        self.assertEqual(requires_grads, [False, False, True, False])

    def test_batch_flatten_4d_tensor(self):
        """Tests that each image in a batch is flattened to 1D array"""
        tensor_4d = torch.ones(size=(4, 3, 2, 2))
        flattened_tensor = self.perm_avg._batch_flatten(tensor_4d)

        self.assertEqual(list(flattened_tensor.shape), [4, 12])

    def test_get_perm_matrix_naive_true(self):
        """Tests that all permutation matrices should be the identity,
        when naive=True
        """
        self.perm_avg.naive = True
        param_0_0 = self.perm_avg.get_perm_matrix(0, 0)
        param_1_0 = self.perm_avg.get_perm_matrix(1, 0)
        param_0_1 = self.perm_avg.get_perm_matrix(0, 1)
        param_1_1 = self.perm_avg.get_perm_matrix(1, 1)

        self.assertEqual(torch.sum(param_0_0).item(), 3)
        self.assertEqual(torch.sum(param_1_0).item(), 3)
        self.assertEqual(torch.sum(param_0_1).item(), 2)
        self.assertEqual(torch.sum(param_1_1).item(), 2)
    
    def test_get_perm_matrix_naive_false(self):
        """Tests that the sinkorn layer is applied correctly when
        naive=False"""
        self.perm_avg.naive = False
        self.perm_avg.param_matrices[1, 0] = torch.tensor(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]
        )
        param_0_0 = self.perm_avg.get_perm_matrix(0, 0)
        param_1_0 = self.perm_avg.get_perm_matrix(1, 0)
        param_0_1 = self.perm_avg.get_perm_matrix(0, 1)
        param_1_1 = self.perm_avg.get_perm_matrix(1, 1)

        self.assertEqual(torch.sum(param_0_0).item(), 3)
        self.assertEqual(round(torch.sum(param_1_0).item()), 3)
        self.assertEqual(torch.sum(param_0_1).item(), 2)
        self.assertEqual(torch.sum(param_1_1).item(), 2)
        
    def test_calc_model_layer_output_naive(self):
        """Tests model-layer output when the permutation is the
        identity (no shuffling)"""
        self.perm_avg.naive = True
        batch_flattened = torch.ones(size=(2, 3))

        output = self.perm_avg._calc_model_layer_output(
            batch_flattened,
            model_index=1,
            layer_index=0
        )

        self.assertEqual(torch.sum(output).item(), 18)

    def test_calc_model_layer_output(self):
        """Tests that model-layer output is permuted"""
        perm_matrix = torch.tensor(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0]
            ],
            dtype=torch.float32
        )
        with patch.object(PermAVG, 'get_perm_matrix', return_value=perm_matrix):
            perm_avg = PermAVG(models=self.models, naive=False)

            with torch.no_grad():
                perm_avg.models[1].linears[0].weight = nn.Parameter(
                    torch.tensor(
                        [
                            [1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]
                        ],
                        dtype=torch.float32
                    )
                )
            batch_flattened = torch.ones(size=(2, 3))

            output = self.perm_avg._calc_model_layer_output(
                batch_flattened,
                model_index=1,
                layer_index=0
            )

            self.assertEqual(list(output[0]), [3, 12, 7.5])
    
    def test_calc_model_layer_output(self):
        """Tests that input and model-layer output is permuted"""
        perm_matrix_0 = torch.tensor(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0]
            ],
            dtype=torch.float32
        )
        perm_matrix_1 = torch.tensor(
            [
                [0, 1],
                [1, 0]
            ],
            dtype=torch.float32
        )
        perm_matrices = [perm_matrix_0, perm_matrix_1]

        def mock_perm_matrices(*args, **kwargs):
            return perm_matrices.pop()

        with patch.object(PermAVG, 'get_perm_matrix', side_effect=mock_perm_matrices):
            perm_avg = PermAVG(models=self.models, naive=False)

            with torch.no_grad():
                perm_avg.models[1].linears[1].weight = nn.Parameter(
                    torch.tensor(
                        [
                            [1, 2, 3],
                            [4, 5, 6]
                        ],
                        dtype=torch.float32
                    )
                )
            batch_flattened = torch.tensor(
                [
                    [1, 2, 3],
                    [1, 2, 3]
                ],
                dtype=torch.float32
            )

            output = self.perm_avg._calc_model_layer_output(
                batch_flattened,
                model_index=1,
                layer_index=1
            )

            self.assertEqual(list(output[0]), [15.5, 6.5])

    def test_calc_layer_output(self):
        """Tests that output is correct for a single layer over
        all models
        """
        self.perm_avg.naive = True
        batch_flattened = torch.ones(size=(2, 3))
        output = self.perm_avg._calc_layer_output(
            batch_flattened, 
            layer_index=0
        )

        self.assertEqual(list(output[0]), [4.5, 4.5, 4.5])
    
    def test_forward(self):
        self.perm_avg.naive = True
        input = torch.ones(size=(2,3,1,1))
        output = self.perm_avg(input)

        self.assertEqual(list(output[0]), [20.25, 20.25])

    def test_forward_relu(self):
        """Tests that relu is called"""
        self.perm_avg.naive = True
        input = torch.ones(size=(2, 3, 1, 1)) * -1
        output = self.perm_avg(input)

        self.assertEqual(list(output[0]), [0, 0])

    def test_forward_relu_last_output(self):
        """Tests that relu is not called on last output"""
        output_0 = torch.ones(size=(3, 3))
        output_1 = torch.ones(size=(2, 2)) * -1
        layer_outputs = [output_0, output_1]
        def mock_outputs(*args, **kwargs):
            return layer_outputs.pop()

        with patch.object(PermAVG, 'forward', side_effect=mock_outputs):
            perm_avg = PermAVG(self.models, naive=True)
            input = torch.ones(size=(2, 3, 1, 1))
            output = perm_avg(input)

        self.assertEqual(list(output[0]), [-1, -1])
        

if __name__ == "__main__":
    unittest.main()