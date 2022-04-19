import unittest

import torch

from models.deep_ensemble import DeepEnsemble
from models.mlp import MLP


class TestDeepEnsemble(unittest.TestCase):

    def setUp(self):
        model_0 = MLP([[3, 3], [3, 2]])
        model_0.init_weights_as_int(1)

        model_1 = MLP([[3, 3], [3, 2]])
        model_1.init_weights_as_int(2)
        
        models = [model_0, model_1]

        self.deep_ensemble = DeepEnsemble(models)

    def test_forward(self):
        input = torch.ones(size=(2,1,1,3))

        output = self.deep_ensemble(input)

        self.assertEqual(output[0].tolist(), [22.5, 22.5])


if __name__ == "__main__":
    unittest.main()

