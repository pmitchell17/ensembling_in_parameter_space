import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from loss.knowledge_distillation_loss import KDLoss
from models.deep_ensemble import DeepEnsemble
from models.mlp import MLP


class TestKDLoss(unittest.TestCase):

    def setUp(self):
        model_0 = MLP([[3, 3], [3, 2]])
        model_0.init_weights_as_int(1)

        model_1 = MLP([[3, 3], [3, 2]])
        model_1.init_weights_as_int(2)
        
        models = [model_0, model_1]

        deep_ensemble = DeepEnsemble(models)
        self.kd_loss = KDLoss(deep_ensemble, alpha=0.5, temp=3)

    def test_calc_dist_p(self):
        student_outputs = torch.tensor(
            [
                [0, 0, 3],
                [3, 0, 0]
            ],
            dtype=torch.float32
        )
        p = self.kd_loss.calc_dist_p(student_outputs)

        outputs = student_outputs / 3
        probs = (
            torch.exp(outputs) / 
            torch.sum(torch.exp(outputs), dim=1).unsqueeze(dim=1)
        )
        log_probs = torch.log(probs)
        
        p_sum = round(torch.sum(p).item(),2)
        log_probs_sum = round(torch.sum(log_probs).item(),2)
        self.assertEqual(p_sum, log_probs_sum)

    def test_calc_dist_q(self):
        images = torch.tensor(
            [
                [0, 0, 1],
                [1, 0, 0]
            ],
            dtype=torch.float32
        )
        q = self.kd_loss.calc_dist_q(images)

        outputs = torch.tensor([[7.5, 7.5], [7.5, 7.5]])
        outputs /= 3
        probs = (
            torch.exp(outputs) /
            torch.sum(torch.exp(outputs), dim=1).unsqueeze(dim=1)
        )

        q_sum = torch.sum(q)
        probs_sum = torch.sum(probs)
        self.assertEqual(q_sum, probs_sum)
    
    def test_kl_divergence(self):
        """Tests that the KL divergence between two distributions
        is equal to 0"""
        p = torch.tensor(
            [
                [0.25, 0.25, 0.5],
                [0.5, 0.25, 0.25]
            ]
        )
        q = torch.log(
            torch.tensor(
                [
                    [0.25, 0.25, 0.5],
                    [0.25, 0.25, 0.5]
                ]
            )
        )
        kl_div = self.kd_loss.calc_kl_divergence(p, q)
        
        self.assertEqual(kl_div.item(), 0)

    def test_forwaard(self):
        "Tests weighted sum of KL and CE loss"
        dummy_images = torch.ones(size=(2, 1, 1, 3))
        dummy_student_outputs = torch.tensor(
            [[0.33, 0.33, 0.33], [0.33, 0.33, 0.33]]
        )
        dummy_labels = torch.ones([1, 1, 1])
        self.kd_loss.calc_kl_divergence = mock.Mock(return_value=torch.tensor(10))
        self.kd_loss.calc_cross_entropy = mock.Mock(return_value=torch.tensor(10))

        self.kd_loss.alpha = 0.5
        loss = self.kd_loss.forward(dummy_student_outputs, dummy_images, dummy_student_outputs)
        
        self.assertEqual(loss.item(), 10)
        

if __name__ == "__main__":
    unittest.main()


