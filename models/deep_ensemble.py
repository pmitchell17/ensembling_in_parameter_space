import torch.nn as nn

class DeepEnsemble(nn.Module):
    """An implementation of a deep-ensemble
    
    The deep-ensemble is composed of M models. An inference
    is made by averaging over the outputs of each of the M
    models.
    
    """

    def __init__(self, models):
        """
        
        Args:
            models (list[MLP]): A list of MLP models, which
                all have the same dimensions.

        """
        super().__init__()

        self.models = models

    def forward(self, input):
        """Average over outputs of all M models"""
        num_models = len(self.models)
        outputs = 0
        for model in self.models:
            outputs += model(input) / num_models
        return outputs

