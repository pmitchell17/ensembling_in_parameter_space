import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mlp import MLP
from models.sinkhorn import SinkhornStableNormalizer


class PermAVG(nn.Module):
    """Implementation of the proposed 'PermAVG' model

    The PermAVG model is an ensemble of multiple MLP models,
    which are averaged in parameter space. The model learns
    the optimal permutation matrices such that the average
    of the permuted parameter matrices is optimal. Compared
    to deep ensembles which average in function space and thus
    require M forward passes and memory for storing M models in
    Memory, the PermAVG model requires a single forward pass
    and storing a single model.
    """

    def __init__(self, models, naive=False, sinkhorn_temp=1):
        """
        Args:
            models (list(MLP)): A list of mlp models
            naive (bool): naive=True refers to averaging without
                applying learned permutation matrices. Hence, we
                are creating an ensemble by "naively" averaging
                over the parameters of the M mlp models. It is
                useful for comparing "naive" averaging vs. averaging
                over permuted paramater matrices.
            sinkhorn_temp (float): The temperature which should be
                used in the sinkhorn algorithm. This controls the
                softness of the Doubly-Stochastic-Matrix. Low values
                lead to more mass concentration on fewer matrix cells.

        """
        super().__init__()

        self.models = models
        self.num_models = len(models)
        self.num_layers = len(self.models[0].linears)
        self.naive = naive
        self.sinkhorn = SinkhornStableNormalizer(temp=sinkhorn_temp)
        self.relu = nn.ReLU()

        self._init_param_matrices()
        self._freeze_model_weights()

    def _freeze_model_weights(self):
        """Freeze the weights of the M mlp models
        
        We only want to learn the optimal permutation of
        the weights. Therefore we freeze the gradients
        of the parameters of the M mlp models.
        """
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

    def _init_param_matrices(self):
        """Create parameter cost matrices which will parameterize permutation
            matrices.

        Any square matrix with positive values has a unique mapping to
        a Doubly-Stochastic-Matrix (DSM). A DSM is a continuous relaxation
        of a permutation matrix, ideal for gradient based learning. 
        """
        self.param_matrices = dict()
        for model_index, model in enumerate(self.models):
            for layer_index, layer in enumerate(self.models[model_index].linears):
                layer_out_units = layer.out_features
                param_matrix = self._create_param_matrix(
                    model_index=model_index,
                    layer_index=layer_index, 
                    param_size=layer_out_units
                )
                self.param_matrices[model_index, layer_index] = param_matrix
        
    def _create_param_matrix(self, model_index, layer_index, param_size):
        """Create a parameter cost matrix

        Note that we do not need to learn permutation matrices for the
        first model, nor for the final layer of any model.

        Args:
            model_index (int): The index of the model in the models list
            layer_index (int): The index of the model layer
            param_size (int): The size D of the square parameter matrix.
                Hence, the dimensions are D*D. Note, that this corresponds
                to the number of out units of the given layer.
        
        Returns:
            param_matrix (nn.Paramter): A parameter matrix for the given
                model and layer index.
        
        """
        if self.naive or model_index == 0 or layer_index == self.num_layers - 1:
            return nn.Parameter(torch.eye(n=param_size), requires_grad=False)
        return nn.Parameter(torch.eye(n=param_size), requires_grad=True)

    def _batch_flatten(self, input):
        """Flatten each image in batch to 1D vector

        Args:
            input (torch.tensor): Batch of images

        Returns:
            input (torch.tensor): Batch of flattened images
        """
        if input.dim() > 2:
            input = input.reshape(input.shape[0],-1)
        return input
    
    def get_perm_matrix(self, model_index, layer_index):
        """ Convert parameter cost matrix to a permutation matrix

        The cost matrix is converted to a DSM matrix by applying
        the sinkhorn algorithm.

        Args:
            model_index (int): The index of the model in the models list
            layer_index (int): The index of the model layer 
        
        Returns:
            permutation_matrix (torch.tensor): A doubly-stochastic
                permutation matrix
        """
        param_matrix = self.param_matrices[(model_index, layer_index)]
        if self.naive or model_index == 0 or layer_index == self.num_layers - 1:
            return param_matrix 
        return self.sinkhorn(param_matrix)       
    
    def _calc_model_layer_output(self, input, model_index, layer_index):
        """Calculate the output for a given model and layer

        Note that permuting the parameter matrix is equivalent to permuting
        the output of a layer. 

        Args:
            input (torch.tensor): A batch of flattened images
            model_index (int): The index of the model in the models list
            layer_index (int): The index of the model layer 

        Returns:
            permuted_output (torch.tensor): The permuted output of the model
                at the given layer

        """
        perm_matrix = self.get_perm_matrix(model_index, layer_index)
        if layer_index != 0:
            prev_perm_matrix = self.get_perm_matrix(model_index, layer_index - 1)
            input = F.linear(input, prev_perm_matrix.T, None)
        model_layer = self.models[model_index].linears[layer_index]
        model_layer_output = model_layer(input)
        permuted_output = \
            F.linear(model_layer_output, perm_matrix, None) * (1 / self.num_models)
        return permuted_output

    def _calc_layer_output(self, input, layer_index):
        """Calculate the output of all models for a given layer

        The output of a layer is the average over all models output
        
        Args:
            input (torch.tensor): A batch of flattened images
            layer_index (int): The index of the model layer

        Returns:
            layer_output (torch.tensor): The output for a given
                layer 

        """
        layer_output = 0
        for model_index in range(self.num_models):
            model_layer_output = self._calc_model_layer_output(
                input, 
                model_index, 
                layer_index
            )
            layer_output += model_layer_output
        return layer_output

    def forward(self, input):
        """Evaluates an input
        
        Args:
            input (torch.tensor): A batch of images
        
        Returns:
            ouput (torch.tensor): Output of PermAVG model
            
        """
        input = self._batch_flatten(input)
        for layer_index in range(self.num_layers):
            if layer_index != 0:
                input = self.relu(input)
            layer_output = self._calc_layer_output(input, layer_index)
            input = layer_output
        return layer_output

    def _get_permuted_layer_weights(self, model_index, layer_index):
        layer = self.models[model_index].linears[layer_index]
        perm_matrix = self.get_perm_matrix(model_index, layer_index)
        permuted_weights = perm_matrix @ layer.weight
        permuted_bias = perm_matrix @ layer.bias
        return permuted_weights, permuted_bias

    def get_weights(self, permuted=True):
        weights_dict = dict()
        for model_index, model in enumerate(self.models):
            for layer_index, layer in enumerate(model.linears):
                if permuted:
                    weights, bias = \
                    self._get_permuted_layer_weights(model_index, layer_index)
                else:
                    weights, bias = layer.weight, layer.bias
                weights_dict[model_index, layer_index] = \
                    {
                        "weights": weights,
                        "bias": bias
                    }
        return weights_dict

    def export(self):
        in_out_units = [
            (linear.in_features, linear.out_features) for linear in self.models[0].linears
        ]

        export_model = MLP(in_out_units)
        for layer_index in range(self.num_layers):
            avg_layer_weights = 0
            avg_layer_bias = 0
            for model_index, _ in enumerate(self.models):
                permuted_weights, permuted_bias = \
                    self._get_permuted_layer_weights(model_index, layer_index)
                avg_layer_weights += permuted_weights / self.num_models
                avg_layer_bias += permuted_bias / self.num_models
            with torch.no_grad():
                export_model.linears[layer_index].weight = nn.Parameter(
                    avg_layer_weights,
                    requires_grad=False
                )
                export_model.linears[layer_index].bias = nn.Parameter(
                    avg_layer_bias,
                    requires_grad=False
                )
        return export_model

        

