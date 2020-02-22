"""
CNN model implementation.
"""
import torch.nn as nn


class CNN(nn.Module):
    """
    CNN module implementation.
    """
    def __init__(self, params):
        """
        CNN model initialization method.

        :param dict params: Dictionary of model parameters.
        """
        super(CNN, self).__init__()
        self.layers = []
        self.params = params
        self.__init_cnns()

    def forward(self, x):
        """
        Forward operation for the multi-layer CNN model.

        :param x: Input tensor with dimensions (B, T, M, N).
        :return: Output vector.
        """
        y = x.clone()
        for layer in self.layers:
            y = layer(y).clone()
        return y

    def __init_cnns(self):
        """
        Constructs the model variables and sub-structures.

        :return: None
        """
        self.layers = []
        for layer_params in self.params["layers"]:
            self.layers.extend([nn.Conv2d(**layer_params), nn.ReLU()])
