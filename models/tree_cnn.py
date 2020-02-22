"""
TreeCNN model implementation.
"""
import torch
import torch.nn as nn
from torch.optim import Adam

from models.tree import Tree
from models.cnn import CNN


class TreeCNN(nn.Module):
    """
    TreeCNN model implementation.
    """
    def __init__(self, params):
        """
        TreeCNN model initialization.

        :param params: Dictionary of model parameters.
        """
        super(TreeCNN, self).__init__()
        self.params = params

        self.__init_model()
        param_list = [{"params": self.tree.parameters(), "lr": params["lr"]*0.1}]
        param_list.extend([{"params": self.cnns[k].parameters(), "lr": params["lr"]}
                           for k in range(self.tree.num_regions)])

        self.optimizer = Adam(params=param_list)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        """
        Computes the predictions for a given input tensor.

        :param x: Input tensor of past crimes and side information. (B, D, M, N)
        :return: Predictions for the future.
        """
        region_map = self.tree.regions
        preds = torch.zeros(x.shape[0], self.params["output_dim"], *self.params["input_size"])
        for k in range(self.tree.num_regions):
            model_output = self.cnns[k](x)
            preds += torch.mul(model_output, region_map[:, :, :, k].unsqueeze(dim=1)
                               .repeat(1, self.params["output_dim"], 1, 1))
        return preds

    def fit(self, x, y):
        """
        Fits the model to a given batch of input tensors.

        :param x: Tensor of input images (T, D, M, N).
        :param y: Tensor of output images (T, D, M, N).
        :return: Loss.
        """
        self.optimizer.zero_grad()
        y_hat = self(x)
        loss = 1/x.shape[0] * self.criterion(y_hat, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, x):
        """
        Returns a batch of predictions without performing gradient updates.

        :param x: Batch of inputs (T, D, M, N).
        :return: Batch of outputs (T, D, M, N).
        """
        with torch.no_grad():
            y = self(x)
        return y

    def __init_model(self):
        """
        Constructs the model with the given set of model parameters.

        :return: None
        """
        self.tree = Tree(self.params["tree_params"])
        self.cnns = []
        for k in range(self.tree.num_regions):
            self.cnns.append(CNN(self.params["cnn_params"]))
