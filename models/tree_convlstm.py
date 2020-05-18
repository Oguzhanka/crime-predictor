"""
TreeCNN model implementation.
"""
import torch
import torch.nn as nn
from torch.optim import Adam

import numpy as np
import matplotlib.pyplot as plt

from models.tree import Tree
from models.convlstm import ConvLSTM


class TreeConvlstm(nn.Module):
    """
    TreeCNN model implementation.
    """
    def __init__(self, params):
        """
        TreeCNN model initialization.

        :param params: Dictionary of model parameters.
        """
        super(TreeConvlstm, self).__init__()
        self.params = params

        self.__init_model()
        param_list = [{"params": self.tree.parameters(), "lr": params["lr"]*0.001}]
        param_list.extend([{"params": self.cnns[k].parameters(), "lr": params["lr"]}
                           for k in range(self.tree.num_regions)])

        self.optimizer = Adam(params=param_list)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        """
        Computes the predictions for a given input tensor.

        :param x: Input tensor of past crimes and side information. (T, D, M, N)
        :return: Predictions for the future.
        """
        region_map = self.tree.regions

        model_preds = []
        for x_t in x:
            x_t = x_t[None, :, :, :]
            preds = torch.zeros(1, self.params["output_dim"], *self.params["input_size"])
            for k in range(self.tree.num_regions):
                model_output = self.cnns[k](x_t)
                preds += torch.mul(model_output, region_map[:, :, :, k].unsqueeze(dim=1)
                                   .repeat(1, self.params["output_dim"], 1, 1))
            model_preds.append(preds)

        model_preds = torch.cat(model_preds, dim=0)
        return model_preds

    def fit(self, x, y):
        """
        Fits the model to a given batch of input tensors.

        :param x: Tensor of input images (T, D, M, N).
        :param y: Tensor of output images (T, D, M, N).
        :return: Loss.
        """
        losses = 0.0
        num_seqs = int(x.shape[0] / self.params["sequence_length"]) + 1
        for i in range(num_seqs):
            self.optimizer.zero_grad()
            self.detach()
            self.reset()
            x_cur = x[i*self.params["sequence_length"]:(i+1)*self.params["sequence_length"]]
            y_cur = y[i*self.params["sequence_length"]:(i+1)*self.params["sequence_length"]]

            y_hat = self(x_cur)
            loss = self.criterion(y_hat, y_cur)
            loss.backward()
            self.optimizer.step()
            loss_ = loss.item()

            losses += loss_ / num_seqs
        return losses

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
            self.cnns.append(ConvLSTM(self.params["cnn_params"]))

    def detach(self):
        for k in range(self.tree.num_regions):
            self.cnns[k].detach()

    def reset(self):
        for k in range(self.tree.num_regions):
            self.cnns[k].reset()

    def visualize(self):
        """

        :return:
        """
        region_map = self.tree.regions
        new_map = np.zeros(self.params["tree_params"]["input_size"])
        for k in range(self.tree.num_regions):
            new_map += 10 * k * region_map[0, :, :, k].detach().numpy()

        plt.imshow(new_map)
        plt.show()
