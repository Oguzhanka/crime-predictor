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
        TreeCNN model initialization. Parameters are given as a dictionary. Details about
        the content of the dictionary can be found in the config.py script.

        :param params: Dictionary of model parameters.
        """
        super(TreeConvlstm, self).__init__()
        self.params = params

        self.__init_model()     # Initializes the parts of the model.
        param_list = [{"params": self.tree.parameters(), "lr": params["lr"]*0.001}]
        param_list.extend([{"params": self.cnns[k].parameters(), "lr": params["lr"]}
                           for k in range(self.tree.num_regions)])

        self.optimizer = Adam(params=param_list)    # Adam optimizer.
        self.criterion = nn.MSELoss()               # MSE loss.

    def forward(self, x):
        """
        Computes the predictions for a given input tensor.

        :param x: Input tensor of past crimes and side information. (T, D, M, N)
        :return: Predictions for the future.
        """
        region_map = self.tree.regions

        model_preds = []
        for x_t in x:                   # For each sample in subsequence...
            x_t = x_t[None, :, :, :]
            preds = torch.zeros(1, self.params["output_dim"], *self.params["input_size"])   # Initialize predictions.
            for k in range(self.tree.num_regions):                                          # For each subregion...
                model_output = self.cnns[k](x_t)                                            # Compute the raw output.
                preds += torch.mul(model_output, region_map[:, :, :, k].unsqueeze(dim=1)    # Mask with the regional
                                   .repeat(1, self.params["output_dim"], 1, 1))             # mask and sum preds.
            model_preds.append(preds)

        model_preds = torch.cat(model_preds, dim=0)         # Merge predictions back.
        return model_preds

    def fit(self, x, w, y):
        """
        Fits the model to a given batch of input tensors.

        :param x: Tensor of input images (T, D, M, N).
        :param y: Tensor of output images (T, D, M, N).
        :return: Loss.
        """
        x = torch.cat([x, w], dim=1)

        losses = 0.0
        num_seqs = int(x.shape[0] / self.params["sequence_length"]) + 1     # Compute the number of subsequences.
        for i in range(num_seqs):           # For each subsequence...
            self.optimizer.zero_grad()      # Reset optimizer gradients.
            self.detach()                   # Remove gradient connections with the previous subsequence.
            self.reset()                    # Reset state vectors.
            x_cur = x[i*self.params["sequence_length"]:(i+1)*self.params["sequence_length"]]    # Current inputs.
            y_cur = y[i*self.params["sequence_length"]:(i+1)*self.params["sequence_length"]]    # Current labels.

            y_hat = self(x_cur)             # Compute predictions.
            weight_loss = 0.0               # L2-weight
            for k in range(self.tree.num_regions):  # For each subregion...
                weight_loss += self.cnns[k].conv.weight.norm()  # Increment the L2-weight term with the loss

            loss = self.criterion(y_hat, y_cur) + 1e-4 * weight_loss    # Compute the overall loss.
            loss.backward()                                             # Update model parameters.
            self.optimizer.step()
            loss_ = loss.item()

            losses += loss_ / num_seqs      # Compute mean loss.
        return losses

    def predict(self, x, w):
        """
        Returns a batch of predictions without performing gradient updates.

        :param x: Batch of inputs (T, D, M, N).
        :return: Batch of outputs (T, D, M, N).
        """
        x = torch.cat([x, w], dim=1)    # Merge inputs and side information.

        with torch.no_grad():           # Disable gradient computations.
            y = self(x)                 # Compute predictions.
        return y

    def __init_model(self):
        """
        Constructs the model with the given set of model parameters.

        :return: None
        """
        self.tree = Tree(self.params["tree_params"])
        self.cnns = []
        for k in range(self.tree.num_regions):                      # For each subregion...
            self.cnns.append(ConvLSTM(self.params["cnn_params"]))   # Initialize and add a model.

    def detach(self):
        """
        Detaches each LSTM state vector for all subregions.
        :return: None.
        """
        for k in range(self.tree.num_regions):      # For each subregion...
            self.cnns[k].detach()                   # Detach the state vector.

    def reset(self):
        """
        Resets the state vector for all models.
        :return: None
        """
        for k in range(self.tree.num_regions):      # For each subregion...
            self.cnns[k].reset()                    # Reset the state vector.

    def visualize(self):
        """
        Visualizes the spatial subregions represented by the decision tree.
        :return: None
        """
        region_map = self.tree.regions      # Get the regional map.
        new_map = np.zeros(self.params["tree_params"]["input_size"])        # Initialize a visualization array.
        for k in range(self.tree.num_regions):                              # For each subregion...
            new_map += 10 * k * region_map[0, :, :, k].detach().numpy()     # Place the subregion on the array.

        plt.imshow(new_map)                 # Visualize the map array.
        plt.show()

    def hist(self):
        """
        Computes the histogram of weights for the weights corresponding to the weather events.
        :return: Weight histograms.
        """

        hists = {}
        for k in range(self.tree.num_regions):
            hists[k] = self.cnns[k].hist()

        hist = []
        bounds = np.linspace(-0.5, 0.5, 1000)
        plt.figure()
        plt.subplot(6, 6, 1)
        for i in range(33):
            plt.subplot(6, 6, i+1)
            vals = np.concatenate([hists[k][i] for k in range(self.tree.num_regions)])
            h, _ = np.histogram(vals, bounds)
            hist.append(h)
            plt.bar(bounds[1:], h)

        return hist, bounds
