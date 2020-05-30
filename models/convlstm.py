"""
ConvLSTM model implementation.
"""
import torch
import torch.nn as nn
from torch.optim import Adam


class ConvLSTM(nn.Module):
    """
    ConvLSTM module implementation.
    """
    def __init__(self, params):
        """
        ConvLSTM model initialization method. Arguments are taken as a dictionary. Details
        about the content of the dictionary can be accessed from the config.py.

        :param dict params: Dictionary of model parameters.
        """
        super(ConvLSTM, self).__init__()
        self.params = params
        self.__init_cnns()

        self.h = None   # H state vector.
        self.c = None   # C state vector.
        self.reset()    # Initializes the state vectors with zero values.

        self.optimizer = Adam(params=self.parameters(), lr=self.params["lr"])   # Optimizer for the training.
        self.criterion = nn.MSELoss()                                           # MSE loss.

    def forward(self, x):
        """
        Forward operation for the single-layer CONVLSTM model. Takes the current frame
        as input and generates the prediction for the future frame. State is updated
        when this function is called.

        :param torch.Tensor x: Input tensor with dimensions (B, D, M, N).
        :return: Output vector.
        """
        merged = torch.cat([x, self.h], dim=1)      # Merge [X, H]
        conved = self.conv(merged)                  # Convolve W * [X, H]
        cc_f, cc_i, cc_g, cc_o = torch.split(conved, self.params["hidden_dim"], 1)  # Separate each gate.

        self.c = self.c * self.gate_f(cc_f) + self.gate_i(cc_i) * self.gate_g(cc_g)     # c_t = c_t-1 * f_t + g_t * i_t
        self.h = self.gate_o(cc_o) * self.gate_o2(self.c)                               # h_t = f(c_t) * o_t

        y = self.conv_out(self.h)   # Output dimension reduction by channel projection.
        return y

    def fit(self, x, w, y):
        """
        Fits the model to a given batch of input tensors for a single step.

        :param x: Tensor of input images (T, D, M, N).
        :param y: Tensor of output images (T, D, M, N).
        :return: Loss.
        """
        x = torch.cat([x, w], dim=1)
        losses = 0.0
        num_seqs = int(x.shape[0] / self.params["sequence_length"]) + 1
        for i in range(num_seqs):           # For each subsequence...
            self.optimizer.zero_grad()      # Reset gradients.
            self.detach()                   # Remove gradient ties with the previous subsequence.
            self.reset()
            x_cur = x[i * self.params["sequence_length"]:(i + 1) * self.params["sequence_length"]]  # Current X values.
            y_cur = y[i * self.params["sequence_length"]:(i + 1) * self.params["sequence_length"]]  # Current labels.

            y_hat = torch.cat([self(x_part[None, :]) for x_part in x_cur])  # Predictions combined into a tensor.
            loss = self.criterion(y_hat, y_cur)                     # Loss is computed from the whole subsequence.
            loss.backward()                                         # Update weights.
            self.optimizer.step()
            loss_ = loss.item()                                     # Get the loss value.

            losses += loss_ / num_seqs                              # Compute the mean loss.
        return losses

    def predict(self, x, w):
        """
        Returns a batch of predictions without performing gradient updates.

        :param x: Batch of inputs (T, D, M, N).
        :return: Batch of outputs (T, D, M, N).
        """
        x = torch.cat([x, w], dim=1)
        with torch.no_grad():           # Disable gradient computations.
            y = torch.cat([self(x_part[None, :]) for x_part in x])  # Predictions combined into a tensor.
        return y

    def __init_cnns(self):
        """
        Constructs the model variables and sub-structures. Both the CONVLLSTM convolution and
        the channel projection layer are initialized. Also the internal gates are initialized.

        :return: None
        """
        # ConvLSTM internal convolution operation.
        self.conv = nn.Conv2d(in_channels=self.params["hidden_dim"] + self.params["input_dim"],
                              out_channels=4 * self.params["hidden_dim"],
                              kernel_size=self.params["kernel_size"],
                              padding=self.params["kernel_size"] // 2)

        # Channel projection convolution.
        self.conv_out = nn.Conv2d(in_channels=self.params["hidden_dim"],
                                  out_channels=1,
                                  kernel_size=self.params["kernel_size"],
                                  padding=self.params["kernel_size"] // 2)

        self.gate_f = nn.Sigmoid()  # Forget gate.
        self.gate_i = nn.Sigmoid()  # Input gate.
        self.gate_g = nn.Tanh()     # Modulation gate.
        self.gate_o = nn.Sigmoid()  # Output gate.
        self.gate_o2 = nn.Tanh()    # Projection output gate.

    def reset(self):
        """
        Resets the state vectors back to zero vectors with the standard dimensions. In place
        operation.

        :return: None
        """
        self.h = torch.zeros(self.params["batch_size"], self.params["hidden_dim"],
                             *self.params["input_size"])
        self.c = torch.zeros_like(self.h)

    def detach(self):
        """
        Detaches the state vectors and reassigns them back to their places. Called in between
        the subsequences to break gradient connections.
        :return:
        """
        self.h = self.h.detach()
        self.c = self.c.detach()

    def hist(self):
        """
        Computes the histogram of weights for the weights corresponding to the weather events.
        :return: Weight histograms.
        """
        weather_weights = self.conv.weight

        weights = []
        for i in range(1, 34):      # For each weather event...
            weights.append(weather_weights[:, i].detach().numpy())

        return weights
