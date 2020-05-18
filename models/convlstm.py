"""
ConvLSTM model implementation.
"""
import torch
import torch.nn as nn


class ConvLSTM(nn.Module):
    """
    ConvLSTM module implementation.
    """
    def __init__(self, params):
        """
        ConvLSTM model initialization method.

        :param dict params: Dictionary of model parameters.
        """
        super(ConvLSTM, self).__init__()
        self.layers = []
        self.params = params
        self.__init_cnns()

        self.h = None
        self.c = None
        self.reset()

    def forward(self, x):
        """
        Forward operation for the multi-layer CNN model.

        :param torch.Tensor x: Input tensor with dimensions (B, D, M, N).
        :return: Output vector.
        """
        merged = torch.cat([x, self.h], dim=1)
        conved = self.conv(merged)
        cc_f, cc_i, cc_g, cc_o = torch.split(conved, self.params["hidden_dim"], 1)

        self.c = self.c * self.gate_f(cc_f) + self.gate_i(cc_i) * self.gate_g(cc_g)
        self.h = self.gate_o(cc_o) * self.gate_o2(self.c)

        y = self.conv_out(self.h)
        return y

    def __init_cnns(self):
        """
        Constructs the model variables and sub-structures.

        :return: None
        """
        self.conv = nn.Conv2d(in_channels=self.params["hidden_dim"] + self.params["input_dim"],
                              out_channels=4 * self.params["hidden_dim"],
                              kernel_size=self.params["kernel_size"],
                              padding=self.params["kernel_size"] // 2)

        self.conv_out = nn.Conv2d(in_channels=self.params["hidden_dim"],
                                  out_channels=1,
                                  kernel_size=self.params["kernel_size"],
                                  padding=self.params["kernel_size"] // 2)

        self.gate_f = nn.Sigmoid()
        self.gate_i = nn.Sigmoid()
        self.gate_g = nn.Tanh()
        self.gate_o = nn.Sigmoid()
        self.gate_o2 = nn.Tanh()

    def reset(self):
        """

        :return:
        """
        self.h = torch.zeros(self.params["batch_size"], self.params["hidden_dim"],
                             *self.params["input_size"])
        self.c = torch.zeros_like(self.h)

    def detach(self):
        self.h = self.h.detach()
        self.c = self.c.detach()
