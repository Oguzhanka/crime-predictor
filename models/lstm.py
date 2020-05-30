import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, hidden_size, input_size):
        """
        Standard LSTM model implementation. Only takes hidden size and input size as arguments.
        :param hidden_size: Size of the hidden vector.
        :param input_size:  Size of the input vector data.
        """
        super().__init__()
        params = {"hidden_dim": hidden_size,
                  "input_dim": input_size}
        self.params = params

        self.h = None           # State vector.
        self.c = None
        self.__init_weights()   # Initializes the model weights.
        self.reset()            # Resets the state vector.

    def forward(self, x):
        """
        Forward operation for the LSTM model. State vector is also updated during the process.

        :param x: (B, D) Input vector.
        :return: State vector.
        """
        merged = torch.cat([x, self.h], dim=1)
        weighted = merged @ self.W

        c_f, c_i, c_g, c_o = torch.split(weighted, self.params["hidden_dim"], dim=1)
        self.c = self.c * self.g_f(c_f) + self.g_i(c_i) * self.g_g(c_g)
        self.h = self.g_o(c_o) * self.g_h(self.c)

        return self.h

    def __init_weights(self):
        """
        Initializes the weights of the model and initializes a function for each LSTM gate.
        :return: None.
        """
        self.W = torch.rand(self.params["hidden_dim"] + self.params["input_dim"],
                            4 * self.params["hidden_dim"])
        self.W = nn.Parameter(self.W)

        self.g_f = nn.Sigmoid()
        self.g_i = nn.Sigmoid()
        self.g_g = nn.Tanh()
        self.g_o = nn.Sigmoid()
        self.g_h = nn.Tanh()

    def detach(self):
        """
        Detaches state vectors and reassigns them back to their places.
        :return: None.
        """
        self.h = self.h.detach()
        self.c = self.c.detach()

    def reset(self):
        """
        Resets the state vectors.
        :return: None.
        """
        self.h = torch.zeros(1, self.params["hidden_dim"])
        self.c = torch.zeros(1, self.params["hidden_dim"])
