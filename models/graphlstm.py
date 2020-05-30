import torch
import pickle
import torch.nn as nn
from torch.optim import Adam

from models.lstm import LSTM


class GraphLSTM(nn.Module):
    def __init__(self, params):
        """
        Graph-LSTM model implementation (SOTA). Takes the parameters for the configurable
        parameters and the connection graph from the arguments.
        :param params: Dictionary of parameters.
        """
        super().__init__()

        self.params = params

        self.graph = pickle.load(open("./data/graph_{}_{}.pkl".format(params["input_size"][0],
                                                                      params["input_size"][1]),
                                      "rb"))
        self.graph[(-0.5 < self.graph) & (self.graph < 0.5)] = 0

        self.edges = {}             # Edge connections.
        self.lstms = {}             # LSTMs for the edges and vertices.
        self.__construct_net()      # Constructs the model.

        modules = []
        for _, models in self.lstms.items():    # For each LSTM family...
            modules.extend(models)              # Append the models to the existing ones.
        self.module_list = nn.ModuleList(modules)
        self.conv = nn.Conv2d(self.params["hidden_dim"], self.params["output_dim"], 1)  # Channel projection stage.

        self.optimizer = Adam(params=self.parameters(), lr=self.params["lr"])           # Adam optimizer.
        self.criterion = nn.MSELoss()                                                   # MSE loss.

    def __construct_net(self):
        """
        Constructs the edge and vertex LSTM models according to the graph dictionary.
        :return: None
        """
        for i in range(self.params["input_size"][0]*self.params["input_size"][1]):  # For each cell...
            x_i = i // self.params["input_size"][1]     # Get the x and y indices.
            y_i = i % self.params["input_size"][1]
            self.edges[(x_i, y_i)] = [(x_i, y_i)]       # Append the current position to the graph.
            self.lstms[(x_i, y_i)] = [LSTM(input_size=self.params["input_dim"],     # Add the corresponding LSTM.
                                           hidden_size=self.params["hidden_dim"])]

            for j in range(self.params["input_size"][0] * self.params["input_size"][1]):    # For each cell..
                x_j = j // self.params["input_size"][1]     # Get the x and y indices.
                y_j = j % self.params["input_size"][1]

                if self.graph[i][j] != 0:                       # If there is a connection for the pair...
                    self.edges[(x_i, y_i)].append((x_j, y_j))   # Append to the graph.
                    self.lstms[(x_i, y_i)].append(LSTM(input_size=self.params["input_dim"],
                                                       hidden_size=self.params["hidden_dim"]))  # Add the LSTM.

    def forward(self, x):
        """
        Forward operation for the GraphLSTM model.

        :param x: (B, D, M, N)
        :return:
        """
        output_frame = torch.zeros(x.shape[0], self.params["hidden_dim"], x.shape[2], x.shape[3])

        for i in range(self.params["input_size"][0]*self.params["input_size"][1]):  # For each cell...
            x_i = i // self.params["input_size"][1]     # Get the x and y indices.
            y_i = i % self.params["input_size"][1]

            output = torch.zeros(1, self.params["hidden_dim"], 1, 1)    # Initialize the prediction array.

            for source, model in zip(self.edges[(x_i, y_i)], self.lstms[(x_i, y_i)]):   # For each connected vertex...
                source_index = source[0] * self.params["input_size"][1] + source[1]
                source_input = x[:, :, source[0], source[1]]    # Get the corresponding input.
                weight = self.graph[i, source_index]            # Get the weight from the graph.
                out = model(source_input)                       # Compute the output for that vertex.

                output[:, :, 0, 0] += weight * out              # Add the weighted input

            output_frame[:, :, x_i, y_i] = output[:, :, 0, 0]   # Fill the prediction array.

        output_frame = self.conv(output_frame)      # Channel projection array.
        return output_frame

    def fit(self, x, w, y):
        """
        Fits the model to a given batch of input tensors.

        :param x: Tensor of input images (T, D, M, N).
        :param y: Tensor of output images (T, D, M, N).
        :return: Loss.
        """

        losses = 0.0
        num_seqs = int(x.shape[0] / self.params["sequence_length"]) + 1     # Compute the number of subsequences.

        for i in range(num_seqs):           # For each subsequence...
            self.optimizer.zero_grad()      # Reset the optimizer gradients.
            self.detach()                   # Detach the state vetors.
            self.reset()
            x_cur = x[i * self.params["sequence_length"]:(i + 1) * self.params["sequence_length"]]  # Current inputs.
            y_cur = y[i * self.params["sequence_length"]:(i + 1) * self.params["sequence_length"]]  # Current labels.

            y_hat = torch.cat([self(x_par[None, :]) for x_par in x_cur])    # Compute predictions.
            loss = self.criterion(y_hat, y_cur)                             # Compute loss and update model.
            loss.backward()                                                 # Update parameters.
            self.optimizer.step()
            loss_ = loss.item()                 # Get the loss value.

            losses += loss_ / num_seqs          # Compute the mean loss.
        return losses

    def predict(self, x, w):
        """
        Returns a batch of predictions without performing gradient updates.

        :param x: Batch of inputs (T, D, M, N).
        :return: Batch of outputs (T, D, M, N).
        """

        with torch.no_grad():       # Disable gradient computations.
            y = torch.cat([self(x_par[None, :]) for x_par in x])    # Compute predictions.
        return y

    def detach(self):
        """
        Detaches the state vectors.
        :return: None
        """
        for _, models in self.lstms.items():    # For each model family...
            for model in models:                # For each LSTM model...
                model.detach()                  # Detach the state vectors.

    def reset(self):
        """
        Resets the state vectors.
        :return: None.
        """
        for _, models in self.lstms.items():    # For each model family...
            for model in models:                # For each LSTM model...
                model.reset()                   # Reset the state vectors.
