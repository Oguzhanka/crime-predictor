import random
import torch
import torch.nn as nn
from torch.nn import Parameter


class Tree(nn.Module):
    """
    Partitioning decision tree module.
    """
    def __init__(self, params):
        """
        Soft decision tree initialization with the given dictionary of parameters.

        :param params: Dictionary of parameters.
        """
        super(Tree, self).__init__()
        self.params = params

        self.depth = params["depth"]                    # Depth of the tree.
        self.num_regions = 2 ** params["depth"]         # Number of regions.
        self.num_features = params["num_features"]      # Default to 2. (latitude and longitude)
        self.boundaries = params["boundaries"]          # Boundaries of the spatial region.
        self.input_size = params["input_size"]

        self.hardness = params["hardness"]              # Controls the soft boundaries between the regions.
        self.region_mode = params["region_mode"]        # Initialization mode.
        self.region_rot = params["region_rot"]

        self.tree_layers = []
        self.tree_bias = []
        self.tree_score = []
        self.sign_score = []

        self.tree_structure()
        self.tree_params = nn.ParameterList([*self.tree_layers, *self.tree_bias])

    def forward(self, input_):
        """
        Forward function for the soft decision tree which returns a vector of spatial
        scores for the given region.

        :param tuple input_: Tuple of spatial dimensions (x, y).
        :return: Spatial score vector indicating region.
        """
        input_tensor = torch.Tensor(input_).unsqueeze(dim=1)
        product_score = torch.ones(1, self.num_regions)
        prev_score = torch.ones(1, self.num_regions)

        for level in range(self.depth):                                             # For each level...
            level_score = torch.ones(self.num_regions)                              # Initialize the scores map.
            weighted_score = torch.matmul(self.tree_layers[level], input_tensor)    # Weighted scores.
            biased_score = torch.add(weighted_score, self.tree_bias[level])         # Bias added.
            exped_score = torch.sigmoid(biased_score).repeat(1, 2).view(-1, 1)      # Predictions after gate.
            node_score = torch.sub(self.tree_score[level], exped_score)
            child_score = torch.mul(node_score, self.sign_score[level])

            for i in range(self.num_regions):                                       # For each subregion...
                level_score[i] = child_score[i // (self.num_regions // 2 ** (level + 1))]   # Leaf scores are computed.

            product_score = torch.mul(prev_score, level_score)
            prev_score = product_score.clone()

        return product_score[-1]

    @property
    def regions(self):
        """
        Returns a 2-d array of spatial regions
        :return: Spatial region map.
        """
        region_map = torch.zeros(*self.params["input_size"], self.num_regions)
        for x in range(self.params["input_size"][0]):
            for y in range(self.params["input_size"][1]):
                region_map[x, y, :] = self((self.params["input_size"][0] - x, y)).clone()

        return region_map[None, :, :, :]

    def tree_structure(self):
        """
        Constructs a soft decision tree with the specified parameters.

        :return: None
        """
        boundaries = [self.boundaries]

        selected_feature = 1
        for level in range(self.depth):

            if self.region_mode == 1 or self.region_mode == 2:
                if self.region_mode == 1:
                    select_feats = [random.randint(0, 1) for _ in range(2 ** level)]
                else:
                    select_feats = [selected_feature for _ in range(2 ** level)]
                    selected_feature = 1 if selected_feature == 0 else 0

                level_weights = [[1, self.region_rot] if select_feats[i] == 0
                                 else [self.region_rot, 1] for i in range(2 ** level)]

                bias_vals = [-(bounds[select_feats[i]][0] + bounds[select_feats[i]][1]) / 2 for i, bounds in
                             enumerate(boundaries)]

                boundary = [
                    [[[bounds[0][0], -bias_vals[i]], bounds[1]], [[-bias_vals[i], bounds[0][1]], bounds[1]]]
                    if select_feats[i] == 0 else
                    [[bounds[0], [bounds[1][0], -bias_vals[i]]], [bounds[0], [-bias_vals[i], bounds[1][1]]]]
                    for i, bounds in zip(range(2 ** level), boundaries)]

                boundaries = []
                for i in boundary:
                    boundaries.extend([i[0], i[1]])

                level_tensor = self.hardness * torch.Tensor(level_weights)
                bias_tensor = self.hardness * torch.Tensor(bias_vals).unsqueeze(dim=1)
            else:
                level_tensor = self.hardness * torch.rand(2 ** level, self.num_features)
                bias_tensor = self.hardness * torch.rand(2 ** level, 1)

            self.tree_layers.append(Parameter(level_tensor, requires_grad=True))
            self.tree_bias.append(Parameter(bias_tensor, requires_grad=True))
            level_score = torch.Tensor([1.0 if i % 2 == 1 else 0.0 for i in range(2 ** (level + 1))])
            sign_score = torch.Tensor([-1.0 if i % 2 == 0 else 1.0 for i in range(2 ** (level + 1))]).unsqueeze(1)
            self.tree_score.append(level_score.unsqueeze(dim=1))
            self.sign_score.append(sign_score)
