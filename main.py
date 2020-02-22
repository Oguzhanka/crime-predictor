"""
Main script for testing the model flow.
"""
import torch
import config
from models.tree_cnn import TreeCNN


if __name__ == "__main__":
    params = config.TreeCNNParams().__dict__

    model = TreeCNN(params)

    dummy_input = torch.randn(1, 5, 10, 30)
    output = model(dummy_input)
