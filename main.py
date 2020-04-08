"""
Main script for testing the model flow.
"""
import config
from models.tree_cnn import TreeCNN
from data_processor import DataProcessor
from fake_data_generator import generate_batch
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_processor = DataProcessor(24, 10, 8)
    data_processor.read_to_file("test")
    # tensor = data_processor.read_file_to_tensor("data/crimes")
    #
    # params = config.TreeCNNParams().__dict__
    #
    # model = TreeCNN(params)
    #
    # for e in range(150):
    #     x, y = generate_batch(config.INPUT_WINDOW_LEN,
    #                           config.KSTEP,
    #                           params["batch_size"],
    #                           config.INPUT_SIZE)
    #     loss = model.fit(x, y)
    #     print("Loss: " + str(float(loss)))
