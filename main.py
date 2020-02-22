"""
Main script for testing the model flow.
"""
import config
from models.tree_cnn import TreeCNN

from fake_data_generator import generate_batch


if __name__ == "__main__":
    params = config.TreeCNNParams().__dict__

    model = TreeCNN(params)

    for e in range(150):
        x, y = generate_batch(config.INPUT_WINDOW_LEN,
                              config.KSTEP,
                              params["batch_size"],
                              config.INPUT_SIZE)
        loss = model.fit(x, y)
        print("Loss: " + str(float(loss)))
