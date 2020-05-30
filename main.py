"""
Main script for testing the model flow.
"""
import config

from models.convlstm import ConvLSTM
from models.graphlstm import GraphLSTM
from models.tree_convlstm import TreeConvlstm

from transformers import WeatherTransformer, CrimeTransformer
from batch_generator import BatchGenerator

from validation import pr_curve
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


if __name__ == "__main__":
    """
    Main function used to start the initialization, training and testing of the models. Takes
    no arguments from the std. All arguments and configurable parameters are tuned from the
    config.py script.
    """

    # Data processing parameters are read.
    params = config.DataParams().__dict__
    params.update({"file_name": params["weather_file"]})

    # Weather side information transformer initialized with parameters and transformed.
    weather_transform = WeatherTransformer(params)
    weather_data = weather_transform.load_data("processed_weather_{}_{}_{}".format(params["temporal_resolution"],
                                                                                   params["spatial_resolution"][0],
                                                                                   params["spatial_resolution"][1]))

    # Crime data transformer initialized with parameters and transformed.
    params.update({"file_name": params["crime_file"]})
    crime_transform = CrimeTransformer(params)
    crime_data = crime_transform.load_data("processed_crime_{}_{}_{}".format(params["temporal_resolution"],
                                                                             params["spatial_resolution"][0],
                                                                             params["spatial_resolution"][1]))
    # Batch generator module initialized.
    batch_generator = BatchGenerator(params, crime_data, weather_data)

    # Model parameters are obtained from config.py
    params = config.TreeCNNParams().__dict__
    if config.MODEL_TYPE == "treeconv":             # Our model. Consists of a decision tree and several ConvLSTMs.
        model = TreeConvlstm(params)

    elif config.MODEL_TYPE == "convlstm":           # Baseline model. Simple ConvLSTM model.
        model = ConvLSTM(params["cnn_params"])

    elif config.MODEL_TYPE == "graphlstm":          # SOTA model. Graph based LSTM model.
        params = config.GraphLSTMParams().__dict__
        model = GraphLSTM(params)

    plt.ion()                                       # Plotting object for the validation AP.
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.draw()

    for e in range(config.EPOCHS):                               # For every epoch...
        x, w, y = batch_generator.get_batch("train")    # Read training data.
        loss = model.fit(x, w, y)                       # Perform update on the training data. Also compute loss.
        print("EPOCH: {}".format(e))                    # Print the epoch number.
        print("Loss: " + str(float(loss)))              # Print the loss value for the current training epoch.
        print("")

        x, w, y = batch_generator.get_batch("validation")   # Get the validation set.
        y_hat = model.predict(x, w)                         # Compute the predictions for the validation set.
        pr_curve(y_hat, y, ax)                              # Compute the AP score for the validation score.

    x, w, y = batch_generator.get_batch("test")   # Get the test set.
    y_hat = model.predict(x, w)                   # Compute the predictions for the test set.
    pr_curve(y_hat, y, ax)                        # Compute the AP score for the test score.

plt.ioff()
plt.show()
