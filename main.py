"""
Main script for testing the model flow.
"""
import torch
import config
from models.tree_convlstm import TreeConvlstm
from transformers import WeatherTransformer, CrimeTransformer
from batch_generator import BatchGenerator
from validation import pr_curve


if __name__ == "__main__":
    params = config.DataParams().__dict__
    params.update({"file_name": params["weather_file"]})
    weather_transform = WeatherTransformer(params)
    weather_data = weather_transform.load_data("processed_weather_{}_{}_{}".format(params["temporal_resolution"],
                                                                                   params["spatial_resolution"][0],
                                                                                   params["spatial_resolution"][1]))

    params.update({"file_name": params["crime_file"]})
    crime_transform = CrimeTransformer(params)
    crime_data = crime_transform.load_data("processed_crime_{}_{}_{}".format(params["temporal_resolution"],
                                                                             params["spatial_resolution"][0],
                                                                             params["spatial_resolution"][1]))
    batch_generator = BatchGenerator(params, crime_data, weather_data)

    params = config.TreeCNNParams().__dict__
    model = TreeConvlstm(params)

    for e in range(1500):
        x, w, y = batch_generator.get_batch("train")
        x = torch.cat([x, w], dim=1)
        loss = model.fit(x, y)
        print("Loss: " + str(float(loss)))

        x, w, y = batch_generator.get_batch("validation")
        x = torch.cat([x, w], dim=1)
        y_hat = model.predict(x)
        pr_curve(y_hat, y)
        model.visualize()
