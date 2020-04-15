"""
Main script for testing the model flow.
"""
import config
from models.tree_cnn import TreeCNN
from transformers import WeatherTransformer, CrimeTransformer
from batch_generator import BatchGenerator

if __name__ == "__main__":
    params = config.DataParams().__dict__
    params.update({"file_name": params["weather_file"]})
    weather_transform = WeatherTransformer(params)
    weather_data = weather_transform.load_data("processed_weather")

    params.update({"file_name": params["crime_file"]})
    crime_transform = CrimeTransformer(params)
    crime_data = crime_transform.load_data("processed_crime")
    batch_generator = BatchGenerator(params, crime_data, weather_data)

    params = config.TreeCNNParams().__dict__
    model = TreeCNN(params)

    for e in range(150):
        x, w, y = batch_generator.get_batch("train")
        loss = model.fit(x, y)
        print("Loss: " + str(float(loss)))
