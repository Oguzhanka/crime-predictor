import pandas as pd
import numpy as np
import datetime
import pickle
import torch
import os


class Transformer:
    def __init__(self, params):
        self.file_path = "./data/" + params["file_name"] + ".csv"

        self.start_date = params["start_date"]
        self.end_date = params["end_date"]
        self.temporal_resolution = params["temporal_resolution"]
        self.spatial_resolution = params["spatial_resolution"]

        self.spatial_boundaries = params["spatial_boundaries"]
        self.lat_width = None
        self.lon_width = None
        self.compute_bounds()

        self.data = None
        self.preread_data()

    def compute_bounds(self):
        lat_width = (self.spatial_boundaries[0][1] - self.spatial_boundaries[0][0]) / self.spatial_resolution[0]
        lon_width = (self.spatial_boundaries[1][1] - self.spatial_boundaries[1][0]) / self.spatial_resolution[1]

        self.lat_width = lat_width
        self.lon_width = lon_width

    def preread_data(self):
        raise NotImplementedError

    def transform(self):
        raise NotImplementedError

    def load_data(self, data_path):
        if os.path.exists("./data/" + data_path + ".pkl"):
            tensor_data = pickle.load(open("./data/" + data_path + ".pkl", "rb"))
        else:
            tensor_data = self.transform()
            pickle.dump(tensor_data, open("./data/" + data_path + ".pkl", "wb"))
        return tensor_data


class CrimeTransformer(Transformer):
    def __init__(self, params):
        super(CrimeTransformer, self).__init__(params)
        self.params = params
        self.preread_data()

    def transform(self):
        num_temp_grids = int((self.end_date - self.start_date).total_seconds() // 3600 / self.temporal_resolution + 1)
        tensor_data = torch.zeros((num_temp_grids, *self.spatial_resolution, 1))

        min_data = self.data["Date"].min()
        for i, row in self.data.iterrows():
            t_ind = int((row["Date"] - min_data).total_seconds() // 3600 / self.temporal_resolution)
            x_ind = int((self.spatial_boundaries[0][1] - row["Latitude"]) / self.lat_width)
            y_ind = self.spatial_resolution[1] - 1 - int((self.spatial_boundaries[1][1] - row["Longitude"]) / self.lon_width)

            tensor_data[t_ind, x_ind, y_ind, 0] += 1

        return tensor_data

    def preread_data(self):
        raw_data = pd.read_csv(self.file_path)
        filtered_data = raw_data[["Date", "Latitude", "Longitude"]]
        filtered_data["Date"] = pd.to_datetime(filtered_data["Date"])
        cropped_data = filtered_data[filtered_data["Date"].between(self.start_date,
                                                                   self.end_date)]
        cropped_data = cropped_data[cropped_data["Latitude"].between(self.spatial_boundaries[0][0],
                                                                     self.spatial_boundaries[0][1])]
        cropped_data = cropped_data[cropped_data["Longitude"].between(self.spatial_boundaries[1][0],
                                                                      self.spatial_boundaries[1][1])]

        self.data = cropped_data


class WeatherTransformer(Transformer):
    def __init__(self, params):
        super(WeatherTransformer, self).__init__(params)
        self.params = params

    def preread_data(self):
        raw_data = pd.read_csv(self.file_path)
        raw_data["Date"] = pd.to_datetime(raw_data["Date"])
        self.data = raw_data

    def transform(self):
        self.data = pd.get_dummies(self.data, columns=["Chicago"])
        num_temp_grids = int((self.end_date - self.start_date).total_seconds() // 3600 / self.temporal_resolution + 1)
        num_dims = 33
        tensor_data = torch.zeros((num_temp_grids, *self.spatial_resolution, num_dims))

        min_data = self.data["Date"].min()
        for i, row in self.data.iterrows():
            t_ind = int((row["Date"] - min_data).total_seconds() // 3600 / self.temporal_resolution)
            tensor_data[t_ind, :, :, np.where(row)[0][-1] - 1] = 1

        return tensor_data


if __name__ == "__main__":
    params = {"start_date": datetime.datetime(year=2012, month=10, day=1),
              "end_date": datetime.datetime(year=2017, month=11, day=30),
              "file_name": "weather",
              "temporal_resolution": 24,
              "spatial_resolution": [30, 10],
              "spatial_boundaries": [[41.6100, 42.042910333],
                                     [-87.939568, -87.501442]]}

    transformer = WeatherTransformer(params)
    tensor_data = transformer.load_data("processed_weather")
    print(tensor_data.shape)

    params = {"start_date": datetime.datetime(year=2012, month=10, day=1),
              "end_date": datetime.datetime(year=2017, month=11, day=30),
              "file_name": "cropped_crime",
              "temporal_resolution": 24,
              "spatial_resolution": [30, 10],
              "spatial_boundaries": [[41.6100, 42.042910333],
                                     [-87.939568, -87.501442]]}
    transformer = CrimeTransformer(params)
    tensor_data = transformer.load_data("processed_crime")
    import matplotlib.pyplot as plt
    plt.imshow(tensor_data.sum(dim=0).sum(dim=2))
    plt.show()
