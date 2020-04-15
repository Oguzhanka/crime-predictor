import torch
import numpy as np


class BatchGenerator:
    def __init__(self, params, crime_data, weather_data):
        self.params = params
        self.data = None
        self.side = None
        self.label = None
        self.merge_data(crime_data, weather_data)

        self.train_bound = self.label.shape[0] * self.params["ratio"][0]
        self.val_bound = self.train_bound + self.label.shape[0] * self.params["ratio"][1]
        self.test_bound = self.label.shape[0]

    def merge_data(self, data_main, data_side):
        self.data = data_main[:-self.params["input_len"] - 1, :, :, 0]
        self.side = data_side[:-self.params["input_len"] - 1]
        self.label = data_main[self.params["input_len"] + 1:, :, :, 0]

    def get_batch(self, type_):
        data_indices = []
        if type_ == "train":
            data_indices = np.random.randint(self.params["input_len"], self.train_bound - 1,
                                             self.params["batch_size"])

        elif type_ == "val":
            data_indices = np.arange(self.train_bound, self.val_bound - 1)

        elif type_ == "test":
            data_indices = np.arange(self.val_bound, self.test_bound - 1)

        batch_x = []
        batch_w = []
        batch_y = []
        for idx in data_indices:
            batch_x.append(self.data[idx-self.params["input_len"]:idx])
            batch_w.append(self.side[idx])
            batch_y.append(self.label[idx])

        batch_x = torch.stack(batch_x, 0)
        batch_w = torch.stack(batch_w, 0)
        batch_y = torch.stack(batch_y, 0)[:, None, :, :]
        return batch_x, batch_w, batch_y
