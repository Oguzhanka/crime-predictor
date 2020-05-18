import torch
import numpy as np


class BatchGenerator:
    def __init__(self, params, crime_data, weather_data):
        self.params = params
        self.data = None
        self.side = None
        self.label = None

        self.label_type = self.params["label_type"]
        self.merge_data(crime_data, weather_data)

        self.train_bound = int(self.label.shape[0] * self.params["ratio"][0])
        self.val_bound = int(self.train_bound + self.label.shape[0] * self.params["ratio"][1])
        self.test_bound = int(self.label.shape[0])

    def merge_data(self, data_main, data_side):
        self.data = data_main[:-self.params["input_len"] - 1, :, :, 0]
        self.side = data_side[:-self.params["input_len"] - 1]
        self.label = data_main[self.params["input_len"] + 1:, :, :, 0]

        self.data = (self.data - self.data.mean()) / self.data.std()
        if self.label_type == "binary":
            self.label[self.label > 1] = 1
            print("MEAN: {}".format(self.label.mean()))
        else:
            self.label = (self.label - self.label.mean()) / self.label.std()

    def get_batch(self, type_):
        data_indices = []
        if type_ == "train":

            if self.params["batch_size"] == -1:
                batch_x = self.data[:self.train_bound, None, :, :]
                batch_w = self.side[:self.train_bound].permute(0, 3, 1, 2)
                batch_y = self.label[:self.train_bound, None, :, :]

                return batch_x, batch_w, batch_y

            data_indices = np.random.randint(self.params["input_len"], self.train_bound - 1,
                                             self.params["batch_size"])

        elif type_ == "validation":
            data_indices = np.arange(self.train_bound, self.val_bound - 1)

        elif type_ == "test":
            data_indices = np.arange(self.val_bound, self.test_bound - 1)

        batch_x = []
        batch_w = []
        batch_y = []
        for idx in data_indices:
            batch_x.append(self.data[idx-self.params["input_len"]:idx+1])
            batch_w.append(self.side[idx])
            batch_y.append(self.label[idx])

        batch_x = torch.stack(batch_x, 0)
        batch_w = torch.stack(batch_w, 0).permute(0, 3, 1, 2)
        batch_y = torch.stack(batch_y, 0)[:, None, :, :]
        return batch_x, batch_w, batch_y
