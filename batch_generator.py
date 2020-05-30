import torch
import numpy as np


class BatchGenerator:
    def __init__(self, params, crime_data, weather_data):
        """
        Batch generator module implementation. Takes the crime data and weather data
        during initialization. Train-test ratios are computed according to their sizes.

        :param params: Configurable parameters dictionary.
        :param crime_data: Transformed crime data.
        :param weather_data: Transformed weather data.
        """
        self.params = params
        self.data = None
        self.side = None
        self.label = None

        self.label_type = self.params["label_type"]     # Type of the labeling.
        self.merge_data(crime_data, weather_data)       # Merges the crime and weather data.

        self.train_bound = int(self.label.shape[0] * self.params["ratio"][0])   # Train data size.
        self.val_bound = int(self.train_bound + self.label.shape[0] * self.params["ratio"][1])
        self.test_bound = int(self.label.shape[0])

    def merge_data(self, data_main, data_side):
        """
        Prepares the labels, data and side information for the specified configurations.
        :param data_main: Main data ued for prediction.
        :param data_side: Side information used for prediction.
        :return: None
        """
        self.data = data_main[:-self.params["input_len"] - 1, :, :, 0]
        self.side = data_side[:-self.params["input_len"] - 1]
        self.label = data_main[self.params["input_len"] + 1:, :, :, 0]

        self.data = (self.data - self.data.mean()) / self.data.std()    # Data standardization.
        if self.label_type == "binary":                                 # Label binarization.
            self.label[self.label > 1] = 1
            print("DATA STATS: ")
            print("POSITIVE RATIO: {}".format(self.label.mean()))
            print("")
        else:
            self.label = (self.label - self.label.mean()) / self.label.std()    # Label standardization.

    def get_batch(self, type_):
        """
        Returns a batch from the specified set, i.e. train, test or validation sets.
        :param type_: Set to be returned for prediction and testing.
        :return: Batch of data, side information and label.
        """
        data_indices = []
        if type_ == "train":    # Training data.

            if self.params["batch_size"] == -1:     # If no batch size specified...
                batch_x = self.data[:self.train_bound, None, :, :]              # Return all data.
                batch_w = self.side[:self.train_bound].permute(0, 3, 1, 2)
                batch_y = self.label[:self.train_bound, None, :, :]

                return batch_x, batch_w, batch_y

            data_indices = np.random.randint(self.params["input_len"], self.train_bound - 1,    # Else randomly sample.
                                             self.params["batch_size"])

        elif type_ == "validation":         # Validation data.
            data_indices = np.arange(self.train_bound, self.val_bound - 1)  # Return all set.

        elif type_ == "test":               # Test data.
            data_indices = np.arange(self.val_bound, self.test_bound - 1)   # Return all set.

        batch_x = []
        batch_w = []
        batch_y = []
        for idx in data_indices:            # For each sample in the selected set...
            batch_x.append(self.data[idx-self.params["input_len"]:idx+1])   # Append it to the batch data.
            batch_w.append(self.side[idx])
            batch_y.append(self.label[idx])

        batch_x = torch.stack(batch_x, 0)   # Convert to torch tensor.
        batch_w = torch.stack(batch_w, 0).permute(0, 3, 1, 2)
        batch_y = torch.stack(batch_y, 0)[:, None, :, :]
        return batch_x, batch_w, batch_y
