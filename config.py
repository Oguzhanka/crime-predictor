"""
Configuration file.
"""
import datetime


BOUNDARIES = [[0, 30], [0, 30]]
INPUT_SIZE = [30, 30]

INPUT_WINDOW_LEN = 5
KSTEP = 1


class DataParams:
    """
    Data processing params.
    """
    def __init__(self):
        self.start_date = datetime.datetime(year=2012, month=10, day=1)
        self.end_date = datetime.datetime(year=2017, month=11, day=30)
        self.weather_file = "weather"
        self.temporal_resolution = 24
        self.spatial_resolution = [BOUNDARIES[0][1], BOUNDARIES[1][1]]
        self.spatial_boundaries = [[41.6100, 42.042910333],
                                   [-87.939568, -87.501442]]

        self.crime_file = "cropped_crime"
        self.batch_size = 64
        self.input_len = INPUT_WINDOW_LEN
        self.ratio = [0.7, 0.2, 0.1]


class TreeParams:
    """
    Partitioning tree parameters.
    """
    def __init__(self):
        self.num_features = 2
        self.depth = 2
        self.hardness = 5
        self.region_mode = 1
        self.region_rot = 0.1

        self.boundaries = BOUNDARIES
        self.input_size = INPUT_SIZE
        self.output_dim = KSTEP


class CNNParams:
    """
    CNN multi-layer model parameters.
    """
    def __init__(self):
        self.layers = [{"in_channels": INPUT_WINDOW_LEN, "out_channels": 64, "kernel_size": 5,
                        "stride": 1, "padding": 2, "dilation": 1, "groups": 1, "bias": True},
                       {"in_channels": 64, "out_channels": 128, "kernel_size": 3,
                        "stride": 1, "padding": 1, "dilation": 1, "groups": 1, "bias": True},
                       {"in_channels": 128, "out_channels": 64, "kernel_size": 3,
                        "stride": 1, "padding": 1, "dilation": 1, "groups": 1, "bias": True},
                       {"in_channels": 64, "out_channels": 32, "kernel_size": 3,
                        "stride": 1, "padding": 1, "dilation": 1, "groups": 1, "bias": True},
                       {"in_channels": 32, "out_channels": KSTEP, "kernel_size": 3,
                        "stride": 1, "padding": 1, "dilation": 1, "groups": 1, "bias": True}]


class TreeCNNParams:
    """
    TreeCNN model hyper-parameters.
    """
    def __init__(self):
        self.tree_params = TreeParams().__dict__
        self.cnn_params = CNNParams().__dict__

        self.input_size = INPUT_SIZE
        self.output_dim = KSTEP
        self.batch_size = 128
        self.lr = 1e-4
