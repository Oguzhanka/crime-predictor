"""
Configuration file.
"""
import datetime


INPUT_SIZE = [16, 16]
BOUNDARIES = [[0, INPUT_SIZE[0]], [0, INPUT_SIZE[1]]]

INPUT_WINDOW_LEN = 0
KSTEP = 1
MODEL_TYPE = "treeconv"


class DataParams:
    """
    Data processing params.
    """
    def __init__(self):
        self.start_date = datetime.datetime(year=2012, month=10, day=1)
        self.end_date = datetime.datetime(year=2017, month=11, day=30)
        self.weather_file = "weather"
        self.temporal_resolution = 12
        self.spatial_resolution = [BOUNDARIES[0][1], BOUNDARIES[1][1]]
        self.spatial_boundaries = [[41.6100, 42.042910333],
                                   [-87.939568, -87.501442]]

        self.crime_file = "cropped_crime"
        self.batch_size = -1
        self.input_len = INPUT_WINDOW_LEN
        self.ratio = [0.7, 0.2, 0.1]
        self.label_type = "binary"


class TreeParams:
    """
    Partitioning tree parameters.
    """
    def __init__(self):
        self.num_features = 2
        self.depth = 4
        self.hardness = 20
        self.region_mode = 2
        self.region_rot = 0.1

        self.boundaries = BOUNDARIES
        self.input_size = INPUT_SIZE
        self.output_dim = KSTEP


class CNNParams:
    """
    CNN multi-layer model parameters.
    """
    def __init__(self):
        self.kernel_size = 3
        self.hidden_dim = 16
        self.batch_size = 1
        self.input_size = INPUT_SIZE
        self.input_dim = 34
        self.sequence_length = 13
        self.lr = 1e-5


class TreeCNNParams:
    """
    TreeCNN model hyper-parameters.
    """
    def __init__(self):
        self.tree_params = TreeParams().__dict__
        self.cnn_params = CNNParams().__dict__

        self.input_size = INPUT_SIZE
        self.output_dim = KSTEP
        self.lr = self.cnn_params["lr"]
        self.sequence_length = self.cnn_params["sequence_length"]


class GraphLSTMParams:
    """
    GraphLSTM model hyper-parameters.
    """
    def __init__(self):
        self.input_size = INPUT_SIZE
        self.input_dim = 1
        self.hidden_dim = 16
        self.output_dim = KSTEP
        self.sequence_length = 13
        self.lr = 1e-5
