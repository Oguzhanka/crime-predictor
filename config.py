"""
Configuration file.
"""
import datetime


INPUT_SIZE = [16, 16]                                   # Spatial cell numbers for two axes.
BOUNDARIES = [[0, INPUT_SIZE[0]], [0, INPUT_SIZE[1]]]   # Boundaries of the decision tree (do not change).

INPUT_WINDOW_LEN = 0                                    # Input window length K-frames.
KSTEP = 1                                               # Output window length K-frames.
MODEL_TYPE = "treeconv"                                 # Type of the model.

EPOCHS = 200                                            # Number of epochs.


class DataParams:
    """
    Data processing params.
    """
    def __init__(self):
        # Start and end dates for temporal cropping.
        self.start_date = datetime.datetime(year=2012, month=10, day=1)
        self.end_date = datetime.datetime(year=2017, month=11, day=30)

        # Name of the weather source file.
        self.weather_file = "weather"

        # Temporal preprocessing resolution (hours).
        self.temporal_resolution = 12

        # Spatial preprocessing resolution (km2).
        self.spatial_resolution = [BOUNDARIES[0][1], BOUNDARIES[1][1]]

        # Spatial boundaries of the region.
        self.spatial_boundaries = [[41.6100, 42.042910333],
                                   [-87.939568, -87.501442]]
        # Name of the crime source file.
        self.crime_file = "cropped_crime"

        # Batch size. Selected as -1 if the whole set is fed.
        self.batch_size = -1
        self.input_len = INPUT_WINDOW_LEN

        # Train, validation and test ratios for the sequence.
        self.ratio = [0.7, 0.2, 0.1]

        # Type of the label range. Can be count or binary.
        self.label_type = "binary"


class TreeParams:
    """
    Partitioning tree parameters.
    """
    def __init__(self):
        self.num_features = 2

        # Depth of the decision tree.
        self.depth = 4

        # Initial hardness/softness of the boundaries of the decision nodes.
        self.hardness = 20

        # Initialization mode for decision boundaries. 0 for random rotation initializations,
        # 1 for random 0 or 90 degree rotation initializations and 2 for switching 0 and 90
        # degree initializations.
        self.region_mode = 2
        self.region_rot = 0.1

        # General parameters copied.
        self.boundaries = BOUNDARIES
        self.input_size = INPUT_SIZE
        self.output_dim = KSTEP


class CNNParams:
    """
    CNN multi-layer model parameters.
    """
    def __init__(self):
        # Kernel size for the convolutions.
        self.kernel_size = 3

        # Hidden dimension for the LSTM state.
        self.hidden_dim = 16

        # Batch size is always 1 as there is only one sequence.
        self.batch_size = 1

        # Input spatial size for frames.
        self.input_size = INPUT_SIZE

        # Dimension of the input. Sum of the crime data and weather dimensions.
        self.input_dim = 34

        # Subsequence length to perform updates.
        self.sequence_length = 13

        # Learning rate.
        self.lr = 1e-5


class TreeCNNParams:
    """
    TreeCNN model hyper-parameters.
    """
    def __init__(self):
        # Take the arguments for both the tree and the convolutional LSTM.
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
        # Input dimension. Only the crime data.
        self.input_dim = 1

        # Hidden dimension of the LSTM state vectors.
        self.hidden_dim = 16

        # Output window length.
        self.output_dim = KSTEP

        # Length of the subsequences.
        self.sequence_length = 13

        # Learning rate.
        self.lr = 1e-5
