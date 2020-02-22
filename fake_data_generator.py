"""
Fake data generator for mocking the actual data
input which will be passed to the models.
"""
import torch


def generate_batch(input_len, output_len, batch_size, input_size):
    """
    Generates a batch of fake data.

    :param input_len: Temporal length of the input windows.
    :param output_len: Temporal length of the output windows.
    :param batch_size: Size of a batch.
    :param input_size: Spatial size of the input tensors.
    :return: Batch.
    """
    x = torch.zeros((batch_size, input_len, *input_size))
    y = 5*torch.ones((batch_size, output_len, *input_size))
    return x, y
