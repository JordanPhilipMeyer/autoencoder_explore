import torch
from CP_AutoEncoder import combined_criterion


def test_combined_criterion():
    expected = [[0.5, 0.1, 1, 0], [0.5, 0.1, 1, 0], [0.5, 0.1, 1, 0]]
    expected = torch.FloatTensor(expected)
    actual = [[0.5, 0.1, 1, 0], [0.5, 0.1, 1, 0], [0.5, 0.1, 1, 0]]
    actual = torch.FloatTensor(actual)
    assert combined_criterion(actual, expected, [0, 1], [2, 3]) == 0
