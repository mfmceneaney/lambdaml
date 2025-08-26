# ----------------------------------------------------------------------------------------------------#
# TRAIN
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pytest

# Local imports
from lambdaml.train import *

@pytest.fixture(name="total_epochs")
def total_epochs_fixture():
    return 100

@pytest.fixture(name="epoch")
def epoch_fixture():
    return 50

@pytest.fixture(name="alpha_fn_coefficient")
def alpha_fn_coefficient_fixture():
    return 0.05

def test_alpha_fn(epoch, total_epochs, coefficient=alpha_fn_coefficient):
    assert (alpha_fn(epoch, total_epochs, coefficient) == pytest.approx(
        coefficient * (2.0 / (1.0 + np.exp(-10 * epoch / total_epochs)) - 1)
    ))


