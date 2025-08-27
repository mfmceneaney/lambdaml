# ----------------------------------------------------------------------------------------------------#
# TRAIN
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import pytest

# Local imports
from lambdaml.train import *

@pytest.fixture(name="nepochs")
def nepochs_fixture():
    return 100

@pytest.fixture(name="epoch")
def epoch_fixture():
    return 50

@pytest.fixture(name="coeff")
def coeff_fixture():
    return 0.05

def test_sigmoid_growth(epoch, nepochs, coeff):
    assert (sigmoid_growth(epoch, nepochs, coeff) == pytest.approx(
        coeff * (2.0 / (1.0 + np.exp(-10 * epoch / nepochs)) - 1)
    ))


