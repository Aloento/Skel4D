import torch
from diffusers.utils.import_utils import is_xformers_available


def train():
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert is_xformers_available(), "XFormers is required for training."
