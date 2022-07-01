import numpy as np


def load_pricing_imagearr_data(config):
    arr = np.load('data/processed/pricing_imagearr.npy')
    return arr.reshape(arr.shape[0], -1)


def load_pricing_vgg_data(config):
    arr = np.load('data/processed/pricing_vggarr.npy')
    return arr.reshape(arr.shape[0], -1)

def load_pricing_with_features():
    pass