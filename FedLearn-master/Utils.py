import numpy as np
import torch
import torch.nn as nn


def copy_model_params(model_from: nn.Module, model_to: nn.Module,layer=0):
    for w_from, w_to in zip(model_from.parameters(), model_to.parameters()):
        w_to.data = torch.clone(w_from.data)

def copy_model_first_k_layers_params(model_from: nn.Module, model_to: nn.Module,k=0):
    for w_from, w_to in zip(model_from.parameters(), model_to.parameters()):
        if k>0:
            w_to.data = torch.clone(w_from.data)
            k=k-1


class BasicDataset:
    """
    Basic class for a dataset
    Can get a batch of the data; When the whole dataset is iterated once, it will shuffle itself
    """
    def __init__(self, data: np.ndarray):
        self.data = data
        self.data_size = len(data)
        self.current_queue = [_ for _ in range(len(data))]
        np.random.shuffle(self.current_queue)

    def get_batch(self, batch_size: int):
        if len(self.current_queue) < batch_size:
            self.current_queue = [_ for _ in range(len(self.data))]
            np.random.shuffle(self.current_queue)

        batch_idx = self.current_queue[:batch_size]
        self.current_queue = self.current_queue[batch_size:]
        return self.data[batch_idx]

    def get_batch_xy(self, batch_size: int):
        raise NotImplementedError()

