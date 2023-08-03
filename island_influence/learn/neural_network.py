"""
@title

@description

"""
import copy
import uuid
from pathlib import Path

import numpy as np
import torch
from numpy.random import default_rng
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from island_influence import project_properties


def linear_stack(n_inputs, n_hidden, n_outputs):
    hidden_size = int((n_inputs + n_outputs) / 2)
    network = nn.Sequential(
        nn.Linear(n_inputs, n_outputs)
    )
    for idx in range(n_hidden):
        network.append(nn.Linear(hidden_size, hidden_size))
    network.append(nn.Linear(hidden_size, n_outputs))
    return network


def linear_layer(n_inputs, n_hidden, n_outputs):
    network = nn.Sequential(
        nn.Linear(n_inputs, n_outputs)
    )
    return network


def linear_relu_stack(n_inputs, n_hidden, n_outputs):
    hidden_size = int((n_inputs + n_outputs) / 2)
    network = nn.Sequential(
        nn.Linear(n_inputs, hidden_size),
        nn.ReLU()
    )

    for idx in range(n_hidden):
        network.append(nn.Linear(hidden_size, hidden_size))
        network.append(nn.ReLU())

    network.append(nn.Linear(hidden_size, n_outputs))
    return network


def load_pytorch_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model


class NeuralNetwork(nn.Module):

    @property
    def name(self):
        return f'{self.network_func.__name__}_NN_{str(self.network_id)[-4:]}'

    def __init__(self, n_inputs, n_outputs, n_hidden=2, network_func=linear_layer):
        super(NeuralNetwork, self).__init__()
        self.network_id = uuid.uuid1().int

        self.network_func = network_func

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.flatten = nn.Flatten()
        self.network = self.network_func(n_inputs=n_inputs, n_hidden=n_hidden, n_outputs=n_outputs)

        self.parent = None
        return

    def __repr__(self):
        base_repr = f'{self.name}'
        if hasattr(self, 'fitness'):
            base_repr = f'{base_repr}, {self.fitness=}'
        return base_repr

    def copy(self):
        # https://discuss.pytorch.org/t/deep-copying-pytorch-modules/13514/2
        new_copy = copy.deepcopy(self)
        new_copy.network_id = uuid.uuid1().int
        return new_copy

    def mutate_gaussian(self, mutation_scalar=0.1, probability_to_mutate=0.05):
        rng = default_rng()
        with torch.no_grad():
            param_vector = parameters_to_vector(self.parameters())

            for each_val in param_vector:
                rand_val = rng.random()
                if rand_val <= probability_to_mutate:
                    # todo  base proportion on current weight rather than scaled random sample
                    noise = torch.randn(each_val.size()) * mutation_scalar
                    each_val.add_(noise)

            vector_to_parameters(param_vector, self.parameters())
        return

    def device(self):
        dev = next(self.parameters()).device
        return dev

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if x.dtype is not torch.float32:
            x = x.float()

        if x.shape[0] != self.n_inputs:
            # if input does not have the correct shape
            # x = torch.zeros([1, self.n_inputs], dtype=torch.float32)
            raise ValueError(f'Input does not have correct shape: {x.shape=} | {self.n_inputs=}')

        logits = self.network(x)
        return logits

    def save_model(self, save_dir=None, tag=''):
        # todo optimize saving pytorch model
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training
        if save_dir is None:
            save_dir = project_properties.model_dir

        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        if tag != '':
            tag = f'_{tag}'

        save_name = Path(save_dir, f'{self.name}_model{tag}.pt')
        torch.save(self, save_name)
        return save_name
