"""
@title

@description

"""
import argparse
import time

import numpy as np
import torch
from tqdm import trange

from island_influence.learn.neural_network import NeuralNetwork


def display_weights(nn_weights):
    if isinstance(nn_weights, NeuralNetwork):
        nn_weights = nn_weights.weight_vectors()

    for each_layer in nn_weights:
        print(each_layer)
    return


def main(main_args):
    n_inputs = 4
    n_outputs = 2
    n_hidden = 0

    model_0 = NeuralNetwork(n_inputs=n_inputs, n_outputs=n_outputs, n_hidden=n_hidden)
    model_1 = NeuralNetwork(n_inputs=n_inputs, n_outputs=n_outputs, n_hidden=n_hidden)

    print(f'Using device: {model_0.device()}\n{model_0}')
    print(f'{model_0.fitness=}')
    model_0.fitness = 5
    print(f'{model_0.fitness=}')

    save_name = model_0.save_model()
    print(f'Saved PyTorch model state to {save_name}')

    np_vect = np.random.rand(n_inputs)
    pt_vect = torch.from_numpy(np_vect)
    bad_np_vect = np.random.rand(n_inputs)
    bad_py_vect = torch.from_numpy(bad_np_vect)

    output = model_0(pt_vect)
    print(f'{output=}')

    bad_output = model_0(bad_py_vect)
    print(f'{bad_output=}')

    test_vects = [
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, 0],
    ]
    for vect in test_vects:
        np_vect = np.asarray(vect)
        pt_vect = torch.from_numpy(np_vect)
        output = model_0(pt_vect)
        # print(f'{vect} | {output=}')

    # print('=' * 80)
    # print('Original weights')
    # print('=' * 80)
    # print(f'Model 0')
    # display_weights(model_0.weight_vectors())
    # print('=' * 80)
    # print(f'Model 1')
    # display_weights(model_1.weight_vectors())
    # print('=' * 80)
    # print('=' * 80)
    # model_0.replace_layers(model_1, [0])
    # print('=' * 80)
    # print('Replace model_0 layer 0 with model_1 layer 0')
    # print('=' * 80)
    # print(f'Model 0')
    # display_weights(model_0.weight_vectors())
    # print('=' * 80)
    # print(f'Model 1')
    # display_weights(model_1.weight_vectors())
    # print('=' * 80)
    # print('=' * 80)
    # model_0.mutate_gaussian(probability_to_mutate=1.0, mutation_scalar=1)
    # print('=' * 80)
    # print('Mutate weights in model_0')
    # print('=' * 80)
    # print(f'Model 0')
    # display_weights(model_0.weight_vectors())
    # print('=' * 80)
    # print(f'Model 1')
    # display_weights(model_1.weight_vectors())
    # print('=' * 80)
    # print('=' * 80)
    ##############################################
    timing_iters = 10_000
    times = []
    for _ in trange(timing_iters):
        start_time = time.process_time_ns()
        new_model = model_0.copy()
        end_time = time.process_time_ns()
        times.append(end_time - start_time)
    print(f'Copy test: {np.average(times)}')

    times = []
    for _ in trange(timing_iters):
        start_time = time.process_time_ns()
        new_model = model_0._deepcopy()
        end_time = time.process_time_ns()
        times.append(end_time - start_time)
    print(f'Deepcopy test: {np.average(times)}')

    times = []
    for _ in trange(timing_iters):
        start_time = time.process_time_ns()
        new_model = model_0._assign_weights()
        end_time = time.process_time_ns()
        times.append(end_time - start_time)
    print(f'Assign weights test: {np.average(times)}')

    print('=' * 80)
    print('Original weights')
    print(f'Model 0')
    print(f'{id(model_0)}: {model_0.network_id}')
    display_weights(model_0)
    print('=' * 80)
    print(f'Copy weights')
    new_model = model_0.copy()
    print(f'{id(new_model) == id(model_0)}: {new_model.network_id == model_0.network_id}: {id(new_model)}: {new_model.network_id}')
    print(f'{new_model.fitness=} | {model_0.fitness=} | {new_model.learner} | {model_0.learner}')
    display_weights(new_model)
    print('=' * 80)
    print(f'Copy1 weights')
    new_model = model_0.copy1()
    print(f'{id(new_model) == id(model_0)}: {new_model.network_id == model_0.network_id}: {id(new_model)}: {new_model.network_id}')
    print(f'{new_model.fitness=} | {model_0.fitness=} | {new_model.learner} | {model_0.learner}')
    display_weights(new_model)
    print('=' * 80)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
