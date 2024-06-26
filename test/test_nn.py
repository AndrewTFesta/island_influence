"""
@title

@description

"""
import argparse
import time

import numpy as np
import torch
from torch import cuda
from tqdm import trange

from island_influence.learn.neural_network import NeuralNetwork


def display_weights(nn_weights):
    if isinstance(nn_weights, NeuralNetwork):
        nn_weights = nn_weights.weight_vectors()

    for each_layer in nn_weights:
        print(each_layer)
    return


def test_device(model_0: NeuralNetwork):
    print(f'{cuda.is_available()=}')
    print(f'Using device: {model_0.device()}')
    return


def test_persistence(model_0: NeuralNetwork):
    save_name = model_0.save_model()
    print(f'Saved PyTorch model state to {save_name}')
    return


def test_mutate(model_0: NeuralNetwork, model_1: NeuralNetwork):
    print('=' * 80)
    print('Original weights')
    print('=' * 80)
    print(f'Model 0')
    display_weights(model_0)
    print('=' * 80)
    print(f'Model 1')
    display_weights(model_1)
    print('=' * 80)
    print('=' * 80)
    # model_0.replace_layers(model_1, [0])
    # print('=' * 80)
    # print('Replace model_0 layer 0 with model_1 layer 0')
    # print('=' * 80)
    # print(f'Model 0')
    # display_weights(model_0)
    # print('=' * 80)
    # print(f'Model 1')
    # display_weights(model_1)
    # print('=' * 80)
    # print('=' * 80)
    # model_0.mutate_gaussian(probability_to_mutate=1.0, mutation_scalar=1)
    # print('=' * 80)
    # print('Mutate weights in model_0')
    # print('=' * 80)
    # print(f'Model 0')
    # display_weights(model_0)
    # print('=' * 80)
    # print(f'Model 1')
    # display_weights(model_1)
    # print('=' * 80)
    # print('=' * 80)
    print('Mutate_gaussian in model_0')
    probability_to_mutate = 0.5
    vals_altered, num_weights = model_0.mutate_gaussian(probability_to_mutate=probability_to_mutate, mutation_scalar=0.5)
    print(f'Number of values altered: {probability_to_mutate=} | {vals_altered=}  | {num_weights=} | {vals_altered / num_weights}')

    num_tests = 10_000
    probability_to_mutate = 0.25
    total_altered = 0
    total_vals = 0
    for _ in range(num_tests):
        vals_altered, num_weights = model_0.mutate_gaussian(probability_to_mutate=probability_to_mutate, mutation_scalar=0.5)
        total_altered += vals_altered
        total_vals += num_weights
    print(f'Number of values altered: {probability_to_mutate=} | {total_altered=} | {total_vals=}  | {total_altered / total_vals}')
    print('=' * 80)
    probability_to_mutate = 0.5
    total_altered = 0
    total_vals = 0
    for _ in range(num_tests):
        vals_altered, num_weights = model_0.mutate_gaussian(probability_to_mutate=probability_to_mutate, mutation_scalar=0.5)
        total_altered += vals_altered
        total_vals += num_weights
    print(f'Number of values altered: {probability_to_mutate=} | {total_altered=} | {total_vals=}  | {total_altered / total_vals}')
    print('=' * 80)
    probability_to_mutate = 0.75
    total_altered = 0
    total_vals = 0
    for _ in range(num_tests):
        vals_altered, num_weights = model_0.mutate_gaussian(probability_to_mutate=probability_to_mutate, mutation_scalar=0.5)
        total_altered += vals_altered
        total_vals += num_weights
    print(f'Number of values altered: {probability_to_mutate=} | {total_altered=} | {total_vals=}  | {total_altered / total_vals}')
    print('=' * 80)
    print('=' * 80)
    print('Mutate_gaussian_individual in model_0')
    probability_to_mutate = 0.5
    vals_altered, num_weights = model_0.mutate_gaussian_individual(probability_to_mutate=probability_to_mutate, mutation_scalar=0.5)
    print(f'Number of values altered: {probability_to_mutate=} | {vals_altered=}  | {num_weights=} | {vals_altered / num_weights}')

    num_tests = 10_000
    probability_to_mutate = 0.25
    total_altered = 0
    total_vals = 0
    for _ in range(num_tests):
        vals_altered, num_weights = model_0.mutate_gaussian_individual(probability_to_mutate=probability_to_mutate, mutation_scalar=0.5)
        total_altered += vals_altered
        total_vals += num_weights
    print(f'Number of values altered: {probability_to_mutate=} | {total_altered=} | {total_vals=}  | {total_altered / total_vals}')
    print('=' * 80)
    probability_to_mutate = 0.5
    total_altered = 0
    total_vals = 0
    for _ in range(num_tests):
        vals_altered, num_weights = model_0.mutate_gaussian_individual(probability_to_mutate=probability_to_mutate, mutation_scalar=0.5)
        total_altered += vals_altered
        total_vals += num_weights
    print(f'Number of values altered: {probability_to_mutate=} | {total_altered=} | {total_vals=}  | {total_altered / total_vals}')
    print('=' * 80)
    probability_to_mutate = 0.75
    total_altered = 0
    total_vals = 0
    for _ in range(num_tests):
        vals_altered, num_weights = model_0.mutate_gaussian_individual(probability_to_mutate=probability_to_mutate, mutation_scalar=0.5)
        total_altered += vals_altered
        total_vals += num_weights
    print(f'Number of values altered: {probability_to_mutate=} | {total_altered=} | {total_vals=}  | {total_altered / total_vals}')
    print('=' * 80)
    return


def test_copy(model_0: NeuralNetwork):
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
    new_model = model_0._deepcopy()
    print(f'{id(new_model) == id(model_0)}: {new_model.network_id == model_0.network_id}: {id(new_model)}: {new_model.network_id}')
    print(f'{new_model.fitness=} | {model_0.fitness=} | {new_model.learner} | {model_0.learner}')
    display_weights(new_model)
    print('=' * 80)
    return


def test_timings(model_0: NeuralNetwork, n_inputs):
    timing_iters = 10_000
    # times = []
    # for _ in trange(timing_iters):
    #     start_time = time.process_time_ns()
    #     new_model = model_0.copy()
    #     end_time = time.process_time_ns()
    #     times.append(end_time - start_time)
    # print(f'Copy test: {np.average(times)}')

    # times = []
    # for _ in trange(timing_iters):
    #     start_time = time.process_time_ns()
    #     new_model = model_0._deepcopy()
    #     end_time = time.process_time_ns()
    #     times.append(end_time - start_time)
    # print(f'Deepcopy test: {np.average(times)}')

    # np_vect = np.random.rand(n_inputs)
    # print(f'Using cpu')
    # times = []
    # for _ in trange(timing_iters):
    #     start_time = time.process_time_ns()
    #     output = model_0(np_vect)
    #     end_time = time.process_time_ns()
    #     times.append(end_time - start_time)
    # print(f'Forward test (Numpy): average={np.average(times)} | total={np.sum(times)}')
    #
    # times = []
    # for _ in trange(timing_iters):
    #     start_time = time.process_time_ns()
    #     pt_vect = torch.from_numpy(np_vect)
    #     output = model_0(pt_vect)
    #     end_time = time.process_time_ns()
    #     times.append(end_time - start_time)
    # print(f'Forward test (Tensor): average={np.average(times)} | total={np.sum(times)}')
    #
    # times = []
    # for _ in trange(timing_iters):
    #     start_time = time.process_time_ns()
    #     output = model_0.forward1(np_vect)
    #     end_time = time.process_time_ns()
    #     times.append(end_time - start_time)
    # print(f'Forward test: average={np.average(times)} | total={np.sum(times)}')

    times = []
    probability_to_mutate = 0.75
    for _ in trange(timing_iters):
        start_time = time.process_time_ns()
        vals_altered, num_weights = model_0.mutate_gaussian(probability_to_mutate=probability_to_mutate, mutation_scalar=0.5)
        end_time = time.process_time_ns()
        times.append(end_time - start_time)
    print(f'Mutate_gaussian test: average={np.average(times) / 1e+9} | total={np.sum(times) / 1e+9}')

    times = []
    probability_to_mutate = 0.75
    for _ in trange(timing_iters):
        start_time = time.process_time_ns()
        vals_altered, num_weights = model_0.mutate_gaussian_individual(probability_to_mutate=probability_to_mutate, mutation_scalar=0.5)
        end_time = time.process_time_ns()
        times.append(end_time - start_time)
    print(f'Mutate_gaussian_individual test: average={np.average(times) / 1e+9} | total={np.sum(times) / 1e+9}')

    # if cuda.is_available():
    #     model_0.to('cuda:0')
    #     print(f'Using gpu')
    #     times = []
    #     for _ in trange(timing_iters):
    #         start_time = time.process_time_ns()
    #         output = model_0(np_vect)
    #         end_time = time.process_time_ns()
    #         times.append(end_time - start_time)
    #     print(f'Forward test (Numpy): average={np.average(times)} | total={np.sum(times)}')
    #
    #     times = []
    #     for _ in trange(timing_iters):
    #         start_time = time.process_time_ns()
    #         pt_vect = torch.from_numpy(np_vect).cuda()
    #         output = model_0(pt_vect)
    #         end_time = time.process_time_ns()
    #         times.append(end_time - start_time)
    #     print(f'Forward test (Tensor): average={np.average(times)} | total={np.sum(times)}')
    #
    #     input_mat = np.tile(np_vect, (timing_iters, 1))
    #     start_time = time.process_time_ns()
    #     output = model_0(input_mat)
    #     end_time = time.process_time_ns()
    #     tot_time = end_time - start_time
    #     print(f'Forward test (vector input): average={tot_time / timing_iters} | total={tot_time}')
    #     print(output)
    return


def test_forward(model_0: NeuralNetwork, n_inputs):
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
        print(f'{vect} | {output=}')
    return


def test_fitness(model_0: NeuralNetwork):
    print(f'{model_0.fitness=}')
    model_0.fitness = 5
    print(f'{model_0.fitness=}')
    return


def main(main_args):
    n_inputs = 4
    n_outputs = 2
    n_hidden = 0

    model_0 = NeuralNetwork(n_inputs=n_inputs, n_outputs=n_outputs, n_hidden=n_hidden)
    model_1 = NeuralNetwork(n_inputs=n_inputs, n_outputs=n_outputs, n_hidden=n_hidden)

    # test_fitness(model_0)
    # test_copy(model_0)
    test_device(model_0)
    test_mutate(model_0, model_1)
    # test_persistence(model_0)
    # test_forward(model_0, n_inputs)
    test_timings(model_0, n_inputs)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
