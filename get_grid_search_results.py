import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from lib.FramesDataset import FramesDataset
from lib import network

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using", DEVICE)

data_paths = [
    './datasets/processed_dataset_20px_45tsteps_part1.npy',
    './datasets/processed_dataset_20px_45tsteps_part2.npy',
    './datasets/processed_dataset_20px_45tsteps_part3.npy'
]

test_dataset = FramesDataset(data_paths, 'test', 4)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

print("Test dataset length:", len(test_dataset))

file_paths = [
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-2000epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-06L1-0.0beta-FalseDale-Nonepath-20210331-011945",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-06L1-0.1beta-FalseDale-20210401-225322",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-06L1-0.2beta-FalseDale-20210401-223344",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-06L1-0.3beta-FalseDale-20210401-230731",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-06L1-0.4beta-FalseDale-20210401-223252",
    "model-0.0beta-hierarchicalmode-1600units-45tsteps-0.0005lr-FalseDale-20framesize-2000epochs-0.25gradclip-1e-06L1-Nonepath-4warmup-20210330-095151",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-1e-06L1-0.1beta-FalseDale-20210402-023944",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-1e-06L1-0.2beta-FalseDale-20210401-234622",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-1e-06L1-0.3beta-FalseDale-20210402-002425",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-1e-06L1-0.4beta-FalseDale-20210402-005735",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-2000epochs-1600units-0.0005lr-0.25gradclip-5.62341325190349e-07L1-0.0beta-FalseDale-Nonepath-20210330-070034",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-5.62341325190349e-07L1-0.1beta-FalseDale-20210402-145109",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-5.62341325190349e-07L1-0.2beta-FalseDale-20210402-003513",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-5.62341325190349e-07L1-0.4beta-FalseDale-20210402-130824",
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1700epochs-1600units-0.0005lr-0.25gradclip-3.162277660168379e-07L1-0.0beta-FalseDale-20210401-072002",
    "model-1600units-FalseDale-0.4beta-4warmup-0.0005lr-3.162277660168379e-07L1-hierarchicalmode-20framesize-2000epochs-0.25gradclip-45tsteps-20210401-212441"
]
beta_values = [0, 0.1, 0.2, 0.3, 0.4, 0, 0.1, 0.2, 0.3, 0.4, 0, 0.1, 0.2, 0.4, 0, 0.4]
L1_values = [-5.5, -5.5, -5.5, -5.5, -5.5, -6, -6, -6, -6, -6, -6.25, -6.25, -6.25, -6.25, -6.5, -6.5]

file_paths = [
    "model-hierarchicalmode-20framesize-45tsteps-4warmup-1000epochs-1600units-0.0005lr-0.25gradclip-5.62341325190349e-07L1-0.3beta-FalseDale-20210402-105619"
]
beta_values = [0.3]
L1_values = [-6.25]

for path_i, path in enumerate(file_paths):
    hyperparameters = {
        "mode": "hierarchical",
        "framesize": 20,
        "tsteps": 45,
        "warmup": 4,
        "epochs": 2000,
        "units": 1600,
        "Dale": False,
        "path": "./models/grid-search-models/" + path,
        "beta": beta_values[path_i],
        "L1": 10**(L1_values[path_i])
    }

    model = network.RecurrentTemporalPrediction.load(
        hidden_units = hyperparameters["units"],
        frame_size = hyperparameters["framesize"],
        warmup = hyperparameters["warmup"],
        mode = hyperparameters["mode"],
        Dale = hyperparameters["Dale"],
        path = hyperparameters["path"] + '.pt'
    )
    print("Loaded checkpoint from", hyperparameters["path"])

    model = model.to(DEVICE)

    train_history = {
        'loss': [],
        'MSE_1': [],
        'MSE_2': [],
        'L1': []
    }

    with open(hyperparameters["path"] + '.pickle', 'rb') as p:
        history_data = pickle.load(p)
        train_history = history_data['train']

    print("Loaded loss history from", hyperparameters["path"])
    print(len(train_history["loss"]), "epochs trained")


    test_history = {
        'i': 0,
        'loss': 0,
        'MSE_1': 0,
        'MSE_2': 0,
        'L1': 0
    }

    for batch_n, data in enumerate(test_data_loader):
        model.eval()
        with torch.no_grad():
            inputs, frame_targets = data
            inputs, frame_targets = inputs.to(DEVICE), frame_targets.to(DEVICE)

            outputs, hidden_states = model(inputs)

            loss, MSE_1, MSE_2, L1 = model.loss_fn(
                outputs = outputs,
                frame_targets = frame_targets,
                hidden_states = hidden_states,
                L1_lambda = hyperparameters["L1"],
                beta = hyperparameters["beta"]
            )
                
            test_history["i"] += 1
            test_history["loss"] += loss.item()
            test_history["MSE_1"] += MSE_1.item()
            test_history["MSE_2"] += MSE_2.item()
            test_history["L1"] += L1.item()

    test_history['loss'] = test_history["loss"] / test_history["i"]
    test_history['MSE_1'] = test_history["MSE_1"] / test_history["i"]
    test_history['MSE_2'] = test_history["MSE_2"] / test_history["i"]
    test_history['L1'] = test_history["L1"] / test_history["i"]

    with open('./grid-search-results/' + path + '.pickle', 'wb') as p:
        pickle.dump(test_history, p, protocol=4)

    print('\n\n')
    print(test_history['loss'])
    print('\n\n')

