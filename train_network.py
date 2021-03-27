import argparse
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

from lib.FramesDataset import FramesDataset
from lib import network

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using", DEVICE)

parser = argparse.ArgumentParser()
parser.add_argument('--beta')
parser.add_argument('--L1')
parser.add_argument('--path')
args = parser.parse_args()

hyperparameters = {
    "mode": "hierarchical",
    "framesize": 20,
    "tsteps": 45,
    "warmup": 4,
    "epochs": 2000,
    "units": 1600,
    "lr": 5*10**-4,
    "gradclip": 0.25,
    "L1": 10**(float(args.L1)),
    "beta": float(args.beta),
    "Dale": False,
    "path": args.path
}

paths = [
    './datasets/processed_dataset_20px_45tsteps_part1.npy',
    './datasets/processed_dataset_20px_45tsteps_part2.npy',
    './datasets/processed_dataset_20px_45tsteps_part3.npy'
]


train_dataset = FramesDataset(paths, 'train', hyperparameters["warmup"])
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

val_dataset = FramesDataset(paths, 'val', hyperparameters["warmup"])
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

print("Training dataset length:", len(train_dataset))
print("Validation dataset length:", len(val_dataset))

model = None
if hyperparameters["path"]:
    model = network.RecurrentTemporalPrediction.load(
        hidden_units = hyperparameters["units"],
        frame_size = hyperparameters["framesize"],
        warmup = hyperparameters["warmup"],
        mode = hyperparameters["mode"],
        Dale = hyperparameters["Dale"],
        path = hyperparameters["path"] + '.pt'
    )
    print("Loaded checkpoint from", hyperparameters["path"])
else:
    model = network.RecurrentTemporalPrediction(
        hidden_units = hyperparameters["units"],
        frame_size = hyperparameters["framesize"],
        warmup = hyperparameters["warmup"],
        mode = hyperparameters["mode"],
        Dale = hyperparameters["Dale"]
    )
    print("Creating new model")
model = model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=hyperparameters["lr"])

train_history = {
    'loss': [],
    'MSE_1': [],
    'MSE_2': [],
    'L1': []
}
val_history = {
    'loss': [],
    'MSE_1': [],
    'MSE_2': [],
    'L1': []
}

# If path is set, load previous train/val history
if hyperparameters["path"]:
    with open(hyperparameters["path"] + '.pickle', 'rb') as p:
        history_data = pickle.load(p)
        train_history = history_data['train']
        val_history = history_data['val_history']

        # Set epochs for how many left since last checkpoint
        hyperparameters["epochs"] -= len(train_history['loss'])
    print("Loaded loss history from", hyperparameters["path"])
    print(hyperparameters["epochs"], "epochs left")

for epoch in range(1, hyperparameters["epochs"]+1):
    # Train dataset
    running_train_history = {
        'i': 0,
        'loss': 0,
        'MSE_1': 0,
        'MSE_2': 0,
        'L1': 0
    }
    for batch_n, data in enumerate(train_data_loader):
        inputs, frame_targets = data
        inputs, frame_targets = inputs.to(DEVICE), frame_targets.to(DEVICE)

        model.train()
        optimizer.zero_grad()
        outputs, hidden_states = model(inputs)

        loss, MSE_1, MSE_2, L1 = model.loss_fn(
            outputs = outputs,
            frame_targets = frame_targets,
            hidden_states = hidden_states,
            L1_lambda = hyperparameters["L1"],
            beta = hyperparameters["beta"]
        )
        
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), hyperparameters["gradclip"])
        model.mask_gradients()
        optimizer.step()
        
        running_train_history["i"] += 1
        running_train_history["loss"] += loss.item()
        running_train_history["MSE_1"] += MSE_1.item()
        running_train_history["MSE_2"] += MSE_2.item()
        running_train_history["L1"] += L1.item()

    train_history['loss'].append(running_train_history["loss"] / running_train_history["i"])
    train_history['MSE_1'].append(running_train_history["MSE_1"] / running_train_history["i"])
    train_history['MSE_2'].append(running_train_history["MSE_2"] / running_train_history["i"])
    train_history['L1'].append(running_train_history["L1"] / running_train_history["i"])
    
    # Validation dataset
    running_val_history = {
        'i': 0,
        'loss': 0,
        'MSE_1': 0,
        'MSE_2': 0,
        'L1': 0
    }

    for batch_n, data in enumerate(val_data_loader):
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
                
            running_val_history["i"] += 1
            running_val_history["loss"] += loss.item()
            running_val_history["MSE_1"] += MSE_1.item()
            running_val_history["MSE_2"] += MSE_2.item()
            running_val_history["L1"] += L1.item()

    val_history['loss'].append(running_val_history["loss"] / running_val_history["i"])
    val_history['MSE_1'].append(running_val_history["MSE_1"] / running_val_history["i"])
    val_history['MSE_2'].append(running_val_history["MSE_2"] / running_val_history["i"])
    val_history['L1'].append(running_val_history["L1"] / running_val_history["i"])

    print('Epoch: {}/{}.............'.format(epoch, hyperparameters["epochs"]), end=' ')
    print("Loss: {:.4f}".format(train_history['loss'][-1]))

    #  Save check points every 100 epochs
    if epoch != hyperparameters["epochs"] and epoch % 100 == 0 :
        model.save(hyperparameters, { "train": train_history, "val_history": val_history })

# Finally, save model after all epochs completed / early stopping engaged
model.save(hyperparameters, { "train": train_history, "val_history": val_history })

