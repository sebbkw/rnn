import torch
import torch.nn as nn
import torch.optim as optim

from lib.FramesDataset import FramesDataset
from lib import network

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using", DEVICE)

hyperparameters = {
    "mode": "hierarchical",
    "framesize": 15,
    "tsteps": 20,
    "warmup": 4,
    "epochs": 2000,
    "units": 1600,
    "lr": 10**-3,
    "gradclip": 0.25,
    "L1": 10**-6
}

train_dataset = FramesDataset('./datasets/processed_dataset_15px_20tsteps_101500.npy', 'all', hyperparameters["warmup"])
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
print("Training dataset length:", len(train_dataset))

model = network.RecurrentTemporalPrediction(
    hidden_units = hyperparameters["units"],
    frame_size = hyperparameters["framesize"],
    warmup = hyperparameters["warmup"]
)
model = model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=hyperparameters["lr"])

loss_history = []

for epoch in range(1, hyperparameters["epochs"] + 1):
    running_loss = 0
    loss_i = 0

    for batch_n, data in enumerate(train_data_loader):
        inputs, frame_targets = data
        inputs, frame_targets = inputs.to(DEVICE), frame_targets.to(DEVICE)

        optimizer.zero_grad()
        outputs, hidden_states = model(inputs)

        loss = model.loss_fn(
            outputs = outputs,
            frame_targets = frame_targets,
            hidden_states = hidden_states,
            L1_lambda = hyperparameters["L1"],
            beta = 0.1
        )
        
        loss.backward()
        model.mask_gradients()
        nn.utils.clip_grad_value_(model.parameters(), hyperparameters["gradclip"])
        optimizer.step()
        
        running_loss += loss.item()
        loss_i += 1
        loss_history.append(running_loss / loss_i)

    print('Epoch: {}/{}.............'.format(epoch, hyperparameters["epochs"]), end=' ')
    print("Loss: {:.4f}".format(loss_history[-1]))

model.save(hyperparameters, loss_history)
