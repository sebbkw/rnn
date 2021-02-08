import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

from lib.FramesDataset import FramesDataset
from lib import network

WARMUP = 20
T_STEPS = 5
FRAME_SIZE = 20

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using", DEVICE)

train_dataset = FramesDataset('./processed_dataset.pkl', 'train', WARMUP)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)

print("Training dataset length:", len(train_dataset))

def main ():
    # Hyperparameters
    hyperparameters = {
        "epochs": 1,
        "units": 50,
        "lr": 3e-4,
        "gradclip": 1.0,
        "L1": 10e-6
    }

    model = network.RNN(hidden_units = hyperparameters["units"], frame_size = FRAME_SIZE, t_steps = T_STEPS)
    model = model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters["lr"])
    
    loss_history = []

    for epoch in range(1, hyperparameters["epochs"] + 1):
        running_loss = 0
        loss_i = 0

        for batch_n, data in enumerate(train_data_loader):
            inputs, targets = data
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            output = model(inputs)
            loss = network.L1_regularisation(
                lam = hyperparameters["L1"],
                loss = criterion(output, targets),
                model = model
            )
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), hyperparameters["gradclip"])
            optimizer.step()

            running_loss += loss.item()
            loss_i += 1

        loss_history.append(running_loss / loss_i)
        print('Epoch: {}/{}.............'.format(epoch, hyperparameters["epochs"]), end=' ')
        print("Loss: {:.4f}".format(loss_history[-1]))

    model.save(hyperparameters, loss_history)

main()
