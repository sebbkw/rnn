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

train_dataset = FramesDataset('./datasets/processed_dataset_small.pkl', 'train', WARMUP)
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)

print("Training dataset length:", len(train_dataset))

def main (n_epochs, lr, hidden_units):
    model = network.RNN(hidden_units = hidden_units, frame_size = FRAME_SIZE, t_steps = T_STEPS)
    model = model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Hyperparameters
    n_epochs = n_epochs
    lr = lr
    clip_value = 1.0
    L1 = 10e-6

    loss_history = []

    for epoch in range(1, n_epochs + 1):
        running_loss = 0
        loss_i = 0

        for batch_n, data in enumerate(train_data_loader):
            inputs, targets = data
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            output = model(inputs)
            loss = network.L1_regularisation(L1, criterion(output, targets), model)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value)  #Gradient Value Clipping
            optimizer.step()

            running_loss += loss.item()
            loss_i += 1

        loss_history.append(running_loss / loss_i)
        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss_history[-1]))

    file_name = 'model-%iepochs-%ihidden_units-' % (n_epochs, hidden_units)
    model.save(file_name)

    plt.plot(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('./models/' + file_name + '.png')

main(n_epochs=3000, lr=1e-3, hidden_units=500)
