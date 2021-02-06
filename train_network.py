import torch
import torch.nn as nn
import torch.optim as optim

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

def main (n_epochs, lr):
    model = network.RNN(hidden_units = 400, frame_size = FRAME_SIZE, t_steps = T_STEPS)
    model = model.to(DEVICE)

    n_epochs = 2
    lr=3e-4

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        for batch_n, data in enumerate(train_data_loader):
            inputs, targets = data
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            output = model(inputs)
            loss = network.L1_regularisation(10e-6, criterion(output, targets), model)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)  #Gradient Value Clipping
            optimizer.step()

        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))

    model.save('model-%iepochs-' % (n_epochs))
    
main(n_epochs=2000, lr=3e-4)