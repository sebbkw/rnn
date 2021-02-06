import numpy as np
import pickle
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

WARMUP = 20
T_STEPS = 5
FRAME_SIZE = 20

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using", DEVICE)

class FramesDataset (torch.utils.data.Dataset):
    def __init__ (self, path, split_type):
        with open(path, 'rb') as file:
            dataset = pickle.load(file)

        n = len(dataset)
        splits = {
            "train": slice(0, int(n*0.8)),
            "val": slice(int(n*0.8), int(n*0.9)),
            "test": slice(int(n*0.9), None)
        }

        dataset = dataset[splits[split_type]]
        dataset = torch.from_numpy(np.array(dataset))
        dataset = dataset.type(torch.FloatTensor)

        self.dataset = dataset

    def __len__ (self):
        return len(self.dataset)

    def __getitem__ (self, i):
        window = self.dataset[i]
        x = window[:WARMUP, :]
        y = window[WARMUP:, :]

        return x, y

train_dataset = FramesDataset('./datasets/processed_dataset.pkl', 'train')
train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128)

print("Training dataset length:", len(train_dataset))

class RNN_Network (nn.Module):
    def __init__ (self, hidden_units):
        super(RNN_Network, self).__init__()

        self.hidden_units = hidden_units

        self.rnn = nn.RNNCell(
            input_size = FRAME_SIZE**2,
            hidden_size = hidden_units,
            nonlinearity = 'relu'
        )
        self.fc = nn.Linear(hidden_units, FRAME_SIZE**2)

        # Initialise RNN weights with identity matrix
        self.rnn.weight_hh = torch.nn.Parameter(torch.eye(hidden_units, hidden_units))

    def forward (self, inputs):
        predictions = []
        hidden_state = torch.zeros((inputs.shape[0], self.hidden_units)).to(DEVICE)

        # Warm up period
        for i in range(WARMUP):
            frame_batch = inputs[:, i, :]
            hidden_state = self.rnn(frame_batch, hidden_state)
        prediction = self.fc(hidden_state)
        predictions.append(prediction)

        # Autoregressive predictions
        for t in range(T_STEPS):
            hidden_state = self.rnn(prediction, hidden_state)
            prediction = self.fc(hidden_state)
            predictions.append(prediction)

        return torch.transpose(torch.stack(predictions), 0, 1)

    def save (self, file_name = None):
        path_name = './models/' + (file_name or 'model-') + time.strftime('%Y%m%d-%H%M%S') + '.pt'
        torch.save(self.state_dict(), path_name)
        print('Saved model as ' + path_name)

    @staticmethod
    def load (hidden_units, path):
        model = RNN_Network(hidden_units)
        model.load_state_dict(torch.load(path))
        model.eval()

        return model.to(DEVICE)

def L1_regularisation (lam, loss, model):
    weights = torch.Tensor([]).to(DEVICE)
    for name, params in model.named_parameters():
        if name.endswith('weight'):
            weights = torch.cat((weights, params.flatten()), 0)

    return loss + lam*weights.abs().sum()

def main (n_epochs, lr):
    model = RNN_Network(hidden_units = 400)
    model = model.to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, n_epochs + 1):
        for batch_n, data in enumerate(train_data_loader):
            inputs, targets = data
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            output = model(inputs)
            loss = L1_regularisation(10e-6, criterion(output, targets), model)
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            optimizer.step()

        print('Epoch: {}/{}.............'.format(epoch, n_epochs), end=' ')
        print("Loss: {:.4f}".format(loss.item()))

    model.save('model-%iepochs-' % (n_epochs))

main(n_epochs=2000, lr=3e-4)
