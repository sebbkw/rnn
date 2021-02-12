import numpy as np
import time
import pickle
from matplotlib import pyplot as plt

import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def L1_regularisation (lam, loss, model):
    weights = torch.Tensor([]).to(DEVICE)
    for name, params in model.named_parameters():
        if 'weight' in name:
            weights = torch.cat((weights, params.flatten()), 0)

    return loss + lam*weights.abs().sum()

def get_file_name (file_name_params = {}, file_type = 'pt'):
    time_str = time.strftime('%Y%m%d-%H%M%S')
    path_name = './models/model'

    for param in file_name_params:
        value = file_name_params[param]
        path_name += '-{}{}'.format(value, param)

    path_name += '-{}.{}'.format(time_str, file_type)

    return path_name

class RecurrentTemporalPrediction (nn.Module):
    def __init__ (self, hidden_units, frame_size, warmup):
        super(RecurrentTemporalPrediction, self).__init__()

        self.warmup = warmup
        self.rnn = nn.RNN(
            input_size = frame_size**2,
            hidden_size = hidden_units,
            num_layers = 1,
            nonlinearity = 'relu',
            batch_first = True
        )
        self.fc = nn.Linear(hidden_units, frame_size**2)

        # Initialise RNN weights with identity matrix
        self.state_dict()['rnn.weight_hh_l0'][:] = torch.nn.Parameter(torch.eye(hidden_units, hidden_units)) / 100

    def forward (self, inputs):
        out, hidden = self.rnn(inputs)
        out = self.fc(out[:, self.warmup:, :])

        return out, hidden

    def save (self, file_name_params = {}, loss_history = None):
        model_file_name = get_file_name(file_name_params, 'pt')
        loss_file_name = get_file_name(file_name_params, 'pickle')

        torch.save(self.state_dict(), model_file_name)

        if loss_history:
            with open(loss_file_name, 'wb') as p:
                pickle.dump(loss_history, p, protocol=4)

        print('Saved model as ' + model_file_name)

    @classmethod
    def load (cls, hidden_units, frame_size, warmup, path):
        model = cls(hidden_units, frame_size, warmup)
        model.load_state_dict(torch.load(path, map_location=torch.device(DEVICE)))
        model.eval()

        return model.to(DEVICE)
