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
        if name.endswith('weight'):
            weights = torch.cat((weights, params.flatten()), 0)
    
    return loss + lam*weights.abs().sum()

def get_file_name (file_name_params = {}, file_type = 'pt'):
    time_str = time.strftime('%Y%m%d-%H%M%S')
    path_name = './models/model'

    for param in file_name_params:
        value = file_name_params[param]
        path_name += f'-{value}{param}'

    path_name += f'-{time_str}.{file_type}'
    
    return path_name

class RNN (nn.Module):
    def __init__ (self, hidden_units, frame_size, t_steps):
        super(RNN, self).__init__()
        
        self.hidden_units = hidden_units
        self.frame_size = frame_size
        self.t_steps = t_steps
        
        self.rnn = nn.RNNCell(
            input_size = frame_size**2,
            hidden_size = hidden_units,
            nonlinearity = 'relu'
        )     
        self.fc = nn.Linear(hidden_units, frame_size**2)
        
        # Initialise RNN weights with identity matrix
        self.rnn.weight_hh = torch.nn.Parameter(torch.eye(hidden_units, hidden_units))
        
    def forward (self, inputs):
        predictions = []
        hidden_states = []
        
        hidden_state = torch.zeros((inputs.shape[0], self.hidden_units)).to(DEVICE)
        
        warmup_length = inputs.shape[1] - self.t_steps
        warmup_inputs = inputs[:, :warmup_length, :]
        prediction_inputs = inputs[:, warmup_length:, :]
        
        # Warm up period
        for i in range(warmup_length):
            frame_batch = inputs[:, i, :]
            hidden_state = self.rnn(frame_batch, hidden_state)
        prediction = self.fc(hidden_state)

        # Predictions
        for t in range(self.t_steps):
            hidden_state = self.rnn(prediction_inputs[:, t, :], hidden_state)
            prediction = self.fc(hidden_state)
            
            hidden_states.append(hidden_state)
            predictions.append(prediction)
            
        return (
            torch.transpose(torch.stack(predictions), 0, 1),
            torch.transpose(torch.stack(hidden_states), 0, 1)
        )
        
    def save (self, file_name_params = {}, loss_history = None):
        model_file_name = get_file_name(file_name_params, 'pt')
        loss_file_name = get_file_name(file_name_params, 'pickle')

        torch.save(self.state_dict(), model_file_name)

        if loss_history:
            with open(loss_file_name, 'wb') as p:
                pickle.dump(loss_history, p, protocol=4)

        print('Saved model as ' + model_file_name)
    
    @classmethod
    def load (cls, hidden_units, frame_size, t_steps, path):
        model = cls(hidden_units, frame_size, t_steps)
        model.load_state_dict(torch.load(path, map_location=torch.device(DEVICE)))
        model.eval()

        return model.to(DEVICE)
