import numpy as np
import time
from matplotlib import pyplot as plt

import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        hidden_state = torch.zeros((inputs.shape[0], self.hidden_units)).to(DEVICE)
        
        # Warm up period
        warmup_length = inputs.shape[1]
        for i in range(warmup_length):
            frame_batch = inputs[:, i, :]
            hidden_state = self.rnn(frame_batch, hidden_state)
        prediction = self.fc(hidden_state)
        predictions.append(prediction)

        # Autoregressive predictions
        for t in range(self.t_steps):
            hidden_state = self.rnn(prediction, hidden_state)
            prediction = self.fc(hidden_state)
            predictions.append(prediction)
            
        return torch.transpose(torch.stack(predictions), 0, 1)
        
    def save (self, file_name = None):
        path_name = './models/' + (file_name or 'model-') + time.strftime('%Y%m%d-%H%M%S') + '.pt'
        torch.save(self.state_dict(), path_name)
        print('Saved model as ' + path_name)
    
    @classmethod
    def load (cls, hidden_units, frame_size, t_steps, path):
        model = cls(hidden_units, frame_size, t_steps)
        model.load_state_dict(torch.load(path, map_location=torch.device(DEVICE)))
        model.eval()

        return model.to(DEVICE)
    
def L1_regularisation (lam, loss, model):
    weights = torch.Tensor([]).to(DEVICE)
    for name, params in model.named_parameters():
        if name.endswith('weight'):
            weights = torch.cat((weights, params.flatten()), 0)
    
    return loss + lam*weights.abs().sum()