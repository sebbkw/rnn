import numpy as np
import time
import pickle
from matplotlib import pyplot as plt

import torch
import torch.nn as nn

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_file_name (file_name_params = {}, file_type = 'pt'):
    time_str = time.strftime('%Y%m%d-%H%M%S')
    path_name = './models/model'

    for param in file_name_params:
        value = file_name_params[param]
        path_name += '-{}{}'.format(value, param)

    path_name += '-{}.{}'.format(time_str, file_type)

    return path_name

class RecurrentTemporalPrediction (nn.Module):
    def __init__ (self, hidden_units, frame_size, warmup, mode):
        super(RecurrentTemporalPrediction, self).__init__()

        self.inhibitory_ratio = 0.1

        self.hidden_units = hidden_units
        self.frame_size = frame_size
        self.warmup = warmup
        self.mode = mode

        self.hidden_units_group = hidden_units // 2
        self.output_units_group1 = frame_size**2 # Predicts frame
        self.output_units_group2 = self.hidden_units_group # Predicts group 1 hidden units

        if mode == 'control':
            self.output_units = self.output_units_group1
        elif mode == 'hierarchical':
            self.output_units = self.output_units_group1 + self.output_units_group2

        self.rnn = nn.RNN(
            input_size = frame_size**2,
            hidden_size = hidden_units,
            num_layers = 1,
            nonlinearity = 'relu',
            batch_first = True
        )
        self.fc = nn.Linear(
            in_features = hidden_units,
            out_features = self.output_units
        )

        # Initialise RNN weights with identity matrix
        self.state_dict()['rnn.weight_hh_l0'][:] = nn.Parameter(torch.eye(hidden_units, hidden_units)) / 100

        # Mask to zero out weights for group 2 RNN input connections
        self.rnn_mask = torch.ones(self.rnn.weight_ih_l0.shape).to(DEVICE)
        self.rnn_mask[self.hidden_units_group:, :] = 0

        # Mask to zero out weights from group 2 FC input connections
        self.fc_mask_control = torch.ones(self.fc.weight.shape).to(DEVICE)
        self.fc_mask_control[:, self.output_units_group1:] = 0

        # Mask to zero out RNN-FC weights for connections from group 1 to group 2, and group 2 to group 1 
        self.fc_mask_hierarchical = torch.ones(self.fc.weight.shape).to(DEVICE)
        self.fc_mask_hierarchical[self.output_units_group1:, :self.hidden_units_group] = 0
        self.fc_mask_hierarchical[:self.output_units_group1, self.hidden_units_group:] = 0

    def forward (self, inputs):
        # Mask weights
        self.rnn.weight_ih_l0.data.mul_(self.rnn_mask)
        if self.mode ==  'control':
            self.fc.weight.data.mul_(self.fc_mask_control)
        elif self.mode == 'hierarchical':
            self.fc.weight.data.mul_(self.fc_mask_hierarchical)

        self.clamp_hidden_weights()

        # Forward pass
        rnn_outputs, _ = self.rnn(inputs)
        fc_outputs = self.fc(rnn_outputs[:, self.warmup:, :]) # Discard warmup outputs
        fc_outputs = fc_outputs[:, :-1, :] # Discard last output as there is no frame target 
                                           # Rather, last tstep used to generate final hidden state

        return fc_outputs, rnn_outputs

    def mask_gradients (self):
        self.rnn.weight_ih_l0.grad.data.mul_(self.rnn_mask)
        if self.mode ==  'control':
            self.fc.weight.grad.data.mul_(self.fc_mask_control)
        elif self.mode == 'hierarchical':
            self.fc.weight.grad.data.mul_(self.fc_mask_hierarchical)

    def clamp_hidden_weights (self):
        group1_inhibitory_units = int(self.hidden_units_group*self.inhibitory_ratio)
        group2_inhibitory_units = int(self.hidden_units_group+self.hidden_units_group*self.inhibitory_ratio)
        
        with torch.no_grad():
            hh_weights = self.rnn.weight_hh_l0.clone()

            # First, clamp weights to a minimum of zero
            out = hh_weights.clamp(min = 0) #torch.abs(hh_weights)

            # Next, clamp 10% of local connections to a maximum of zero
            out[:self.hidden_units_group, :group1_inhibitory_units] = hh_weights[:self.hidden_units_group, :group1_inhibitory_units].clamp(max=0) # Group 1
            out[self.hidden_units_group:, self.hidden_units_group:group2_inhibitory_units] = hh_weights[self.hidden_units_group:, self.hidden_units_group:group2_inhibitory_units].clamp(max=0) # Group 2

            # Update parameters
            self.rnn.weight_hh_l0.copy_(out)
    def L1_regularisation (self, lam):
        weights = torch.Tensor([]).to(DEVICE)
        for name, params in self.named_parameters():
            if 'weight' in name:
                weights = torch.cat((weights, params.flatten()), 0)

        return lam*weights.abs().sum()

    def loss_fn (self, outputs, frame_targets, hidden_states, L1_lambda, beta):
        MSE = nn.MSELoss()

        output_units_group1 = outputs[:, :, :self.output_units_group1]
        output_units_group2 = outputs[:, :, self.output_units_group1:]
        # Shift forward by warmup (discarded) + 1 (offset for prediction)
        hidden_state_targets = hidden_states[:, self.warmup+1:, :self.hidden_units_group]

        L1 = self.L1_regularisation(L1_lambda)

        if self.mode == 'control':
            return MSE(output_units_group1, frame_targets) + L1
        elif self.mode == 'hierarchical':
            return (
                MSE(output_units_group1, frame_targets)*(1-beta) +
                MSE(output_units_group2, hidden_state_targets)*beta +
                L1
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
    def load (cls, hidden_units, frame_size, warmup, mode, path):
        model = cls(hidden_units, frame_size, warmup, mode)
        model.load_state_dict(torch.load(path, map_location=torch.device(DEVICE)), strict=False)
        model.eval()

        return model.to(DEVICE)