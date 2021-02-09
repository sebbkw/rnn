import numpy as np
import pickle
import torch

class FramesDataset (torch.utils.data.Dataset):
    def __init__ (self, path, split_type, warmup):
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
        self.warmup = warmup
        
    def __len__ (self):
        return len(self.dataset)
    
    def __getitem__ (self, i):
        window = self.dataset[i]
        x = window[:-1, :] # All but last frame
        y = window[self.warmup+1:, :] # Only frames after warmup, shifted by t=1

        return x, y