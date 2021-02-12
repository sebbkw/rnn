import numpy as np
import torch

class FramesDataset (torch.utils.data.Dataset):
    def __init__ (self, path, split_type, warmup):
        dataset = np.load(path, mmap_mode='r+')

        n = len(dataset)
        splits = {
            "all": slice(None, None),
            "train": slice(0, int(n*0.8)),
            "val": slice(int(n*0.8), int(n*0.9)),
            "test": slice(int(n*0.9), None)
        }

        self.dataset = dataset[splits[split_type]]  
        self.warmup = warmup
        
    def __len__ (self):
        return len(self.dataset)
    
    def __getitem__ (self, i):
        window = self.dataset[i]
        window = torch.from_numpy(window)
        window = window.type(torch.FloatTensor)
        
        x = window[:-1, :] # All but last frame
        y = window[self.warmup+1:, :] # Only frames after warmup, shifted by t=1

        return x, y
