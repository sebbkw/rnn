import numpy as np
import torch
import torch.utils.data

class FramesDataset (torch.utils.data.Dataset):
    def __init__ (self, paths, split_type, warmup):
        datasets = []

        for path in paths:
            dataset = np.load(path, mmap_mode='r')
            n = len(dataset)
            splits = {
                "all": slice(None, None),
                "train": slice(0, int(n*0.8)),
                "val": slice(int(n*0.8), int(n*0.9)),
                "test": slice(int(n*0.9), None)
            }

            datasets.append(dataset[splits[split_type]])

        self.datasets = datasets
        self.warmup = warmup
        self.count = 0
        
    def __len__ (self):
        total_len = 0

        for dataset in self.datasets:
            total_len += len(dataset)

        return total_len
    
    def __getitem__ (self, i):
        dataset_len = len(self.datasets[0])
        dataset_i = i // dataset_len
        window_i = i - dataset_len*dataset_i

        window = self.datasets[dataset_i][window_i]
        window = torch.from_numpy(window)
        window = window.type(torch.FloatTensor)
        
        x = window[:, :] # All
        y = window[self.warmup+1:, :] # Only frames after warmup, shifted by t=1

        return x, y
