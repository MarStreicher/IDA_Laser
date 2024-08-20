import torch
from torch.utils.data import Dataset, DataLoader

class MatDataset(Dataset):
    def __init__(self, mat_data, input_key, label_key = None):
        self.inputs = mat_data[input_key]
        self.labels = mat_data[label_key] if label_key else None
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self,index):
        x = self.inputs[index]
        
        if self.labels is not None:
            y = self.labels[index]
        else:
            y = None

        return x,y if y is not None else x