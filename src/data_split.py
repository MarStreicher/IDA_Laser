import torch
import numpy as np
from torch.utils.data import random_split, DataLoader
import pickle
import os
import seaborn as sns
import pandas as pd
from scipy.io import loadmat


import sys
from dataset import BaseDataset

file_path = "data/laser.mat"
mat_dict = loadmat(file_path)

dataset = BaseDataset(mat_dict, "X", "Y")

train_size = int(0.8*len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_indices = train_dataset.indices
test_indices = test_dataset.indices

with open('data_split_indices.pkl', 'wb') as f:
    pickle.dump({'train_indices': train_indices, 'test_indices': test_indices}, f)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# with open('dataloaders.pkl', 'wb') as f:
#     pickle.dump({'train_loader': train_loader, 'test_loader': test_loader}, f)

torch.manual_seed(42) 