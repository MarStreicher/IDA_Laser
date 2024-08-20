from torch.utils.data import DataLoader
from scipy.io import loadmat

# Own imports
from mat_dataset import MatDataset


file_path = "data/laser.mat"
mat_dict = loadmat(file_path)

dataset = MatDataset(mat_dict, input_key = "X", label_key = "Y")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for inputs, labels in dataloader:
    print(inputs.shape, labels.shape)


dataset = MatDataset(mat_dict, input_key = "X", label_key = "Y", feature_engineering='r2')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for inputs, labels in dataloader:
    print(inputs.shape, labels.shape)