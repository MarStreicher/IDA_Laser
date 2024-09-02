import torch
import numpy as np
from torch.utils.data import Dataset, random_split, Subset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
import os

class BaseDataset(Dataset):
    def __init__(self, mat_data, input_key, label_key = None):
        """
        mat_data: Loaded data in dictionary-like format.
        input_key: Key to retrieve input data.
        label_key: Key to retrieve label data (default: None).
        """
        self.inputs = mat_data[input_key]
        self.labels = mat_data[label_key] if label_key else None

        train_size = int(0.8*len(self.inputs))
        test_size = len(self.inputs)-train_size

        file_path = '/Users/marleenstreicher/Documents/git/IDA_Laser/IDA_Laser/data_split_indices.pkl'
        print(f"Checking path: {file_path}")
        print(f"Path exists: {os.path.exists(file_path)}")       
        if os.path.exists('/Users/marleenstreicher/Documents/git/IDA_Laser/IDA_Laser/data_split_indices.pkl'):
            with open(file_path, 'rb') as file:
                split_data = pickle.load(file)

                self.train_indices = split_data['train_indices']
                self.test_indices = split_data['test_indices']
        else:
            train_dataset, test_dataset = random_split(self.inputs, [train_size, test_size])

            self.train_indices = train_dataset.indices
            self.test_indices = test_dataset.indices

            with open('data_split_indices.pkl', 'wb') as f:
                pickle.dump({'train_indices': self.train_indices, 'test_indices': self.test_indices}, f)

        self.train_inputs = self.inputs[self.train_indices]
        self.test_inputs = self.inputs[self.test_indices]

        if self.labels is not None:
            self.train_labels = self.labels[self.train_indices]
            self.test_labels = self.labels[self.test_indices]
        else:
            self.train_labels = None
            self.test_labels = None

    def __len__(self):
        """Calculate the length of the input data."""
        return len(self.inputs)
    
    def __getitem__(self,index):
        """Get the item of a specified index."""
        x = self.inputs[index]
        
        if self.labels is not None:
            y = self.labels[index]
        else:
            y = None

        return x,y if y is not None else x
    
    def filter_by_label(self,filter_label = None):
        """A function that filters data based on a specific label value."""
        if filter_label is not None:
            indices = np.where(self.labels == filter_label)[0]
            filtered_inputs = self.inputs[indices]
            filtered_labels = self.labels[indices] if self.labels is not None else None
            return filtered_inputs, filtered_labels
        else:
            raise ValueError("Please provide a filter_label.")
        
class FeatureEngineeredDataset(BaseDataset):
    def __init__(self, mat_data, input_key, label_key = None, feature_engineering = None):
        super().__init__(mat_data, input_key, label_key)
        """A dataset class that applies feature engineering to the input data."""

        if feature_engineering is not None:
            if feature_engineering == 'r2':
                self.inputs = self.calculate_r2(self.inputs)

                self.train_inputs = self.inputs[self.train_indices]
                self.test_inputs = self.inputs[self.test_indices]

    def calculate_r2(self, input):
        """Calculate R^2 scores for given lasers and a time frame (sec)."""
        time_frame = np.arange(0,60)
        r2_scores = []
        model = LinearRegression()
        for instance in input:
            instance_reshaped = instance.reshape(-1, 1)  
            model.fit(instance_reshaped, time_frame)
            y_pred = model.predict(instance_reshaped)
            r2 = r2_score(time_frame, y_pred)
            r2_scores.append(r2)
        return np.array(r2_scores)


    