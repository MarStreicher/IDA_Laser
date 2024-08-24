import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

class BaseDataset(Dataset):
    def __init__(self, mat_data, input_key, label_key = None):
        """
        mat_data: Loaded data in dictionary-like format.
        input_key: Key to retrieve input data.
        label_key: Key to retrieve label data (default: None).
        """
        self.inputs = mat_data[input_key]
        self.labels = mat_data[label_key] if label_key else None
    
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

class FilteredDataset(BaseDataset):
    def __init__(self, mat_data, input_key, label_key, filter_label = None):
        super().__init__(mat_data, input_key, label_key)
        """A dataset class that filters data based on a specific label value."""

        if filter_label is not None:
            indices = np.where(self.labels == filter_label)[0]
            self.inputs = self.inputs[indices]
            self.labels = self.labels[indices]
        else:
            raise ValueError("Please provide a label_key and a filter_label.")

class FeatureEngineeredDataset(BaseDataset):
    def __init__(self, mat_data, input_key, label_key = None, feature_engineering = None):
        super().__init__(mat_data, input_key, label_key)
        """A dataset class that applies feature engineering to the input data."""

        if feature_engineering is not None:
            if feature_engineering == 'r2':
                self.inputs = self.calculate_r2(self.inputs)

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

    