import numpy as np
from collections import Counter

def entropy(y):
    if isinstance(y, np.ndarray):
        y = y.flatten()  
        y = y.tolist() 
    
    label_counts = Counter(y)
    entropy_value = 0.0
    for count in label_counts.values():
        p = float(count) / len(y)
        if p != 0:
            entropy_value -= p * np.log2(p)
    
    return entropy_value
