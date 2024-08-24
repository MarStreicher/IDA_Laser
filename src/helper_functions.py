import numpy as np

class Helper():
    def __init__(self):
        pass

    @staticmethod
    def calculate_entropy(labels):
        """Calculate the entropy of the label distribution."""

        total = len(labels)
        if total == 0:
            return 0.0
        
        positive = np.where(labels > 0)[0]
        negative = np.where(labels <= 0)[0]

        entropy = 0.0

        for group in [positive, negative]:
            group_size = len(group)
            if group_size > 0:
                p = group_size/total
                entropy += p*np.log2(p)
        
        return -entropy
    
    @staticmethod
    def calculate_conditional_entropy(labels, inputs, threshold):
        """Calculate the conditional entropy for one feature with continious values."""
        
        results = {
        'inputs_above_threshold': Helper.calculate_entropy(labels[inputs > threshold]),
        'inputs_below_threshold': Helper.calculate_entropy(labels[inputs <= threshold])
        }
            
        return results

    @staticmethod
    def calculate_continous_information_gain(labels, inputs, threshold):
        """Calculate the information gain for one featue with continiuos values."""

        total = len(labels)
        if total == 0:
            return 0.0

        labels_above_threshold = labels[inputs > threshold]
        labels_below_threshold = labels[inputs <= threshold]

        weighted_conditional_entropy = 0.0
        for group in [labels_above_threshold, labels_below_threshold]:
            conditional_entropy = 0.0
            group_size = len(group)
            if group_size > 0:
                p = group_size/total
                conditional_entropy = Helper.calculate_entropy(group)
                weighted_conditional_entropy -= p*conditional_entropy

        entropy = Helper.calculate_entropy(labels)

        information_gain = 0.0
        information_gain = entropy + weighted_conditional_entropy

        return information_gain
    
    @staticmethod
    def get_information_gain_values(labels, inputs, thresholds):
        """Calculate the information gain values for different threshold values."""

        results = dict()
        results = {
            'threshoulds': thresholds,
            'information_gain_values': None,
            'best_threshould': None,
            'best_information_gain': None
        }

        information_gains = np.zeros(len(thresholds))

        best_information_gain = 0.0
        best_threshold = 0.0

        for i,value in enumerate(thresholds):
            information_gain = Helper.calculate_continous_information_gain(labels, inputs, threshold=value)
            information_gains[i] = information_gain

            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_threshold = value

        results['information_gain_values'] = information_gains
        results['best_threshold'] = best_threshold
        results['best_information_gain'] = best_information_gain

        return results







        


