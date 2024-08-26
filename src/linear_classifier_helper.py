import numpy as np

class LinearClassifierHelper():
    def __init__(self):
        pass

    @staticmethod
    def calculate_euclidean_distance(theta, theta_new):
        """Calculate the Euclidean distance between two vectors."""
        differences = theta-theta_new
        squared_sum = 0

        for diff in differences:
            squared_sum += diff ** 2
        
        euclidean_distance = squared_sum ** 0.5 #sqrt
        return euclidean_distance
    
    @staticmethod
    def calculate_gradient(input, label, theta, lambda_value, samples):
        """
        Calculate the gradient of the hinge loss with L2 regularization.
        
        Parameters:
        input (numpy array): Feature vector (for a single instance).
        labels (int): Label for the instance (-1 or 1).
        theta (numpy array): Parameter vector.
        lambda_value (float): Regularization parameter.
        samples (int): Total number of samples in the dataset.
        
        Returns:
        numpy array: The calculated gradient.
        """
        gradient = 0.0

        margin = 1 - label * np.dot(input, theta)

        if margin > 0:
            gradient -= label*input
        
        gradient += (lambda_value/samples) * theta

        return gradient


    @staticmethod
    def calculate_stepsize(alpha_0, decay_rate, t):
        """
        Calculate the step size (learning rate) at iteration t.
        
        Parameters:
        alpha_0 (float): Initial learning rate.
        decay_rate (float): Rate at which the learning rate decreases.
        t (int): Current step/iteration number.
        
        Returns:
        float: Calculated step size.
        """
        return alpha_0 / (1+decay_rate * t)

    @staticmethod
    def reg_erm_stoch(inputs, labels, lambda_value = 0.1, epsilon = 1e-4, alpha_0 = 0.001, decay_rate = 0.001):
        """Function for performing empirical risk minimization by using stochastic gradient descent method."""

        samples = inputs.shape[0]

        theta = 0.0  
        theta_new = 0.0
        step = 0
        gradient = 0
        alpha = 0
        euclidean_distance = float('inf')


        while(euclidean_distance > epsilon):

            indices = np.arange(samples)
            np.random.shuffle(indices)
            inputs = inputs[indices]
            labels = labels[indices]

            for i in range(samples):
                theta = theta_new
                alpha = LinearClassifierHelper.calculate_stepsize(alpha_0, decay_rate, step)
                gradient = LinearClassifierHelper.calculate_gradient(inputs[i], labels[i], theta, lambda_value, samples)

                theta_new = theta - alpha*gradient

                euclidean_distance = LinearClassifierHelper.calculate_euclidean_distance(theta, theta_new)
                step += 1
        
        return theta_new
    
    @staticmethod
    def predict(inputs, theta):
        """Predict the class labels for a set of inputs using the learned parameter theta."""

        raw_predictions = inputs * theta
        predictions = np.sign(raw_predictions)
        predictions[predictions == 0] = 1

        return predictions
    
    @staticmethod
    def evaluate(predictions, true_labels):
        """Evaluate the accuracy of the model."""

        accuracy = np.mean(predictions == true_labels)
        return accuracy







