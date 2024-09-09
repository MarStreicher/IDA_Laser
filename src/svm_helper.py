import numpy as np
from colorama import Fore, Back, Style, init
import matplotlib.pyplot as plt
from termcolor import colored

init(autoreset=True)

class SvmHelper():
    def __init__(self):
        pass

    @staticmethod
    def calculate_euclidean_distance(theta, theta_new):
        """Calculate the Euclidean distance between two integers or two vectors."""

        if isinstance(theta,(int,float)) and isinstance(theta_new,(int,float)):
            return abs(theta-theta_new)

        elif isinstance(theta,(np.ndarray)) and isinstance(theta_new,(np.ndarray)):
            differences = theta-theta_new
            squared_sum = 0
            for diff in differences:
                squared_sum += diff ** 2
                euclidean_distance = squared_sum ** 0.5 #sqrt
                return euclidean_distance
        else:
            raise TypeError("Both inputs must be either numbers or numpy array.")

    @staticmethod
    def hinge_loss(labels,predictions):

        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)

        margins = 1-labels*predictions

        hinge_losses = np.maximum(0,margins)

        #loss_gradients = np.where(hinge_losses > 0, -np.multiply(inputs,labels), 0) # Lecture
        loss_gradients = np.where(hinge_losses > 0, -labels, 0) # Exercise

        return hinge_losses, loss_gradients
    
    @staticmethod
    def l2_regularizer(theta, lbda, linear=False):

        #regularizer = (lbda/(2*samples_number))*np.dot(theta.T,theta) # Lecture
        regularizer = (lbda/2)*np.dot(theta.T,theta) # Exercise.

        regularizer_gradient = lbda * theta

        if linear == True:
            regularizer_gradient[-1] = 0  

        return regularizer, regularizer_gradient
        
    @staticmethod
    def d_DTW(input, input_copy, distance_function):
        """
        Function used to calculate the d_DTW distance via dynamic programming.
        
        Args:
        input: A numpy array representing the first sequence.
        input_copy: A numpy array representing the second sequence.
        distance_function: A function that takes two arguments (elements of the sequences) and returns their distance.
        """

        if not isinstance(input,(np.ndarray)) and not isinstance(input_copy,(np.ndarray)):
            raise TypeError("Both inputs must be numpy arrays.")

        if input.size == 0 and input_copy.size == 0:
            return 0.0
        if input.size == 0 or input_copy.size == 0:
            return np.infty

        length_input = len(input)
        length_input_copy = len(input_copy)

        rows = length_input+1
        columns = length_input_copy+1

        table = np.zeros((rows,columns))
        table[0,0] = 0

        for i in range(1,(rows)):
            table[i,0] = np.infty
        for i in range(1,(columns)):
            table[0,i] = np.infty

        for i in range(1,rows):
            for j in range(1,columns):
                distance = distance_function(input[i-1], input_copy[j-1])
                minimum = min(table[i-1,j-1], table[i-1,j], table[i,j-1])
                table[i, j] = distance + minimum
        
        return table[length_input, length_input_copy]

    @staticmethod
    def create_kernel_matrix(kernel_function, inputs, inputs_copy):
        """
        Function used to build the kernel matrix out of the given kernel function.

        Args:
        kernel_function: A function that computes the kernel value between two sequences.
        inputs: A collection of sequences (numpy arrays).
        inputs_copy: Another collection of sequences (possibly the same as inputs).
    
        Returns:
        kernel_matrix: A Gram matrix of kernel values.
        """
        ###### exponetial and lambda is still missing here
        rows = len(inputs)
        columns = len(inputs_copy)

        kernel_matrix = np.empty((rows, columns))
        for i in range(rows):
            for j in range(i, columns):
                kernel_matrix[i,j] = kernel_function(inputs[i], inputs_copy[j])
                if i < columns and j < rows:
                    kernel_matrix[j, i] = kernel_matrix[i, j]

        return kernel_matrix

    @staticmethod
    def predict(inputs, theta):
        """Predict the class labels for a set of inputs using the learned parameter theta."""

        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)
        inputs = np.hstack((inputs, np.ones((inputs.shape[0], 1))))
        
        raw_predictions = np.dot(inputs,theta)
        if raw_predictions.ndim == 1:
            raw_predictions = raw_predictions.reshape(-1, 1)

        predictions = np.where(raw_predictions >= 0, 1, -1)

        return predictions, raw_predictions, inputs, theta
    
    @staticmethod
    def predict_kernel(theta, test_inputs, train_inputs, kernel_function):
        K = SvmHelper.create_kernel_matrix(kernel_function, test_inputs, train_inputs)

        if K is None:
            raise ValueError("Kernel matrix is None. Ensure create_kernel_matrix is working properly.")
    
        predictions = np.dot(K, theta)
        predictions[predictions >= 0] = 1
        predictions[predictions < 0] = -1
    
        return predictions

    @staticmethod
    def plot_metrics(hinge_loss_history, regularizer_history, distance_history):
        """Plot the metrics collected during training."""
        iterations = range(1, len(hinge_loss_history) + 1)

        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(iterations, hinge_loss_history, label="Hinge Loss", color="blue")
        plt.xlabel("Iteration")
        plt.ylabel("Hinge Loss")
        plt.title("Hinge Loss over Iterations")
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(iterations, regularizer_history, label="Regularizer", color="green")
        plt.xlabel("Iteration")
        plt.ylabel("Regularizer")
        plt.title("Regularizer over Iterations")
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(iterations, distance_history, label="Distance (theta old - theta new)", color="red")
        plt.xlabel("Iteration")
        plt.ylabel("Distance")
        plt.title("Euclidean Distance between Theta Updates")
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def regularised_kernel_erm_batch(inputs, labels, kernel_function, max_iterations=200, epsilon = 0.001, alpha = 1.0, lbda=1, verbose = False, figure = False):
        """Function for performing empirical risk minimization by using stochastic gradient descent method."""

        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)

        if kernel_function == 'linear':
            inputs = np.hstack([inputs, np.ones((inputs.shape[0], 1))])  

        samples_number = inputs.shape[0]
        feature_number = inputs.shape[1]-1
        column_number = inputs.shape[1]

        theta = np.random.randn(feature_number+1)

        if not kernel_function == 'linear':
            kernel_matrix = SvmHelper.create_kernel_matrix(kernel_function, inputs, inputs)
            theta = np.random.randn(kernel_matrix.shape[1])

        previous_gradients = None

        if verbose == True:
            print(colored('regularised_erm_batch starts ...','green'))
            print(f'Number of samples: {samples_number}')

            if kernel_function == 'linear':
                print(f'Number of features: {feature_number}')
            else:
                print(f'Number of columns: {column_number}')  

            print(f'Initialised theta: {theta}')
            print(colored('Start training ...', 'green'))
        
        if figure == True:
            hinge_loss_history = []
            regularizer_history = []
            distance_history = []

        for iteration in range(max_iterations):
            theta_old = theta

            if kernel_function == 'linear':
                predictions = np.dot(inputs,theta)
            else:
                predictions = np.dot(kernel_matrix.T,theta)

            hinge_losses, loss_gradients = SvmHelper.hinge_loss(labels,predictions)

            if kernel_function == 'linear':
                regularizer, regularizer_gradient = SvmHelper.l2_regularizer(theta, lbda, True)
            else:
                regularizer, regularizer_gradient = SvmHelper.l2_regularizer(theta, lbda)

            if verbose == True:
                print('Average Hinge loss: {}'.format(np.mean(hinge_losses)))
                print(f'Regularizer: {regularizer}')

            if kernel_function == 'linear':
                gradient = np.dot(inputs.T, loss_gradients).flatten() + regularizer_gradient
            else:
                loss_gradients = loss_gradients.flatten()
                gradient = loss_gradients + regularizer_gradient

            if previous_gradients is not None and not kernel_function == 'linear':
                alpha = 0.9**iteration
   
            theta = theta_old - alpha*gradient

            distance = SvmHelper.calculate_euclidean_distance(theta_old,theta)

            if figure == True:
                hinge_loss_history.append(np.mean(hinge_losses))
                regularizer_history.append(regularizer)
                distance_history.append(distance)

            if kernel_function == 'linear':
                if distance < epsilon:
                    break

            previous_gradients = gradient
        
        if theta.ndim == 1:
            theta = theta.reshape(-1, 1)
        
        if verbose == True:
            print(colored('Training end.','green'))
            print(f'Theta: {theta}')

        if figure == True:
            SvmHelper.plot_metrics(hinge_loss_history, regularizer_history, distance_history)

        return theta
    

        













