import numpy as np
from colorama import Fore, Back, Style, init
import matplotlib.pyplot as plt
init(autoreset=True)

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
    def hinge_loss(inputs,labels,theta):

        predictions = np.dot(inputs,theta)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        if labels.ndim == 1:
            labels = labels.reshape(-1, 1)

        margins = 1-labels*predictions

        hinge_losses = np.maximum(0,margins)

        #loss_gradients = np.where(hinge_losses > 0, -np.multiply(inputs,labels), 0) --> Have to found solution and Kontext!!!?
        loss_gradients = np.where(hinge_losses > 0, -labels, 0) # -> Used in the exercise.

        return hinge_losses, loss_gradients
    
    @staticmethod
    def l2_regularizer(samples_number, theta, lbda):

        regularizer = (lbda/(2*samples_number))*np.dot(theta.T,theta)

        #regularizer = (lbda/2)*np.dot(theta.T,theta) -> Used in theexercise.

        regularizer_gradient = lbda * theta
        regularizer_gradient[-1] = 0  

        return regularizer, regularizer_gradient
        
    @staticmethod
    def regularised_erm_batch(inputs, labels, max_iterations=200, epsilon = 0.001, alpha = 1.0, lbda=1, verbose = False, view = False):
        """Function for performing empirical risk minimization by using stochastic gradient descent method."""
        if verbose == True:
            print(Fore.GREEN+'regularised_erm_batch starts ...')

        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)

        inputs = np.hstack([inputs, np.ones((inputs.shape[0], 1))])  

        samples_number = inputs.shape[0]
        feature_number = inputs.shape[1]
        theta = np.random.randn(feature_number)

        previous_gradients = None
        learning_rate = alpha

        if verbose == True:
            print(f'Number of samples is: {samples_number}')
            print(f'Number of features is: {feature_number-1}')
            print(f'Initialised theta is: {theta}')
            print(Fore.GREEN+'Start training ...')
        
        if view == True:
            hinge_loss_history = []
            regularizer_history = []
            distance_history = []

        for iteration in range(max_iterations):
            theta_old = theta

            hinge_losses, loss_gradients = LinearClassifierHelper.hinge_loss(inputs,labels,theta)
            regularizer, regularizer_gradient = LinearClassifierHelper.l2_regularizer(feature_number, theta, lbda)

            if verbose == True:
                print('Average Hinge loss: {}'.format(np.mean(hinge_losses)))
                print(f'Regularizer: {regularizer}')

        
            # loss_gradients = np.sum(loss_gradients, axis=0)
            # if loss_gradients.ndim == 1:
            #     loss_gradients = loss_gradients.reshape(-1, 1)

            gradient = np.dot(inputs.T, loss_gradients).flatten() + regularizer_gradient

            if previous_gradients is not None:
                grad_diff = previous_gradients - gradient
                grad_adjustment = (np.dot(previous_gradients.T, previous_gradients)) / (np.dot(grad_diff.T, previous_gradients))
                learning_rate *= grad_adjustment
            
            theta = theta_old - alpha*gradient

            distance = LinearClassifierHelper.calculate_euclidean_distance(theta_old,theta)

            if view == True:
                hinge_loss_history.append(np.mean(hinge_losses))
                regularizer_history.append(regularizer)
                distance_history.append(distance)

            if distance < epsilon:
                break

            gradient_old = gradient
        
        if theta.ndim == 1:
            theta = theta.reshape(-1, 1)
        
        if verbose == True:
            print(Fore.GREEN+'Training end.')
            print(f'Theta: {theta}')

        if view == True:
            LinearClassifierHelper.plot_metrics(hinge_loss_history, regularizer_history, distance_history)

        return theta
    
    @staticmethod
    def predict(inputs, theta):
        """Predict the class labels for a set of inputs using the learned parameter theta."""

        if theta is None:
            print("Theta is None.")
            return None, None, inputs, theta

        if inputs.ndim == 1:
            inputs = inputs.reshape(-1, 1)
        inputs = np.hstack((inputs, np.ones((inputs.shape[0], 1))))
        
        raw_predictions = np.dot(inputs,theta)
        if raw_predictions.ndim == 1:
            raw_predictions = raw_predictions.reshape(-1, 1)

        predictions = np.where(raw_predictions >= 0, 1, -1)

        return predictions, raw_predictions, inputs, theta
    
    @staticmethod
    def evaluate(predictions, true_labels):
        """Evaluate the accuracy of the model."""

        accuracy = np.mean(predictions == true_labels)
        return accuracy

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








