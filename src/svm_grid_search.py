import time
import numpy as np
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_curve, average_precision_score, roc_curve, roc_auc_score
from svm_helper import SvmHelper
import os

class SvmGridSearch:
    def __init__(self, dataset, param_grid=None, kernel_param_grid=None):
        self.dataset = dataset
        self.param_grid = param_grid
        self.kernel_param_grid = kernel_param_grid
    
    def send_imessage(self, phone_number, message):
        """Send an iMessage to the given phone number using AppleScript."""
        applescript_command = f'''
        tell application "Messages"
            set targetService to 1st account whose service type = iMessage
            set targetBuddy to buddy "{phone_number}" of targetService
            send "{message}" to targetBuddy
        end tell
        '''
        os.system(f'osascript -e \'{applescript_command}\'')

    def train_model_with_params(self, params, kernel_function, kernel_params=None, verbose=False):
        if kernel_params is None:
            raise ValueError("kernel_params cannot be None")

        train_result = SvmHelper.regularised_kernel_erm_batch(
            inputs=self.dataset.train_validate_inputs,
            labels=self.dataset.train_validate_labels,
            kernel_function=kernel_function,
            max_iterations=100,
            epsilon=params['epsilon'], 
            alpha=params['alpha_0'], 
            lbda=params['lambda_value'],
            decay=params['decay'],
            **kernel_params  
        )

        theta = train_result['theta']

        predictions, raw_predictions = SvmHelper.predict_kernel(
            theta=theta,
            kernel_function=kernel_function,
            test_inputs=self.dataset.validate_inputs,
            train_inputs=self.dataset.train_validate_inputs
        )

        accuracy = accuracy_score(self.dataset.validate_labels, predictions)

        if verbose:
            print(f"Accuracy with current params: {accuracy}")
        
        return accuracy

    def create_accuracy_plot(self, accuracies, selected_kernel):
        iterations = range(1, len(accuracies) + 1)
        plt.plot(iterations, accuracies, label=f"{selected_kernel} - Accuracy", color="green")
        plt.xlabel("Iteration")
        plt.ylabel("Accuracy")
        plt.title(f"{selected_kernel} - Accuracy")
        plt.grid(True)
        plt.show()

    def grid_search(self, selected_kernel, verbose=False):

        best_params = None
        best_accuracy = 0.0
        accuracies_history = []
        start_time = time.time()
        elapsed_time = None

        result = {
            'selected_kernel': selected_kernel,
            'best_accuracy': best_accuracy,
            'accuracies_history': accuracies_history,
            'best_params': best_params,
            'elapsed_time': elapsed_time
        }

        kernel_functions = {
            'dtw': lambda input, input_copy: SvmHelper.dtw_kernel(input, input_copy),
            'polynomial': lambda input, input_copy, **kwargs: SvmHelper.polynomial_kernel(
                input, 
                input_copy, 
                alpha=kernel_params.get('kernel_alpha'), 
                degree=kernel_params.get('degree'),
                c=kernel_params.get('c')
            ),
            'rbf': lambda input, input_copy, **kwargs: SvmHelper.rbf_kernel(input, input_copy, lbda=kernel_params.get('gamma'))
        }

        for model_params in ParameterGrid(self.param_grid):
            if selected_kernel in self.kernel_param_grid:
                for kernel_params in ParameterGrid(self.kernel_param_grid[selected_kernel]):
                    kernel_function = kernel_functions[selected_kernel]
                    accuracy = self.train_model_with_params(model_params, kernel_function=kernel_function, kernel_params=kernel_params)
                    accuracies_history.append(accuracy)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_params = {**model_params, **kernel_params}

        end_time = time.time()
        elapsed_time = end_time - start_time

        if verbose == True:
            print("Best parameters found:", best_params)
            print("Best accuracy:", best_accuracy)
            print(f"Time taken for grid search: {elapsed_time:.2f} seconds")
            self.create_accuracy_plot(accuracies_history, selected_kernel)

        result['best_accuracy'] = best_accuracy
        result['accuracies_history'] = accuracies_history
        result['best_params'] = best_params
        result['elapsed_time'] = elapsed_time

        return result


    def plot_training_metrics(self, selected_kernel, hinge_loss_history, regularizer_history, distance_history, output_dir, filename):
        """Plot training metrics for the selected kernel and save the plot to the output directory."""
        iterations = range(1, len(hinge_loss_history) + 1)

        plt.figure(figsize=(12, 8))

        plt.subplot(3, 1, 1)
        plt.plot(iterations, hinge_loss_history, label="Hinge Loss", color="blue")
        plt.xlabel("Iteration")
        plt.ylabel("Hinge Loss")
        plt.title(f"{selected_kernel} - Hinge Loss (Best Accuracy)")
        plt.grid(True)

        plt.subplot(3, 1, 2)
        plt.plot(iterations, regularizer_history, label="Regularizer", color="green")
        plt.xlabel("Iteration")
        plt.ylabel("Regularizer")
        plt.title(f"{selected_kernel} - Regularizer (Best Accuracy)")
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.plot(iterations, distance_history, label="Distance (theta old - theta new)", color="red")
        plt.xlabel("Iteration")
        plt.ylabel("Distance")
        plt.title(f"{selected_kernel} - Distance between Theta Updates (Best Accuracy)")
        plt.grid(True)

        plt.tight_layout()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plot_file = os.path.join(output_dir, filename)
        plt.savefig(plot_file)
        plt.close()

        return plot_file

    def svm_precision_recall_curve(self, labels, raw_predictions):
        labels = np.where(labels == 1, 1, 0)

        precision, recall, thresholds = precision_recall_curve(labels, raw_predictions)
        average_precision = average_precision_score(labels, raw_predictions)

        return precision, recall, thresholds, average_precision
    
    def svm_roc_curve(self, labels, raw_predictions):
        labels = np.where(labels == 1, 1, 0)  

        fpr, tpr, thresholds = roc_curve(labels, raw_predictions)
        auc = roc_auc_score(labels, raw_predictions)

        return fpr, tpr, thresholds, auc
 
    def train_and_test(self, selected_kernel, best_params, max_iterations=100, output_dir="../../figures"):
        accuracies_history = []
        theta_history = []
        best_theta = None
        best_accuracy = 0.0

        best_hinge_loss_history = []
        best_regularizer_history = []
        best_distance_history = []

        precision = None
        recall = None
        thresholds = None
        average_precision = None

        fpr = None
        tpr = None
        roc_thresholds = None
        auc = None

        result = {
            'selected_kernel': selected_kernel,
            'best_accuracy': best_accuracy,
            'best_theta': best_theta,
            'accuracies_history': accuracies_history,
            'theta_history': theta_history,
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'hinge_loss_history': best_hinge_loss_history,
            'regularizer_history': best_regularizer_history,
            'distance_history': best_distance_history,
            'fpr': fpr,
            'tpr': tpr,
            'roc_thresholds': roc_thresholds,
            'auc': auc
        }

        for i in range(max_iterations):

            model_specific_params = {
                'alpha': best_params['alpha_0'],
                'epsilon': best_params['epsilon'],
                'lbda': best_params['lambda_value'],
                'decay': best_params['decay']
            }

            if selected_kernel == 'dtw':
                kernel_function = lambda input, input_copy: SvmHelper.dtw_kernel(input, input_copy)
            elif selected_kernel == 'polynomial':
                kernel_function = lambda input, input_copy, **kwargs: SvmHelper.polynomial_kernel(
                    input, input_copy, 
                    alpha=best_params.get('kernel_alpha'), 
                    degree=best_params.get('degree'), 
                    c=best_params.get('c')
                )
            elif selected_kernel == 'rbf':
                kernel_function = lambda input, input_copy, **kwargs: SvmHelper.rbf_kernel(
                    input, input_copy, 
                    lbda=best_params.get('gamma')
                )

            train_result = SvmHelper.regularised_kernel_erm_batch(
                inputs=self.dataset.train_inputs,
                labels=self.dataset.train_labels,
                kernel_function=kernel_function, 
                max_iterations=200,
                alpha=model_specific_params['alpha'], 
                epsilon=model_specific_params['epsilon'], 
                lbda=model_specific_params['lbda'],
                decay=model_specific_params['decay'],
                verbose=False,
                figure=False
            )
            theta = train_result['theta']

            predictions, raw_predictions = SvmHelper.predict_kernel(theta=theta, kernel_function=kernel_function, test_inputs=self.dataset.test_inputs, train_inputs=self.dataset.train_inputs)
            accuracy = accuracy_score(self.dataset.test_labels, predictions)
            accuracies_history.append(accuracy)
            theta_history.append(theta)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_theta = theta

                best_hinge_loss_history = train_result['hinge_loss_history']
                best_regularizer_history = train_result['regularizer_history']
                best_distance_history = train_result['distance_history']

        plot_file = self.plot_training_metrics(
            selected_kernel=selected_kernel,
            hinge_loss_history=best_hinge_loss_history,
            regularizer_history=best_regularizer_history,
            distance_history=best_distance_history,
            output_dir=output_dir,
            filename=f'{selected_kernel}_best_accuracy_training_plot.png'
        )

        precision, recall, thresholds, average_precision = self.svm_precision_recall_curve(self.dataset.test_labels, raw_predictions)
        fpr, tpr, roc_thresholds, auc = self.svm_roc_curve(self.dataset.test_labels, raw_predictions)

        result['plot_path'] = plot_file
        result['best_accuracy'] = best_accuracy
        result['best_theta'] = best_theta
        result['accuracies_history'] = accuracies_history
        result['theta_history'] = theta_history
        result['precision'] = precision
        result['recall'] = recall
        result['thresholds'] = thresholds
        result['average_precision'] = average_precision
        result['hinge_loss_history'] = best_hinge_loss_history
        result['regularizer_history'] = best_regularizer_history
        result['distance_history'] = best_distance_history
        result['fpr'] = fpr
        result['tpr'] = tpr
        result['roc_thresholds'] = roc_thresholds
        result['auc'] = auc

        return result

    def test(self, selected_kernel, best_params, best_theta):

        model_specific_params = {
            'alpha': best_params['alpha_0'],
            'epsilon': best_params['epsilon'],
            'lbda': best_params['lambda_value'],
            'decay': best_params['decay']
        }

        if selected_kernel == 'dtw':
            kernel_function = lambda input, input_copy: SvmHelper.dtw_kernel(input, input_copy)
        elif selected_kernel == 'polynomial':
            kernel_function = lambda input, input_copy, **kwargs: SvmHelper.polynomial_kernel(
                input, input_copy, 
                alpha=best_params.get('kernel_alpha'), 
                degree=best_params.get('degree'), 
                c=best_params.get('c')
            )
        elif selected_kernel == 'rbf':
            kernel_function = lambda input, input_copy, **kwargs: SvmHelper.rbf_kernel(
                input, input_copy, 
                lbda=best_params.get('gamma')
            )

        predictions, raw_predictions = SvmHelper.predict_kernel(theta=best_theta, kernel_function=kernel_function, test_inputs=self.dataset.test_inputs, train_inputs=self.dataset.train_inputs)
        accuracy = accuracy_score(self.dataset.test_labels, predictions)

        return accuracy
    



