import time
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from svm_helper import SvmHelper
import os

class SvmGridSearch:
    def __init__(self, dataset, param_grid, kernel_param_grid):
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
            inputs=self.dataset.train_inputs,
            labels=self.dataset.train_labels,
            kernel_function=kernel_function,
            max_iterations=100,
            epsilon=params['epsilon'], 
            alpha=params['alpha_0'], 
            lbda=params['lambda_value'],
            decay=params['decay'],
            **kernel_params  
        )

        theta = train_result['theta']

        predictions = SvmHelper.predict_kernel(
            theta=theta,
            kernel_function=kernel_function,
            test_inputs=self.dataset.test_inputs,
            train_inputs=self.dataset.train_inputs
        )

        accuracy = accuracy_score(self.dataset.test_labels, predictions)

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
    
    def train_and_test(self, selected_kernel, best_params, max_iterations=100, output_dir="../../figures"):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        accuracies_history = []
        theta_history = []
        hinge_loss_history = []
        distance_history = []
        regularizer_history = []
        best_theta = None
        best_accuracy = 0.0
        plot_path = output_dir

        result = {
            'selected_kernel': selected_kernel,
            'best_accuracy': best_accuracy,
            'best_theta': best_theta,
            'accuracies_history': accuracies_history,
            'theta_history': theta_history
        }

        for i in range(max_iterations):
            print(f"{selected_kernel} - Run {i+1}:")

            model_specific_params = {
                'alpha': best_params['alpha_0'],
                'epsilon': best_params['epsilon'],
                'lbda': best_params['lambda_value'],
                'decay': best_params['decay']
            }

            if selected_kernel == 'dtw':
                kernel_function = lambda input, input_copy: SvmHelper.kernel(input, input_copy)
            elif selected_kernel == 'polynomial':
                kernel_function = lambda input, input_copy, **kwargs: SvmHelper.kernel(
                    input, input_copy, 
                    alpha=best_params.get('kernel_alpha'), 
                    degree=best_params.get('degree'), 
                    c=best_params.get('c')
                    )
            elif selected_kernel == 'rbf':
                kernel_function = lambda input, input_copy, **kwargs: SvmHelper.kernel(
                    input, input_copy, 
                    lbda=best_params.get('gamma')
                    )

            train_result = SvmHelper.regularised_kernel_erm_batch(
                inputs=dataset.train_inputs,
                labels=dataset.train_labels,
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

            predictions = SvmHelper.predict_kernel(theta=theta, kernel_function=kernel_function, test_inputs=dataset.test_inputs, train_inputs=dataset.train_inputs)
            accuracy = accuracy_score(dataset.test_labels, predictions)
            accuracies_history.append(accuracy)
            theta_history.append(theta)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_theta = theta

                hinge_loss_history = train_result['hinge_loss_history']
                regularizer_history = train_result['regularizer_history']
                distance_history = train_result['distance_history']

                iterations = range(1, len(hinge_loss_history) + 1)

                plt.figure(figsize=(12, 8))

                plt.subplot(3, 1, 1)
                plt.plot(iterations, hinge_loss_history, label="Hinge Loss", color="blue")
                plt.xlabel("Iteration")
                plt.ylabel("Hinge Loss")
                plt.title(f"{selected_kernel} - Hinge Loss over Iterations")
                plt.grid(True)

                plt.subplot(3, 1, 2)
                plt.plot(iterations, regularizer_history, label="Regularizer", color="green")
                plt.xlabel("Iteration")
                plt.ylabel("Regularizer")
                plt.title(f"{selected_kernel} - Regularizer over Iterations")
                plt.grid(True)

                plt.subplot(3, 1, 3)
                plt.plot(iterations, distance_history, label="Distance (theta old - theta new)", color="red")
                plt.xlabel("Iteration")
                plt.ylabel("Distance")
                plt.title(f"{selected_kernel} - Euclidean Distance between Theta Updates")
                plt.grid(True)

                plt.tight_layout()
                plt.show()

        print(accuracies_history)

        plot_file = os.path.join(output_dir, f'{selected_kernel}_training_plot.png')
        plt.savefig(plot_file)
        plt.close()  

        result['plot_path'] = plot_file
        result['best_accuracy'] = best_accuracy
        result['best_theta'] = best_theta
        result['accuracies_history'] = accuracies_history
        result['theta_history'] = theta_history

        return result

