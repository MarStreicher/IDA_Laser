import numpy as np
from colorama import Fore, Back, Style, init
init(autoreset=True)

class SvmKernelHelper():
    def __init__(self):
        pass

    @staticmethod
    def hinge_loss():
        return
    
    @staticmethod
    def r2_regularizer():
        return
    
    @staticmethod
    def dtw_kernel():
        return
    
    @staticmethod
    def reg_erm_kernel(inputs, labels, max_iteration=200):

        K = SvmKernelHelper.dtw_kernel()

        number_instances = K.shape[0]
        beta = np.random.randn(number_instances)
        print(f'Initialised beta is: {beta}')

        print(Fore.GREEN+'Start training ...')
        for iteration in range(max_iteration):
            prediction = np.dot(K.T, beta)
            loss, loss_gradient = loss(prediction, labels)

            print(f'Mean training loss: '+ str(np.mean(loss)))

            ####








        return


    