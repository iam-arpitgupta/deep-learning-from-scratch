# we can see that there is a surge of accuracy using this as compared to GD 

import numpy as np

class Optimized_Mom_GD:
    def __init__(self , learning_rate = 1 , decay = 0. , momentum = 0.):
        self.learning_rate =  learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum


    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \ 
            (1. / (1. + self.decay * self.iterations))


    def update_params(self , layer):
        # if momnetum is 0 then dont use it and if not 
        if self.momentum:
            if not hasattr(layer , 'weight_momentum'):
                    layer.weight_momentum = np.zeros_like(layer_weights)
                    layer.biases_momentum = np.zeros_like(layer_biases)


            # updatin Gradients
            weights_update = self.momentum * layer.weight_momentum - \
                            self.current_learning_rate * layer_weights
            layer.weight_momentum = weights_update

            biases_update = self.momentum * layer.biases_momentum - \
                            self.current_learning_rate * layer_weights
            layer.biases_momentum = biases_update

        else:
                layer_weights = layer_weights + -self.current_learning_rate*  layer_dweights 
                layer_biases = layer_biases + -self.current_learning_rate *  layer_dbiases


    def post_update_params(self):
        self.iterations += 1
