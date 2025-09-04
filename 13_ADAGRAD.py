# for the development of ADAM optimizer we need to understand the use of ADAGRAD optimizer 
# that different weights or the different parameters can have different learning rate
import numpy as np
from Whole_backprop import LayerDense , ActivationSoftmax , ReluActivation, ActivationSoftmax_crossEntropLoss

class AdaGrad:
    def __init__(self , learning_rate, decay = 0. , epislon = 1e-5):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay 
        self.iterations = 0 
        self.epislon = epislon

    def pre_update_params(self):
        # changing the learning rate 
        #alpha => current learning rate 
        if self.decay:
            self.current_learning_rate = self.learning_rate * \ 
            (1. / (1. + self.decay * self.iterations))


    def update_params(self , layer):
            # if not cache then assign 0 to it 
            if not hasattr(layer , 'weight_cache'):
                    layer.weight_momentum = np.zeros_like(layer_weights)
                    layer.biases_momentum = np.zeros_like(layer_biases)

            # update the cache
            # cache = cache * prams_grad ^ 2 
            layer.weight_cache += layer.dweights ** 2 
            layer.biases_cache += layer.dbiases ** 2 

            # aplha * gradient / under_root(cache + gradient)
            layer.weights += -self.current_learning_rate * layer.dweights /  (np.sqrt(layer.weight_cache) + self.epislon)


            layer.biases += -self.current_learning_rate * layer.biases /  (np.sqrt(layer.biases_cache) + self.epislon)

    
    def post_update_params(self):
        self.iterations += 1  



    # Create dataset
X, y = spiral_data( samples = 100 , classes = 3 )
# Create Dense layer with 2 input features and 3 output values
dense1 = LayerDense( 2 , 3 )
# Create ReLU activation (to be used with Dense layer):
activation1 = ReluActivation()
# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = LayerDense( 64 , 3 )
# Create Softmax classifier's combined loss and activation
loss_activation = ActivationSoftmax_crossEntropLoss()

# create optimizer:
optimizer = AdaGrad(decay = 1e-4)

for epoch in range(1000):
    # Perform a forward pass of our training data through this layer
    dense1.forward(X)
    # Perform a forward pass through activation function
    # takes the output of first dense layer here
    activation1.forward(dense1.output)
    # Perform a forward pass through second Dense layer
    # takes outputs of activation function of first layer as inputs
    dense2.forward(activation1.output)
    # Perform a forward pass through the activation/loss function
    # takes the output of second dense layer here and re
    loss = loss_activation.forward(dense2.output, y)
    #calculate accuracy
    predictions = np.argmax(loss_activation.output, axis = 1 )
    if len (y.shape) == 2 :
    y = np.argmax(y, axis = 1 )
    accuracy = np.mean(predictions == y)
    # Print accuracy
    print ( 'acc:' , accuracy)
    # Backward pass
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # update 
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
