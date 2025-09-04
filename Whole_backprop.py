# coding the whole nn in python for a classification problem
import numpy as np
# layer dense forward and backward pass
class LayerDense:
    def __init__(self , input , layers):
        self.weights = 0.01 * np.random.radn(input , layers)
        self.biases = np.zeroes(1, layers)


    def forward(self , input):
        self.input = input
        self.output = np.dot(input , self.weights) + self.biases
    # dvalues -> dl_dz
    def backward(self , dvalues):
        self.dweights =  np.dot(self.input.T , dvalues)
        self.biases = np.sum(dvalues , axis =0 ,keep_dims = True)
        self.inputs = np.dot(dvalues,self.weights.T)

# relu activation forward and back
class ReluActivation:
    def forward(self , input):
        self.input = input 
        self.output = np.maximum(0, input)

    def backward(self, dvalues):
        # first we assign the output to  dl_dz wrt relu output 
        self.dinput = dvalues.copy()
        self.dinput[self.input <= 0] = 0


# we have a combine class for softmax and cross entropy 
class ActivationSoftmax:
    def forward(self, input):
        exp_values = np.exp(input - np.max(input , axis = 1 , keep_dims = True))
        probs = exp_values / np.sum(exp_values , axis = 1 , keep_dims = True)
        self.output = probs


class CrossEntropyLoss:
    def forward(self, y_pred , y_true):
        samples = len(y_pred)

        #clip
        y_pred_clipped = np.clip(y_pred , 1e-5,1e-5)

        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[
                range(samples),
                y_true
            ]

        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(
                y_pred_clipped * y_true,
                axis = 1
            )

        negative_log_likehood = -np.log(correct_confidence)
        return negative_log_likehood
    
    def calculate(self,output,y):
        sample_loss = self.forward(output,y)
        data_loss = np.mean(sample_loss)
        return data_loss
    
    def backward(self, dvalues , y_true):
        sample = len(dvalues)
        # these are the disceret values 
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true ,axis =1)

        self.dinputs = dvalues.copy()
        # range[samples] -> no of total batches or row in the matrix 
        self.dinputs[range(sample) , y_true] -= 1
        # noramilze gradients
        self.dinputs = self.dinputs / range(sample)



class ActivationSoftmax_crossEntropLoss:
    def __init__(self):
        self.activation = ActivationSoftmax()
        self.loss = CrossEntropyLoss()

    def forward(self, input, y_true):
        self.activation.forward(input)
        self.output = self.activation.output
        # cal the loss 
        return self.loss.calculate(self.output , y_true)

        
    def backward(self,dvalues , y_true):
        sample = len(dvalues)
        # these are the disceret values 
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true ,axis =1)

        self.dinputs = dvalues.copy()
        # range[samples] -> no of total batches or row in the matrix 
        # predicted - ground_truth
        self.dinputs[range(sample) , y_true] -= 1
        # noramilze gradients
        self.dinputs = self.dinputs / range(sample)

    

    # Create dataset
X, y = spiral_data( samples = 100 , classes = 3 )
# Create Dense layer with 2 input features and 3 output values
dense1 = LayerDense( 2 , 3 )
# Create ReLU activation (to be used with Dense layer):
activation1 = ReluActivation()
# Create second Dense layer with 3 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = LayerDense( 3 , 3 )
# Create Softmax classifier's combined loss and activation
loss_activation = ActivationSoftmax_crossEntropLoss()
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


# Print gradients
print (dense1.dweights)
print (dense1.dbiases)
print (dense2.dweights)
print (dense2.dbiases)