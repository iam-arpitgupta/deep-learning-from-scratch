# RELU 
# relu , is not for classification problem , as itf does not give any probablistic output 
# in numpy we use np.max(o,x) for relu 
# range [0,x]
# is used to introduce some non linearlity in the code 
# in order to capture the non linear functions
import numpy as np 
from classes_nn import nn

# python code 
inputs = [ 0 , 2 ,- 1 , 3.3 ]
output = []
for i in inputs:
    output.append( max ( 0 , i))
print(output)

# numpy code

inputs = [0.2,0.5,0.6]
output = np.maximum(0,inputs)

# class implementation 
class ReLu:
    def forward(self , X):
        self.output = np.maximum(0,X)


# Create dataset
X, y = spiraldata( samples = 100 , classes = 3 )
# Create Dense layer with 2 input features and 3 output values
dense1 = nn(2,3)
# Create ReLU activation (to be used with Dense layer):
activation1 = ReLu()
# Make a forward pass of our training data through this layer
dense1.forward(X)
# Forward pass through activation func.
# Takes in output from previous layer
activation1.forward(dense1.output)


# softmax activation function 
input = [[1, 2, -3 , 4] , 
         [5, 6, -8 , 9] ,
         [3,5, 6 , -9]]

# subtracting the max value form each col 
# to un-normalize the prob 
exp_values = np.exp(np.max(inputs , axis = 1 , keep_dims = True))#max from each row and taking exp 
proba = exp_values / np.sum(exp_values , axis = 1 , keep_dims=True)
np.sum(proba,axis =1)# sum ==1 


# class

class Softmax:
    def forward(self , x):
        exp_values = np.exp(x - np.sum(x , axis = 1 , keep_dims = True))
        proba  = exp_values / np.sum(exp_values , axis = 1, keep_dims = True)
        self.output = proba

    

## summing up together 
# coding the entire forward pass 
# without loss 
import numpy as np

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output

class ReLUActivation:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        return self.output

class SoftmaxActivation:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output

model = [
    LayerDense(2, 3),
    ReLUActivation(),
    LayerDense(3, 3),
    SoftmaxActivation()
]

inputs = [[1.0, 2.0]]
out = inputs
for layer in model:
    out = layer.forward(out)

print(out)


