# back prop in nn 
import numpy as np 


inputs = np.array([1, 2, 3, 4])

# Initial weights and biases
weights = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2]
])

biases = np.array([0.1, 0.2, 0.3])

learning_rate = 0.001

def relu(x):
    return np.maximum(0 ,x)

def relu_derivative(x):
    return np.where(x > 0 ,1 , 0)



# Training loop
for iteration in range(200):
    # Forward pass
    z = np.dot(weights, inputs) + biases
    a = relu(z)
    y = np.sum(a)

    # Calculate loss
    loss = y ** 2


# back prop in multi - layer nn and with multiple inputs 
# weights value 
dl_dz = np.array([[1,1,1],
         [2,2,2],
         [3,3,3]])

inputs = np.array([1,2,3,4])

dweights = np.dot(inputs.T , dl_dz)

# bias 
biases = np.array([1,2,3])

dbiases = np.sum(dl_dz , axis =0 , keep_dims = True)

# Initial weights and biases
weights = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.5, 0.6, 0.7, 0.8],
    [0.9, 1.0, 1.1, 1.2]
])

# gradient of loss wrt to inputs
dinputs = np.dot(dl_dz , weights . T)


# using the backward func 
import numpy as np

class LayerDense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
    
    def backward(self, dl_dz):
        self.dweights = np.dot(inputs.T , dl_dz)
        self.dbiases = np.sum(dl_dz , axis =0 , keep_dims = True)
        self.dinputs = np.dot(dl_dz , weights . T)


# backeard in relu activation 
class ReLUActivation:
    def forward(self, inputs):
        self.inputs = inputs 
        self.output = np.maximum(0, inputs)
        return self.output
    
    def backward(self ,dvalues):
        self.dinputs = dvalues.copy()
        # if the val of inputs < 0 then it is == 0
        self.dinputs[self.inputs <= 0 ] = 0

class SoftmaxActivation:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output



