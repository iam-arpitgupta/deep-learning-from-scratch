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