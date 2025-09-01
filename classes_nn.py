# implementing the dense neural network layer using classes 
# simple , without any activation func
import numpy as np
class nn:
    def __init__(self,n_inputs , n_neurons):
        input = self.n_input 
        neuron = self.n_neurons
        self.weights = 0.01 * np.random.randn(input , neuron)
        self.biases = np.zeros((1,neuron))

    def forward(self , input):
    
        self.result =  np.dot(input,self.neuron) + self.biases


dense1 = nn(2,3)
dense1.forward(X)

