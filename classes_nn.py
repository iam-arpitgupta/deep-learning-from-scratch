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


#Broadcasting role 
matrix = [[1,2,3,5],
           [4,7,9,3],
           [9,7,6,5]]


# sum 
np.sum(matrix , axis = 0) #all rows , dim =3 
np.sum(matrix , axis = 1) #all cols , dim =3 


# with dim 
np.sum(matrix , axis = 0 , keep_dim = True) #all rows , dim = 1 X 3 2D array
np.sum(matrix , axis = 1 , keep_dim = True) #all col , dim =  3  X 1 2D array




