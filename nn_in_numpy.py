import numpy as np 
# the dot product b.w vector and vector is neuron
input = [1,2,3]
weights = [0.5,0.7,0.8]
bias = 1

output = np.dot[input, weights]
result = output + bias 

# the dot product b.w vector and matrix is layer [mutiple neurons]

input = [1, 2, 3 , 4]
# weights[0] = [[1,2,3,5], weights[1] = [4,7,9,3],  weights[3] =  [9,7,6,5]
weights = [[1,2,3,5],
           [4,7,9,3],
           [9,7,6,5]]

biases = [3,4,5]
# will perform the whooel operation 
output = np.dot(weights , input) + biases
print(output)


# batch coding -> matrix multiplication 
input = [[1, 2, 3 , 4] , 
         [5, 6, 8 , 9] ,
         [3,5, 6 , 9]]

weights = [[1,2,3,5],
           [4,7,9,3],
           [9,7,6,5]]
# dim (3,4) (3,4) cant be mltiplied 
# we need to take the transpose to make it to [4,3]
biases = [3,4,5]
output = np.dot(input ,np.array(weights).T) +biases 



# working with multiple layers [including the hidden layer ]
input = [[1, 2, 3 , 4] , 
         [5, 6, 8 , 9] ,
         [3,5, 6 , 9]]

weights = [[1,2,3,5],
           [4,7,9,3],
           [9,7,6,5]]
# dim (3,4) (3,4) cant be mltiplied 
# we need to take the transpose to make it to [4,3]
biases = [3,4,5]
biases2 = [9,8,6]
output = np.dot(input ,np.array(weights).T) +biases 
# code for the 1 layer 
# convert the list to numpy arr 
input_arr = np.array(input)
weights_arr = np.array(weights)
biases_arr = np.array(biases)
biases_arr2 = np.array(biases2)

output = np.dot(input_arr , weights_arr.T) + biases_arr


input = [[1, 2, 3 , 4] , 
         [5, 6, 8 , 9] ,
         [3,5, 6 , 9]]

weights = [[1,2,3,5],
           [4,7,9,3],
           [9,7,6,5]]

weights2 = [[1,2,2,2],
           [4,7,4,3],
           [9,7,6,5]]
input_arr = np.array(input)
weights_arr = np.array(weights)
weights2_arr = np.array(weights2)

biases_arr = np.array(biases)
# different layers 
layer1_output = np.dot(input ,np.array(weights).T) +biases_arr
layer2_output =  np.dot(layer1_output ,np.array(weights).T) +biases_arr2

