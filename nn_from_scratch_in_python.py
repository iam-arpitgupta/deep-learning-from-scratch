# simple preceptron 
input = [1,2,3]
weights = [0.5,0.7,0.8]
bias = 1 # since onlt neuron is there 
#x1w1 + x2w2 +bi
# as the number of input terms increases weights increases , bias remains the same 
output = input[0]*weights[0] + input[1]*weights[1] +input[2]*weights[2]


# multiple neurons ,
#since each neuron have 4 inputs we are taking 3 neurons so we have 12 value of weights 
input = [1, 2, 3 , 4]
# weights[0] = [[1,2,3,5], weights[1] = [4,7,9,3],  weights[3] =  [9,7,6,5]
weights = [[1,2,3,5],
           [4,7,9,3],
           [9,7,6,5]]

weights1 = weights[0],
weights2 = weights[1], 
weights3 =  weights[2]

bias1 = 2
bias2 = 4
bias3 = 5
output = [#neuron1 
        input[1] * weights1[1]+
        input[0] * weights1[0]+
        input[2] * weights1[2]+
        input[3] * weights1[3]+
        bias1,
        # neuron 2
        input[1] * weights2[1]+
        input[0] * weights2[0]+
        input[2] * weights2[2]+
        input[3] * weights2[3]+
        bias2,
        #neuron3
        input[1] * weights3[1]+
        input[0] * weights3[0]+
        input[2] * weights3[2]+
        input[3] * weights3[3]+
        bias3
]

# modified -> using loops 

input = [1, 2, 3 , 4]
# weights[0] = [[1,2,3,5], weights[1] = [4,7,9,3],  weights[3] =  [9,7,6,5]
weights = [[1,2,3,5],
           [4,7,9,3],
           [9,7,6,5]]

biases = [2,3,4]

layer_output = []

for neuro_weight , neuro_bias in zip(weights , biases):
    neuron_output = 0
    for n_input , weight in zip(input,neuro_weight):
        neuron_output += n_input * weight
        neuron_output += neuro_bias
        layer_output.append(neuron_output)
    print(layer_output)

    
        



