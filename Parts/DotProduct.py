import numpy as np

inputs = [[1, 2, 3, 4], 
          [1, 2, 3, 4], 
          [1, 2, 3, 4]]

weights  = [[0.1, 0.2, 0.3, 0.4], 
             [0.2, 0.3, 0.4, 0.5], 
             [0.3, 0.4, 0.5, 0.6]]

bias = [1, 1, 1]

output = np.dot(inputs, np.array(weights).T) + bias

print(output)

'''
layer_outputs = []

for neuron_weights, neuron_biases in zip(weights, bias):
    neuron_output = 0
    for n_input, weights in zip(inputs, neuron_weights):
        neuron_output += n_input * weights
    neuron_output += neuron_biases
    layer_outputs.append(neuron_output)

print(layer_outputs)
'''