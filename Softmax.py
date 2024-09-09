# first step to teaching a NN to learn is to figure out how wrong it is
# perfect output would have the correct end neuron with a 1.00, meaning it is 100% correct, with all other neurons at 0.00
# cannot just use a ReLU activation function by itself; if any of the end numbers are negative, it would clip it to 0.00
# this would make it impossible to learn if both values are negative, how wrong was it?
# use expoential function: y = e^x, using Euler's number
# normalize these numbers after exponential
''' WITHOUT NUMPY
import math

layer_outputs = [4.8, 1.21, 2.385]

E = math.e #2.71828182846

exp_values = []

for output in layer_outputs:
    exp_values.append(E**output) #append E^layer_output[i] into exp_values

print(exp_values)

norm_base = sum(exp_values)
norm_values = []

for value in exp_values:
    norm_values.append(value / norm_base)

print(norm_values)
print(sum(norm_values)) 
'''
# with numpy
# overall: input -> exponentiate -> normalize -> output
# exponentiate and normalization together is known as softmax

''' ONLY ONE LAYER
import numpy as np

layer_outputs = [4.8, 1.21, 2.385]

exp_values = np.exp(layer_outputs)
norm_values = exp_values / np.sum(exp_values)

print(norm_values)
print(sum(norm_values))
'''

# BATCH USING EXP
import numpy as np

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True) #we want the sum of the rows, so its the sum of each individual batch
# keepdims keeps the sum in 3x3 matrix form instead of 1x3 matrix

print(norm_values) 

# exp values get too big and cause overflow; combat by taking largest value and subtracting it from all values
# largest value becomes 0, and all other values are less than 0 which when changed to exp is a small number close to 0 but not quite
# the actual output stays the exact same, this is just to protect from overflow cases