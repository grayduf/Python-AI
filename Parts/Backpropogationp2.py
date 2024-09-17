import numpy as np

# passed in gradient from the next layer
# using a vector of 1s for this example
dvalues = np.array([[1., 1., 1.]])

# 3 sets of weights, 1 for each neuron
# 4 inputs, so we have 4 weights 
# keep weights transposed
weights = np.array([[0.2, 0.8, -0.5, 1],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# Sum weights related to the given input multiplied by the gradient related to the given neuron
# same thing as: dx0 = sum([weights[0][0]*dvalues[0][0], weights[0][1]*dvalues[0][1], weights[0][2]*dvalues[0][2]])
# this is because weights and dvalues are both NumPy arrays 
dx0 = sum(weights[0]*dvalues[0])
dx1 = sum(weights[1]*dvalues[0])
dx2 = sum(weights[2]*dvalues[0])
dx3 = sum(weights[3]*dvalues[0])

dinputs1 = np.array([dx0, dx1, dx2, dx3])
print("sum of multiplication: ", dinputs1)

# the dot product is the simplification of the code above
dinputs2 = np.dot(dvalues[0], weights.T)
print("dot product:           ", dinputs1)