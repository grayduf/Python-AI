import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

# Dense Layer class
class Layer_Dense:
    # Initialize layer
    def __init__(self, n_inputs, n_neurons):
        # Set weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    # Forward pass
    def forward(self, inputs):
        # Set output for layer, from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases

# ReLU Activation class
class Activation_ReLU:
    # Forward pass 
    def forward(self, inputs):
        # Calculate output from largest of 0 or inputs
        self.output = np.maximum(0, inputs)

# Softmax Activation class
class Activation_Softmax:
    # Forward pass
    def forward(self, inputs):
        # Calculate un-normalized probabilities (axis=1 means column wise; keepdims means it will stay in the same dimensions)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize probabilities 
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        # Set the output
        self.output = probabilities

# Common Loss class
class Loss:
    # Calculate pass
    def calculate(self, output, y):
        # calc sample loss
        sample_losses = self.forward(output, y)
        # calc mean loss
        data_loss = np.mean(sample_losses)
        # return loss
        return data_loss
    
# Cross-Entropy Loss
class Loss_CategoricalCrossEntropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0, and bias one way or the other
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # Find probabilities for target values, but only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]

        # Mask values, but only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Find Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossEntropy()

# Helper variables
lowest_loss = 9999999 # some initial value
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for iteration in range(10000):
    # Generate a new set of weights for iteration
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_function.calculate(activation2.output, y)

    predictions = np.argmax(activation2.output, axis=1)
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        print('New set of weights found, iteration:', iteration, 'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()