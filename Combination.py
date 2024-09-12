import numpy as np
import nnfs
from nnfs.datasets import spiral_data

# Initialize random seed and dataset for tests
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

# Set data, X = samples; y = classes
X, y = spiral_data(samples=100, classes=3)

# Set dense layer 1 of 2 input features and 3 output features
dense1 = Layer_Dense(2, 3)
# Set ReLU activation
activation1 = Activation_ReLU()

# Set dense layer 2, set activation 2
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

# Create loss function
loss_function = Loss_CategoricalCrossEntropy()

# Call the forward pass for dense1 with input data
dense1.forward(X)
# Call forward pass for ReLU activation1 with dense1 output
activation1.forward(dense1.output)

# Next layer in set
dense2.forward(activation1.output)
activation2.forward(dense2.output)

# Print output of the first couple of samples
print(activation2.output[:5])

# Calculate Loss using output from second layer and correct classes
loss = loss_function.calculate(activation2.output, y)

# Print loss
print("Loss: ", loss)

# Calculate prediction values along columns
predictions = np.argmax(activation2.output, axis=1)
# If one-hot encoded, convert
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
# Find average accuracy
accuracy = np.mean(predictions == y)
# Print accuracy
print("Accuracy: ", accuracy)