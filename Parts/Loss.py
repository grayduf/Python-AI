# could calculate loss using only true or false and optimize for accuracy, but that wont let the NN know how wrong it was and learn better. 
# the normal Loss fucntion of choice when using softmax, is Categorical Cross-Entropy: -sum(target val * (log(predicted value)))
# this simplifies to: -log(target class' predicted value)
# one hot-encoding: one vector with the amount of classes as the number of entries and the label is the index of the class that is 1
# note that the "log" mentioned is natural log using Euler's number

import math

softmax_output = [0.7, 0.1, 0.2]
target_output = [1, 0, 0] # target_class = 0

loss = -(math.log(softmax_output[0]) * target_output[0] + #multiplied by 1
         math.log(softmax_output[1]) * target_output[1] + #becomes 0
         math.log(softmax_output[2]) * target_output[2])  #become 0

print(loss)

loss = -(math.log(softmax_output[0]))
print(loss)

# basically, softmax_output is the confidence in the answer and as the confidence is higher, the loss is lower and as the confidence is lower, the loss is higher
# when finding loss if the confidence is 0.0, it will be infinite which causes a lot of problems 
# can fix this by clipping the confidence to something like 1e-7, which might create a bias. Thus we clip the high end by 1 - 1e-7. Use np.clip