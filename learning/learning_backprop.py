import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

input = [1, 0.5, 0.8]

weights = [0.2, 0.8, -0.5]

bias = 0.1

#forward pass

output = input[0] * weights[0] + input[1] * weights[1] + input[2] * weights[2] + bias
output = sigmoid(output)

#back pass

answer = 1

#we want de/dw, which is the error with respect to weights
#if we have this, we can just change weights in the direction of the lower error (negative gradient)

# de/dw = de/da * da/dz * dz/dw

#this is de/da (error with respect to activation output)
error = answer - output


#da/dz (activation derivative with respect to weighted sum (output))
#this is equal to the sigmoid derivative function

#this is de/da * da/dz (activation derivative with respect to weighted sum (output))
delta = error * sigmoid_derivative(output)

learning_rate = 0.01

#dz/dw (derivative of weighted sum with respect to each weight)
#mathematically, this is just x (the input)

#using our above equation: de/dw = de/da * da/dz * dz/dw

#de/dw = error * sigmoid_derivative(output) * input (for a given weight)

#so to update, it is:

weights = [weights[i] - (learning_rate * input[i] * delta) for i in range(len(input))]

bias = bias - (learning_rate * delta)

print(weights)



