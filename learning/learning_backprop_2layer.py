import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    max_x = max(x)
    exp_x = [math.exp(x_i - max_x) for x_i in x]
    sum_exp_x = sum(exp_x)
    softmax = [exp_x_i / sum_exp_x for exp_x_i in exp_x]
    return softmax


#neural network structure:
# 3 input neurons
# 2 hidden neurons
# 2 output neurons


input = [1, 0.5, 0.8]
actual_answer = [0, 1]

weights_input_to_hidden = [[0.2, 0.8, -0.5],
                           [0.3, -0.2, 0.1]]

weights_hidden_to_output = [[1, -0.1],
                           [0.9, 0.4]]

bias_hidden_1 = 0.1
bias_hidden_2 = 0.3

bias_output_1 = 0.4
bias_output_2 = 0.6

#forward pass
#calculate hidden layer
#calcuate weighted sum
hidden_1 = sum([input[i] * weights_input_to_hidden[0][i] for i in range(len(input))]) + bias_hidden_1
hidden_2 = sum([input[i] * weights_input_to_hidden[1][i] for i in range(len(input))]) + bias_hidden_2

#apply activation function
hidden_1 = sigmoid(hidden_1)
hidden_2 = sigmoid(hidden_2)

hidden_layer = [hidden_1, hidden_2]

#calculate output layer

output_1 = sum([hidden_layer[i] * weights_hidden_to_output[0][i] for i in range(len(weights_hidden_to_output))]) + bias_output_1
output_2 = sum([hidden_layer[i] * weights_hidden_to_output[1][i] for i in range(len(weights_hidden_to_output))]) + bias_output_2

#apply softmax
softmax_vals = softmax([output_1, output_2])
output_1 = softmax_vals[0]
output_2 = softmax_vals[1]

#now we can do a backpass ------------------------------------------------------
print("before training answer: ", output_1, output_2)
for i in range(10000):



    #with softmax, delta for output layer is output-label
    output_layer_delta = []
    output_layer_delta.append(output_1 - actual_answer[0])
    output_layer_delta.append(output_2 - actual_answer[1])

    # now we have deltas for the output neurons, these essentially become the
    # new neuron values for the backpass, going from output to input
    # can think of it like a inverse-neuron value that is propagated back

    hidden_layer_delta = []
    hidden_layer_delta.append(sum([weights_hidden_to_output[0][i] * output_layer_delta[i] for i in range(0,2)]))
    hidden_layer_delta.append(sum([weights_hidden_to_output[1][i] * output_layer_delta[i] for i in range(0,2)]))

    # but for this we now need to times it by derivative of activation function
    # we do this for two reasons i can think of:
    # 1. the chain rule wants this: de/dw = de/da * da/dz * dz/dw
    # ^ it is the da/dz part
    # 2. It helps update unsure neurons faster, as the derivative
    # is as its highest at 0.5, and lowest at 0.1 or 0.9 etc

    hidden_layer_delta[0] *= sigmoid_derivative(hidden_1)
    hidden_layer_delta[1] *= sigmoid_derivative(hidden_2)

    # we dont go any further, no point calculating deltas for the input
    # the input isnt "wrong"

    # now change weights and biases
    # de/dw = de/da * da/dz * dz/dw
    # de/da = delta from previous layer
    # da/dz = derivative of activation function
    # dz/dw = value of neuron where weight is coming from


    # we do weight -= (learning_rate * delta * (value of neuron where weight is COMING FROM))

    # in essence, we have found the function de/dw and delta * (value of neuron where weight is COMING FROM)
    # gives us the gradient of the loss function with respect to the weight
    # then, we just move in the opposite direction of weight to minmize our loss/cost/error

    # and we do bias -= (learning_rate * delta) (we dont need to multiply by neuron value 
    # of where it is coming from, as a bias doesnt really come from a neuron technically)

    learning_rate = 0.01

    #update weights:

    weights_input_to_hidden[0][0] -= learning_rate * input[0] * hidden_layer_delta[0]
    weights_input_to_hidden[0][1] -= learning_rate * input[1] * hidden_layer_delta[0]
    weights_input_to_hidden[0][2] -= learning_rate * input[2] * hidden_layer_delta[0]

    weights_input_to_hidden[1][0] -= learning_rate * input[0] * hidden_layer_delta[1]
    weights_input_to_hidden[1][1] -= learning_rate * input[1] * hidden_layer_delta[1]
    weights_input_to_hidden[1][2] -= learning_rate * input[2] * hidden_layer_delta[1]

    weights_hidden_to_output[0][0] -= learning_rate * hidden_1 * output_layer_delta[0]
    weights_hidden_to_output[0][1] -= learning_rate * hidden_1 * output_layer_delta[1]
    weights_hidden_to_output[1][0] -= learning_rate * hidden_2 * output_layer_delta[0]
    weights_hidden_to_output[1][1] -= learning_rate * hidden_2 * output_layer_delta[1]

    #update biases
    bias_hidden_1 -= learning_rate * hidden_layer_delta[0]
    bias_hidden_2 -= learning_rate * hidden_layer_delta[1]
    bias_output_1 -= learning_rate * output_layer_delta[0]
    bias_output_2 -= learning_rate * output_layer_delta[1]


# with updated weights and biases, do another forward pass:


#forward pass
#calculate hidden layer
#calcuate weighted sum
hidden_1 = sum([input[i] * weights_input_to_hidden[0][i] for i in range(len(input))]) + bias_hidden_1
hidden_2 = sum([input[i] * weights_input_to_hidden[1][i] for i in range(len(input))]) + bias_hidden_2

#apply activation function
hidden_1 = sigmoid(hidden_1)
hidden_2 = sigmoid(hidden_2)

hidden_layer = [hidden_1, hidden_2]

#calculate output layer

output_1 = sum([hidden_layer[i] * weights_hidden_to_output[0][i] for i in range(len(weights_hidden_to_output))]) + bias_output_1
output_2 = sum([hidden_layer[i] * weights_hidden_to_output[1][i] for i in range(len(weights_hidden_to_output))]) + bias_output_2

#apply softmax
softmax_vals = softmax([output_1, output_2])
output_1 = softmax_vals[0]
output_2 = softmax_vals[1]
print("after training answer: ", output_1, output_2)

#final weights

print("weights: ", weights_input_to_hidden, weights_hidden_to_output)

