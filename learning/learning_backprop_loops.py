import math
import random

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

def cross_entropy_loss(output, label):
    actual_answer = [0, 1] if label == 1 else [1, 0]
    return -sum(actual_answer[i] * math.log(output[i] + 1e-15) for i in range(num_output_neurons))



#neural network structure:
# 3 input neurons
# 2 hidden neurons
# 2 output neurons

num_input_neurons = 3
num_hidden_neurons = 2
num_output_neurons = 2


inputs = [[0.4, 0.3, 0.8], [0.51, 0.2, 0.9], [0.2, 0.8, 0.9]]
labels = [0, 1, 1]

weights_input_to_hidden = [[random.uniform(-1, 1) for _ in range(num_hidden_neurons)] for _ in range(num_input_neurons)] # 2 x 3
weights_hidden_to_output = [[random.uniform(-1, 1) for _ in range(num_output_neurons)] for _ in range(num_hidden_neurons)] # 2 x 2

biases_hidden = [random.uniform(-1, 1) for _ in range(num_hidden_neurons)]
biases_output = [random.uniform(-1, 1) for _ in range(num_output_neurons)]


def forward_pass(input_layer):
    #calculate hidden layer
    hidden_layer_1_activations = []
    for i in range(num_input_neurons):
        weighted_sum = 0
        for j in range(num_hidden_neurons):
            weighted_sum += input_layer[i] * weights_input_to_hidden[i][j]
        hidden_layer_1_activations.append(weighted_sum)
    
    #biases and sigmoid for hidden layer
    for i in range(num_hidden_neurons):
        hidden_layer_1_activations[i] = sigmoid(hidden_layer_1_activations[i] + biases_hidden[i])
    
    #calculate output layer
    output_layer_1_activations = []
    for i in range(num_hidden_neurons):
        weighted_sum = 0
        for j in range(num_output_neurons):
            weighted_sum += hidden_layer_1_activations[i] * weights_hidden_to_output[i][j]
        output_layer_1_activations.append(weighted_sum)
    
    #biases for output layer
    for i in range(num_output_neurons):
        output_layer_1_activations[i] += biases_output[i]
    
    return hidden_layer_1_activations, softmax(output_layer_1_activations)
     

def back_pass(inputs, labels, iterations=1):
    for _ in range(iterations):
        for idx in range(len(inputs)):
            input_layer = inputs[idx]
            label = labels[idx]

            actual_answer = [0, 1] if label == 1 else [1, 0]

            hidden_layer_1_activations, output_layer_1_activations = forward_pass(input_layer)
            output_layer_deltas = [output_layer_1_activations[i] - actual_answer[i] for i in range(num_output_neurons)]

            #calculate hidden layer deltas
            hidden_layer_deltas = []
            for i in range(num_hidden_neurons):
                delta = 0
                for j in range(num_output_neurons):
                    delta += weights_hidden_to_output[i][j] * output_layer_deltas[j]
                delta *= sigmoid_derivative(hidden_layer_1_activations[i])
                hidden_layer_deltas.append(delta)

            
            #update weights and biases
            # we do weight -= (learning_rate * delta of where weight is going to * ( activation value of neuron where weight is COMING FROM))
            # and we do bias -= (learning_rate * delta) (we dont need to multiply by neuron value 
            # of where it is coming from, as a bias doesnt really come from a neuron technically)

            learning_rate = 0.01

            for i in range(num_input_neurons):
                for j in range(num_hidden_neurons):
                    weights_input_to_hidden[i][j] -= learning_rate * input_layer[i] * hidden_layer_deltas[j]
            for i in range(num_hidden_neurons):
                biases_hidden[i] -= learning_rate * hidden_layer_deltas[i]

            for i in range(num_hidden_neurons):
                for j in range(num_output_neurons):
                    weights_hidden_to_output[i][j] -= learning_rate * hidden_layer_1_activations[i] * output_layer_deltas[j]
            for i in range(num_output_neurons):
                biases_output[i] -= learning_rate * output_layer_deltas[i]

def evaluate_performance(inputs, labels):
    total_loss = 0
    for input_layer, label in zip(inputs, labels):
        _, output_layer_activations = forward_pass(input_layer)
        total_loss += cross_entropy_loss(output_layer_activations, label)
    return total_loss / len(inputs)

def evaluate_accuracy(inputs, labels):
    total_correct = 0
    for input_layer, label in zip(inputs, labels):
        _, output_layer_activations = forward_pass(input_layer)
        if output_layer_activations.index(max(output_layer_activations)) == label:
            total_correct += 1
    return total_correct / len(inputs)

#performace without backpass:

initial_loss = evaluate_performance(inputs, labels)
initial_accuracy = evaluate_accuracy(inputs, labels)
print(f"Initial loss: {initial_loss}")
print(f"Initial accuracy: {initial_accuracy}")

# Performance with backpass
back_pass(inputs, labels, 1000)

final_loss = evaluate_performance(inputs, labels)
final_accuracy = evaluate_accuracy(inputs, labels)
print(f"Final loss: {final_loss}")
print(f"Final accuracy: {final_accuracy}")







