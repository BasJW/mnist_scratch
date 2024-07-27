import pickle

def relu(x):
    return max(0, x)

def compute_neuron_values_relu(layer1_size, layer2_size, layer1, weights, biases):
    layer2 = []
    for i in range(layer2_size): # 16
        weighted_sum = 0
        for j in range(layer1_size): # 784
            weighted_sum += layer1[j] * weights[i][j]
        weighted_sum += biases[i]
        if layer2_size == 10: #if it is output, dont relu
            layer2.append(weighted_sum)
        else:
            layer2.append(relu(weighted_sum))
    return layer2


def classify(image):
    
    image = [image[i][j] for i in range(28) for j in range(28)]
    input_layer_size = 784
    hidden_layer_size = 16
    output_layer_size = 10
    with open('nn/weights_biases.pkl', 'rb') as f:
        weights_biases_g = pickle.load(f)

    weights_input_to_hidden1 = weights_biases_g['weights_input_to_hidden1']
    biases_hidden1 = weights_biases_g['biases_hidden1']
    weights_hidden1_to_hidden2 = weights_biases_g['weights_hidden1_to_hidden2']
    biases_hidden2 = weights_biases_g['biases_hidden2']
    weights_hidden2_to_output = weights_biases_g['weights_hidden2_to_output']
    biases_output = weights_biases_g['biases_output']


    hidden_layer_1 = compute_neuron_values_relu(input_layer_size,
                                        hidden_layer_size,
                                        image,
                                        weights_input_to_hidden1,
                                        biases_hidden1)


    hidden_layer_2 = compute_neuron_values_relu(hidden_layer_size,
                                        hidden_layer_size,
                                        hidden_layer_1,
                                        weights_hidden1_to_hidden2,
                                        biases_hidden2)


    output_layer = compute_neuron_values_relu(hidden_layer_size,
                                        output_layer_size,
                                        hidden_layer_2,
                                        weights_hidden2_to_output,
                                        biases_output)
    
    

    #get index of max value of output list
    #print(output_layer)
    return output_layer.index(max(output_layer))
    