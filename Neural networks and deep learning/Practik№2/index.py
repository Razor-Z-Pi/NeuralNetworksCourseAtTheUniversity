import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

training_inputs = np.array ([[0,0,1],
                            [1,1,1],
                            [1,0,1],
                            [0,1,1]])

training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)
synaptic_weights = 2 * np.random.random((3,1)) - 1

for i in range(10000):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    err = training_outputs - outputs
    adjustments = np.dot(input_layer.T, err * (outputs))
    synaptic_weights += adjustments

print ("веса после обучения")
print(synaptic_weights)

print("result")
print(outputs)

print("Проверка работоспособности нейросети на незнакомом примере!!!")
new_input = np.array([[0, 0, 1]])
print(sigmoid(np.dot(new_input, synaptic_weights)))