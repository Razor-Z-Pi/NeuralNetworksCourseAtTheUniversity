import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-x))

# XOR
training_inputs = np.array([[0, 0],
                            [0, 1],
                            [1, 0],
                            [1, 1]])

training_outputs = np.array([[0], [1], [1], [0]])

np.random.seed(1)
synaptic_weights = 2 * np.random.random((2,1)) - 1

for i in range(20000):
    input_layer = training_inputs
    outputs = sigmoid(np.dot(input_layer, synaptic_weights))
    err = training_outputs - outputs
    adjustments = np.dot(input_layer.T, err * outputs * (1 - outputs))  # производная сигмоиды
    synaptic_weights += adjustments

print("Веса после обучения на XOR:")
print(synaptic_weights)

print("\nРезультаты после обучения:")
for inp in training_inputs:
    out = sigmoid(np.dot(inp, synaptic_weights))
    print(f"{inp} => {out[0]:.4f}")

print("\nПроверка новых примеров:")
new_inputs = np.array([[0,0], [1,1], [0,1], [1,0]])
for inp in new_inputs:
    out = sigmoid(np.dot(inp, synaptic_weights))
    print(f"{inp} => {out[0]:.4f}")