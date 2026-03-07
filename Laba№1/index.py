import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# Входные данные
text = np.loadtxt("data_perceptron.txt")

# Разделение точек данных и меток
data = text[:, :2]
labels = text[:, 2].reshape((text.shape[0], 1))

plt.figure()
plt.scatter(data[:, 0], data[:, 1])
plt.xlabel("Размерность 1")
plt.ylabel("Размерность 2")
plt.title("Входные данные:")

# Макс. и мин. значение для измерений
dim1_min, dim1_max, dim2_min, dim2_max = 0, 1, 0, 1

# Количество нейронов в входном слое
num_output = labels.shape[1]

# Определения перцептрона с двумя входными нейронома (они двухмерные)
dim1 = [dim1_min, dim1_max]
dim2 = [dim2_min, dim2_max]
perceptron = nl.net.newp([dim1, dim2], num_output)

# Тренеруем перцептрон с использованием наших данных
error_progress = perceptron.train(data, labels, epochs = 100, show = 20, lr = 0.03)

plt.figure()
plt.plot(error_progress)
plt.xlabel("Количество эпох")
plt.ylabel("Ошибка обучения")
plt.title("Изменения ошибки обучения")
plt.grid()

plt.show()