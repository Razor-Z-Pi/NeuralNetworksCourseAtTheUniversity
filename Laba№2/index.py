import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

# Загрузка входных данных
text = np.loadtxt("data_simple_nn.txt")

data = text[:, 0:2]
labels = text[:, 2:]

# Построение графика входных данных
plt.figure()
plt.scatter(data[:, 0], data[:,1])
plt.xlabel("Размерность 1")
plt.ylabel("Размерность 2")
plt.title("Входные данные")

# Минимальное и максимальное значения для каждого измерения
diml_min, diml_max = data[:,0].min(), data[:, 0].max ()
dim2_min, dim2_max = data[:, 1].min(), data[:,1].max ()

# Определение количества нейронов в выходном слое
num_output = labels.shape[1]

# Определение однослойной нейронной сети
diml = [diml_min, diml_max]
dim2 = [dim2_min, dim2_max]
nn = nl.net.newp([diml, dim2], num_output)


# Обучение нейронной сети
error_progress = nn.train(data, labels, epochs = 100, show = 20, lr = 0.03)

# Построение графика продвижения процесса обучения
plt.figure()
plt.plot(error_progress)
plt.xlabel("Количество эпох")
plt.ylabel("Ошибка обучения")
plt.title("Изменение ошибки обучения")
plt.grid()

plt.show()

# Выполнение классификатора на тестовых точках данных
print('\nTest results:')
data_test = [[0.4, 4.3] , [4.4, 0.6], [4.7, 8.1]]
for item in data_test:
    print(item, '-->', nn.sim([item]) [0])
