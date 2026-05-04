import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl
import os

# Проверка наличия файла и его создание при необходимости
file_name = 'data_vector_quantization.txt'

if not os.path.exists(file_name):
    print(f"Файл {file_name} не найден. Создаю...")
    np.random.seed(42)
    data_list = []
    for i in range(200):
        x = np.random.uniform(0, 10)
        y = np.random.uniform(0, 10)
        if x < 5 and y < 5:
            label = [1, 0, 0, 0]
        elif x >= 5 and y < 5:
            label = [0, 1, 0, 0]
        elif x < 5 and y >= 5:
            label = [0, 0, 1, 0]
        else:
            label = [0, 0, 0, 1]
        data_list.append([x, y] + label)
    data_array = np.array(data_list)
    np.savetxt(file_name, data_array)
    print(f"Файл {file_name} создан")

# Загрузка входных данных
text = np.loadtxt('data_vector_quantization.txt')

# Разделение на данные и метки
data = text[:, 0:2]
labels = text[:, 2:]

print(f"Загружено точек: {len(data)}")
print(f"Размер данных: {data.shape}")
print(f"Размер меток: {labels.shape}")

# Создаем нейронную сеть с 2 входами и 4 выходами
nn = nl.net.newp(nl.tool.minmax(data), 4)

# Обучение нейронной сети
print("Начало обучения...")
error = nn.train(data, labels, epochs = 500, goal = 0.01, show = 50)
print("Обучение завершено")

# Создание входной сетки
xx, yy = np.meshgrid(np.arange(0, 10, 0.2), np.arange(0, 10, 0.2))
xx.shape = xx.size, 1
yy.shape = yy.size, 1
grid_xy = np.concatenate((xx, yy), axis = 1)

# Оценка входной сетки точек
grid_eval = nn.sim(grid_xy)

# Используем argmax для определения класса
grid_classes = np.argmax(grid_eval, axis = 1)

# Определение 4 классов из исходных данных
class_1 = data[labels[:,0] == 1]
class_2 = data[labels[:,1] == 1]
class_3 = data[labels[:,2] == 1]
class_4 = data[labels[:,3] == 1]

# Определение сеток x-y для всех 4 классов
grid_1 = grid_xy[grid_classes == 0]
grid_2 = grid_xy[grid_classes == 1]
grid_3 = grid_xy[grid_classes == 2]
grid_4 = grid_xy[grid_classes == 3]

plt.figure(figsize = (10, 8))

# Исходные данные (черные точки)
plt.plot(class_1[:,0], class_1[:,1], 'ko', markersize = 6, label='Класс 1')
plt.plot(class_2[:,0], class_2[:,1], 'ko', markersize = 6, label='Класс 2')
plt.plot(class_3[:,0], class_3[:,1], 'ko', markersize = 6, label='Класс 3')
plt.plot(class_4[:,0], class_4[:,1], 'ko', markersize = 6, label='Класс 4')

# Результаты классификации
if len(grid_1) > 0:
    plt.plot(grid_1[:,0], grid_1[:,1], 'm.', markersize = 2, label='Область класса 1')
if len(grid_2) > 0:
    plt.plot(grid_2[:,0], grid_2[:,1], 'bx', markersize = 2, label='Область класса 2')
if len(grid_3) > 0:
    plt.plot(grid_3[:,0], grid_3[:,1], 'c^', markersize = 2, label='Область класса 3')
if len(grid_4) > 0:
    plt.plot(grid_4[:,0], grid_4[:,1], 'y+', markersize = 2, label='Область класса 4')

plt.axis([0, 10, 0, 10])
plt.xlabel('Измерение 1')
plt.ylabel('Измерение 2')
plt.title('Векторное квантование (Персептрон)')
plt.legend()
plt.grid(True, alpha = 0.3)

plt.show()

# Вывод точности классификации
predictions = nn.sim(data)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(labels, axis = 1)
accuracy = np.mean(predicted_classes == actual_classes)
print(f"\nТочность на обучающих данных: {accuracy * 100:.2f}%")