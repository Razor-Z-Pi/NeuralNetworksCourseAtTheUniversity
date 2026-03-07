import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных из файла
data = np.loadtxt("data_perceptron.txt")
x = data[:, :2]  # Признаки
y = data[:, 2]   # Метки (0 или 1)

# Параметры обучения
learning_rate = 0.01
n_iterations = 20
n_samples, n_features = x.shape

# Инициализация весов и смещения
weights = np.random.randn(n_features) * 0.01
bias = 0

# Функция активации
def activation(x):
    return np.where(x >= 0, 1, 0)

# Список для хранения ошибок на каждой эпохе
errors_history = []

# Обучение перцептрона
for epoch in range(n_iterations):
    errors = 0
    
    for idx in range(n_samples):
        # Линейная комбинация
        linear_output = np.dot(x[idx], weights) + bias
        
        # Предсказание
        y_predicted = activation(linear_output)
        
        # Правило обновления перцептрона
        update = learning_rate * (y[idx] - y_predicted)
        weights += update * x[idx]
        bias += update
        
        # Считаем ошибки (а именно количество неверных предсказаний)
        if update != 0:
            errors += 1
    
    errors_history.append(errors)
    print(f"Эпоха {epoch + 1}: ошибок = {errors}")

print("\n" + "*" * 40)
print("Вывод результата обучения:")
print(f"Веса: [{weights[0]:.4f}, {weights[1]:.4f}]")
print(f"Смещение: {bias:.4f}")

# Проверка точности на обучающих данных
predictions = activation(np.dot(x, weights) + bias)
accuracy = np.mean(predictions == y)
print(f"Точность на обучающих данных: {accuracy * 100:.2f}%")

# Визуализация динамики ошибки
plt.figure(figsize = (12, 5))

# График № 1: Изменение ошибки по эпохам
plt.subplot(1, 2, 1)
plt.plot(range(1, len(errors_history) + 1), errors_history, marker = 'o', 
         color = 'blue', linewidth = 2, markersize = 8)
plt.xlabel('Эпохи')
plt.ylabel('Количество ошибок')
plt.title('Изменение ошибки обучения по эпохам')
plt.grid(True, alpha = 0.3)
plt.xticks(range(1, n_iterations + 1))

# График № 2: Визуализация данных и разделяющей прямой
plt.subplot(1, 2, 2)

# Отображение точек данных с цветовой кодировкой классов
for i in range(n_samples):
    if y[i] == 1:
        plt.scatter(x[i, 0], x[i, 1], color = 'red', marker = 'o', s = 100, 
                   edgecolors = 'black', linewidth = 2, label = 'Класс 1' if i == 0 else '') # Тернарный оператор удобно в поисках условия первой точки в массиве данных не посредственно в самом методе
    else:
        plt.scatter(x[i, 0], x[i, 1], color = 'blue', marker = 's', s = 100, 
                   edgecolors = 'black', linewidth = 2, label = 'Класс 0' if i == 0 else '')

# Построение разделяющей прямой (где w1*x1 + w2*x2 + b = 0)!!!
x_min, x_max = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
x_line = np.linspace(x_min, x_max, 100)
if weights[1] != 0:  # Избегаем деления на ноль
    y_line = -(weights[0] * x_line + bias) / weights[1]
    plt.plot(x_line, y_line, 'g--', linewidth = 2, label = 'Разделяющая прямая')

plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title('Визуализация данных и разделяющей прямой')
plt.legend(loc = 'upper left')
plt.grid(True, alpha = 0.3)
plt.xlim([x_min, x_max])
plt.ylim([x[:, 1].min() - 0.1, x[:, 1].max() + 0.1])

plt.tight_layout()
plt.show()

# Дополнительно: эксперимент с разным количеством эпох
print("\n" + "*" * 40)
print("Эксперемент: влияние количества эпох на ошибку")

epochs_to_test = [1, 2, 3, 5, 10, 20, 50]
results = []

for n_epoch in epochs_to_test:
    # Сброс весов
    test_weights = np.random.randn(n_features) * 0.01
    test_bias = 0
    
    for epoch in range(n_epoch):
        epoch_errors = 0
        for idx in range(n_samples):
            linear_output = np.dot(x[idx], test_weights) + test_bias
            y_predicted = activation(linear_output)
            update = learning_rate * (y[idx] - y_predicted)
            test_weights += update * x[idx]
            test_bias += update
            if update != 0:
                epoch_errors += 1
    
    # Проверка точности
    test_predictions = activation(np.dot(x, test_weights) + test_bias)
    test_accuracy = np.mean(test_predictions == y)
    results.append((n_epoch, epoch_errors, test_accuracy * 100))

print("Эпохи | Ошибки на последней эпохе | Точность (%)")
print("*" * 45)
for n_epoch, err, acc in results:
    print(f"{n_epoch:6d} | {err:23d} | {acc:11.2f}%")