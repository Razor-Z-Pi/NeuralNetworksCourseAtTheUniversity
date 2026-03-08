import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

print("*" * 60)
print("Эксперименты с однослойной нейронной сетью")
print("*" * 60)

# Загрузка данных
text = np.loadtxt("data_simple_nn.txt")
data = text[:, 0:2]
labels = text[:, 2:]

print(f"\nРазмерность данных:")
print(f"Количество образцов: {data.shape[0]}")
print(f"Количество признаков: {data.shape[1]}")
print(f"Количество выходных нейронов: {labels.shape[1]}")

# Анализ распределения по классам
class_counts = {}
for i in range(labels.shape[0]):
    class_tuple = tuple(labels[i].astype(int))
    class_counts[class_tuple] = class_counts.get(class_tuple, 0) + 1

print(f"\nРаспределение по классам:")
for class_type, count in class_counts.items():
    print(f"Класс {class_type}: {count} образцов")

plt.figure(figsize=(12, 8))

# Цвета для разных классов
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
class_colors = {}

for i in range(labels.shape[0]):
    class_tuple = tuple(labels[i].astype(int))
    if class_tuple not in class_colors:
        class_colors[class_tuple] = colors[len(class_colors) % len(colors)]
    
    plt.scatter(data[i, 0], data[i, 1], 
               color = class_colors[class_tuple], 
               marker = 'o', s = 150, edgecolors = 'black', linewidth = 2,
               label = f'Класс {class_tuple}' if class_tuple not in plt.gca().get_legend_handles_labels()[1] else '') # выводим первые данные из массива с помощью тернарной конструкции

plt.xlabel('Признак 1 (X1)', fontsize = 12)
plt.ylabel('Признак 2 (X2)', fontsize = 12)
plt.title('Визуализация входных данных по классам', fontsize = 14)
plt.legend(loc = 'upper left', fontsize = 10)
plt.grid(True, alpha = 0.3)

plt.show()

print("\n" + "*" * 60)
print("Эксп. № 1: Базовое обучение (lr = 0.03, epochs = 100)")
print("*" * 60)

# Определение сети
dim1_min, dim1_max = data[:,0].min(), data[:,0].max()
dim2_min, dim2_max = data[:,1].min(), data[:,1].max()
num_output = labels.shape[1]

nn = nl.net.newp([[dim1_min, dim1_max], [dim2_min, dim2_max]], num_output)

# Обучение
error_progress = nn.train(data, labels, epochs = 100, show = 20, lr = 0.03)

# График ошибки
plt.figure(figsize = (10, 6))
plt.plot(error_progress, 'b-', linewidth = 2)
plt.xlabel('Эпоха', fontsize = 12)
plt.ylabel('Ошибка обучения', fontsize = 12)
plt.title('Изменение ошибки обучения (базовый эксперимент)', fontsize = 14)
plt.grid(True, alpha = 0.3)
plt.xlim(0, len(error_progress))

plt.show()

print(f"\nФинальная ошибка: {error_progress[-1]}")

print("\n" + "*" * 60)
print("Эксп. № 2: Влияние скорости обучения")
print("*" * 60)

learning_rates = [0.001, 0.01, 0.03, 0.05, 0.1, 0.2]
lr_errors = {}

plt.figure(figsize = (12, 8))

for lr in learning_rates:
    nn_lr = nl.net.newp([[dim1_min, dim1_max], [dim2_min, dim2_max]], num_output)
    errors = nn_lr.train(data, labels, epochs = 50, show = 50, lr = lr)
    lr_errors[lr] = errors
    plt.plot(errors, 'o-', linewidth = 2, markersize = 4, label = f'lr = {lr}')

plt.xlabel('Эпоха', fontsize = 12)
plt.ylabel('Ошибка обучения', fontsize = 12)
plt.title('Рис. 3: Влияние скорости обучения на сходимость', fontsize = 14)
plt.legend(fontsize = 10)
plt.grid(True, alpha = 0.3)

plt.show()

print("\nФинальные ошибки при разных скоростях обучения:")
for lr in learning_rates:
    print(f"lr = {lr}: {lr_errors[lr][-1]}")

print("\n" + "*" * 60)
print("Эксп. 3: Влияние количества эпох")
print("*" * 60)

epochs_list = [10, 20, 30, 50, 100, 200]
epoch_errors = {}

plt.figure(figsize = (12, 8))

for n_epochs in epochs_list:
    nn_epoch = nl.net.newp([[dim1_min, dim1_max], [dim2_min, dim2_max]], num_output)
    errors = nn_epoch.train(data, labels, epochs = n_epochs, show = n_epochs, lr = 0.03)
    epoch_errors[n_epochs] = errors[-1]
    
    # Показывать только финальные значения для наглядности
    plt.plot(n_epochs, errors[-1], 'o', markersize = 10, label = f'{n_epochs} эпох' if n_epochs in [10, 50, 200] else '')

plt.xlabel('Количество эпох', fontsize = 12)
plt.ylabel('Финальная ошибка', fontsize = 12)
plt.title('Зависимость финальной ошибки от числа эпох', fontsize = 14)
plt.xscale('log')
plt.grid(True, alpha = 0.3)
plt.legend()

plt.show()

print("\nЗависимость ошибки от числа эпох:")
for n_epochs in epochs_list:
    print(f"{n_epochs} эпох: ошибка = {epoch_errors[n_epochs]}")

print("\n" + "*" * 60)
print("Тестирование на новых точках")
print("*" * 60)

# Обучаем лучшую модель
best_nn = nl.net.newp([[dim1_min, dim1_max], [dim2_min, dim2_max]], num_output)
best_nn.train(data, labels, epochs = 100, show = 100, lr = 0.03)

# Тестовые точки
test_points = [
    [2.0, 2.0],    # Центр
    [0.5, 4.5],    # Ближе к первому кластеру
    [4.5, 0.5],    # Ближе ко второму кластеру
    [4.5, 7.5],    # Ближе к третьему кластеру
    [7.5, 4.5],    # Ближе к четвертому кластеру
    [3.0, 3.0],    # Переходная зона
    [1.0, 1.0],    # Нижний левый угол
    [8.0, 8.0],    # Верхний правый угол
    [0.4, 4.3],    # Из data_simple_nn.txt
    [4.4, 0.6],    # Из data_simple_nn.txt
    [4.7, 8.1]     # Из data_simple_nn.txt
]

print("\nРезультаты классификации тестовых точек:")
print("*" * 50)
print(f"{'Точка (x1, x2)':>20} | {'Результат':>30}")
print("*" * 50)

# Визуализация с тестовыми точками
plt.figure(figsize = (14, 10))

# Исходные данные
for i in range(labels.shape[0]):
    class_tuple = tuple(labels[i].astype(int))
    color = class_colors.get(class_tuple, 'gray')
    plt.scatter(data[i, 0], data[i, 1], 
               color = color, marker = 'o', s = 150, alpha = 0.6,
               edgecolors = 'black', linewidth = 2)

# Тестовые точки
for idx, point in enumerate(test_points):
    result = best_nn.sim([point])[0]
    result_tuple = tuple(result.astype(int))
    color = class_colors.get(result_tuple, 'gray')
    plt.scatter(point[0], point[1], 
               color = color, marker = 'D', s = 200, 
               edgecolors = 'black', linewidth = 2,
               label = f'Тест {idx + 1}' if idx < 3 else '')
    
    # Вывод результатов
    result_str = f"Класс {result_tuple}"
    print(f"({point[0]:4.1f}, {point[1]:4.1f}) --> {result_str:>25}")

plt.xlabel('Признак 1 (X1)', fontsize = 12)
plt.ylabel('Признак 2 (X2)', fontsize = 12)
plt.title('Классификация тестовых точек', fontsize = 14)
plt.legend()
plt.grid(True, alpha = 0.3)

plt.show()

print("\n" + "*" * 60)
print("Статистический анализ (Будет 10 запусков)")
print("*" * 60)

final_errors = []
training_times = []

for i in range(10):
    import time
    start_time = time.time()
    
    nn_stat = nl.net.newp([[dim1_min, dim1_max], [dim2_min, dim2_max]], num_output)
    errors = nn_stat.train(data, labels, epochs = 100, show = 100, lr = 0.03)
    
    end_time = time.time()
    
    final_errors.append(errors[-1])
    training_times.append(end_time - start_time)
    
    print(f"Запуск {i + 1:2d}: финальная ошибка = {errors[-1]}, время = {training_times[-1]:.3f} сек")

print(f"\nСтатистика по 10 запускам:")
print(f"Средняя ошибка: {np.mean(final_errors):.2f}")
print(f"Минимальная ошибка: {np.min(final_errors)}")
print(f"Максимальная ошибка: {np.max(final_errors)}")
print(f"Среднее время обучения: {np.mean(training_times):.3f} сек")

print("\n" + "*" * 60)
print("Визуализация разделяющих поверхностей")
print("*" * 60)

# Создаем сетку для визуализации
x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

# Предсказания для каждой точки сетки
grid_points = np.c_[xx.ravel(), yy.ravel()]
predictions = []

for point in grid_points:
    pred = best_nn.sim([point])[0]
    # Преобразуем в одно число для визуализации (сумма битов)
    pred_class = np.sum(pred.astype(int) * np.array([2, 1]))  # Для 2-битного выхода
    predictions.append(pred_class)

predictions = np.array(predictions).reshape(xx.shape)

plt.figure(figsize = (14, 10))

# Контурный график
plt.contourf(xx, yy, predictions, alpha = 0.3, cmap = 'tab10')
plt.colorbar(label='Класс')

# Исходные данные
for i in range(labels.shape[0]):
    class_tuple = tuple(labels[i].astype(int))
    color = class_colors.get(class_tuple, 'gray')
    plt.scatter(data[i, 0], data[i, 1], 
               color = color, marker = 'o', s = 200,
               edgecolors = 'black', linewidth = 2)

plt.xlabel('Признак 1 (X1)', fontsize = 12)
plt.ylabel('Признак 2 (X2)', fontsize = 12)
plt.title('Разделяющие поверхности однослойной сети', fontsize = 14)
plt.grid(True, alpha = 0.3)

plt.show()