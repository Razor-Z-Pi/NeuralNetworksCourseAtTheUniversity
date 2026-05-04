import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

min_val = -15         
max_val = 15           
num_points = 130      
x = np.linspace(min_val, max_val, num_points)
y = 3 * np.square(x) + 5
y /= np.linalg.norm(y)

# Формируем данные для обучения
data = x.reshape(num_points, 1)   
labels = y.reshape(num_points, 1) 

plt.figure(figsize = (8, 6))
plt.scatter(data, labels, alpha = 0.7, c = 'blue', edgecolors = 'k')
plt.xlabel('Размер 1 (x)')
plt.ylabel('Размерность 2 (y)')
plt.title('Входные данные: y = 3*x² + 5 (нормализованные)')
plt.grid(True, alpha = 0.3)

nn = nl.net.newff([[min_val, max_val]], [10, 6, 1])

nn.trainf = nl.train.train_gd

nn.lr = 0.01

print("Начало обучения нейронной сети...")
error_progress = nn.train(
    data,           # Входные данные
    labels,         # Целевые значения
    epochs = 2000,    # Количество эпох обучения
    show = 100,       # Показывать ошибку каждые 100 эпох
    goal = 0.01       # Целевая ошибка (остановиться при достижении)
)
print("Обучение завершено!!!")

output = nn.sim(data)          # Прогоняем данные через сеть
y_pred = output.reshape(num_points)  # Приводим к плоскому виду

plt.figure(figsize = (8, 6))
plt.plot(error_progress, linewidth = 2, color = 'green')
plt.xlabel('Количество эпох')
plt.ylabel('Ошибка (MSE)')
plt.title('Прогресс ошибки обучения')
plt.grid(True, alpha = 0.3)
plt.yscale('log')  # Логарифмическая шкала для лучшей видимости
x_dense = np.linspace(min_val, max_val, num_points * 2)
y_dense_pred = nn.sim(x_dense.reshape(x_dense.size, 1)).reshape(x_dense.size)
plt.figure(figsize = (10, 7))
plt.plot(x_dense, y_dense_pred, '-', 
         linewidth = 2, color = 'red', label = 'Прогнозируемая (плотная)')
plt.plot(x, y, ':', 
         linewidth = 2, color = 'blue', label = 'Истинные ценности')
plt.plot(x, y_pred, 'p', 
         markersize = 6, color = 'orange', label = 'Прогнозируемое значение (в обучающих точках)')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Фактические значения против прогнозируемых (аппроксимация с помощью нейронной сети)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
mse = np.mean((y - y_pred) ** 2)
print(f"\nФинальная среднеквадратичная ошибка (MSE): {mse:.6f}")
print(f"Целевая ошибка (goal): 0.01")
print(f"Достигнута ли цель? {'Да' if mse <= 0.01 else 'Нет'}")