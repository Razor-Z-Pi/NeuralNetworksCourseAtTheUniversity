import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
data = np.loadtxt("data_perceptron.txt")
x = data[:, :2]
y = data[:, 2]

print("data_perceptron.txt:")
print("*" * 50)
print("X1  |  X2  |  y")
print("-"*20)
for i in range(len(data)):
    print(f"{data[i, 0]:.2f}  {data[i, 1]:.2f}  {int(data[i, 2])}")
print("*" * 50)

# Параметры эксперимента
learning_rate = 0.1
max_epochs = 30

# Функция активации
def activation(x):
    return np.where(x >= 0, 1, 0)

# Функция для обучения с возвратом истории ошибок
def train_perceptron(x, y, learning_rate, n_epochs):
    n_samples, n_features = x.shape
    weights = np.random.randn(n_features) * 0.1
    bias = 0
    
    errors_history = []
    weights_history = []
    bias_history = []
    
    for epoch in range(n_epochs):
        epoch_errors = 0
        
        for idx in range(n_samples):
            linear_output = np.dot(x[idx], weights) + bias
            y_predicted = activation(linear_output)
            update = learning_rate * (y[idx] - y_predicted)
            
            weights += update * x[idx]
            bias += update
            
            if update != 0:
                epoch_errors += 1
        
        errors_history.append(epoch_errors)
        weights_history.append(weights.copy())
        bias_history.append(bias)
        
        if epoch < 5 or epoch % 5 == 4:  # Печатаем первые 5 эпох и каждую 5-ю
            print(f"Эпоха {epoch + 1:2d}: ошибок = {epoch_errors:2d}, веса = [{weights[0]:.3f}, {weights[1]:.3f}], bias = {bias:.3f}")
    
    return weights, bias, errors_history, weights_history, bias_history

# Обучение
print("\nПроцесс обучения:")
print("*" * 60)
final_weights, final_bias, errors, weights_hist, bias_hist = train_perceptron(x, y, learning_rate, max_epochs)

print("*" * 60)
print(f"Итоговые параметры:")
print(f"Веса: w1 = {final_weights[0]:.4f}, w2 = {final_weights[1]:.4f}")
print(f"Смещение: b = {final_bias:.4f}")

# Проверка точности
predictions = activation(np.dot(x, final_weights) + final_bias)
accuracy = np.mean(predictions == y)
print(f"Точность на обучающих данных: {accuracy * 100:.2f} %")

# Изменение ошибки по эпохам
plt.figure(figsize = (15, 10))

plt.subplot(2, 3, 1)
plt.plot(range(1, len(errors) + 1), errors, 'bo-', linewidth = 2, markersize = 4)
plt.axhline(y = 0, color = 'r', linestyle = '--', alpha = 0.5)
plt.xlabel('Эпоха')
plt.ylabel('Количество ошибок')
plt.title('Изменение ошибки обучения')
plt.grid(True, alpha = 0.3)
plt.xticks(range(1, max_epochs + 1, 2))
plt.xlim(0, max_epochs + 1)

plt.show()

# Исследование влияния скорости обучения
learning_rates = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
epochs_to_test = 20

plt.subplot(2, 3, 2)
for lr in learning_rates:
    _, _, errors_lr, _, _ = train_perceptron(x, y, lr, epochs_to_test) # использую "_" как временную переменную (заглушку) для записи и вывода, современный лаконичный способ написания чистого кода в python!!!
    plt.plot(range(1, len(errors_lr) + 1), errors_lr, 'o-', linewidth = 2, label = f'lr = {lr}')

plt.xlabel('Эпоха')
plt.ylabel('Количество ошибок')
plt.title('Влияние скорости обучения (learning rate)')
plt.legend()
plt.grid(True, alpha = 0.3)

plt.show()

# Исследование влияния начальной инициализации весов
plt.subplot(2, 3, 3)
n_experiments = 5
lr_fixed = 0.1

for exp in range(n_experiments):
    _, _, errors_init, _, _ = train_perceptron(x, y, lr_fixed, epochs_to_test)
    plt.plot(range(1, len(errors_init) + 1), errors_init, 'o-', linewidth = 2, 
             label = f'Эксперимент {exp + 1}')

plt.xlabel('Эпоха')
plt.ylabel('Количество ошибок')
plt.title('Влияние случайной инициализации весов')
plt.legend()
plt.grid(True, alpha = 0.3)

plt.show()

# Анализ сходимости для разного количества эпох
epoch_values = [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]
convergence_results = []

plt.subplot(2, 3, 4)

for n_epochs in epoch_values:
    weights_temp, bias_temp, errors_temp, _, _ = train_perceptron(x, y, 0.1, n_epochs)
    predictions_temp = activation(np.dot(x, weights_temp) + bias_temp)
    accuracy_temp = np.mean(predictions_temp == y) * 100
    final_error = errors_temp[-1] if errors_temp else 0
    convergence_results.append((n_epochs, final_error, accuracy_temp))

# График зависимости точности от числа эпох
epochs_list = [r[0] for r in convergence_results]
accuracy_list = [r[2] for r in convergence_results]

plt.plot(epochs_list, accuracy_list, 'go-', linewidth = 2, markersize = 4)
plt.xlabel('Количество эпох')
plt.ylabel('Точность (%)')
plt.title('Зависимость точности от числа эпох')
plt.grid(True, alpha = 0.3)
plt.xscale('log')

plt.show()

# Таблица результатов
print("\n" + "*" * 60)
print("Результаты эксперементов с разными эпохами:")
print("*" * 60)
print(f"{'Эпохи':>8} | {'Ошибки на последней эпохе':>25} | {'Точность':>10}")
print("*" * 60)
for n_epochs, final_err, acc in convergence_results:
    print(f"{n_epochs:8d} | {final_err:25d} | {acc:9.2f}%")