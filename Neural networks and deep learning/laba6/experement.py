import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

model = tf.keras.Sequential([
    keras.layers.Dense(units = 1, input_shape = [1])
])

model.compile(optimizer="sgd", loss="mean_squared_error")

# Данные: X -> Y (линейная зависимость)
# Тут можно заметить закономерность: Y = 5*X + 1?
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype = float)
ys = np.array([-4.0, 1.0, 6.0, 11.0, 16.0, 21.0], dtype = float)

print("Начинаем обучение...")
history = model.fit(xs, ys, epochs = 500, verbose = 0)

print("Обучение завершено!!!")

# Предсказание
prediction = model.predict(tf.constant([10.0]))
print(f"\nПредсказание для X = 10.0: {prediction[0][0]:.2f}")

weights = model.get_weights()
print(f"\nВеса модели:")
print(f"Weight (W): {weights[0][0][0]:.4f}")
print(f"Bias (b): {weights[1][0]:.4f}")
print(f"Формула: Y = {weights[0][0][0]:.4f} * X + {weights[1][0]:.4f}")

# Теоретическая формула для анализа данных
# Из точек видно: при X=0 -> Y=1, при X=1 -> Y=6, при X=2 -> Y=11
# То есть Y = 5*X + 1
print(f"\nТеоретическая формула: Y = 5.00 * X + 1.00")

# Эксперимент 1: Визуализация обучения
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
x_test = np.linspace(-2, 5, 100)
y_pred = model.predict(tf.constant(x_test), verbose = 0)
plt.scatter(xs, ys, color = 'red', label = 'Исходные данные', s = 100)
plt.plot(x_test, y_pred, 'b-', label = 'Предсказание модели', linewidth = 2)
plt.xlabel('X', fontsize = 12)
plt.ylabel('Y', fontsize = 12)
plt.title('Линия регрессии', fontsize = 14)
plt.legend()
plt.grid(True, alpha = 0.3)

# Эксперимент 2: Тестирование на новых значениях
test_values = np.array([-5, -3, 1.5, 5.5, 7, 12], dtype = float)
predictions = model.predict(tf.constant(test_values), verbose = 0)
print("\nЭксперимент: Предсказания для новых значений:")
for x, y_pred in zip(test_values, predictions):
    y_true = 5 * x + 1  # истинная формула
    print(f"X = {x:5.1f} => Предсказано: {y_pred[0]:6.2f} | Истинное: {y_true:6.2f} | Ошибка: {abs(y_pred[0]-y_true):6.2f}")

# Эксперимент 3: Анализ ошибки обучения
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'])
plt.xlabel('Эпоха', fontsize = 12)
plt.ylabel('Потери (MSE)', fontsize = 12)
plt.title('Кривая обучения', fontsize = 14)
plt.yscale('log')
plt.grid(True, alpha = 0.3)

# Эксперимент 4: Исследование влияния количества эпох
print("\nЭксперимент: Влияние количества эпох обучения:")
for epochs in [10, 50, 100, 500]:
    model_temp = tf.keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])
    model_temp.compile(optimizer = "sgd", loss = "mean_squared_error")
    model_temp.fit(xs, ys, epochs = epochs, verbose = 0)
    pred = model_temp.predict(tf.constant([10.0]), verbose = 0)
    weights_temp = model_temp.get_weights()
    print(f"Эпох: {epochs:3d} => Предсказание для X = 10: {pred[0][0]:7.2f} | "
          f"W = {weights_temp[0][0][0]:.4f}, b = {weights_temp[1][0]:.4f}")

# Эксперимент 5: Визуализация ошибки предсказания
plt.subplot(1, 3, 3)
errors = []
x_range = np.arange(-5, 15, 1)
for x in x_range:
    pred = model.predict(tf.constant([x]), verbose = 0)[0][0]
    true = 5 * x + 1
    errors.append(pred - true)
plt.bar(x_range, errors, alpha = 0.7, color = 'purple')
plt.axhline(y = 0, color = 'red', linestyle = '--', linewidth = 2)
plt.xlabel('X', fontsize = 12)
plt.ylabel('Ошибка предсказания', fontsize = 12)
plt.title('Абсолютная ошибка модели', fontsize = 14)
plt.grid(True, alpha = 0.3)

plt.tight_layout()
plt.show()

# Эксперимент 6: Сравнение с разными оптимизаторами
print("\nЭксперимент: Сравнение оптимизаторов:")
optimizers = ['sgd', 'adam', 'rmsprop']
for opt in optimizers:
    model_opt = tf.keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])
    model_opt.compile(optimizer = opt, loss = "mean_squared_error")
    history_opt = model_opt.fit(xs, ys, epochs = 200, verbose = 0)
    pred_opt = model_opt.predict(tf.constant([10.0]), verbose = 0)[0][0]
    print(f"{opt:7s} => Предсказание для X = 10: {pred_opt:.2f}, "
          f"Финальная ошибка: {history_opt.history['loss'][-1]:.6f}")

# Эксперимент 7: Анализ производительности
print("\nЭксперимент: Обучение с шумом в данных")
np.random.seed(42)
noise = np.random.normal(0, 1, len(xs))
ys_noisy = ys + noise

model_noisy = tf.keras.Sequential([keras.layers.Dense(units = 1, input_shape = [1])])
model_noisy.compile(optimizer = "sgd", loss = "mean_squared_error")
model_noisy.fit(xs, ys_noisy, epochs = 500, verbose = 0)

print("Данные без шума:   ", end = "")
print(f"W = {model.get_weights()[0][0][0]:.4f}, b = {model.get_weights()[1][0]:.4f}")
print("Данные с шумом:    ", end = "")
print(f"W = {model_noisy.get_weights()[0][0][0]:.4f}, b = {model_noisy.get_weights()[1][0]:.4f}")
print(f"Истинная формула: W = 5.0000, b = 1.0000")