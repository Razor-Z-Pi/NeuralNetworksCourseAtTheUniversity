import numpy as np
import matplotlib.pyplot as plt

# Создание обучающего набора данных с символами
symbols = {
    'L': np.array([[1, 0, 0],  # Определяем символ 'L' как 3x3 массив
                   [1, 0, 0],  # Вторая строка
                   [1, 1, 1]]),  # Третья строка
    'T': np.array([[1, 1, 1],  # Определяем символ 'T'
                   [0, 1, 0],  # Вторая строка
                   [0, 1, 0]]),  # Третья строка
    'X': np.array([[1, 0, 1],  # Определяем символ 'X'
                   [0, 1, 0],  # Вторая строка
                   [1, 0, 1]])  # Третья строка
}

# Функция для вычисления Хэммингова расстояния
def hamming_distance(a, b):
    return np.sum(a != b)

# Функция для распознавания символа
def recognize_symbol(input_symbol):
    distances = {}
    for symbol, template in symbols.items():
        distances[symbol] = hamming_distance(input_symbol, template)

    # Находим символ с минимальным расстоянием
    recognized_symbol = min(distances, key = distances.get)
    return recognized_symbol

input_symbol = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 1, 0]])
recognized = recognize_symbol(input_symbol)
print(f'Распознанный символ: {recognized}')

# Визуализация
fig, ax = plt.subplots()  # Создаем фигуру и оси для графика
ax.imshow(input_symbol, cmap = 'gray', vmin = 0, vmax = 1)  # Отображаем входной символ с цветовой картой
ax.set_title('Входной символ')
plt.show()