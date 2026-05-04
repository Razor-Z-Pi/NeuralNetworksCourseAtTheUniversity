import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

def generate_custom_signals(num_points, num_signals=4):
    """
    Генерация произвольных синусоидальных сигналов с разными параметрами
    
    Параметры:
    - num_points: количество точек в каждом сигнале
    - num_signals: количество различных сигналов
    """
    signals = []
    amplitudes = []
    
    np.random.seed(42)  # Для воспроизводимости результатов
    
    for i in range(num_signals):
        # Генерация случайных параметров синусоиды
        amplitude = np.random.uniform(0.5, 5.0)      # Амплитуда от 0.5 до 5
        frequency = np.random.uniform(0.5, 2.0)      # Частота от 0.5 до 2
        phase = np.random.uniform(0, 2 * np.pi)       # Фаза от 0 до 2pi
        
        # Создание синусоидального сигнала
        t = np.arange(0, num_points) / 10  # Временная шкала
        signal = amplitude * np.sin(frequency * t + phase)
        
        # Целевая амплитуда (постоянная для каждого сигнала)
        target_amp = amplitude * np.ones(num_points)
        
        signals.append(signal)
        amplitudes.append(target_amp)
        
        print(f"Сигнал {i + 1}: амплитуда = {amplitude:.2f}, частота = {frequency:.2f}, фаза = {phase:.2f}")
    
    # Объединение данных
    signal_data = np.array(signals).reshape(num_points * num_signals, 1)
    amp_data = np.array(amplitudes).reshape(num_points * num_signals, 1)
    
    return signal_data, amp_data

def plot_signals_3d(signals_list, title = "Сгенерированные сигналы"):
    """3D визуализация всех сгенерированных сигналов"""
    fig = plt.figure(figsize = (12, 6))
    ax = fig.add_subplot(111, projection = '3d')
    
    for i, signal in enumerate(signals_list):
        t = np.arange(len(signal))
        ax.plot(t, np.ones(len(signal)) * i, signal, label = f'Сигнал {i + 1}')
    
    ax.set_xlabel('Время')
    ax.set_ylabel('Номер сигнала')
    ax.set_zlabel('Амплитуда')
    ax.set_title(title)
    ax.legend()
    return fig

def train_and_visualize(signal_data, amp_data, num_epochs=1200):
    """Обучение RNN и визуализация результатов"""
    # Создание рекуррентной сети
    nn = nl.net.newelm([[-2, 2]], [15, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
    
    nn.layers[0].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
    nn.layers[1].initf = nl.init.InitRand([-0.1, 0.1], "wb")
    nn.init()
    
    print("\nНачало обучения RNN...")
    error_progress = nn.train(signal_data, amp_data, epochs=num_epochs, show = 200, goal = 0.005)
    print("Обучение завершено!!!")
    
    # Предсказание
    predictions = nn.sim(signal_data)
    
    return nn, error_progress, predictions

def plot_results(error_progress, original, predictions, title_prefix=""):
    """Визуализация результатов обучения и предсказаний"""
    fig, axes = plt.subplots(2, 2, figsize = (14, 10))
    
    # График ошибки обучения
    axes[0, 0].plot(error_progress, 'b-', linewidth = 2)
    axes[0, 0].set_xlabel('Эпоха обучения')
    axes[0, 0].set_ylabel('Среднеквадратичная ошибка (MSE)')
    axes[0, 0].set_title(f'{title_prefix}Прогресс обучения RNN')
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True)
    
    # Сравнение оригинал против предсказание (все данные)
    axes[0, 1].plot(original, 'b-', label = 'Оригинальный сигнал', alpha = 0.7)
    axes[0, 1].plot(predictions, 'r--', label = 'Предсказание RNN', alpha = 0.7)
    axes[0, 1].set_xlabel('Номер образца')
    axes[0, 1].set_ylabel('Амплитуда')
    axes[0, 1].set_title(f'{title_prefix}Сравнение на всех данных')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Участок с первыми 100 точками (для детального рассмотрения)
    axes[1, 0].plot(original[:100], 'b-', label = 'Оригинал', linewidth = 2)
    axes[1, 0].plot(predictions[:100], 'r--', label = 'Предсказание', linewidth = 2)
    axes[1, 0].set_xlabel('Номер образца')
    axes[1, 0].set_ylabel('Амплитуда')
    axes[1, 0].set_title(f'{title_prefix}Детальный вид (первые 100 точек)')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Гистограмма ошибок
    errors = original.flatten() - predictions.flatten()
    axes[1, 1].hist(errors, bins = 30, edgecolor = 'black', alpha = 0.7)
    axes[1, 1].set_xlabel('Ошибка предсказания')
    axes[1, 1].set_ylabel('Частота')
    axes[1, 1].set_title(f'{title_prefix}Распределение ошибок (MSE = {np.mean(errors**2):.4f})')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    return fig

print("_" * 70)
print("Эксперемент № 1: Обучение на синусоидальных сигналах")
print("_" * 70)
    
# Генерация сигналов для обучения
num_points_train = 50
signal_train, amp_train = generate_custom_signals(num_points_train, num_signals = 4)

# Визуализация сгенерированных сигналов
signals_list = signal_train.reshape(4, num_points_train)
plot_signals_3d(signals_list, "Сгенерированные синусоидальные сигналы (обучение)")
    
# Обучение RNN
nn, error_progress, predictions = train_and_visualize(signal_train, amp_train, num_epochs = 1500)
plot_results(error_progress, amp_train, predictions, "Обучение: ")
    
print("\n" + "_" * 70)
print("Эксперемент № 2: Тестирование на новых сигналах")
print("_" * 70)
    
# Генерация новых сигналов для тестирования
num_points_test = 40
signal_test, amp_test = generate_custom_signals(num_points_test, num_signals = 3)
test_predictions = nn.sim(signal_test)
    
# Визуализация тестовых сигналов
test_signals_list = signal_test.reshape(3, num_points_test)
plot_signals_3d(test_signals_list, "Тестовые синусоидальные сигналы")
plot_results([], amp_test, test_predictions, "Тестирование: ")
    
# Сравнение обучения на разных количествах эпох
print("\n" + "_" * 70)
print("Эксперемент № 3: Сравнение при разном количестве эпох")
print("_" * 70)
    

fig_comparison, axes = plt.subplots(1, 2, figsize = (12, 5))
    
epochs_list = [500, 1000, 1500]
colors = ['red', 'green', 'blue']
    
for epochs, color in zip(epochs_list, colors):
    # Быстрое обучение с разным количеством эпох
    nn_temp = nl.net.newelm([[-2, 2]], [10, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
    nn_temp.layers[0].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
    nn_temp.layers[1].initf = nl.init.InitRand([-0.1, 0.1], "wb")
    nn_temp.init()
    
    progress = nn_temp.train(signal_train, amp_train, epochs = epochs, show = epochs, goal = 0.01)
    axes[0].plot(progress, color = color, label = f'{epochs} эпох', linewidth = 2)
    
axes[0].set_xlabel('Эпоха')
axes[0].set_ylabel('MSE')
axes[0].set_title('Сравнение скорости обучения')
axes[0].set_yscale('log')
axes[0].legend()
axes[0].grid(True)
    
# Финальная ошибка при разных размерах скрытого слоя
hidden_sizes = [5, 10, 15, 20]
final_errors = []
    
for hidden in hidden_sizes:
    nn_temp = nl.net.newelm([[-2, 2]], [hidden, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
    nn_temp.init()
    progress = nn_temp.train(signal_train, amp_train, epochs = 800, show = 800, goal = 0.01)
    final_errors.append(progress[-1] if len(progress) > 0 else 1.0)
    
axes[1].bar(hidden_sizes, final_errors, color = 'skyblue', edgecolor = 'black')
axes[1].set_xlabel('Количество нейронов в скрытом слое')
axes[1].set_ylabel('Финальная ошибка (MSE)')
axes[1].set_title('Влияние размера скрытого слоя на ошибку')
axes[1].grid(True, axis = 'y')
    
plt.tight_layout()
plt.show()
    
print("\n" + "_" * 70)
print("Результаты эксперементов:")
print("_" * 70)
print(f"1) Финальная ошибка обучения: {error_progress[-1]:.6f}")
print(f"2) RNN успешно обучилась предсказывать амплитуды синусоидальных сигналов")
print(f"3) Качество предсказания на тестовых данных: MSE = {np.mean((amp_test.flatten() - test_predictions.flatten())**2):.6f}")
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    