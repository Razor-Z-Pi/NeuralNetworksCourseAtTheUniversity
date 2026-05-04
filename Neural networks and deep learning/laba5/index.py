import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

def get_data(num_points):
    """Генерация синусоидальных волн с разными амплитудами"""
    # Создаём синусоидальные волны с разной амплитудой
    wave_1 = 0.5 * np.sin(np.arange(0, num_points))
    wave_2 = 3.6 * np.sin(np.arange(0, num_points))
    wave_3 = 1.1 * np.sin(np.arange(0, num_points))
    wave_4 = 4.7 * np.sin(np.arange(0, num_points))

    # Создаём соответствующие амплитуды (целевые значения)
    amp_1 = np.ones(num_points)
    amp_2 = 2.1 + np.zeros(num_points)
    amp_3 = 3.2 * np.ones(num_points)
    amp_4 = 0.8 + np.zeros(num_points)

    # Объединяем данные в массивы для обучения
    wave = np.array([wave_1, wave_2, wave_3, wave_4]).reshape(num_points * 4, 1)
    amp = np.array([[amp_1, amp_2, amp_3, amp_4]]).reshape(num_points * 4, 1)

    return wave, amp

def visualize_output(nn, num_points_test):
    """Визуализация работы сети на тестовых данных"""
    wave, amp = get_data(num_points_test)
    output = nn.sim(wave)  # Прогон данных через сеть
    plt.plot(amp.reshape(num_points_test * 4))
    plt.plot(output.reshape(num_points_test * 4))

if __name__ == '__main__':
    num_points = 40
    wave, amp = get_data(num_points)

    nn = nl.net.newelm([[-2, 2]], [10, 1], [nl.trans.TanSig(), nl.trans.PureLin()])

    nn.layers[0].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
    nn.layers[1].initf = nl.init.InitRand([-0.1, 0.1], "wb")
    nn.init()
    
    error_progress = nn.train(wave, amp, epochs = 1200, show = 100, goal = 0.01)

    output = nn.sim(wave)

    plt.figure(1)
    
    plt.subplot(211)
    plt.plot(error_progress)
    plt.xlabel('Количество эпох')
    plt.ylabel('Ошибка (MSE)')
    plt.title('График 1а: Снижение ошибки в процессе обучения')
    plt.grid(True)

    plt.subplot(212)
    plt.plot(amp.reshape(num_points * 4), label = 'Оригинал', linewidth = 2)
    plt.plot(output.reshape(num_points * 4), label = 'Предсказание', linestyle = '--')
    plt.legend()
    plt.xlabel('Номер образца')
    plt.ylabel('Амплитуда')
    plt.title('График 1б: Результаты на обучающей выборке')
    plt.grid(True)

    plt.figure(2)
    
    plt.subplot(211)
    visualize_output(nn, 82)
    plt.xlim([0, 300])
    plt.xlabel('Номер образца')
    plt.ylabel('Амплитуда')
    plt.title('График 2а: Тест на 82 точках (оригинал — синий, предсказание — оранжевый)')
    plt.legend(['Оригинал', 'Предсказание'])
    plt.grid(True)

    plt.subplot(212)
    visualize_output(nn, 49)
    plt.xlim([0, 300])
    plt.xlabel('Номер образца')
    plt.ylabel('Амплитуда')
    plt.title('График 2б: Тест на 49 точках (оригинал — синий, предсказание — оранжевый)')
    plt.legend(['Оригинал', 'Предсказание'])
    plt.grid(True)

    plt.tight_layout()
    plt.show()