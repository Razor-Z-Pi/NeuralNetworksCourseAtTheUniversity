import numpy as np
import matplotlib.pyplot as plt

class MultilayerNeuralNetwork:  
    def __init__(self, layer_sizes, learning_rate=0.01):
        """
        Инициализация сети
        
        Параметры:
        layer_sizes: список размеров слоев [входной_слой, скрытый1, скрытый2, ..., выходной_слой]
        learning_rate: скорость обучения
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []      # Список матриц весов
        self.biases = []       # Список векторов смещений
        
        # Инициализация весов и смещений случайными значениями
        for i in range(len(layer_sizes) - 1):
            # Инициализация весов по методу Ксавье (Xavier initialization)
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i + 1]))
            w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i + 1]))
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        """Сигмоидная функция активации"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # clip для избежания переполнения
    
    def sigmoid_derivative(self, x):
        """Производная сигмоидной функции"""
        return x * (1 - x)
    
    def forward(self, X):
        """
        Прямой проход
        
        Возвращает:
        activations: активации всех слоев
        """
        self.activations = [X]  # Сохраняем активации каждого слоя
        
        for i in range(len(self.weights)):
            # Линейное преобразование
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            # Функция активации
            a = self.sigmoid(z)
            self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, y_true):
        """
        Обратный проход (вычисление градиентов)
        
        Возвращает:
        gradients_weights, gradients_biases
        """
        m = y_true.shape[0]  # Количество примеров
        gradients_weights = []
        gradients_biases = []
        
        # Градиент для выходного слоя
        delta = self.activations[-1] - y_true
        
        # Проход от выходного слоя к входному
        for i in range(len(self.weights) - 1, -1, -1):
            # Градиент для текущего слоя
            grad_w = np.dot(self.activations[i].T, delta) / m
            grad_b = np.sum(delta, axis = 0, keepdims = True) / m
            
            gradients_weights.insert(0, grad_w)
            gradients_biases.insert(0, grad_b)
            
            if i > 0:
                # Распространение ошибки на предыдущий слой
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.activations[i])
        
        return gradients_weights, gradients_biases
    
    def update_weights(self, gradients_weights, gradients_biases):
        """Обновление весов и смещений"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_weights[i]
            self.biases[i] -= self.learning_rate * gradients_biases[i]
    
    def train(self, X, y, epochs = 1000, batch_size = 32, verbose = True):
        """
        Обучение сети методом градиентного спуска
        
        Возвращает:
        losses: история ошибок
        """
        n_samples = X.shape[0]
        losses = []
        
        for epoch in range(epochs):
            # Перемешивание данных
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            # Мини-пакетное обучение
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Прямой проход
                predictions = self.forward(X_batch)
                
                # Вычисление ошибки (MSE)
                loss = np.mean((predictions - y_batch) ** 2)
                epoch_loss += loss * len(X_batch)
                
                # Обратный проход
                gradients_w, gradients_b = self.backward(y_batch)
                
                self.update_weights(gradients_w, gradients_b)
            
            # Средняя ошибка за эпоху
            epoch_loss /= n_samples
            losses.append(epoch_loss)
            
            if verbose and (epoch + 1) % 100 == 0:
                print(f"Эпоха {epoch + 1} / {epochs}, Ошибка: {epoch_loss:.6f}")
        
        return losses
    
    def predict(self, X):
        """Предсказание"""
        return self.forward(X)

def generate_dataset(func, min_val, max_val, num_points, noise = 0):
    """
    Генерация набора данных на основе произвольной функции
    
    Параметры:
    func: функция y = f(x)
    min_val, max_val: диапазон x
    num_points: количество точек
    noise: уровень шума
    """
    x = np.linspace(min_val, max_val, num_points)
    y = func(x)
    
    # Добавление шума
    if noise > 0:
        y += np.random.normal(0, noise, num_points)
    
    # Нормализация
    y = (y - np.mean(y)) / np.std(y)
    
    return x, y

# Эксперемент № 1: Сложная периодическая функция
def complex_function1(x):
    return 2 * np.sin(0.5 * x) + 0.5 * np.sin(2 * x) + 0.3 * np.sin(5 * x)


# Эксперемент № 2: Гауссова функция с двумя пиками
def complex_function2(x):
    return 3 * np.exp(-((x + 3) ** 2) / 20) + 2 * np.exp(-((x - 5) ** 2) / 15)


# Эксперемент № 3: Полиномиальная функция (Полином 4-й степени с осцилляциями)
def complex_function3(x):
    return 0.001 * x**4 - 0.05 * x**3 - 0.5 * x**2 + 2 * x + 1


def train_and_visualize(func, func_name, min_val, max_val, num_points = 200, epochs = 3000):
    """
    Обучение нейронной сети на заданной функции и визуализация результатов
    """
    x, y = generate_dataset(func, min_val, max_val, num_points, noise = 5)
    
    # Подготовка данных
    X_train = x.reshape(-1, 1)
    y_train = y.reshape(-1, 1)
    
    # Создание сети
    # Тут будем экспериментировать с разными архитектурами :)
    architectures = [
        ([1, 5, 1], 0.05, "Малая сеть (5 нейронов)"),
        ([1, 10, 5, 1], 0.03, "Средняя сеть (10 -> 5 нейронов)"),
        ([1, 20, 10, 5, 1], 0.01, "Большая сеть (20 -> 10 -> 5 нейронов)")
    ]
    
    fig, axes = plt.subplots(2, 3, figsize = (15, 10))
    fig.suptitle(f'Эксперименты с функцией: {func_name}', fontsize = 16)
    
    for idx, (architecture, lr, desc) in enumerate(architectures):
        # Создание и обучение сети
        nn = MultilayerNeuralNetwork(architecture, learning_rate=lr)
        print(f"\n{'_'*60}")
        print(f"Обучение сети: {desc}")
        print(f"Архитектура: {architecture}")
        print(f"Скорость обучения: {lr}")
        
        losses = nn.train(X_train, y_train, epochs=epochs, batch_size = 16, verbose = False)
        
        # Предсказание
        x_dense = np.linspace(min_val - 2, max_val + 2, 500)
        X_dense = x_dense.reshape(-1, 1)
        y_dense_pred = nn.predict(X_dense)
        
        # Визуализация процесса обучения
        ax1 = axes[0, idx]
        ax1.plot(losses, linewidth = 2, color = 'green')
        ax1.set_xlabel('Эпоха')
        ax1.set_ylabel('Ошибка (MSE)')
        ax1.set_title(f'{desc}\nФинальная ошибка: {losses[-1]:.6f}')
        ax1.set_yscale('log')
        ax1.grid(True, alpha = 0.3)
        ax1.axhline(y = 0.01, color = 'r', linestyle = '-', alpha = 0.5)
        
        # Визуализация результатов
        ax2 = axes[1, idx]
        ax2.scatter(x, y, alpha = 0.4, s = 20, label = 'Фактические данные', color = 'blue')
        ax2.plot(x_dense, y_dense_pred, '-', linewidth = 2, color = 'red', label = 'Предсказание')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title(f'Аппроксимация функции')
        ax2.legend(fontsize = 8)
        ax2.grid(True, alpha = 0.3)
        
        # Вывод статистики
        y_pred = nn.predict(X_train)
        mse = np.mean((y_pred - y_train) ** 2)
        print(f"  Финальная MSE: {mse:.6f}")
        print(f"  Начальная MSE: {losses[0]:.6f}")
    
    plt.tight_layout()
    plt.show()
    
    return losses

print("\n" + "_" * 70)
print("Эксперемент № 1: Сложная периодическая функция")
print("_" * 70)
train_and_visualize(
    complex_function1, 
    "f(x) = 2sin(0.5x) + 0.5sin(2x) + 0.3sin(5x)", 
    -10, 10, 
    num_points = 300, 
    epochs = 3000
)
    

print("\n" + "_" * 70)
print("Эксперемент № 2: Гауссова функция с двумя пиками")
print("_" * 70)
train_and_visualize(
    complex_function2, 
    "f(x) = 3e^{-(x+3)^2/20} + 2e^{-(x-5)^2/15}", 
    -12, 15, 
    num_points = 250, 
    epochs = 3000
)
    

print("\n" + "_" * 70)
print("Эксперемент № 3: Полиномиальная функция 4-й степени")
print("_" * 70)
train_and_visualize(
    complex_function3, 
    "f(x) = 0.001x^4 - 0.05x^3 - 0.5x^2 + 2x + 1", 
    -15, 20, 
    num_points = 200, 
    epochs = 3000
)
    

print("\n" + "_" * 70)
print("Все эксперементы завершены")
print("_" * 70)
    

# Дополнительный эксперимент: сравнение разных функций активации
print("\n" + "_" * 70)
print("Доп. эксп.: Сравнение архитектур")
print("_" * 70)
    

# Тест на переобучение
x = np.linspace(-5, 5, 100)
y = np.sin(x) * np.cos(2 * x) + 0.1 * np.random.randn(100)
X = x.reshape(-1, 1)
y = y.reshape(-1, 1)
    

architectures_comparison = [
    ([1, 3, 1], 0.1, "Слишком простая"),
    ([1, 10, 1], 0.05, "Оптимальная"),
    ([1, 50, 50, 1], 0.01, "Склонная к переобучению")
]
    

plt.figure(figsize = (15, 5))
    

for idx, (arch, lr, desc) in enumerate(architectures_comparison):
    nn = MultilayerNeuralNetwork(arch, learning_rate = lr)
    losses = nn.train(X, y, epochs = 1000, batch_size = 16, verbose = False)
    
    plt.subplot(1, 3, idx + 1)
    plt.plot(losses, linewidth = 2)
    plt.xlabel('Эпоха')
    plt.ylabel('Ошибка')
    plt.title(f'{desc}\nАрхитектура: {arch}')
    plt.yscale('log')
    plt.grid(True, alpha = 0.3)
    

plt.tight_layout()
plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    