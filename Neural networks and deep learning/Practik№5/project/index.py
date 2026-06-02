"""
Импорт необходимых библиотек для создания веб-сервера и нейронной сети
"""
import numpy as np
from flask import Flask, render_template, request, jsonify
import json

app = Flask(__name__)

# Генерация датасетов (аналогично TensorFlow Playground)
def generate_dataset(dataset_type = 'circle', n_samples = 300, noise = 0.1):
    """
    Генерация различных наборов данных для экспериментов
    
    Параметры:
        dataset_type : str - тип датасета ('circle', 'xor', 'gaussian', 'spiral')
        n_samples : int - количество точек
        noise : float - уровень шума в данных
    
    Возвращает:
        X : np.array - координаты точек (n_samples, 2)
        y : np.array - метки классов (n_samples,)
    """
    np.random.seed(42)
    
    if dataset_type == 'circle':
        # Генерация концентрических окружностей
        n_per_class = n_samples // 2
        
        # Внутренний круг (класс 0)
        radius_inner = np.random.uniform(0, 0.4, n_per_class)
        angle_inner = np.random.uniform(0, 2 * np.pi, n_per_class)
        x1_inner = radius_inner * np.cos(angle_inner) + np.random.normal(0, noise, n_per_class)
        x2_inner = radius_inner * np.sin(angle_inner) + np.random.normal(0, noise, n_per_class)
        
        # Внешнее кольцо (класс 1)
        radius_outer = np.random.uniform(0.6, 1.0, n_per_class)
        angle_outer = np.random.uniform(0, 2 * np.pi, n_per_class)
        x1_outer = radius_outer * np.cos(angle_outer) + np.random.normal(0, noise, n_per_class)
        x2_outer = radius_outer * np.sin(angle_outer) + np.random.normal(0, noise, n_per_class)
        
        X = np.vstack([
            np.column_stack([x1_inner, x2_inner]),
            np.column_stack([x1_outer, x2_outer])
        ])
        y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
        
    elif dataset_type == 'xor':
        # Генерация XOR-подобного датасета
        n_per_quadrant = n_samples // 4
        
        # Четыре квадранта с чередующимися классами
        centers = [(0.5, 0.5), (-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5)]
        labels = [1, 1, 0, 0]
        
        X_list = []
        y_list = []
        
        for (cx, cy), label in zip(centers, labels):
            x1 = np.random.normal(cx, 0.15 + noise, n_per_quadrant)
            x2 = np.random.normal(cy, 0.15 + noise, n_per_quadrant)
            X_list.append(np.column_stack([x1, x2]))
            y_list.append(np.full(n_per_quadrant, label))
        
        X = np.vstack(X_list)
        y = np.hstack(y_list)
        
    elif dataset_type == 'gaussian':
        # Гауссовские облака (два кластера)
        n_per_class = n_samples // 2
        
        X1 = np.random.normal(0, 0.5 + noise, (n_per_class, 2))
        X2 = np.random.normal(1, 0.5 + noise, (n_per_class, 2))
        
        X = np.vstack([X1, X2])
        y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
        
    elif dataset_type == 'spiral':
        # Спиральный датасет
        n_per_class = n_samples // 2
        
        theta1 = np.random.uniform(0, 2 * np.pi, n_per_class)
        r1 = np.linspace(0, 1, n_per_class) + np.random.normal(0, noise, n_per_class)
        x1_1 = r1 * np.cos(theta1)
        x2_1 = r1 * np.sin(theta1)
        
        theta2 = np.random.uniform(0, 2 * np.pi, n_per_class) + np.pi
        r2 = np.linspace(0, 1, n_per_class) + np.random.normal(0, noise, n_per_class)
        x1_2 = r2 * np.cos(theta2)
        x2_2 = r2 * np.sin(theta2)
        
        X = np.vstack([
            np.column_stack([x1_1, x2_1]),
            np.column_stack([x1_2, x2_2])
        ])
        y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])
    
    return X, y


class NeuralNetwork:
    """
    Простая нейронная сеть с одним скрытым слоем для демонстрации обучения
    
    Архитектура:
        Входной слой -> Скрытый слой (ReLU) -> Выходной слой (Sigmoid)
    """
    
    def __init__(self, input_size = 2, hidden_size = 4, output_size = 1, learning_rate = 0.03):
        """
        Инициализация параметров нейронной сети
        
        Параметры:
            input_size : int - размер входного слоя
            hidden_size : int - количество нейронов в скрытом слое
            output_size : int - размер выходного слоя
            learning_rate : float - скорость обучения
        """
        # Инициализация весов методом He (для ReLU)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        
        self.learning_rate = learning_rate
        self.losses = []
        self.accuracies = []
    
    def sigmoid(self, z):
        """Сигмоидная функция активации для выходного слоя"""
        # Защита от переполнения
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def relu(self, z):
        """ReLU функция активации для скрытого слоя"""
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        """Производная ReLU"""
        return (z > 0).astype(float)
    
    def forward(self, X):
        """
        Прямое распространение (forward pass)
        
        Параметры:
            X : np.array - входные данные
        
        Возвращает:
            output : np.array - предсказания сети
            cache : dict - кэш значений для обратного распространения
        """
        # Скрытый слой
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        # Выходной слой
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        cache = {
            'z1': self.z1,
            'a1': self.a1,
            'z2': self.z2,
            'a2': self.a2
        }
        
        return self.a2, cache
    
    def compute_loss(self, y_true, y_pred):
        """
        Вычисление бинарной кросс-энтропии (функция потерь)
        
        Параметры:
            y_true : np.array - истинные метки
            y_pred : np.array - предсказанные вероятности
        
        Возвращает:
            loss : float - значение функции потерь
        """
        # Добавляем небольшое значение для избежания логарифма от 0
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return loss
    
    def backward(self, X, y_true, cache):
        """
        Обратное распространение ошибки (backward pass)
        
        Параметры:
            X : np.array - входные данные
            y_true : np.array - истинные метки
            cache : dict - кэш значений из прямого прохода
        
        Выполняет:
            Обновление весов и смещений через градиентный спуск
        """
        m = X.shape[0]
        
        # Градиент выходного слоя
        dz2 = cache['a2'] - y_true.reshape(-1, 1)
        dW2 = np.dot(cache['a1'].T, dz2) / m
        db2 = np.sum(dz2, axis = 0, keepdims = True) / m
        
        # Градиент скрытого слоя
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(cache['z1'])
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis = 0, keepdims = True) / m
        
        # Обновление параметров (градиентный спуск)
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs = 500, verbose = False):
        """
        Обучение нейронной сети
        
        Параметры:
            X : np.array - обучающие данные
            y : np.array - метки классов
            epochs : int - количество эпох обучения
            verbose : bool - выводить ли прогресс обучения
        
        Возвращает:
            history : dict - история потерь и точности
        """
        for epoch in range(epochs):
            # Прямой проход
            y_pred, cache = self.forward(X)
            
            # Вычисление потерь
            loss = self.compute_loss(y.reshape(-1, 1), y_pred)
            self.losses.append(loss)
            
            # Вычисление точности
            accuracy = np.mean((y_pred > 0.5).astype(float).flatten() == y)
            self.accuracies.append(accuracy)
            
            # Обратный проход
            self.backward(X, y.reshape(-1, 1), cache)
            
            if verbose and epoch % 100 == 0:
                print(f"Эпоха {epoch}, Потери: {loss:.4f}, Точность: {accuracy:.4f}")
        
        return {
            'losses': self.losses,
            'accuracies': self.accuracies
        }
    
    def predict(self, X):
        """
        Предсказание для новых данных
        
        Параметры:
            X : np.array - входные данные
        
        Возвращает:
            predictions : np.array - предсказанные классы (0 или 1)
        """
        y_pred, _ = self.forward(X)
        return (y_pred > 0.5).astype(float).flatten()
    
    def predict_grid(self, x_min = -1, x_max = 1, y_min = -1, y_max = 1, grid_size = 50):
        """
        Генерация сетки предсказаний для визуализации границы решений
        
        Параметры:
            x_min, x_max : float - диапазон по оси X
            y_min, y_max : float - диапазон по оси Y
            grid_size : int - размер сетки
        
        Возвращает:
            grid : np.array - сетка предсказаний (grid_size, grid_size)
        """
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_size),
            np.linspace(y_min, y_max, grid_size)
        )
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        predictions = self.predict(grid_points)
        return predictions.reshape(grid_size, grid_size)


# Глобальные переменные для хранения состояния
current_dataset = None
current_network = None
train_history = None

@app.route('/')
def index():
    """Главная страница с интерфейсом"""
    return render_template('index.html')

@app.route('/api/generate_dataset', methods = ['POST'])
def api_generate_dataset():
    """
    API эндпоинт для генерации датасета
    
    JSON с параметрами:
        - dataset_type : str - тип датасета
        - n_samples : int - количество точек
        - noise : float - уровень шума
    """
    global current_dataset
    
    data = request.json
    dataset_type = data.get('dataset_type', 'circle')
    n_samples = data.get('n_samples', 300)
    noise = data.get('noise', 0.1)
    
    X, y = generate_dataset(dataset_type, n_samples, noise)
    current_dataset = {'X': X.tolist(), 'y': y.tolist()}
    
    return jsonify(current_dataset)

@app.route('/api/train', methods = ['POST'])
def api_train():
    """
    API эндпоинт для обучения нейронной сети
    
    Принимает JSON с параметрами:
        - hidden_size : int - количество нейронов в скрытом слое
        - learning_rate : float - скорость обучения
        - epochs : int - количество эпох
    """
    global current_dataset, current_network, train_history
    
    if current_dataset is None:
        return jsonify({'error': 'Сначала сгенерируйте датасет'}), 400
    
    data = request.json
    hidden_size = data.get('hidden_size', 4)
    learning_rate = data.get('learning_rate', 0.03)
    epochs = data.get('epochs', 500)
    
    X = np.array(current_dataset['X'])
    y = np.array(current_dataset['y'])
    
    # Создание и обучение сети
    current_network = NeuralNetwork(
        input_size = 2,
        hidden_size = hidden_size,
        learning_rate = learning_rate
    )
    
    train_history = current_network.train(X, y, epochs, verbose = True)
    
    # Генерация сетки для визуализации
    grid = current_network.predict_grid().tolist()
    
    return jsonify({
        'losses': train_history['losses'],
        'accuracies': train_history['accuracies'],
        'grid': grid
    })

@app.route('/api/predict_grid', methods = ['POST'])
def api_predict_grid():
    """
    API эндпоинт для получения сетки предсказаний
    
    Используется для обновления визуализации без переобучения
    """
    global current_network
    
    if current_network is None:
        return jsonify({'error': 'Сеть еще не обучена'}), 400
    
    grid = current_network.predict_grid().tolist()
    return jsonify({'grid': grid})

if __name__ == '__main__':
    print("_" * 50)
    print("TensorFlow Playground - Локальная версия")
    print("Откройте http://localhost:5000 в браузере")
    print("_" * 50)
    app.run(debug = True, port = 5000)