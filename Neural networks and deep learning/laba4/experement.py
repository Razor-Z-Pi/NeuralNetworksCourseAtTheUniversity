import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
import os

def generate_data(filename = 'data_vector_quantization.txt', n_points = 400):
    """Генерация различных типов данных для векторной квантизации"""
    np.random.seed(42)
    
    # Типы распределений для разных кластеров
    data_list = []
    labels_list = []
    
    # Кластер: Нормальное распределение в верхнем левом углу
    cluster1_x = np.random.normal(2, 0.8, n_points // 4)
    cluster1_y = np.random.normal(8, 0.8, n_points // 4)
    for i in range(len(cluster1_x)):
        data_list.append([cluster1_x[i], cluster1_y[i]])
        labels_list.append(0)
    
    # Кластер: Нормальное распределение в верхнем правом углу
    cluster2_x = np.random.normal(8, 0.8, n_points // 4)
    cluster2_y = np.random.normal(8, 0.8, n_points // 4)
    for i in range(len(cluster2_x)):
        data_list.append([cluster2_x[i], cluster2_y[i]])
        labels_list.append(1)
    
    # Кластер: Нормальное распределение в нижнем левом углу
    cluster3_x = np.random.normal(2, 0.8, n_points // 4)
    cluster3_y = np.random.normal(2, 0.8, n_points // 4)
    for i in range(len(cluster3_x)):
        data_list.append([cluster3_x[i], cluster3_y[i]])
        labels_list.append(2)
    
    # Кластер: Кольцевое распределение
    radii = np.random.uniform(3, 6, n_points // 4)
    angles = np.random.uniform(0, 2 * np.pi, n_points // 4)
    cluster4_x = 5 + radii * np.cos(angles)
    cluster4_y = 5 + radii * np.sin(angles)
    for i in range(len(cluster4_x)):
        data_list.append([cluster4_x[i], cluster4_y[i]])
        labels_list.append(3)
    
    data = np.array(data_list)
    labels = np.array(labels_list)
    
    output_data = np.column_stack((data, labels))
    np.savetxt(filename, output_data, fmt = '%.4f %.4f %d', 
               header = 'x y class', comments = '')
    
    print(f"Создан файл {filename} с {n_points} точками")
    return data, labels

def vector_quantization(data, n_clusters=4, method='kmeans'):
    """
    Выполняет векторную квантизацию данных
    
    Parameters:
    - data: входные данные (N x 2)
    - n_clusters: количество кластеров
    - method: метод квантизации ('kmeans' или 'knn')
    """
    
    if method == 'kmeans':
        # Метод K-means для векторной квантизации
        vq = KMeans(n_clusters = n_clusters, random_state = 42, n_init = 10)
        vq.fit(data)
        centroids = vq.cluster_centers_
        labels = vq.labels_
        
        # Вычисление ошибки квантизации
        distortion = vq.inertia_ / len(data)
        
        return labels, centroids, distortion
    
    else:
        raise ValueError(f"Неизвестный метод: {method}")


def plot_results(data, true_labels, predicted_labels, centroids, distortion):
    """Построение графиков результатов векторной квантизации"""
    fig = plt.figure(figsize = (16, 12))
    
    # График № 1: Исходные данные (с реальными классами)
    ax1 = fig.add_subplot(2, 2, 1)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for i in range(4):
        mask = true_labels == i
        ax1.scatter(data[mask, 0], data[mask, 1], 
                   c = colors[i], marker = markers[i], 
                   label = f'Класс {i+1}', alpha = 0.6, s = 30)
    ax1.set_xlabel('Измерение 1', fontsize = 11)
    ax1.set_ylabel('Измерение 2', fontsize = 11)
    ax1.set_title('Исходные данные (реальная классификация)', fontsize = 13)
    ax1.legend()
    ax1.grid(True, alpha = 0.3)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    
    # График № 2: Результаты векторной квантизации
    ax2 = fig.add_subplot(2, 2, 2)
    for i in range(4):
        mask = predicted_labels == i
        ax2.scatter(data[mask, 0], data[mask, 1], 
                   c = colors[i], marker = markers[i], 
                   label = f'Кластер {i + 1}', alpha = 0.6, s = 30)
    # Рисуем центроиды
    ax2.scatter(centroids[:, 0], centroids[:, 1], 
               c = 'black', marker = '*', s = 200, 
               edgecolors = 'white', linewidth = 2, label = 'Центроиды')
    ax2.set_xlabel('Измерение 1', fontsize = 11)
    ax2.set_ylabel('Измерение 2', fontsize = 11)
    ax2.set_title('Результат векторной квантизации (K-means)', fontsize = 13)
    ax2.legend()
    ax2.grid(True, alpha = 0.3)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    
    # График № 3: Границы между кластерами (диаграмма Вороного)
    ax3 = fig.add_subplot(2, 2, 3)
    
    # Создание сетки для отображения границ
    xx, yy = np.meshgrid(np.linspace(0, 10, 200), np.linspace(0, 10, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Классификация точек сетки
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(centroids, np.arange(len(centroids)))
    grid_labels = knn.predict(grid_points).reshape(xx.shape)
    
    # Отображение областей
    ax3.contourf(xx, yy, grid_labels, alpha = 0.3, colors = colors[:4])
    ax3.scatter(data[:, 0], data[:, 1], c = 'black', s = 15, alpha = 0.5)
    ax3.scatter(centroids[:, 0], centroids[:, 1], 
               c = 'red', marker = '*', s = 150, edgecolors = 'black', linewidth = 1.5)
    ax3.set_xlabel('Измерение 1', fontsize = 11)
    ax3.set_ylabel('Измерение 2', fontsize = 11)
    ax3.set_title('Границы между кластерами (диаграмма Вороного)', fontsize = 13)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.grid(True, alpha = 0.3)
    
    # График 4: Ошибка квантизации (Elbow метод)
    ax4 = fig.add_subplot(2, 2, 4)
    distortions = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters = k, random_state = 42, n_init = 10)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_ / len(data))
    
    ax4.plot(K_range, distortions, 'bo-', linewidth = 2, markersize = 8)
    ax4.axvline(x = 4, color = 'r', linestyle = '--', label = 'Выбранное k = 4')
    ax4.set_xlabel('Количество кластеров (k)', fontsize = 11)
    ax4.set_ylabel('Средняя ошибка квантизации', fontsize = 11)
    ax4.set_title('Метод локтя для выбора k', fontsize = 13)
    ax4.legend()
    ax4.grid(True, alpha = 0.3)
    
    plt.tight_layout()
    return fig


def plot_quantization_error_evolution(data, n_clusters=4, n_iterations=20):
    """Отображение эволюции ошибки квантизации в процессе обучения"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (14, 5))
    
    # Эволюция ошибки
    errors = []
    best_kmeans = None
    
    for i in range(1, n_iterations + 1):
        kmeans = KMeans(n_clusters = n_clusters, random_state = i, n_init = 1, max_iter = 100)
        kmeans.fit(data)
        errors.append(kmeans.inertia_ / len(data))
    
    ax1.plot(range(1, n_iterations + 1), errors, 'bo-', linewidth = 2, markersize = 6)
    ax1.axhline(y = np.mean(errors), color = 'r', linestyle = '--', 
                label=f'Среднее: {np.mean(errors):.2f}')
    ax1.set_xlabel('Номер запуска', fontsize = 11)
    ax1.set_ylabel('Ошибка квантизации', fontsize = 11)
    ax1.set_title('Эволюция ошибки при разных инициализациях', fontsize = 12)
    ax1.legend()
    ax1.grid(True, alpha = 0.3)
    
    # Гистограмма ошибок
    ax2.hist(errors, bins = 15, color = 'skyblue', edgecolor = 'black', alpha = 0.7)
    ax2.axvline(x = np.mean(errors), color = 'r', linestyle = '--', 
                label = f'Среднее: {np.mean(errors):.2f}')
    ax2.axvline(x = np.min(errors), color = 'g', linestyle = '--', 
                label = f'Минимум: {np.min(errors):.2f}')
    ax2.set_xlabel('Ошибка квантизации', fontsize = 11)
    ax2.set_ylabel('Частота', fontsize = 11)
    ax2.set_title('Распределение ошибок квантизации', fontsize = 12)
    ax2.legend()
    ax2.grid(True, alpha = 0.3)
    
    plt.tight_layout()
    return fig

def print_statistics(data, true_labels, predicted_labels, centroids, distortion):
    """Вывод статистики по результатам квантизации"""
    
    print("\n" + "_" * 70)
    print("Результаты по квантизации")
    print("_" * 70)
    
    print(f"\nОбщее количество точек: {len(data)}")
    print(f"Количество кластеров: {len(centroids)}")
    print(f"Средняя ошибка квантизации: {distortion:.4f}")
    print(f"Коэффициент сжатия: {len(data) * 2 * 8 / (len(centroids) * 2 * 8):.1f}:1")
    
    print("\n@_@ Распределение точек по кластерам @_@")
    unique, counts = np.unique(predicted_labels, return_counts=True)
    for i, count in zip(unique, counts):
        print(f"  Кластер {i + 1}: {count} точек ({count / len(data) * 100:.1f}%)")
    
    print("\n@_@ Центроиды кластеров @_@")
    for i, centroid in enumerate(centroids):
        print(f"  Кластер {i + 1}: ({centroid[0]:.3f}, {centroid[1]:.3f})")
    
    print("\n@_@ Матрица ошибок (сравнение с реальными классами) @_@")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    print("        Кл1  Кл2  Кл3  Кл4")
    for i in range(4):
        print(f"Класс{i + 1} : {cm[i,0]:3d}  {cm[i,1]:3d}  {cm[i,2]:3d}  {cm[i,3]:3d}")
    
    # Точность
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    print(f"\n@_@ Метрики качества @_@")
    print(f" Скорректированный индекс Рэнда  Adjusted Rand Index (ARI): {ari:.4f}")
    print(f" Нормализованная взаимная информация Normalized Mutual Info (NMI): {nmi:.4f}")

# Параметры
filename = 'data_vector_quantization.txt'
n_points = 400
n_clusters = 4
    

print("_" * 70)
print("Эксперементы!!!")
print("_" * 70)
    

if not os.path.exists(filename):
    data, true_labels = generate_data(filename, n_points)
else:
    loaded = np.loadtxt(filename)
    if loaded.ndim == 1:
        loaded = loaded.reshape(1, -1)
    data = loaded[:, :2]
    if loaded.shape[1] >= 3:
        true_labels = loaded[:, 2].astype(int)
    else:
        true_labels = np.zeros(len(data))
    print(f"Загружен файл {filename} с {len(data)} точками")
    

print(f"\nФорма данных: {data.shape}")
print(f"Диапазон X: [{data[:,0].min():.2f}, {data[:,0].max():.2f}]")
print(f"Диапазон Y: [{data[:,1].min():.2f}, {data[:,1].max():.2f}]")
    
print("\nВыполнение векторной квантизации...")
predicted_labels, centroids, distortion = vector_quantization(data, n_clusters)
    
print_statistics(data, true_labels, predicted_labels, centroids, distortion)
    
fig1 = plot_results(data, true_labels, predicted_labels, centroids, distortion)
    
fig2 = plot_quantization_error_evolution(data, n_clusters)
    
plt.show()
    
print("\n" + "@_@" * 70)
print("Эксперимент завершен!!!")
print("@_@" * 70)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    