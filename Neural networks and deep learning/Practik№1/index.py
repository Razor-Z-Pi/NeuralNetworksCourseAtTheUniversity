import numpy as np

class Neuron:
    def __init__ (self, w):         
        self.w = w
                  
    def y(self, x):             # Сумматор  
        s = np.dot(self.w, x)   # Суммируем входы
        return s                # Функция активации
    
                    
Xi = np.array([2, 3])   # Задание значений входам
Wi = np.array([1, 1])   # Веса входных сенсоров
n = Neuron(Wi)
print('S1 =', n.y(Xi)) # Обращение к нейрону
Xi = np.array([5, 6])   # Веса входных сенсоров
print('S2 =', n.y(Xi))  # Обращение к нейрону

Zi = np.array([1, 0, 0, 1])
Yi = np.array([5, 4, 3, 1])
r = Neuron(Yi)
print("S =", r.y(Zi))