import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Neuron:
    def __init__ (self, w):         
        self.w = w
                  
    def y(self, x):             # Сумматор  
        s = np.dot(self.w, x)   # Суммируем входы
        return s                # Функция активации
    
                    
Xi = np.array([0, 0, 1, 1])   # Задание значений входам
Wi = np.array([5, 4, 3, 1])   # Веса входных сенсоров
n = Neuron(Wi)
print('Y =', n.y(Xi)) # Обращение к нейрону