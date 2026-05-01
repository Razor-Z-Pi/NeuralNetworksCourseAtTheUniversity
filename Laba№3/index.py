import numpy as np
import matplotlib.pyplot as plt
import random

class Maze:
    def __init__(self, maze, alpa = 0.1, gamma = 0.9, epsilon = 0.1):
        self.maze = maze
        self.alpha = self.alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((maze.n_rows, maze.n_cols, 4)) # Q-таблицы
    
    def choose_action(self, state):
        pass