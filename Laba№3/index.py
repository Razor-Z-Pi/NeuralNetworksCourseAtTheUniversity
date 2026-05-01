import numpy as np
import matplotlib.pyplot as plt
import random

class Maze:
    def __init__(self, maze):
        self.maze = maze
        self.n_rows, self.n_cols = maze.shape
        self.start = (0, 0)
        self.goal = (self.n_rows - 1, self.n_cols - 1)
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state
    
    def is_done(self):
        return self.state == self.goal
    
    def step(self, action):
        actions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        new_state = (self.state[0] + actions[action][0], self.state[1] + actions[action][1])

        if (0 <= new_state[0] < self.n_rows and 0 <= new_state[1] < self.n_rows and self.maze[new_state] != 1):
            self.state = new_state
        
        reward = 1 if self.is_done() else -1
        return self.state, reward, self.is_done()

class QLearning:
    def __init__(self, maze, alpha = 0.1, gamma = 0.9, epsilon = 0.1):
        self.maze = maze
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((maze.n_rows, maze.n_cols, 4)) # Q-таблицы
    
    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3) # Случайные действия
        else:
            return np.argmax(self.q_table[state]) # Лучшие действия
    
    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        tb_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_delta = td_target = self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_delta

    def train(self, episodes):
        for episode in range(episodes):
            state =  self.maze.reset()
            while not self.maze.is_done():
                action = self.choose_action(state)
                next_state, reward, done = self.maze.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state

maze_array = np.array([ [0, 0, 0, 1, 0],
                        [1, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 0, 1, 0], ])

maze = Maze(maze_array)
agent = QLearning(maze)

agent.train(1000)

plt.imshow(np.max(agent.q_table, axis = 2))
plt.colorbar()
plt.title("Q-Значения")
plt.xlabel("Действия")
plt.ylabel("Состояние")
plt.show()

state = maze.reset()
path = [state]

while not maze.is_done():
    action = np.argmax(agent.q_table[state])
    next_state, _, _ = maze.step(action)
    path.append(next_state)
    state = next_state

print(f"Путь к цели: {path}")