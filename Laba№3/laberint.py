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
        new_state = (self.state[0] + actions[action][0], 
                     self.state[1] + actions[action][1])

        # Проверка на выход за границы и стену
        if (0 <= new_state[0] < self.n_rows and 
            0 <= new_state[1] < self.n_cols and 
            self.maze[new_state] != 1):
            self.state = new_state
        
        # Награда: +10 за достижение цели, -1 за шаг
        reward = 10 if self.is_done() else -1
        return self.state, reward, self.is_done()
    
    def render(self):
        display = np.array(self.maze, dtype=str)
        display[display == '1'] = '|'  # Стена
        display[display == '0'] = '·'  # Путь
        display[self.state] = 'A'      # Агент
        display[self.goal] = 'Win'       # Цель
        for row in display:
            print(' '.join(row))
        print()

class QLearning:
    def __init__(self, maze, alpha = 0.1, gamma = 0.95, epsilon = 0.1):
        self.maze = maze
        self.alpha = alpha      # Скорость обучения
        self.gamma = gamma      # Коэффициент дисконтирования
        self.epsilon = epsilon  # Вероятность исследования
        self.q_table = np.zeros((maze.n_rows, maze.n_cols, 4))
        
        self.goal_reward = 10
    
    def choose_action(self, state):
        """Выбор действия с ε-жадной стратегией"""
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 3)  # Исследование
        else:
            return np.argmax(self.q_table[state])  # Эксплуатация
    
    def update_q_table(self, state, action, reward, next_state):
        """Обновление Q-значения по формуле Q-Learning"""
        best_next_action = np.argmax(self.q_table[next_state])
        
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_delta
    
    def train(self, episodes, verbose=True):
        rewards_history = []
        
        for episode in range(episodes):
            state = self.maze.reset()
            total_reward = 0
            steps = 0
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.maze.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                steps += 1
                
                # Защита от бесконечного цикла
                if steps > 1000:
                    break
            
            rewards_history.append(total_reward)
            
            if verbose and (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:])
                print(f"Эпизод {episode + 1}/{episodes}, "
                      f"Средняя награда: {avg_reward:.2f}, "
                      f"Шагов: {steps}")
        
        return rewards_history
    
    def get_path(self):
        """Получение оптимального пути после обучения"""
        state = self.maze.reset()
        path = [state]
        visited = set([state])
        
        while not self.maze.is_done():
            action = np.argmax(self.q_table[state])
            next_state, _, _ = self.maze.step(action)
            
            # Защита от зацикливания
            if next_state in visited:
                break
            visited.add(next_state)
            
            path.append(next_state)
            state = next_state
        
        return path

maze_array = np.array([[0, 0, 0, 1, 0],
                       [1, 0, 1, 0, 0],
                       [0, 0, 0, 1, 0],
                       [0, 1, 0, 0, 0],
                       [0, 0, 0, 1, 0]])

maze = Maze(maze_array)
agent = QLearning(maze, alpha=0.1, gamma=0.95, epsilon=0.1)

print("Начало обучения...")
rewards = agent.train(5000, verbose=True)

optimal_path = agent.get_path()
print(f"\nОптимальный путь (длина {len(optimal_path)}):")
for i, (row, col) in enumerate(optimal_path):
    print(f"  Шаг {i}: ({row}, {col})")

def visualize_q_values(agent, maze):
    """Комплексная визуализация Q-значений"""
    fig, axes = plt.subplots(2, 2, figsize = (12, 10))
    
    max_q = np.max(agent.q_table, axis = 2)
    im1 = axes[0, 0].imshow(max_q, cmap = 'viridis', interpolation = 'nearest')
    axes[0, 0].set_title('Максимальные Q-значения по всем действиям')
    axes[0, 0].set_xlabel('Колонка')
    axes[0, 0].set_ylabel('Строка')
    plt.colorbar(im1, ax = axes[0, 0])
    
    optimal_actions = np.argmax(agent.q_table, axis=2)
    action_names = ['Вправо', 'Вниз', 'Влево', 'Вверх']
    action_matrix = np.array([[action_names[optimal_actions[r, c]] 
                               for c in range(maze.n_cols)] 
                              for r in range(maze.n_rows)])
    
    maze_display = np.where(maze.maze == 1, 1, 0)
    axes[0, 1].imshow(maze_display, cmap = 'gray_r', alpha = 0.7)
    for r in range(maze.n_rows):
        for c in range(maze.n_cols):
            if maze.maze[r, c] != 1:
                axes[0, 1].text(c, r, action_matrix[r, c], 
                               ha='center', va='center', fontsize = 14)
    axes[0, 1].set_title('Оптимальные направления действий')
    axes[0, 1].set_xlabel('Колонка')
    axes[0, 1].set_ylabel('Строка')
    
    axes[1, 0].plot(rewards)
    axes[1, 0].set_title('Динамика обучения')
    axes[1, 0].set_xlabel('Эпизод')
    axes[1, 0].set_ylabel('Суммарная награда')
    axes[1, 0].grid(True, alpha=0.3)
    
    path = optimal_path
    path_matrix = np.zeros_like(maze.maze, dtype=float)
    for r, c in path:
        path_matrix[r, c] = 1
    
    axes[1, 1].imshow(path_matrix, cmap = 'Blues', alpha = 0.8)
    # Наложение лабиринта
    axes[1, 1].imshow(maze.maze, cmap = 'gray_r', alpha = 0.3)
    
    axes[1, 1].scatter(maze.start[1], maze.start[0], 
                      color = 'green', s = 100, marker = 'o', label = 'Старт')
    axes[1, 1].scatter(maze.goal[1], maze.goal[0], 
                      color = 'red', s = 100, marker = '*', label = 'Цель')
    axes[1, 1].set_title(f'Оптимальный путь (длина: {len(path)})')
    axes[1, 1].set_xlabel('Колонка')
    axes[1, 1].set_ylabel('Строка')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Дополнительная визуализация: Q-значения для каждого действия
    fig, axes = plt.subplots(1, 4, figsize = (16, 4))
    actions = ['Вправо (0)', 'Вниз (1)', 'Влево (2)', 'Вверх (3)']
    
    for i, (ax, action_name) in enumerate(zip(axes, actions)):
        q_values = agent.q_table[:, :, i]
        im = ax.imshow(q_values, cmap = 'plasma', interpolation = 'nearest')
        ax.set_title(f'Q-значения: {action_name}')
        ax.set_xlabel('Колонка')
        ax.set_ylabel('Строка')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()

visualize_q_values(agent, maze)