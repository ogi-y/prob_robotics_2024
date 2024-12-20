import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# 定数
grid_size = 10
alpha = 0.1  # 学習率
gamma = 0.95  # 割引率
epsilon = 0.1  # 探索率
actions = [-1, 0, 1]  # 左移動、停止、右移動

# 報酬
reward_catch = 10
reward_miss = -10
reward_step = -1

# Qテーブルの初期化
q_table = np.zeros((grid_size, grid_size, len(actions)))

# Q学習の実行
def train_q_learning(episodes=500):
    for episode in range(episodes):
        player_pos = grid_size // 2
        item_pos = np.random.randint(0, grid_size)
        item_y = grid_size - 1

        while item_y >= 0:
            # 状態の取得
            state = (player_pos, item_pos)
            
            # ε-greedyポリシーで行動を選択
            if np.random.rand() < epsilon:
                action = np.random.choice(len(actions))  # ランダム行動
            else:
                action = np.argmax(q_table[player_pos, item_pos])  # 最大Q値の行動

            # 行動を実行
            player_pos = max(0, min(grid_size - 1, player_pos + actions[action]))
            item_y -= 1

            # 次の状態
            next_state = (player_pos, item_pos)

            # 報酬の計算
            if item_y == 0:  # アイテムが最下段に到達
                if player_pos == item_pos:
                    reward = reward_catch  # キャッチ成功
                else:
                    reward = reward_miss  # キャッチ失敗
            else:
                reward = reward_step  # 通常の移動ペナルティ

            # Q値の更新
            best_next_action = np.argmax(q_table[next_state[0], next_state[1]])
            q_table[state[0], state[1], action] += alpha * (
                reward + gamma * q_table[next_state[0], next_state[1], best_next_action]
                - q_table[state[0], state[1], action]
            )

            # アイテムが最下段ならリセット
            if item_y == 0:
                item_pos = np.random.randint(0, grid_size)
                item_y = grid_size - 1

# 学習後のエージェント動作をシミュレート
def simulate_trained_agent(frames=20):
    player_pos = grid_size // 2
    item_pos = np.random.randint(0, grid_size)
    item_y = grid_size - 1
    frames_data = []

    for _ in range(frames):
        frames_data.append((player_pos, (item_pos, item_y)))

        # Q値に基づく最適行動を選択
        action = np.argmax(q_table[player_pos, item_pos])
        player_pos = max(0, min(grid_size - 1, player_pos + actions[action]))
        item_y -= 1

        if item_y < 0:  # アイテムが最下段に到達
            item_pos = np.random.randint(0, grid_size)
            item_y = grid_size - 1

    return frames_data

# アニメーション作成関数
def create_animation(frames):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(-1, grid_size)
    ax.set_ylim(-1, grid_size)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.grid(True)

    player_marker, = ax.plot([], [], 'ro', label="Player", markersize=15)
    item_marker, = ax.plot([], [], 'bo', label="Item", markersize=15)
    ax.legend()

    def init():
        player_marker.set_data([], [])
        item_marker.set_data([], [])
        return player_marker, item_marker

    def update(frame):
        player_pos, (item_x, item_y) = frame
        player_marker.set_data([player_pos], [0])
        item_marker.set_data([item_x], [item_y])
        return player_marker, item_marker

    ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=500)
    return ani

# Q学習の訓練
train_q_learning(episodes=500)

# 学習後のエージェントをシミュレート
frames = simulate_trained_agent(frames=20)

# アニメーション作成と保存
ani = create_animation(frames)
ani.save("trained_agent.gif", writer=PillowWriter(fps=2))
print("GIFを保存しました: trained_agent.gif")
