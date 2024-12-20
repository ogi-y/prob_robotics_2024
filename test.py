import numpy as np
import random
import matplotlib.pyplot as plt

# 環境の設定
state_space = 100  # 位置は-5から5までの11個の状態
action_space = 2  # 行動は2つ（左に進む、右に進む）
goal = 5  # 目標位置

# Qテーブルの初期化
Q = np.zeros((state_space, action_space))

# 学習率、割引率、探索率
alpha = 0.1  # 学習率
gamma = 0.9  # 割引率
epsilon = 0.1  # ε-greedy探索率

# 報酬関数
def get_reward(state):
    if state == goal:
        return 100  # 目標位置に到達したときの報酬
    else:
        return -1  # それ以外は少しのペナルティ

# 行動選択（ε-greedy）
def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        # ランダムに行動を選択
        return random.randint(0, 1)
    else:
        # Q値が最大の行動を選択
        return np.argmax(Q[state])

# 1エピソードの学習
def train(epochs=10000):
    for _ in range(epochs):
        state = 5  # 初期位置は0（状態空間の中央）
        
        while True:
            action = choose_action(state)  # 行動を選択
            # 次の状態を計算
            if action == 0:  # 左に進む
                next_state = state - 1 if state > -5 else state
            else:  # 右に進む
                next_state = state + 1 if state < 5 else state
            
            reward = get_reward(next_state)  # 報酬を得る
            # Q値の更新
            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            
            state = next_state  # 次の状態へ移動
            
            if state == goal:  # 目標に到達したら終了
                break

# 学習実行
train()

# Qテーブルの表示
print(Q)

# 最適な行動を表示
optimal_actions = []
for state in range(state_space):
    action = np.argmax(Q[state])
    optimal_actions.append(action)

# Matplotlibでビジュアライズ
fig, ax = plt.subplots(figsize=(10, 4))

# 最適な行動を矢印で表示
for state in range(state_space):
    if optimal_actions[state] == 0:  # 左に進む
        ax.annotate('<', (state - 5, 0), textcoords="offset points", xytext=(0, 10), ha='center', color='red')
    else:  # 右に進む
        ax.annotate('>', (state - 5, 0), textcoords="offset points", xytext=(0, 10), ha='center', color='green')

# 目標位置を示す
ax.annotate('Goal', (goal - 5, 0), textcoords="offset points", xytext=(0, 20), ha='center', color='blue', fontsize=12)

# グラフの設定
ax.set_xlim(-5, 5)
ax.set_ylim(-1, 1)
ax.set_xticks(range(-5, 6))
ax.set_yticks([])
ax.set_title("Optimal Actions on 1D Line (Q-Learning)")

plt.show()
