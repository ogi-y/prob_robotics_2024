import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import env_loader

# 環境の設定
env_settings = env_loader.load_env('env.csv')
n_states = env_settings['n_states']
goal_state = env_settings['goal_state']
start_state = env_settings['start_state']
obstacles = env_settings['obstacles']
water = env_settings['water']
actions = env_settings['actions']

# 最終的な方策を読み込む
policy = np.load('policy/episode_15.npy')  # 最後の方策を読み込み（エピソード10）

eps = 0.3  # 0.1 の確率でランダム行動を選択

# エージェントの状態
state = start_state

# 行動選択（ランダムまたは方策に従う）
def choose_action(state):
    if random.uniform(0, 1) < eps:
        return random.choice([-1, 1])  # ランダムに行動
    else:
        return policy[state]  # 方策に従う

# アニメーションを作成するための関数
def animate_agent(i, line, ax):
    global state  # グローバル変数として状態を使用

    # エージェントが次の状態に移動
    if state != goal_state:
        action = choose_action(state)  # σ に従って行動を選択
        state = max(0, min(n_states - 1, state + action))  # 状態遷移

    # エージェントの位置を更新
    line.set_data([state - 0.5], [0.2])

    # ゴールに到達した場合はアニメーションを停止
    if state == goal_state:
        ax.set_title("Goal Reached!")
    
    return line,  # エージェントの位置のみを返す

# 方策に従ってアニメーションを描画
def plot_animation():
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.set_xlim(-0.5, n_states - 0.5)
    ax.set_ylim(-1, 1)

    # フィールドの直線
    ax.plot(np.arange(n_states), np.zeros(n_states), color='black', lw=2)

    # 障害物の四角
    for obs in obstacles:
        ax.add_patch(patches.Rectangle((obs - 0.5, 0), 1, 0.5, facecolor='red'))

    # ゴールの丸
    ax.add_patch(patches.Circle((goal_state - 0.5, 0.2), 0.2, facecolor='green'))

    # エージェントの位置（初期位置）
    agent, = ax.plot([], [], 'bo', markersize=10)  # エージェント（青い点）

    # 政策の矢印
    for i in range(n_states):
        if policy[i] == 1:  # 右
            ax.text(i, 0.5, "→", ha='center', fontsize=12, color='black')
        elif policy[i] == -1:  # 左
            ax.text(i, 0.5, "←", ha='center', fontsize=12, color='black')

    # 軸を非表示に
    ax.axis('off')
    ax.axis('equal')

    # アニメーションを作成
    ani = animation.FuncAnimation(fig, animate_agent, frames=200, fargs=(agent, ax),
                                  interval=500, repeat=False)

    plt.show()

# アニメーションの表示
plot_animation()
