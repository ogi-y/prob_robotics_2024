import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# 環境設定
grid_width = 10  # グリッドの幅
grid_height = 10  # グリッドの高さ
num_actions = 3  # 左移動, 右移動, 何もしない
epsilon = 0.1  # ε-greedy法のε
alpha = 0.1  # 学習率
gamma = 0.9  # 割引率
episodes = 500  # 学習エピソード数

# Qテーブルの初期化
q_table = np.zeros((grid_width, grid_height, num_actions))

# アイテムの設定
item_drop_position = [3, 5, 6, 1, 3]  # アイテムが落ちる位置（x座標）

# エージェントの初期位置
player_pos = grid_width // 2

# 行動選択
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, num_actions - 1)  # ランダムに行動を選択
    else:
        return np.argmax(q_table[state[0], state[1], :])  # 最大Q値の行動を選択

# 環境のリセット
def reset_environment():
    return [grid_width // 2, 0]  # エージェントを中央に配置、アイテムはy=0からスタート

# 学習
def train_q_learning(episodes):
    for episode in range(episodes):
        state = reset_environment()
        item_y = 10  # アイテムの初期位置
        item_x = random.choice(item_drop_position)
        done = False
        total_reward = 0

        while not done:
            action = choose_action(state, epsilon)  # 行動選択
            # 行動の結果
            if action == 0:  # 左に移動
                player_pos = max(0, state[0] - 1)
            elif action == 1:  # 右に移動
                player_pos = min(grid_width - 1, state[0] + 1)
            else:  # 何もしない
                player_pos = state[0]
            
            # アイテムの位置更新
            item_y -= 1  # アイテムが1フレーム下に落ちる
            if item_y == 0:  # アイテムが下に着いた場合
                if player_pos == item_x:  # アイテムをキャッチ
                    reward = 1  # 報酬を与える
                else:
                    reward = -1  # アイテムをキャッチできなかった場合は罰
                item_y = 10  # 新しいアイテムがy=10から落ちる
                item_x = random.choice(item_drop_position)
            else:
                reward = 0  # アイテムはまだキャッチされていない
            
            # Q値の更新
            next_state = [player_pos, item_y]
            
            # next_state[1] (アイテムのy位置)がgrid_heightより大きくならないように調整
            next_state[1] = min(grid_height - 1, next_state[1])  # 0からgrid_height-1の範囲に収める

            max_next_q = np.max(q_table[next_state[0], next_state[1], :])
            q_table[state[0], state[1], action] = (1 - alpha) * q_table[state[0], state[1], action] + alpha * (reward + gamma * max_next_q)

            # 次の状態に更新
            state = next_state

            # エピソード終了条件（アイテムを3回落とした場合、または制限時間経過）
            if item_y == 0:
                done = True

        # 進行状況
        if episode % 50 == 0:
            print(f"Episode {episode}/{episodes} - Total reward: {total_reward}")

# 学習結果の可視化
def plot_environment(player_pos, item_x, item_y):
    grid = np.zeros((grid_height, grid_width))
    grid[item_y, item_x] = 1  # アイテムの位置
    grid[0, player_pos] = 2  # プレイヤーの位置
    
    plt.imshow(grid, cmap='Blues', origin='lower')
    plt.xticks(np.arange(0, grid_width, 1))
    plt.yticks(np.arange(0, grid_height, 1))
    plt.grid(True)

# シミュレーションと結果表示
def simulate_and_show_result():
    state = reset_environment()
    item_y = 10
    item_x = random.choice(item_drop_position)
    
    fig, ax = plt.subplots()
    ims = []

    for frame in range(50):  # 50フレーム進める
        action = choose_action(state, 0)  # 学習したQテーブルを使って行動選択
        if action == 0:  # 左に移動
            player_pos = max(0, state[0] - 1)
        elif action == 1:  # 右に移動
            player_pos = min(grid_width - 1, state[0] + 1)
        else:  # 何もしない
            player_pos = state[0]
        
        item_y -= 1  # アイテムが1フレーム下に落ちる
        if item_y == 0:  # アイテムが下に着いた場合
            item_y = 10  # 新しいアイテムがy=10から落ちる
            item_x = random.choice(item_drop_position)

        ims.append([plot_environment(player_pos, item_x, item_y)])  # 現在の状態を記録

        # 次の状態に更新
        state = [player_pos, item_y]

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True)
    plt.show()

# 学習の開始
train_q_learning(episodes=500)

# 結果の可視化
simulate_and_show_result()
