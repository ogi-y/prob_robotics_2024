# prob_robotics_2024

## 概要
このリポジトリは、確率ロボティクス2024年度の課題として作成されました。  
1次元の数直線上をエージェントが移動し、Q学習によって最適な方策を学習します。

## 実行方法
以下のコマンドを使用して、必要なPythonライブラリをインストールしてください。
```bash
pip install numpy matplotlib
```
コードの実行は以下のように行います。
```bash
python Q_train.py
```

## 各ファイルの概要
* Q_train.py
  * Q学習を行い、各エピソードを保存します。
* Q_plot_animation_eps.py
  * 学習の結果得られた方策を表示し、それに従いエージェントを移動させますが、確率εでランダムに行動をさせます。（ε-greedy方策）
* Q_plot_eps.py
  * 各エピソードによって得た方策ををε-greedy方策に従って10回ずつ試行し、平均のゴール時間をグラフにします。
* env_loader.py
  * シミュレーション環境を読み込みます。
* env.csv
  * シミュレーション環境が記述されています。

## Q学習の説明
Q学習の更新式は以下です。

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)
$$