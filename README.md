# BipedalWalker PPO ローカル実行

このリポジトリは、貼られていた notebook 形式のコードをローカルで実行しやすいように、学習・評価プロット・動画記録の 3 つの Python スクリプトに分けたものです。

## セットアップ

システム依存関係は環境に合わせて入れてください。Linux なら少なくとも `swig`、`ffmpeg`、`xvfb` が必要になることがあります。

```bash
sudo apt update
sudo apt install -y swig ffmpeg xvfb
pip install -r requirements.txt
```

## 実行

学習:

```bash
python train_bipedal_walker.py --timesteps 1000000
```

PPO / SAC / DDPG を競わせる夜間スイープ:

```bash
python nightly_search_bipedal_walker.py --timesteps 600000 --chunk-timesteps 50000 --eval-episodes 5
```

評価ログのプロット保存:

```bash
python plot_bipedal_walker.py
```

学習済みモデルの動画記録:

```bash
python record_bipedal_walker.py
```

ヘッドレス環境で動画を記録する場合は、必要に応じて `xvfb-run` を付けてください。

```bash
xvfb-run -a python record_bipedal_walker.py
```

## venv作成

```bash
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## 出力

- 学習済みモデル: `ppo_bipedalwalker_result/final_model.zip`
- 最良モデル: `ppo_bipedalwalker_result/best_model/best_model.zip`
- 正規化情報: `ppo_bipedalwalker_result/vecnormalize.pkl`
- 評価ログ: `ppo_bipedalwalker_result/logs/evaluations.npz`
- プロット画像: `ppo_bipedalwalker_result/logs/training_rewards.png` と `ppo_bipedalwalker_result/logs/episode_lengths.png`
- 動画: `bipedal_walker_videos_practice/`
