# DQN Atari - Group 16

Deep Q-Network agent trained on Atari Pong (`ALE/Pong-v5`) using Gymnasium and PyTorch.

## Files

- `train.py` - training script
- `check_env.py` - checks the environment and GPU
- `notebooks/` - notebooks
- `shem_ayioka/` - Shem's model and training log

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run check_env.py
```

## Training

```bash
uv run train.py
```

With custom settings:

```bash
uv run train.py --lr 1e-4 --batch_size 32 --total_steps 250000 \
    --save_path your_name/exp01.pth --log_path your_name/exp01.csv
```

Run `uv run train.py --help` to see all options.

| Option | Default |
|---|---|
| `--env_id` | `ALE/Pong-v5` |
| `--lr` | `1e-4` |
| `--gamma` | `0.99` |
| `--batch_size` | `32` |
| `--eps_start` | `1.0` |
| `--eps_end` | `0.01` |
| `--eps_decay` | `50000` |
| `--total_steps` | `250000` |
| `--buffer_size` | `50000` |
| `--learning_starts` | `10000` |
| `--train_freq` | `1` |
| `--target_update` | `1000` |
| `--save_path` | `dqn_model.pth` |
| `--log_path` | `training_log.csv` |
| `--seed` | `42` |

The model is saved every 50,000 steps and at the end. The CSV log has one row per episode: `episode, episode_length, total_steps, reward, epsilon`.

## Results

| Member | Exp | lr | gamma | batch | epsilon decay | Steps | Avg reward (last 10) |
|---|---|---|---|---|---|---|---|
| Shem Ayioka | exp01 | 1e-4 | 0.99 | 32 | 100k | ~100k | -21.0 |
