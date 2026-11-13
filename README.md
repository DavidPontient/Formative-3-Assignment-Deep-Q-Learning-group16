# DQN Atari - Group 16

Deep Q-Network agent trained on Atari Pong (`ALE/Pong-v5`) using Gymnasium and PyTorch.

## Files

- `train.py` - training script
- `play.py` - watch a trained model play in a window
- `evaluate.py` - score saved models without a window
- `check_env.py` - checks the environment and GPU
- `check_setup.py` - checks the Pong window and PyTorch work
- `run_experiments.ps1` - runs David's experiments on Windows
- `notebooks/` - notebooks
- `shem_ayioka/` - Shem's models, logs and results
- `rachel_toronga/` - Rachel's results

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

## Playing and evaluating

Watch a model play:

```bash
uv run play.py --model shem_ayioka/exp08.pth --episodes 3
```

Score models with no window (5 games each, no exploration):

```bash
uv run evaluate.py shem_ayioka/exp01.pth shem_ayioka/exp08.pth --episodes 5
```

## Results

- Shem (learning rate x batch size): [shem_ayioka/results.md](shem_ayioka/results.md). Best model is `exp08` (lr 5e-5, batch 32), averaging +3.6 over 5 games.
- Rachel (epsilon): [rachel_toronga/Hyperparameter_experiment_results.md](rachel_toronga/Hyperparameter_experiment_results.md)
