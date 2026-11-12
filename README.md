# Deep Q-Learning on Atari Pong — Group 16

Formative 3 assignment: training a Deep Q-Network (DQN) agent to play Atari Pong (`ALE/Pong-v5`) using Gymnasium, ALE and PyTorch.

## Repository layout

```
dqn-atari/
├── train.py            # DQN training script (all hyperparameters are CLI flags)
├── check_env.py        # Quick sanity check: environment, preprocessing, GPU
├── notebooks/          # Jupyter notebooks (analysis / Colab runs)
├── shem_ayioka/        # Shem Ayioka's experiments
│   ├── exp01.pth       #   trained model weights
│   └── exp01.csv       #   per-episode training log
├── src/dqn_atari/      # Package stub (created by uv)
├── pyproject.toml      # Project metadata and dependencies
├── uv.lock             # Locked dependency versions
└── .python-version     # Python 3.11
```

Each group member keeps their own experiments in a folder named after them (e.g. `shem_ayioka/`), with one `.pth` model and one `.csv` log per experiment.

## Setup

Requires **Python 3.11+**. The project is managed with [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:DavidPontient/Formative-3-Assignment-Deep-Q-Learning-group16.git
cd Formative-3-Assignment-Deep-Q-Learning-group16

# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create .venv and install all dependencies from uv.lock
uv sync
```

<details>
<summary>Without uv (plain pip)</summary>

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install "gymnasium[atari]>=1.4.0" "ale-py>=0.12.1" torch numpy matplotlib opencv-python
```

Then replace `uv run` with `python` in the commands below.
</details>

### Check the environment

```bash
uv run check_env.py
```

Expected output: observation shape `(4, 84, 84)`, 6 actions, a random-agent score around -20 / -21, and whether CUDA is available. Training works on CPU but is much faster on a GPU.

## Training

Basic run with default hyperparameters:

```bash
uv run train.py
```

This saves `dqn_model.pth` and `training_log.csv` in the current folder. To keep your experiments organised, pass your own folder and experiment name:

```bash
uv run train.py --lr 1e-4 --batch_size 32 --total_steps 250000 \
    --save_path your_name/exp01.pth --log_path your_name/exp01.csv
```

The output folder is created automatically. Progress is printed after every episode:

```
step    12345 | episode   14 | reward  -20.0 | avg(10)  -20.4 | eps 0.765 | 120s
```

The model is checkpointed to `--save_path` every 50,000 steps and again at the end, so an interrupted run (e.g. a Colab disconnect) still leaves usable weights.

### Hyperparameters

| Flag | Default | Description |
|---|---|---|
| `--env_id` | `ALE/Pong-v5` | Gymnasium Atari environment |
| `--lr` | `1e-4` | Adam learning rate |
| `--gamma` | `0.99` | Discount factor |
| `--batch_size` | `32` | Replay batch size per gradient update |
| `--eps_start` | `1.0` | Initial exploration rate |
| `--eps_end` | `0.01` | Final exploration rate |
| `--eps_decay` | `50000` | Steps over which epsilon decays linearly |
| `--total_steps` | `250000` | Total environment steps |
| `--buffer_size` | `50000` | Replay buffer capacity |
| `--learning_starts` | `10000` | Random steps before learning begins |
| `--train_freq` | `1` | Gradient update every N steps |
| `--target_update` | `1000` | Copy online → target network every N steps |
| `--save_path` | `dqn_model.pth` | Where to save model weights |
| `--log_path` | `training_log.csv` | Where to write the episode log |
| `--seed` | `42` | Random seed |

Run `uv run train.py --help` to see the full list.

### Training log format

Each CSV has one row per episode:

| Column | Meaning |
|---|---|
| `episode` | Episode number |
| `episode_length` | Agent steps in the episode |
| `total_steps` | Cumulative environment steps |
| `reward` | Episode score (Pong: -21 to +21) |
| `epsilon` | Exploration rate at the end of the episode |

## Implementation overview

- **Preprocessing:** grayscale, resize to 84×84, frame skip 4, stack the last 4 frames, giving input shape `(4, 84, 84)`.
- **Q-network:** the Nature-DQN CNN, with three conv layers (32/64/64 filters), then a 512-unit dense layer, then one Q-value per action.
- **Replay buffer:** fixed-size circular buffer stored as `uint8` to save memory; uniform random sampling.
- **Target network:** a frozen copy of the Q-network, synced every `--target_update` steps, used for Bellman targets `r + γ · max Q_target(s', a')`.
- **Exploration:** ε-greedy with linear decay.
- **Loss / optimisation:** Huber (smooth L1) loss, Adam, gradient-norm clipping at 10.
- **Terminal handling:** only `terminated` (not `truncated`) zeroes the bootstrap term.

## Loading a trained model

```python
import torch
from train import QNetwork, make_env

env = make_env("ALE/Pong-v5", render_mode="human")
q_net = QNetwork(env.action_space.n)
q_net.load_state_dict(torch.load("shem_ayioka/exp01.pth", map_location="cpu"))
q_net.eval()

obs, _ = env.reset()
done = False
while not done:
    with torch.no_grad():
        action = q_net(torch.as_tensor(obs).unsqueeze(0)).argmax(1).item()
    obs, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
env.close()
```

## Experiments

| Member | Exp | lr | gamma | batch | eps (start → end, decay) | Steps logged | Avg reward (last 10 eps) | Notes |
|---|---|---|---|---|---|---|---|---|
| Shem Ayioka | exp01 | 1e-4 | 0.99 | 32 | 1.0 → 0.01, 100k | ~100k (114 episodes) | -21.0 | No improvement yet over a random agent; needs more steps |


