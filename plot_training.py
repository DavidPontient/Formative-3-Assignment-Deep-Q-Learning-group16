"""
Plot reward and episode length from a training log.

Usage:
    uv run plot_training.py shem_ayioka/exp08.csv shem_ayioka/exp08_training.png
"""

import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def rolling_mean(values, window=10):
    return [np.mean(values[max(0, i - window + 1):i + 1]) for i in range(len(values))]


log_path, out_path = sys.argv[1], sys.argv[2]
with open(log_path) as f:
    rows = list(csv.DictReader(f))

steps = [int(r["total_steps"]) for r in rows]
rewards = [float(r["reward"]) for r in rows]
lengths = [int(r["episode_length"]) for r in rows]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

ax1.plot(steps, rewards, alpha=0.3, label="episode reward")
ax1.plot(steps, rolling_mean(rewards), label="rolling mean (10)")
ax1.set_ylabel("Reward")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(steps, lengths, alpha=0.3, label="episode length")
ax2.plot(steps, rolling_mean(lengths), label="rolling mean (10)")
ax2.set_ylabel("Episode length (steps)")
ax2.set_xlabel("Total training steps")
ax2.legend()
ax2.grid(alpha=0.3)

fig.suptitle(f"{log_path} - {len(rows)} episodes, {steps[-1]} steps")
fig.tight_layout()
fig.savefig(out_path, dpi=120)
print(f"Saved {out_path}")
