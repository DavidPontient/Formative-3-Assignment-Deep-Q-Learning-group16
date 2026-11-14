"""Evaluate every saved model with greedy play and write results_table_with_eval.csv.

For each experiment it plays several full Pong games with the saved model, always
choosing the action with the highest Q-value (no random moves, no window), and
combines that with the training results from the log file.

Usage:
    uv run evaluate_models.py
    uv run evaluate_models.py --folder david_pontient --episodes 10
"""

import argparse
import csv
import os

import numpy as np
import torch

from train import QNetwork, make_env

# gamma and batch size for each experiment. Everything else is the same in all runs.
SETTINGS = {
    1: (0.99, 32), 2: (0.90, 32), 3: (0.95, 32), 4: (0.97, 32), 5: (0.999, 32),
    6: (0.99, 64), 7: (0.99, 128), 8: (0.95, 64), 9: (0.999, 64), 10: (0.90, 128),
}
LEARNING_RATE = 1e-4
EPS_START, EPS_END, EPS_DECAY = 1.0, 0.01, 50_000


def training_stats(csv_path):
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    rewards = [float(r["reward"]) for r in rows]
    return {
        "total_steps": int(rows[-1]["total_steps"]),
        "episodes": len(rows),
        "train_avg_last10": round(float(np.mean(rewards[-10:])), 2),
        "train_best_episode": max(rewards),
    }


def evaluate(model_path, episodes, env_id):
    env = make_env(env_id)
    model = QNetwork(env.action_space.n)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    scores, lengths = [], []
    for episode in range(episodes):
        state, _ = env.reset(seed=1000 + episode)
        done, score, steps = False, 0.0, 0
        while not done:
            with torch.no_grad():
                x = torch.as_tensor(np.array(state)).unsqueeze(0)
                action = model(x).argmax(dim=1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            steps += 1
            done = terminated or truncated
        scores.append(score)
        lengths.append(steps)
    env.close()
    return scores, lengths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default="david_pontient")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--env_id", default="ALE/Pong-v5")
    args = parser.parse_args()

    header = ["experiment", "learning_rate", "gamma", "batch_size", "epsilon_start",
              "epsilon_end", "epsilon_decay", "total_steps", "episodes",
              "train_avg_last10", "train_best_episode", "eval_episodes",
              "eval_avg_reward", "eval_std", "eval_min", "eval_max", "eval_avg_length"]
    out_path = os.path.join(args.folder, "results_table_with_eval.csv")
    rows = []

    for exp, (gamma, batch) in SETTINGS.items():
        name = f"exp{exp:02d}"
        model_path = os.path.join(args.folder, f"{name}.pth")
        log_path = os.path.join(args.folder, f"{name}.csv")
        if not (os.path.exists(model_path) and os.path.exists(log_path)):
            print(f"Skipping {name}: model or log file not found")
            continue

        print(f"Evaluating {name} ({args.episodes} games)...", flush=True)
        stats = training_stats(log_path)
        scores, lengths = evaluate(model_path, args.episodes, args.env_id)
        print(f"  average score {np.mean(scores):.2f}", flush=True)

        rows.append([
            exp, LEARNING_RATE, gamma, batch, EPS_START, EPS_END, EPS_DECAY,
            stats["total_steps"], stats["episodes"], stats["train_avg_last10"],
            stats["train_best_episode"], args.episodes,
            round(float(np.mean(scores)), 2), round(float(np.std(scores)), 2),
            min(scores), max(scores), round(float(np.mean(lengths)), 1),
        ])

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print("Saved", out_path)


if __name__ == "__main__":
    main()
