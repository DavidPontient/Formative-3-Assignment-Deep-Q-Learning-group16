"""
Evaluate saved models with greedy actions (no exploration), no window.

Usage:
    uv run evaluate.py shem_ayioka/exp01.pth shem_ayioka/exp02.pth --episodes 5
"""

import argparse

import numpy as np
import torch

from train import QNetwork, make_env


def evaluate(model_path, env_id, episodes, seed):
    env = make_env(env_id)
    model = QNetwork(env.action_space.n)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    scores = []
    for i in range(episodes):
        state, _ = env.reset(seed=seed + i)
        done = False
        score = 0.0
        while not done:
            with torch.no_grad():
                x = torch.as_tensor(np.array(state)).unsqueeze(0)
                action = model(x).argmax(dim=1).item()
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            done = terminated or truncated
        scores.append(score)
    env.close()
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--env_id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    for path in args.models:
        scores = evaluate(path, args.env_id, args.episodes, args.seed)
        print(f"{path}: mean {np.mean(scores):.1f}, scores {[int(s) for s in scores]}", flush=True)


if __name__ == "__main__":
    main()
