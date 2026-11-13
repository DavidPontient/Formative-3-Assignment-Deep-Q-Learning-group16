"""Load the trained DQN model and watch it play Pong.

The agent is evaluated greedily: at every step it takes the action with the
highest predicted Q-value. There is no epsilon and no learning here.

It uses make_env and QNetwork from train.py, so the game, the preprocessing
and the network are exactly the same as in training.

Usage:
    uv run play.py
    uv run play.py --model shem_ayioka/exp01.pth --episodes 3
"""

import argparse
import time

import numpy as np
import torch

from train import QNetwork, make_env


def load_model(path, n_actions):
    model = QNetwork(n_actions)                         # same architecture as training
    state_dict = torch.load(path, map_location="cpu")   # works without a GPU
    model.load_state_dict(state_dict)
    model.eval()                                         # evaluation mode
    return model


def play(args):
    env = make_env(args.env_id, render_mode="human")    # same game as training
    model = load_model(args.model, env.action_space.n)

    scores = []
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        done = False
        score, steps = 0.0, 0

        while not done:
            with torch.no_grad():
                x = torch.as_tensor(np.array(state)).unsqueeze(0)
                action = model(x).argmax(dim=1).item()   # greedy action
            state, reward, terminated, truncated, _ = env.step(action)
            score += reward
            steps += 1
            done = terminated or truncated
            if args.delay > 0:
                time.sleep(args.delay)

        scores.append(score)
        print(f"Episode {episode}: score {score:.0f}, steps {steps}")

    env.close()
    print(f"Average score over {len(scores)} episode(s): {np.mean(scores):.2f}")


def main():
    parser = argparse.ArgumentParser(description="Watch the trained DQN agent play")
    parser.add_argument("--env_id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--model", type=str, default="dqn_model.pth")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--delay", type=float, default=0.0,
                        help="seconds to wait after each step if the game runs too fast")
    play(parser.parse_args())


if __name__ == "__main__":
    main()
