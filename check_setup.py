"""Checks that this laptop can open the Pong window and run PyTorch.

It uses make_env from train.py. The small network below is a stand-in with
random weights, only to test the greedy action loop. It is not the group's
QNetwork.

Run: uv run check_setup.py
"""

import numpy as np
import torch
import torch.nn as nn

from train import make_env


class TinyNet(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=8, stride=4), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 20 * 20, n_actions),
        )

    def forward(self, x):
        return self.net(x.float() / 255.0)


def main():
    print("PyTorch version:", torch.__version__)

    env = make_env("ALE/Pong-v5", render_mode="human")
    model = TinyNet(env.action_space.n)
    model.eval()

    state, _ = env.reset()
    for _ in range(300):
        with torch.no_grad():
            x = torch.as_tensor(np.array(state)).unsqueeze(0)
            action = model(x).argmax(dim=1).item()
        state, reward, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            state, _ = env.reset()

    env.close()
    print("Window and PyTorch check passed.")


if __name__ == "__main__":
    main()
