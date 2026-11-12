import gymnasium as gym
import ale_py
import torch

gym.register_envs(ale_py)

# frameskip=1 because AtariPreprocessing handles the frame skipping
env = gym.make("ALE/Pong-v5", frameskip=1)
env = gym.wrappers.AtariPreprocessing(env, frame_skip=4, screen_size=84, grayscale_obs=True)
env = gym.wrappers.FrameStackObservation(env, stack_size=4)

obs, info = env.reset(seed=42)
print("Observation shape:", obs.shape)        # expect (4, 84, 84)
print("Number of actions:", env.action_space.n)  # expect 6

total_reward = 0
done = False
while not done:
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    total_reward += reward
    done = terminated or truncated
print("Random agent score:", total_reward)    # expect around -20 or -21
print("Torch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
env.close()
