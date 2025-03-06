import gymnasium as gym
from stable_baselines3 import PPO
import os

# Create the environment with video recording
video_dir = "videos"
os.makedirs(video_dir, exist_ok=True)

env = gym.make("LunarLander-v3", render_mode="rgb_array")  # Use "rgb_array" for video recording
env = gym.wrappers.RecordVideo(env, video_dir, episode_trigger=lambda e: True)  # Record every episode

# Load the trained model
model_dir = "models/PPO 1741252172"
model_path = f"{model_dir}/180000.zip"
model = PPO.load(model_path, env=env)

# Run the model and save the video
episodes = 10
for ep in range(episodes):
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

env.close()  # Close the environment

print(f"Videos saved in: {video_dir}")
