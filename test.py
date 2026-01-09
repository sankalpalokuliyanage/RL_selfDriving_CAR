import gymnasium as gym
from stable_baselines3 import PPO
from game_env import AdvancedCarEnv

# 1. Load Environment
env = AdvancedCarEnv()

# 2. Load Model
model_path = "ultimate_brain"
print(f"Loading {model_path}...")

try:
    model = PPO.load(model_path, env=env)
except FileNotFoundError:
    print("ERROR: Brain file not found. Did you run train_ultimate.py?")
    exit()

# 3. Watch it Drive
obs, _ = env.reset()
env.render()  # Opens the Full Screen window

print("AI is driving... Press ESC to quit.")

while True:
    # Get action from AI
    action, _states = model.predict(obs, deterministic=True)

    # Execute action
    obs, reward, terminated, truncated, info = env.step(action)

    # Auto-restart on crash
    if terminated:
        print(f"Game Over! Final Reward: {env.total_reward:.1f}")
        obs, _ = env.reset()