import gymnasium as gym
from stable_baselines3 import PPO
from game_env import AdvancedCarEnv  # Imports your new environment

# 1. Create the Environment
env = AdvancedCarEnv()

# 2. Define the Model
# We use a larger learning rate (3e-4) to help it learn the complex road faster
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    batch_size=64,
    n_steps=2048
)

print("--------------------------------")
print("STARTING ULTIMATE TRAINING...")
print("The window will NOT open (faster training).")
print("Recommended training time: 10-20 minutes.")
print("--------------------------------")

# 3. Train for 300,000 steps
# (If it drives badly, increase this to 500,000 or 1,000,000)
model.learn(total_timesteps=300000)

# 4. Save
model_path = "ultimate_brain"
model.save(model_path)

print("--------------------------------")
print(f"Training Complete! Model saved as '{model_path}.zip'")
print("--------------------------------")