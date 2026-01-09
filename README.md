# 🚗 Autonomous Driving with Deep Reinforcement Learning

A custom high-performance self-driving car simulation built with **Python**, **Pygame**, and **Gymnasium**. 
The agent learns to drive, overtake traffic, and navigate curved roads using **Proximal Policy Optimization (PPO)** from Stable Baselines3.

<img width="2551" height="1435" alt="image" src="https://github.com/user-attachments/assets/eb7c82c3-d5e1-4783-9f29-ff2a63d13f5d" />


## 🌟 Features

* **Custom Gym Environment:** Built from scratch inheriting from `gymnasium.Env`.
* **Raycasting LIDAR:** Simulates 7 laser sensors to detect distance to road edges and traffic.
* **Continuous Control:** The AI outputs continuous values for Steering `[-1, 1]` and Throttle `[-1, 1]`.
* **Dynamic Physics:** Includes centrifugal force, friction, and momentum.
* **Procedural Generation:** Endless road with random curves, straightaways, and traffic patterns.
* **Real-Time Telemetry:** Full HUD showing neural network inputs, speed, and reward metrics.

## 🛠️ Tech Stack

* **Language:** Python 3.8+
* **Simulation:** Pygame
* **RL Framework:** Stable Baselines3 (PPO Algorithm)
* **Environment:** Gymnasium (OpenAI Gym standard)

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sankalpalokuliyanage/RL_selfDriving_CAR.git
    cd RL_selfDriving_CAR
    ```

2.  **Install dependencies:**
    ```bash
    pip install gymnasium stable-baselines3 shimmy pygame numpy
    ```

3.  **Assets:**
    Ensure you have `car.png` and `traffic.png` in the root directory.

## 🚀 How to Run

### 1. Train the AI 🧠
To train the agent from scratch. This process usually takes 20-30 minutes to see good results (300k+ steps).
```bash
python train.py
