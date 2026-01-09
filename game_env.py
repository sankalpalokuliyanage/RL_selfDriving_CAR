import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import random
import os
import math


class AdvancedCarEnv(gym.Env):
    def __init__(self):
        super(AdvancedCarEnv, self).__init__()

        # --- Screen Setup (Maximize) ---
        pygame.init()
        pygame.font.init()
        info = pygame.display.Info()
        self.SCREEN_WIDTH = info.current_w
        self.SCREEN_HEIGHT = info.current_h
        self.FPS = 60

        # --- Configs ---
        self.ROAD_WIDTH = int(self.SCREEN_WIDTH * 0.5)
        self.LANE_COUNT = 4
        self.LANE_WIDTH = self.ROAD_WIDTH // self.LANE_COUNT
        self.CAR_WIDTH = int(self.LANE_WIDTH * 0.6)
        self.CAR_HEIGHT = int(self.CAR_WIDTH * 1.8)

        # --- Action Space (Steer, Gas/Brake) ---
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # --- Observation Space (Lidar + State) ---
        # 5 Lidar rays + Speed + Angle + Track Position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )

        # Lidar Setup
        self.RAY_LENGTH = 400
        self.ray_angles = [-60, -30, 0, 30, 60]

        self.window = None
        self.clock = None
        self.font = pygame.font.SysFont("Consolas", 20, bold=True)
        self.big_font = pygame.font.SysFont("Arial", 60, bold=True)

    def load_assets(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.car_img = pygame.image.load(os.path.join(script_dir, "car.png"))
            self.car_img = pygame.transform.scale(self.car_img, (self.CAR_WIDTH, self.CAR_HEIGHT))
            self.traffic_img = pygame.image.load(os.path.join(script_dir, "traffic.png"))
            self.traffic_img = pygame.transform.scale(self.traffic_img, (self.CAR_WIDTH, self.CAR_HEIGHT))
        except:
            self.car_img = None
            self.traffic_img = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.road_center_x = self.SCREEN_WIDTH // 2
        self.curve_angle = 0.0
        self.target_curve = 0.0
        self.frame_count = 0

        # Initialize Road
        self.road_segments = []
        for y in range(0, self.SCREEN_HEIGHT + 100, 20):
            self.road_segments.append({'y': y, 'center_x': self.road_center_x})

        self.car_x = self.SCREEN_WIDTH // 2 - self.CAR_WIDTH // 2
        self.car_y = self.SCREEN_HEIGHT - 250
        self.speed = 0
        self.max_speed = 45

        self.traffic = []
        self._spawn_traffic()

        self.lidar_readings = [1.0] * 5
        self.score = 0
        self.total_reward = 0
        self.last_steer = 0
        self.crash_cause = None  # For UI

        return self._get_obs(), {}

    def _spawn_traffic(self):
        lane = random.randint(0, self.LANE_COUNT - 1)
        spawn_x = self.road_segments[0]['center_x'] if self.road_segments else self.road_center_x
        lane_offset = (lane * self.LANE_WIDTH) - (self.ROAD_WIDTH // 2) + (self.LANE_WIDTH // 2)

        self.traffic.append({
            'x': spawn_x + lane_offset - (self.CAR_WIDTH // 2),
            'y': -300,
            'speed': random.randint(15, 35),
            'lane': lane
        })

    def _cast_rays(self):
        readings = []
        cx = self.car_x + self.CAR_WIDTH // 2
        cy = self.car_y + self.CAR_HEIGHT // 2

        for angle in self.ray_angles:
            rad = math.radians(angle)
            dx, dy = math.sin(rad), -math.cos(rad)
            dist = 1.0

            for i in range(1, 11):
                d = (self.RAY_LENGTH / 10) * i
                check_x, check_y = cx + dx * d, cy + dy * d
                hit = False

                # Road Edge Check
                seg = next((s for s in self.road_segments if abs(s['y'] - check_y) < 20), None)
                if seg:
                    edge_l = seg['center_x'] - self.ROAD_WIDTH // 2
                    edge_r = seg['center_x'] + self.ROAD_WIDTH // 2
                    if check_x < edge_l or check_x > edge_r: hit = True

                # Traffic Check
                for car in self.traffic:
                    if (car['x'] < check_x < car['x'] + self.CAR_WIDTH and
                            car['y'] < check_y < car['y'] + self.CAR_HEIGHT): hit = True

                if hit:
                    dist = i / 10.0
                    break
            readings.append(dist)
        return readings

    def _get_obs(self):
        self.lidar_readings = self._cast_rays()
        # Normalized observation
        return np.array([
            self.speed / self.max_speed,
            self.curve_angle,
            self.last_steer,
            *self.lidar_readings
        ], dtype=np.float32)

    def step(self, action):
        self.frame_count += 1

        steer = float(action[0])
        throttle = float(action[1])
        self.last_steer = steer

        # 1. Road Generation
        if self.frame_count % 250 == 0: self.target_curve = random.uniform(-3.0, 3.0)
        self.curve_angle += (self.target_curve - self.curve_angle) * 0.01

        # 2. Car Physics
        if throttle > 0:
            self.speed += throttle * 0.6
        else:
            self.speed += throttle * 1.5  # Strong brakes
        self.speed = np.clip(self.speed, 0, self.max_speed)

        # Steering is harder at high speeds
        turn_rate = steer * (self.speed * 0.35)
        self.car_x += turn_rate
        self.car_x -= self.curve_angle * (self.speed * 0.28)  # Centrifugal force

        # 3. Move World
        for seg in self.road_segments: seg['y'] += self.speed
        self.road_segments = [s for s in self.road_segments if s['y'] < self.SCREEN_HEIGHT]

        while len(self.road_segments) < (self.SCREEN_HEIGHT // 20) + 5:
            self.road_center_x += self.curve_angle * 5
            max_drift = self.SCREEN_WIDTH // 3
            self.road_center_x = np.clip(self.road_center_x, (self.SCREEN_WIDTH // 2) - max_drift,
                                         (self.SCREEN_WIDTH // 2) + max_drift)
            self.road_segments.append({'y': -20, 'center_x': self.road_center_x})
            self.road_segments.sort(key=lambda s: s['y'])

        # 4. Traffic Physics
        for car in self.traffic:
            car['y'] += (self.speed - car['speed'])
            car['x'] -= self.curve_angle * (self.speed * 0.28)
            # Lane keeping
            seg = next((s for s in self.road_segments if abs(s['y'] - car['y']) < 25), None)
            if seg:
                tx = seg['center_x'] + ((car[
                                             'lane'] * self.LANE_WIDTH) - self.ROAD_WIDTH // 2 + self.LANE_WIDTH // 2) - self.CAR_WIDTH // 2
                car['x'] += (tx - car['x']) * 0.1

        if self.traffic and self.traffic[0]['y'] > self.SCREEN_HEIGHT:
            self.traffic.pop(0);
            self.score += 1
        if not self.traffic or self.traffic[-1]['y'] > 350:
            if random.random() < 0.03: self._spawn_traffic()

        # --- REWARD SYSTEM (The "Perfect" Logic) ---
        reward = 0
        terminated = False

        # A. Speed Reward (Encourage speed, penalize stopping)
        reward += (self.speed / self.max_speed) * 1.0
        if self.speed < 5: reward -= 0.1  # Penalty for moving too slow

        # B. Smoothness Penalty (Penalize jerky steering)
        reward -= abs(steer) * 0.1

        # C. Off-Road Check (Fatal Penalty)
        p_seg = next((s for s in self.road_segments if abs(s['y'] - self.car_y) < 25), None)
        if p_seg:
            limit_l = p_seg['center_x'] - self.ROAD_WIDTH // 2
            limit_r = p_seg['center_x'] + self.ROAD_WIDTH // 2
            if self.car_x < limit_l or self.car_x + self.CAR_WIDTH > limit_r:
                terminated = True
                reward = -50
                self.crash_cause = "OFF ROAD"

        # D. Collision Check (Fatal Penalty)
        player_rect = pygame.Rect(self.car_x, self.car_y, self.CAR_WIDTH, self.CAR_HEIGHT)
        for car in self.traffic:
            if player_rect.colliderect(pygame.Rect(car['x'], car['y'], self.CAR_WIDTH, self.CAR_HEIGHT)):
                terminated = True
                reward = -100
                self.crash_cause = "CRASHED"

        self.total_reward += reward

        if self.window: self.render()
        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        if self.window is None:
            self.window = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.FULLSCREEN)
            self.clock = pygame.time.Clock()
            self.load_assets()

        # 1. Background
        self.window.fill((30, 100, 30))  # Darker Grass

        # 2. Draw Road
        sorted_segs = sorted(self.road_segments, key=lambda s: s['y'])
        pts_l, pts_r = [], []
        for s in sorted_segs:
            pts_l.append((s['center_x'] - self.ROAD_WIDTH // 2, s['y']))
            pts_r.append((s['center_x'] + self.ROAD_WIDTH // 2, s['y']))

        if len(pts_l) > 2:
            pygame.draw.polygon(self.window, (60, 60, 60), pts_l + list(reversed(pts_r)))

        # 3. Draw Lines
        for i in range(len(sorted_segs) - 1):
            if i % 3 == 0: continue  # Optimization
            s1, s2 = sorted_segs[i], sorted_segs[i + 1]
            xl1, xl2 = s1['center_x'] - self.ROAD_WIDTH // 2, s2['center_x'] - self.ROAD_WIDTH // 2
            xr1, xr2 = s1['center_x'] + self.ROAD_WIDTH // 2, s2['center_x'] + self.ROAD_WIDTH // 2

            # White Edge Lines
            pygame.draw.line(self.window, (255, 255, 255), (xl1, s1['y']), (xl2, s2['y']), 6)
            pygame.draw.line(self.window, (255, 255, 255), (xr1, s1['y']), (xr2, s2['y']), 6)

            # Dashed Lane Lines
            if i % 6 < 3:
                for l in range(1, self.LANE_COUNT):
                    off = (l * self.LANE_WIDTH) - (self.ROAD_WIDTH // 2)
                    pygame.draw.line(self.window, (200, 200, 200), (s1['center_x'] + off, s1['y']),
                                     (s2['center_x'] + off, s2['y']), 2)

        # 4. Draw Entities
        for car in self.traffic:
            if self.traffic_img:
                self.window.blit(self.traffic_img, (car['x'], car['y']))
            else:
                pygame.draw.rect(self.window, (200, 50, 50), (car['x'], car['y'], self.CAR_WIDTH, self.CAR_HEIGHT))

        if self.car_img:
            self.window.blit(self.car_img, (self.car_x, self.car_y))
        else:
            pygame.draw.rect(self.window, (0, 255, 255), (self.car_x, self.car_y, self.CAR_WIDTH, self.CAR_HEIGHT))

        # 5. Draw Lidar
        cx, cy = self.car_x + self.CAR_WIDTH // 2, self.car_y + self.CAR_HEIGHT // 2
        for i, dist in enumerate(self.lidar_readings):
            rad = math.radians(self.ray_angles[i])
            color = (0, 255, 0) if dist > 0.8 else (255, 165, 0) if dist > 0.5 else (255, 0, 0)
            ex = cx + math.sin(rad) * (self.RAY_LENGTH * dist)
            ey = cy - math.cos(rad) * (self.RAY_LENGTH * dist)
            pygame.draw.line(self.window, color, (cx, cy), (ex, ey), 2)
            pygame.draw.circle(self.window, color, (int(ex), int(ey)), 4)

        # --- 6. DASHBOARD (The "Advanced" Part) ---
        # Draw a semi-transparent panel
        panel_w, panel_h = 300, 250
        panel = pygame.Surface((panel_w, panel_h))
        panel.set_alpha(200)
        panel.fill((0, 0, 0))
        self.window.blit(panel, (20, 20))

        # Telemetry Text
        texts = [
            f"SPEED:    {int(self.speed)} km/h",
            f"THROTTLE: {self.last_steer:.2f}",  # Actually steer
            f"REWARD:   {self.total_reward:.1f}",
            f"SCORE:    {self.score}",
            "",
            "LIDAR SENSORS:",
            f"L: {self.lidar_readings[0]:.2f}  R: {self.lidar_readings[4]:.2f}",
            f"C: {self.lidar_readings[2]:.2f}"
        ]

        for i, t in enumerate(texts):
            col = (0, 255, 255) if "SPEED" in t else (255, 255, 255)
            label = self.font.render(t, True, col)
            self.window.blit(label, (35, 35 + i * 25))

        # 7. CRASH SCREEN
        if self.crash_cause:
            overlay = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            overlay.set_alpha(150)
            overlay.fill((255, 0, 0))  # Red Flash
            self.window.blit(overlay, (0, 0))

            msg = self.big_font.render(f"{self.crash_cause}!", True, (255, 255, 255))
            self.window.blit(msg, (self.SCREEN_WIDTH // 2 - msg.get_width() // 2, self.SCREEN_HEIGHT // 2))

        pygame.display.update()
        self.clock.tick(self.FPS)

    def close(self):
        if self.window: pygame.quit()


if __name__ == "__main__":
    env = AdvancedCarEnv()
    env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False

        # Manual Test (Arrows)
        keys = pygame.key.get_pressed()
        steer = 0
        gas = 0.5
        if keys[pygame.K_LEFT]: steer = -1.0
        if keys[pygame.K_RIGHT]: steer = 1.0
        if keys[pygame.K_UP]: gas = 1.0
        if keys[pygame.K_DOWN]: gas = -1.0

        env.step([steer, gas])
        env.render()
    env.close()