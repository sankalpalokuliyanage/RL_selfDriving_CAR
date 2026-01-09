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

        # --- Screen Setup ---
        pygame.init()
        pygame.font.init()
        info = pygame.display.Info()
        self.SCREEN_WIDTH = info.current_w
        self.SCREEN_HEIGHT = info.current_h
        self.FPS = 60

        # --- Dimensions ---
        self.ROAD_WIDTH = int(self.SCREEN_WIDTH * 0.5)
        self.LANE_COUNT = 4
        self.LANE_WIDTH = self.ROAD_WIDTH // self.LANE_COUNT
        self.CAR_WIDTH = int(self.LANE_WIDTH * 0.55)
        self.CAR_HEIGHT = int(self.CAR_WIDTH * 1.9)

        # --- AI Config ---
        # Action: [Steering (-1 to 1), Throttle (-1 to 1)]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Observation: [Speed, Angle, LaneDiff, 7x Lidar Rays]
        # Increased Lidar resolution (7 rays instead of 5)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # --- Lidar Settings (LONG RANGE) ---
        self.RAY_LENGTH = 800  # Doubled range!
        self.ray_angles = [-55, -35, -15, 0, 15, 35, 55]  # Wider FOV

        self.window = None
        self.clock = None
        self.font = pygame.font.SysFont("Consolas", 18, bold=True)
        self.big_font = pygame.font.SysFont("Arial", 80, bold=True)

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

        # Create initial straight road
        self.road_segments = []
        for y in range(0, self.SCREEN_HEIGHT + 200, 20):
            self.road_segments.append({'y': y, 'center_x': self.road_center_x})

        self.car_x = self.SCREEN_WIDTH // 2 - self.CAR_WIDTH // 2
        self.car_y = self.SCREEN_HEIGHT - 250
        self.speed = 0
        self.max_speed = 50

        self.traffic = []
        self._spawn_traffic()

        self.lidar_readings = [1.0] * 7
        self.score = 0
        self.total_reward = 0
        self.crash_cause = None

        return self._get_obs(), {}

    def _spawn_traffic(self):
        # Spawning logic optimized: Less chaos, safer gaps
        available_lanes = [0, 1, 2, 3]
        random.shuffle(available_lanes)

        # Spawn 1 or 2 cars max at a time to prevent "walls" of traffic
        num_cars = random.choice([1, 1, 2])

        top_y = self.road_segments[0]['y'] if self.road_segments else -100

        for i in range(num_cars):
            lane = available_lanes[i]
            seg = self.road_segments[0] if self.road_segments else None
            cx = seg['center_x'] if seg else self.road_center_x

            lane_offset = (lane * self.LANE_WIDTH) - (self.ROAD_WIDTH // 2) + (self.LANE_WIDTH // 2)

            self.traffic.append({
                'x': cx + lane_offset - (self.CAR_WIDTH // 2),
                'y': -400 - (random.randint(0, 300)),  # Spawn further up
                'speed': random.randint(20, 40),
                'lane': lane
            })

    def _cast_rays(self):
        readings = []
        cx = self.car_x + self.CAR_WIDTH // 2
        cy = self.car_y + self.CAR_HEIGHT // 2

        for angle in self.ray_angles:
            rad = math.radians(angle)
            sin_a = math.sin(rad)
            cos_a = math.cos(rad)

            dist = 1.0

            # Optimization: March in larger steps (20px) for speed, refine if hit
            step_size = 20
            steps = int(self.RAY_LENGTH / step_size)

            for i in range(1, steps + 1):
                check_dist = i * step_size
                check_x = cx + sin_a * check_dist
                check_y = cy - cos_a * check_dist

                hit = False

                # 1. Road Edges
                # Find closest road segment using index math (faster than loop)
                # Since segments are 20px apart, we can guess the index
                seg_idx = int(check_y / 20)
                if 0 <= seg_idx < len(self.road_segments):
                    seg = self.road_segments[seg_idx]
                    left_edge = seg['center_x'] - self.ROAD_WIDTH // 2
                    right_edge = seg['center_x'] + self.ROAD_WIDTH // 2
                    if check_x < left_edge or check_x > right_edge:
                        hit = True

                # 2. Traffic
                if not hit:
                    for car in self.traffic:
                        # Simple AABB collision check
                        if (car['x'] < check_x < car['x'] + self.CAR_WIDTH and
                                car['y'] < check_y < car['y'] + self.CAR_HEIGHT):
                            hit = True
                            break

                if hit:
                    dist = check_dist / self.RAY_LENGTH
                    break

            readings.append(dist)
        return readings

    def _get_obs(self):
        self.lidar_readings = self._cast_rays()

        # Calculate Lane Deviation (0.0 = center of lane, 1.0 = crossing line)
        # Find current lane center
        current_seg = next((s for s in self.road_segments if abs(s['y'] - self.car_y) < 15), None)
        lane_deviation = 0.0
        if current_seg:
            road_left = current_seg['center_x'] - self.ROAD_WIDTH // 2
            car_center = self.car_x + self.CAR_WIDTH // 2
            dist_from_left = car_center - road_left
            # Normalize to 0-4 range (lane index float)
            lane_pos = dist_from_left / self.LANE_WIDTH
            # Deviation from center of nearest lane (0.0 to 0.5)
            lane_deviation = abs((lane_pos % 1.0) - 0.5) * 2.0

        return np.array([
            self.speed / self.max_speed,
            self.curve_angle,
            lane_deviation,
            *self.lidar_readings
        ], dtype=np.float32)

    def step(self, action):
        self.frame_count += 1

        steer = float(action[0])
        throttle = float(action[1])

        # --- Physics Engine ---
        if throttle > 0:
            self.speed += throttle * 0.8  # Acceleration
        else:
            self.speed += throttle * 1.8  # Braking
        self.speed = np.clip(self.speed, 0, self.max_speed)

        # Road Curve Logic
        if self.frame_count % 300 == 0:
            self.target_curve = random.uniform(-2.5, 2.5)
        self.curve_angle += (self.target_curve - self.curve_angle) * 0.005  # Smoother transitions

        # Car Movement
        turn_force = steer * (self.speed * 0.40)
        centrifugal = self.curve_angle * (self.speed * 0.30)
        self.car_x += (turn_force - centrifugal)

        # World Movement
        # Remove segments that fell off screen
        # Optimization: Don't recreate list every frame
        speed_int = int(self.speed)
        for seg in self.road_segments:
            seg['y'] += speed_int

        # Efficiently manage road buffer
        self.road_segments = [s for s in self.road_segments if s['y'] < self.SCREEN_HEIGHT + 100]

        while len(self.road_segments) < (self.SCREEN_HEIGHT // 20) + 10:
            self.road_center_x += self.curve_angle * 5
            # Keep road somewhat on screen
            self.road_center_x = np.clip(self.road_center_x, 300, self.SCREEN_WIDTH - 300)
            self.road_segments.append({'y': -20, 'center_x': self.road_center_x})
            self.road_segments.sort(key=lambda s: s['y'])

        # Traffic Logic
        for car in self.traffic:
            car['y'] += (self.speed - car['speed'])
            car['x'] -= centrifugal  # Traffic follows road curve physics

            # Simple AI: Stay in lane
            seg = next((s for s in self.road_segments if abs(s['y'] - car['y']) < 25), None)
            if seg:
                lane_x = seg['center_x'] - (self.ROAD_WIDTH // 2) + (car['lane'] * self.LANE_WIDTH) + (
                            self.LANE_WIDTH // 2)
                car['x'] += (lane_x - (car['x'] + self.CAR_WIDTH / 2)) * 0.05

        # Cleanup Traffic
        if self.traffic and self.traffic[0]['y'] > self.SCREEN_HEIGHT + 100:
            self.traffic.pop(0)
            self.score += 1

        if not self.traffic or self.traffic[-1]['y'] > 400:  # Increased gap
            if random.random() < 0.02: self._spawn_traffic()

        # --- REWARD FUNCTION ---
        reward = 0
        terminated = False

        # 1. Speed Reward
        reward += (self.speed / self.max_speed) * 0.5

        # 2. Safety Penalty (Lidar)
        min_lidar = min(self.lidar_readings)
        if min_lidar < 0.2: reward -= 0.5  # Getting too close!

        # 3. Lane Centering Reward
        # We calculated 'lane_deviation' in _get_obs (0.0 is perfect, 1.0 is bad)
        # obs[2] is lane_deviation
        # reward += (1.0 - lane_deviation) * 0.1

        # 4. Crash Checks
        p_rect = pygame.Rect(self.car_x, self.car_y, self.CAR_WIDTH, self.CAR_HEIGHT)

        # Check Traffic
        for car in self.traffic:
            if p_rect.colliderect(pygame.Rect(car['x'], car['y'], self.CAR_WIDTH, self.CAR_HEIGHT)):
                terminated = True
                reward = -50
                self.crash_cause = "TRAFFIC HIT"

        # Check Road Edge
        p_seg = next((s for s in self.road_segments if abs(s['y'] - self.car_y) < 25), None)
        if p_seg:
            limit_l = p_seg['center_x'] - self.ROAD_WIDTH // 2
            limit_r = p_seg['center_x'] + self.ROAD_WIDTH // 2
            if self.car_x < limit_l or self.car_x + self.CAR_WIDTH > limit_r:
                terminated = True
                reward = -50
                self.crash_cause = "OFF ROAD"

        self.total_reward += reward

        if self.window: self.render()
        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        if self.window is None:
            self.window = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.FULLSCREEN)
            self.clock = pygame.time.Clock()
            self.load_assets()

        self.window.fill((20, 80, 20))  # Dark Grass

        # Draw Road Polygon
        points_l, points_r = [], []
        sorted_segs = sorted(self.road_segments, key=lambda s: s['y'])
        for s in sorted_segs:
            points_l.append((s['center_x'] - self.ROAD_WIDTH // 2, s['y']))
            points_r.append((s['center_x'] + self.ROAD_WIDTH // 2, s['y']))

        if len(points_l) > 2:
            pygame.draw.polygon(self.window, (50, 50, 60), points_l + list(reversed(points_r)))

        # Draw Markings
        for i in range(0, len(sorted_segs) - 2, 4):  # Optimize: Draw every 4th segment
            s = sorted_segs[i]
            s_next = sorted_segs[i + 2]

            # Edges
            l1, r1 = s['center_x'] - self.ROAD_WIDTH // 2, s['center_x'] + self.ROAD_WIDTH // 2
            l2, r2 = s_next['center_x'] - self.ROAD_WIDTH // 2, s_next['center_x'] + self.ROAD_WIDTH // 2
            pygame.draw.line(self.window, (200, 200, 200), (l1, s['y']), (l2, s_next['y']), 5)
            pygame.draw.line(self.window, (200, 200, 200), (r1, s['y']), (r2, s_next['y']), 5)

            # Lanes
            if (i // 4) % 2 == 0:
                for lane in range(1, self.LANE_COUNT):
                    off = (lane * self.LANE_WIDTH) - (self.ROAD_WIDTH // 2)
                    pygame.draw.line(self.window, (255, 255, 255),
                                     (s['center_x'] + off, s['y']),
                                     (s_next['center_x'] + off, s_next['y']), 2)

        # Draw Traffic
        for car in self.traffic:
            if self.traffic_img:
                self.window.blit(self.traffic_img, (car['x'], car['y']))
            else:
                pygame.draw.rect(self.window, (200, 50, 50), (car['x'], car['y'], self.CAR_WIDTH, self.CAR_HEIGHT))

        # Draw Player
        if self.car_img:
            self.window.blit(self.car_img, (self.car_x, self.car_y))
        else:
            pygame.draw.rect(self.window, (0, 255, 255), (self.car_x, self.car_y, self.CAR_WIDTH, self.CAR_HEIGHT))

        # Draw Lidar (Visual Only)
        cx, cy = self.car_x + self.CAR_WIDTH // 2, self.car_y + self.CAR_HEIGHT // 2
        for i, dist in enumerate(self.lidar_readings):
            rad = math.radians(self.ray_angles[i])
            # Color gradient based on distance
            color = (0, 255, 0)
            if dist < 0.5: color = (255, 165, 0)
            if dist < 0.2: color = (255, 0, 0)

            end_x = cx + math.sin(rad) * (self.RAY_LENGTH * dist)
            end_y = cy - math.cos(rad) * (self.RAY_LENGTH * dist)
            pygame.draw.line(self.window, color, (cx, cy), (end_x, end_y), 2)

        # HUD
        self._draw_hud()

        # Crash Overlay
        if self.crash_cause:
            s = pygame.Surface((self.SCREEN_WIDTH, self.SCREEN_HEIGHT), pygame.SRCALPHA)
            s.fill((255, 0, 0, 100))
            self.window.blit(s, (0, 0))
            txt = self.big_font.render(self.crash_cause, True, (255, 255, 255))
            self.window.blit(txt, (self.SCREEN_WIDTH // 2 - txt.get_width() // 2, self.SCREEN_HEIGHT // 2))

        pygame.display.update()
        self.clock.tick(self.FPS)

    def _draw_hud(self):
        # Panel
        pygame.draw.rect(self.window, (0, 0, 0), (20, 20, 320, 180))
        pygame.draw.rect(self.window, (255, 255, 255), (20, 20, 320, 180), 2)

        lines = [
            f"SPEED:  {int(self.speed)} KM/H",
            f"REWARD: {self.total_reward:.1f}",
            f"DIST:   {self.score} Cars",
            f"LIDAR:  {min(self.lidar_readings):.2f}"
        ]

        for i, line in enumerate(lines):
            t = self.font.render(line, True, (0, 255, 0))
            self.window.blit(t, (40, 40 + i * 30))

    def close(self):
        if self.window: pygame.quit()


if __name__ == "__main__":
    env = AdvancedCarEnv()
    env.reset()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                env.close()
                exit()

        keys = pygame.key.get_pressed()
        steer = 0
        if keys[pygame.K_LEFT]: steer = -1
        if keys[pygame.K_RIGHT]: steer = 1

        env.step([steer, 0.5])
        env.render()