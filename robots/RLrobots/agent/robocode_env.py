import gymnasium as gym
import numpy as np

class RobocodeEnv(gym.Env):
    """
    A simple Gymnasium environment for a Robocode-like robot battle simulation.
    Observations:
        - x: Robot's x position (float)
        - y: Robot's y position (float)
        - heading: Robot's heading in degrees (float)
        - enemy_bearing: Relative angle to enemy (float)
        - enemy_distance: Distance to enemy (float)
        - enemy_heading: Enemy's heading in degrees (float)
    Actions:
        0: Move forward
        1: Turn left
        2: Turn right
        3: Fire
        4: Do nothing
    """

    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super().__init__()
        # Observation: [x, y, heading, enemy_bearing, enemy_distance, enemy_heading]
        low = np.array([0, 0, 0, -180, 0, 0], dtype=np.float32)
        high = np.array([800, 600, 360, 180, 1000, 360], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(5)
        self.render_mode = render_mode

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.x = np.random.uniform(100, 700)
        self.y = np.random.uniform(100, 500)
        self.heading = np.random.uniform(0, 360)
        self.enemy_x = np.random.uniform(100, 700)
        self.enemy_y = np.random.uniform(100, 500)
        self.enemy_heading = np.random.uniform(0, 360)
        self.done = False
        self.steps = 0
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        dx = self.enemy_x - self.x
        dy = self.enemy_y - self.y
        enemy_distance = np.hypot(dx, dy)
        enemy_bearing = (np.degrees(np.arctan2(dy, dx)) - self.heading) % 360
        if enemy_bearing > 180:
            enemy_bearing -= 360
        obs = np.array([
            self.x,
            self.y,
            self.heading,
            enemy_bearing,
            enemy_distance,
            self.enemy_heading
        ], dtype=np.float32)
        return obs

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        # Move robot
        if action == 0:  # Move forward
            rad = np.deg2rad(self.heading)
            self.x += 10 * np.cos(rad)
            self.y += 10 * np.sin(rad)
        elif action == 1:  # Turn left
            self.heading = (self.heading - 10) % 360
        elif action == 2:  # Turn right
            self.heading = (self.heading + 10) % 360
        elif action == 3:  # Fire
            # Simple hit logic: if facing enemy within 10 degrees and < 100 units
            dx = self.enemy_x - self.x
            dy = self.enemy_y - self.y
            enemy_distance = np.hypot(dx, dy)
            angle_to_enemy = (np.degrees(np.arctan2(dy, dx)) - self.heading) % 360
            if angle_to_enemy > 180:
                angle_to_enemy -= 360
            if abs(angle_to_enemy) < 10 and enemy_distance < 100:
                reward += 10.0  # Hit!
                terminated = True
            else:
                reward -= 0.5  # Missed shot penalty
        elif action == 4:  # Do nothing
            pass

        # Keep robot in bounds
        self.x = np.clip(self.x, 0, 800)
        self.y = np.clip(self.y, 0, 600)

        # Simple enemy logic: move randomly
        self.enemy_heading = (self.enemy_heading + np.random.uniform(-10, 10)) % 360
        rad = np.deg2rad(self.enemy_heading)
        self.enemy_x += 5 * np.cos(rad)
        self.enemy_y += 5 * np.sin(rad)
        self.enemy_x = np.clip(self.enemy_x, 0, 800)
        self.enemy_y = np.clip(self.enemy_y, 0, 600)

        # End episode if too many steps
        self.steps += 1
        if self.steps >= 200:
            truncated = True

        obs = self._get_obs()
        info = {}

        return obs, reward, terminated or truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"Robot: ({self.x:.1f}, {self.y:.1f}) heading {self.heading:.1f}")
            print(f"Enemy: ({self.enemy_x:.1f}, {self.enemy_y:.1f}) heading {self.enemy_heading:.1f}")

    def close(self):
        pass
