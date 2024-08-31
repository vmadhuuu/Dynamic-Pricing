import gymnasium as gym
import os
from gymnasium import spaces
import numpy as np
import pandas as pd

class PricingEnvironment(gym.Env):
    def __init__(self, mode='train'):
        super(PricingEnvironment, self).__init__()
        current_dir = os.path.dirname(__file__)

        if mode == 'train':
            data_path = os.path.join(current_dir, '../data/train_data.csv')
        else:
            data_path = os.path.join(current_dir, '../data/test_data.csv')
        
        self.data = pd.read_csv(data_path)
        self.max_price = self.data['Historical_Cost_of_Ride'].quantile(0.95)

        # Define action and observation spaces
        self.action_space = spaces.Discrete(21)  # 21 discrete actions
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.data.shape[1],), dtype=np.float32)

        self.state = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.data.sample(1, random_state=seed).values.flatten().astype(np.float32)
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        price_adjustment = np.clip((action - 10) / 100, -0.1, 0.1)
        current_price = self.state[2]
        new_price = np.clip(current_price * (1 + price_adjustment), 0, self.max_price * 1.5)
        self.state[2] = new_price

        reward = self.calculate_reward(new_price)
        self.current_step += 1

        terminated = self.check_done()
        truncated = self.current_step >= 1000

        info = {
            'current_step': self.current_step,
            'current_price': new_price,
            'reward': reward
        }

        return self.state, reward, terminated, truncated, info

    def calculate_reward(self, price):
        # Example calculation, can be adjusted as needed
        number_of_rides = 100
        revenue = price * number_of_rides
        base_price = 1.0
        normalized_revenue = min(revenue / (base_price * number_of_rides), 1)
        satisfaction = max(0, 1 - (price - base_price))
        reward = 0.5 * normalized_revenue + 0.5 * satisfaction
        return max(0, min(reward, 1))

    def check_done(self):
        max_steps = 1000
        return self.current_step >= max_steps
