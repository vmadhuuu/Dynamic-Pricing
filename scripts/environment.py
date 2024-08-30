import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PricingEnvironment(gym.Env):
    def __init__(self, mode = 'train'):
        super(PricingEnvironment, self).__init__()

        current_dir = os.path.dirname(__file__)

        if mode == 'train':
            data_path = os.path.join(current_dir, '../data/train_data.csv')
        else:
            data_path = os.path.join(current_dir, '../data/test_data.csv')
        
        print(f"Loading data from {data_path}")

        self.data = pd.read_csv(data_path)

        self.max_price = self.data['Historical_Cost_of_Ride'].quantile(0.95)

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(21)  # 21 discrete actions (e.g., -10% to +10% price adjustment)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.data.shape[1],), dtype=np.float32)

        self.state = None
        self.current_step = 0
        self.previous_price = 1.0

        self.reset()

    def reset(self, seed = None, options = None):
        super().reset(seed = seed)
        # Ensure the returned state is a numpy array of type float32
        self.state = self.data.sample(1, random_state = seed).values.flatten().astype(np.float32)
        self.current_step = 0
        return self.state, {}

    # Price adjustment
    def step(self, action):
        price_adjustment = np.clip((action - 10) / 100, -0.1, 0.1)  # Clamped between -10% and +10%
        current_price = self.state[2]
        new_price = np.clip(current_price * (1 + price_adjustment), 0, self.max_price * 1.5)  # Ensure new price is reasonable
        self.state[2] = new_price

        reward = self.calculate_reward(new_price)
        terminated = self.check_done()
        truncated = self.current_step >= 1000
        self.current_step += 1

        info = {
            'current_step': self.current_step,
            'current_price': new_price,
            'reward': reward
        }

        return self.state, reward, terminated, truncated, info

    # Reward calculation
    def calculate_reward(self, price):
        number_of_rides = 100
        revenue = price * number_of_rides

        # Normalizing revenue directly against a base price
        base_price = 1.0  # This can be adjusted as needed
        normalized_revenue = min(revenue / (base_price * number_of_rides), 1)

        # Simple linear decrease in satisfaction as price increases from the base price
        satisfaction = max(0, 1 - (price - base_price))

        # Assigning equal weights to revenue and satisfaction for simplicity
        revenue_weight = 0.5
        satisfaction_weight = 0.5

        # Simplified reward calculation
        reward = (revenue_weight * normalized_revenue) + (satisfaction_weight * satisfaction)

        # Clip the reward to ensure it's between 0 and 1
        reward = max(0, min(reward, 1))

        return reward



    def check_done(self):
        max_steps = 1000
        return self.current_step >= max_steps