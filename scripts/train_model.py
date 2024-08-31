from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from scripts.environment import PricingEnvironment
import torch as torch
import pandas as pd
import sys
import os

# Add the project's root directory to the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TrainingLoggerCallback(BaseCallback):
    def __init__(self):
        super(TrainingLoggerCallback, self).__init__()
        self.training_logs = {
            'timesteps': [],
            'loss': [],
            'reward': []
        }

    def _on_step(self) -> bool:
        # Correct the line by removing parentheses
        if 'infos' in self.locals and len(self.locals['infos']) > 0:
            # Extract the reward if it exists in the episode info
            ep_info = self.locals['infos'][-1].get('episode')
            if ep_info is not None:
                self.training_logs['reward'].append(ep_info['r'])
            else:
                self.training_logs['reward'].append(None)
            print(f"Step {self.num_timesteps}: Reward logged: {self.training_logs['reward'][-1]}")
        else:
            self.training_logs['reward'].append(None)
            print(f"Step {self.num_timesteps}: No reward info found.")

        # Log loss if it's available in the local variables
        if 'loss' in self.locals:
            self.training_logs['loss'].append(self.locals['loss'])
            print(f"Step {self.num_timesteps}: Loss logged: {self.training_logs['loss'][-1]}")
        else:
            self.training_logs['loss'].append(None)
            print(f"Step {self.num_timesteps}: No loss info found.")

        # Log timesteps
        self.training_logs['timesteps'].append(self.num_timesteps)

        return True






def train_rl_agent():
    env = PricingEnvironment(mode='train')

    # Use the DQN model with MLP Policy
    model = DQN(
        policy='MlpPolicy',
        env=env,
        verbose=1,
        buffer_size=50000,
        batch_size=16,
        learning_rate=1e-3,
        policy_kwargs={
            "net_arch": [32, 32],
            "activation_fn": torch.nn.ReLU,
        }
    )

    # Create the training logger callback
    training_logger_callback = TrainingLoggerCallback()

    # Train the model with the training logger callback
    model.learn(total_timesteps=50000, callback=training_logger_callback)

    # Save the trained model
    model.save("models/dqn_pricing_model")

    # Save training logs to a CSV file
        # Save training logs to a CSV file
    df = pd.DataFrame(training_logger_callback.training_logs)
    df.fillna(method='ffill', inplace=True)  # Forward-fill any missing data
    df.to_csv("training_logs.csv", index=False)
    print("Training complete, model and logs saved!")


if __name__ == "__main__":
    train_rl_agent()
