from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.dqn import MlpPolicy
from scripts.environment import PricingEnvironment
import torch as torch

class GradientClippingCallback(BaseCallback):
    def __init__(self, clip_value: float):
        super(GradientClippingCallback, self).__init__()
        self.clip_value = clip_value

    def _on_step(self) -> bool:
        torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), self.clip_value)
        return True
    

def train_rl_agent_with_mlp():
    env = PricingEnvironment(mode='train')

    model = DQN(
        policy=MlpPolicy,
        env=env,
        verbose=1,
        buffer_size=50000,
        batch_size=32,
        learning_rate=1e-4,  # Adjusted learning rate
        policy_kwargs={
            "net_arch": [32],  # Simplified architecture
            "activation_fn": torch.nn.ReLU,
        }
    )

    model.policy.optimizer = torch.optim.Adam(
        model.policy.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    gradient_clip_callback = GradientClippingCallback(clip_value=1.0)

    model.learn(total_timesteps=50000, callback=gradient_clip_callback)

    model.save("models/dqn_pricing_model")

    print("Training complete and model saved!")

if __name__ == "__main__":
    train_rl_agent_with_mlp()
