import gymnasium as gym
from stable_baselines3 import DQN
from scripts.environment import PricingEnvironment

def evaluate_rl_agent():
    env = PricingEnvironment(mode = 'test')
    env = gym.vector.SyncVectorEnv([lambda: env])
    
    model = DQN.load("models/dqn_pricing_model")

    obs, _ = env.reset()
    total_reward = 0

    for _ in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, terminated, truncated, infos = env.step(action)
        total_reward += rewards.sum()
        if terminated.any() or truncated.any():
            break

    print(f'Evaluation complete. Total reward: {total_reward.sum()}')
    

