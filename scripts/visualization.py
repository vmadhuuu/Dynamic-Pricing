import pandas as pd
import matplotlib.pyplot as plt

# Load training logs from CSV file
training_logs = pd.read_csv('training_logs.csv')

# Visualize loss over time
plt.figure(figsize=(10, 5))
plt.plot(training_logs['timesteps'], training_logs['loss'], label='Loss')
plt.xlabel('Timesteps')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.grid(True)
plt.show()

# Visualize reward over time
plt.figure(figsize=(10, 5))
plt.plot(training_logs['timesteps'], training_logs['reward'], label='Reward')
plt.xlabel('Timesteps')
plt.ylabel('Reward')
plt.title('Training Reward Over Time')
plt.legend()
plt.grid(True)
plt.show()
