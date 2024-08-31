from scripts.data_preprocessing import preprocess_data
from scripts.train_model import train_rl_agent
from scripts.evaluate_model import evaluate_rl_agent

def main():
    """Main function to run the RL project"""

    print("Loading and preprocessing data!")
    preprocess_data(input_file = 'dynamic_pricing.csv')

    print("Training the model!")
    train_rl_agent()

    print("Evaluating the model!")
    evaluate_rl_agent()

if __name__ == "__main__":
    main()