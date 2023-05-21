from agent import train
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import datetime
# Define the range of hyperparameters for the grid search
learning_rates = [0.0007, 0.0008]
gammas = [0.89, 0.91]
ratios = [79, 81]
end_n_games = 180 # you can change this as per your requirements

# Store results
grid_search_results = []

# Calculate total iterations
total = len(learning_rates) * len(gammas) * len(ratios)

# Create a progress bar
pbar = tqdm(total=total)

import torch
import random   

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
# set your seed value
set_seed(42)

for lr in learning_rates:
    for gamma in gammas:
        for ratio in ratios:
            # Print the current parameters
            print(f"Training with lr: {lr}, gamma: {gamma}, ratio: {ratio}")
            
            # Train the agent
            last_scores = train(lr, gamma, end_n_games, ratio)
            
            # Store the results
            grid_search_results.append((lr, gamma, ratio, last_scores))
            
            # Update the progress bar   
            pbar.update(1)

# Now you have a list of tuples, where each tuple represents the learning rate, gamma, record score and mean score from each training session.
# You can print this out or save it to a file to analyze the best hyperparameters.
plt.figure()
plt.title("Grid Search Results")
plt.xlabel("Last 50 games")
plt.ylabel("Scores")
for result in grid_search_results:
    # barplot of the last games
    plt.bar(range(len(result[4])), result[3], alpha=0.8, label=f"lr: {result[0]}, gam: {result[1]}, rtio: {result[2]}")

plt.legend(loc='best')
plt.savefig(f"grid_search_results{datetime.datetime.now()}.png")
plt.show()