from agent import train
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import datetime
# Define the range of hyperparameters for the grid search
learning_rates = [0.0007]
gammas = [0.91, 0.93, 0.95]
ratios = [79]
end_n_games = 220 # you can change this as per your requirements

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

print("Training complete!")
pbar.close()
print( grid_search_results )
# # Now you have a list of tuples, where each tuple represents the learning rate, gamma, record score and mean score from each training session.
# # You can print this out or save it to a file to analyze the best hyperparameters.

all_scores = [result[-1] for result in grid_search_results]
labels = [f'LR={result[0]}, GAM={result[1]}, RT={result[2]}' for result in grid_search_results]


# # Define means and standard deviations for each list
# means = [25, 27, 29, 31]
# std_devs = [10, 10, 10, 10]

# # Simulate all_scores
# all_scores = []
# for mean, std_dev in zip(means, std_devs):
#     # Use numpy to generate gaussian distributed scores, round them and clip to range [0, 60]
#     scores = np.random.normal(loc=mean, scale=std_dev, size=36)
#     scores = np.round(scores)
#     scores = np.clip(scores, 0, 60)
#     all_scores.append(scores.tolist())

# # Simulated labels
# labels = [
#     'LR=0.1, GAMMA=0.9, RATIO=0.5',
#     'LR=0.01, GAMMA=0.95, RATIO=0.6',
#     'LR=0.001, GAMMA=0.85, RATIO=0.7',
#     'LR=0.0001, GAMMA=0.8, RATIO=0.8'
# ]

# Define score segments
score_segments = [(0, 20), (20, 25), (25, 30), (30, 35), (35, 40), (45, 60), (60, 3500)]

bar_width = 0.15
r = np.arange(len(score_segments))

# Count occurrences of scores within each segment
for i, scores in enumerate(all_scores):
    counts = [sum(lower <= s < upper for s in scores) for lower, upper in score_segments]
    plt.bar(r + i * bar_width, counts, width=bar_width, label=labels[i])

plt.xlabel('Score segments')
plt.xticks([r + bar_width for r in range(len(score_segments))], 
           [f'{lower}-{upper}' for lower, upper in score_segments])

plt.legend(loc='best')
plt.savefig(f"grid_search_results_{datetime.datetime.now()}.png")

plt.show()