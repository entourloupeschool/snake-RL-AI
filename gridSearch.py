from agent import train, Agent

# Define the range of hyperparameters for the grid search
learning_rates = [0.001, 0.01, 0.1]
gammas = [0.8, 0.9, 0.95, 0.99]
end_n_games = 10 # you can change this as per your requirements

# Store results
grid_search_results = []

for lr in learning_rates:
    for gamma in gammas:
        # Train the agent
        record, mean_score = train(lr, gamma, end_n_games)
        
        # Store the results
        grid_search_results.append((lr, gamma, record, mean_score))

# Now you have a list of tuples, where each tuple represents the learning rate, gamma, record score and mean score from each training session.
# You can print this out or save it to a file to analyze the best hyperparameters.
for result in grid_search_results:
    print(f"Learning rate: {result[0]}, Gamma: {result[1]}, Record score: {result[2]}, Mean score: {result[3]}")
