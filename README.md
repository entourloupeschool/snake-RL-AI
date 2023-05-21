Python Snake Game with Grid Search

This project is an enhanced version of the primary snake game originally developed by Patrick Loeber. The enhancement includes the implementation of a grid search for hyperparameter tuning, which helps to optimize the performance of the reinforcement learning (RL) algorithm used in the game.

Enhancements

    Grid Search: To find the best set of hyperparameters, a grid search was implemented. This allows us to systematically work through multiple combinations of hyperparameter tunes, cross-validate each and determine which one gives the best performance.

    Network Architecture: An exploration was made to find the best network architecture. Although there wasn't a systematic optimization search, different architectures were tested to see their impact on the game's performance.

    Reward Function: A critical aspect of the project was to refine the reward functions. The objective was to prevent the snake from getting into infinite loops. The reward function tries to entice the snake towards the apple, although caution was taken to prevent unwanted behaviours.

Acknowledgements

The foundational code for this project was based on Patrick Loeber's snake game, which can be found at youtube.com/watch?v=L8ypSXwyBds. Additionally, steps followed from a YouTube tutorial by #FreeCodeCamp were crucial in building upon the original game.
Future Improvements

There is always room for more optimization. Future work could include a more systematic search for the best network architecture or refining the reward function to further improve the game's performance.