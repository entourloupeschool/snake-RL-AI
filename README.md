Python Snake Game with Grid Search

This project is an enhanced version of the primary snake game originally developed by Patrick Loeber. The enhancement includes the implementation of a grid search for hyperparameter tuning, which helps to optimize the performance of the reinforcement learning (RL) algorithm used in the game.

important concepts related to Deep Reinforcement Learning (DRL):

    Data Efficiency and Sample Complexity: DRL models learn by interacting with an environment and receiving feedback in the form of rewards. However, they often require a large amount of data (interactions with the environment) to learn effectively. This means they have high 'sample complexity', and it's often a challenge when collecting a lot of data is difficult or expensive. Researchers are actively seeking ways to improve the 'data efficiency' of DRL models, i.e., to help them learn effectively from less data. Train on less games.

    Exploration-Exploitation Trade-Off: When learning, a DRL model needs to balance two things: exploration (trying new actions to see if they lead to better rewards) and exploitation (repeating actions it already knows will give it a good reward). If a model explores too much, it may never find a good strategy. If it exploits too much, it may miss out on better strategies. The optimal balance can vary depending on the specific problem and environment.

    Hyperparameter Tuning and Training Stability: Hyperparameters in DRL models (like the learning rate, discount factor, etc.) can greatly affect the learning process. However, finding the best values for these parameters can be a complex and time-consuming process. Also, the learning process can be unstable or sensitive to small changes in these parameters, which makes training more challenging.

Enhancements

    Grid Search: To find the best set of hyperparameters, a grid search was implemented. This allows us to systematically work through multiple combinations of hyperparameter tunes, cross-validate each and determine which one gives the best performance.

    Network Architecture: An exploration was made to find the best network architecture. Although there wasn't a systematic optimization search, different architectures were tested to see their impact on the game's performance.

    Reward Function: A critical aspect of the project was to refine the reward functions. The objective was to prevent the snake from getting into infinite loops. The reward function tries to entice the snake towards the apple, although caution was taken to prevent unwanted behaviours.

Acknowledgements

The foundational code for this project was based on Patrick Loeber's snake game, which can be found at youtube.com/watch?v=L8ypSXwyBds. Additionally, steps followed from a YouTube tutorial by #FreeCodeCamp were crucial in building upon the original game.

Future Improvements

There is always room for more optimization. Future work could include a more systematic search for the best network architecture or refining the reward function to further improve the game's performance.