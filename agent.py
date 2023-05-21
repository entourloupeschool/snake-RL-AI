import torch
import numpy as np
import random
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import datetime
import matplotlib.pyplot as plt
from os.path import isfile

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0007
GAMMA = 0.91

class Agent:
    
    def __init__(self, LR, GAMMA, R_EXPO_EXPLO, model_name=None, trainer=None):
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = GAMMA
        self.lr = LR
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()

        # If a model name is provided and the file exists, load the model
        if model_name is not None and isfile(model_name):
            self.model = torch.load(model_name)
        # Otherwise, create a new model
        else:
            self.model = Linear_QNet(11, 256, 256, 3)

        # If a trainer is provided, use it
        if trainer is not None:
            self.trainer = trainer
        # Otherwise, create a new trainer
        else:
            self.trainer = QTrainer(self.model, lr=self.lr, gamma=self.gamma)
            
        self.ratio_exploration_exploitation = R_EXPO_EXPLO

        
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
            
        else:
            mini_sample = self.memory
            
        states, actions, rewards, next_states, dones = zip(*mini_sample) # zip(*list) = transpose
        
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        
        #for state, action, reward, next_state, done in mini_sample:
        #    self.train_short_memory(state, action, reward, next_state, done)
    
    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)
    
    def get_action(self, state):
        final_move = [0,0,0]
        if not self.model.training: # exploitation
            with torch.no_grad():
                state0 = torch.tensor(state, dtype=torch.float)
                prediction = self.model(state0)
                move = torch.argmax(prediction).item()
                final_move[move] = 1
                
        else:  # exploration
            # random moves: tradeoff exploration / exploitation
            self.epsilon = self.ratio_exploration_exploitation - self.n_games
            if random.randint(0, 200) < self.epsilon:
                move = random.randint(0, 2)
                final_move[move] = 1
            else:
                state0 = torch.tensor(state, dtype=torch.float)
                prediction = self.model(state0)
                move = torch.argmax(prediction).item()
                final_move[move] = 1

        return final_move

def train(LR, GAMMA, END_N_GAME=None, R_EXPO_EXPLO=79, model_path=None, trainer=None):
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent(LR, GAMMA, R_EXPO_EXPLO, model_path, trainer)
    game = SnakeGameAI()
    reward = 0
    
    while END_N_GAME is None or agent.n_games < END_N_GAME:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # Set model to training mode
        agent.model.train()

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)


            if score > record:
                record = score
                if record > 30:
                    file_name = f"model-{datetime.datetime.now()}-lr{agent.lr}gam{agent.gamma}rt{agent.ratio_exploration_exploitation}.pth"
                    agent.model.save(file_name)

            print('Game', agent.n_games, 'Score', score, 'Record:', record, 'Epsilon:', agent.epsilon)


    # close and end all matplotlib plots
    plt.close('all')
    plt.ioff()
    
    return plot_scores[-50:]