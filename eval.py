import os
import torch
from pathlib import Path
import datetime
import numpy as np
import time

#Importing agent
from agent import Agent
from agent_conv import Agent_conv

# Gymboard environment
from gym_board import GymBoard
from environment import GameEnv

# Render env
#import play

#env = GymBoard(max_wrong_steps=5, zero_invalid_move_reward=False)
env = GameEnv()


# Let's train & play
use_cuda = torch.cuda.is_available()
# use_cuda = False
if os.getenv("HOSTNAME") == "arcanes": # CUDA is buggy on my machine
    use_cuda = False #this line checks it is my machine and disables it if it is so
print(f"Using CUDA: {use_cuda}")

save_dir = Path("eval") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)


## ----------------------------------------------------------- ##
# Please fix evaluation parameters here.

agent_type = "DDQN"                                   # DQN or DDQN
archi = "conv"                                        # fc or conv
agent_dir = "checkpoints/conv_corner_reward/2048_net_75.chkpt"       # load weights
episodes = 1000                                       # Number of games to play
render = False                                        # True or False

## ----------------------------------------------------------- ##


if use_cuda:
    weights = torch.load(agent_dir)["model"]
else:
    weights = torch.load(agent_dir, map_location=torch.device('cpu'))["model"]

if archi == 'fc':
    agent = Agent(state_dim=(8, 4, 4, 16), action_dim=GymBoard.NB_ACTIONS, agent_type = agent_type, save_dir=save_dir)
    agent.net.load_state_dict(weights)
elif archi == 'conv':
    agent = Agent_conv(state_dim=(1,4,4,16), action_dim=GymBoard.NB_ACTIONS, save_dir=save_dir)
    agent.onlineNet.load_state_dict(weights)

agent.exploration_rate = 0
max_tiles = np.zeros(episodes)
MOVES = {0: "MOVE UP", 1:"MOVE DOWN", 2: "MOVE LEFT", 3: "MOVE RIGHT"}


for e in range(episodes):

    state = env.reset() #gives directly the reset state

    if not e%20:
        print("Episode : ", e, " out of : ", episodes)

    # Play the game!
    while True:

        # Run agent on the state
        action = agent.act(state)

        if(render):
            print(np.power(2, state))
            print(MOVES[action], "\n")
            #play.render_board(np.power(2, state))
            #time.sleep(0.5)


        # Agent performs action
        next_state, reward, done, info = env.step(action)

        # Remember
        agent.cache(state, next_state, action, reward, done)

        # Update state
        state = next_state

        # Check if end of game
        if done:
            if(render):
                print(np.power(2, state))
                print("Game over")

            max_tiles[e] = np.max(np.power(2, next_state))
            break



max_tiles = np.unique(max_tiles, return_counts=True)
print("Max tiles : \n", np.asarray(max_tiles).T)
