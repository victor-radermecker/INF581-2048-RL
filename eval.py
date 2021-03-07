import torch
from pathlib import Path
import datetime
import numpy as np
import time

#Importing agent
from agent import Agent

# Gymboard environment
from gym_board import GymBoard

AGENT_TYPE = ["DQN", "DDQN"]

env = GymBoard(max_wrong_steps=5, zero_invalid_move_reward=False)


# Let's train & play
use_cuda = torch.cuda.is_available()
print(f"Using CUDA: {use_cuda}")

save_dir = Path("eval") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)


agent_type = "DQN"

agent_dir = "checkpoints/DQN_20000/2048_net_80.chkpt"
agent = Agent(state_dim=(8, 4, 4, 16), action_dim=GymBoard.NB_ACTIONS, agent_type = agent_type, save_dir=save_dir)
agent.net.load_state_dict(torch.load(agent_dir)["model"])
agent.exploration_rate = 0

episodes = 1000
max_tiles = np.zeros(episodes)

render = False


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