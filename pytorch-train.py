import torch
from pathlib import Path
import datetime
import os
import numpy as np



#Importing agent and MetricLogger
from agent import Agent
from agent_conv import Agent_conv
from metricLogger import MetricLogger

# Gymboard environment
#from gym_board import GymBoard
from environment import GameEnv

AGENT_TYPE = ["DQN", "DDQN"]

#env = GymBoard(max_wrong_steps=5, zero_invalid_move_reward=False)
env = GameEnv()

env.reset()
next_state, reward, done, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")


# Let's train & play
use_cuda = torch.cuda.is_available()
if os.getenv("HOSTNAME") == "arcanes": # CUDA is buggy on my machine
    use_cuda = False
print(f"Using CUDA: {use_cuda}")
print()

save_dir = Path("checkpoints") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir.mkdir(parents=True)


#agent = Agent(state_dim=(8, 4, 4, 16), action_dim=GymBoard.NB_ACTIONS, agent_type = "DDQN", save_dir=save_dir)
agent = Agent_conv(state_dim=(1,4,4,16), action_dim=GameEnv.NB_ACTIONS, save_dir=save_dir)

resume_training = False
# reward type
base_reward = True       #Taking the base reward
empty_reward = False     #Taking number of white tiles as reward
max_corner_reward = False 
reward_max_tile = False  #Taking max tile as reward
reward_nr_merge = False  #Taking number of tiles merged at each step as reward
reward_new_max_tile = False #Getting a reward for each new max tile created

if(resume_training):
    agent_dir = "checkpoints/DQN_10000/2048_net_67.chkpt"
    print("Resume training from checkpoint : ", agent_dir)
    saved_params = torch.load(agent_dir)
    agent.net.load_state_dict(saved_params["model"])
    agent.exploration_rate = saved_params["exploration_rate"]
    agent.burnin = 100

logger = MetricLogger(save_dir)

episodes = 10000

for e in range(episodes):

    state = env.reset() #gives directly the reset state

    # Play the game!
    while True:

        # Run agent on the state
        action = agent.act(state)

        # Agent performs action
        next_state, reward, done, info = env.step(action)

        # Empty tiles reward
        if not base_reward:
            reward = 0
        if empty_reward:
            reward = reward + np.log2(np.sum(next_state == 0) + 1)
        elif max_corner_reward:
            max_tile = np.amax(next_state)
            if max_tile in [next_state[0,0], next_state[-1,0], next_state[-1,-1], next_state[0,-1]]:
                reward = reward + max_tile
        elif reward_nr_merge:
            if env.max_tile == 2048:
                reward = 1
            else:
                reward = (env.nbr_merge-0.5)/8
        elif reward_max_tile:
            if reward_max_tile:
                    reward = reward + np.log2(env.max_tile)
        elif reward_new_max_tile:
          if env.max_tile > max_tile:
            max_tile = env.max_tile
            reward = + np.log2(max_tile)
        
        
        # Remember
        agent.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = agent.learn()
    
        # Logging
        logger.log_step(reward, loss, q)
        # Update state
        state = next_state

        if done:
            break

    #logger.log_episode(info['score'])
    logger.log_episode(info['score'], info['max_tile'], np.sum(next_state))

    if e % 20 == 0:
        logger.record(episode=e, epsilon=agent.exploration_rate, step=agent.curr_step)
