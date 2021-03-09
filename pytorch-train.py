import torch
from pathlib import Path
import datetime
import os



#Importing agent and MetricLogger
from agent import Agent
from agent_conv import Agent_conv
from metricLogger import MetricLogger

# Gymboard environment
from gym_board import GymBoard
#from environment import GameEnv

AGENT_TYPE = ["DQN", "DDQN"]

env = GymBoard(max_wrong_steps=5, zero_invalid_move_reward=False)
#env = GameEnv()

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
agent = Agent_conv(state_dim=(1,4,4,16), action_dim=GymBoard.NB_ACTIONS, save_dir=save_dir)

resume_training = False
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

        # Remember
        agent.cache(state, next_state, action, reward, done)

        # Learn
        q, loss = agent.learn()
    
        # Logging
        logger.log_step(reward, loss, q)

        # Update state
        state = next_state

        # Check if end of game
        if done:
            break

    logger.log_episode(info['score'])

    if e % 20 == 0:
        logger.record(episode=e, epsilon=agent.exploration_rate, step=agent.curr_step)
