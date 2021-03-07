# import gym_2048
import gym
import numpy as np

from environment import GameEnv

if __name__ == '__main__':
  # env = gym.make('2048-v0')
  env = GameEnv()
  # env.seed(42)

  env.reset()
  # env.render()

  done = False
  moves = 0
  while not done:
    action = np.random.choice(range(4), 1).item()
    next_state, reward, done, info = env.step(action)
    moves += 1
    print(next_state)

    # print('Next Action: "{}"\n\nReward: {}'.format(
    #   gym_2048.Base2048Env.ACTION_STRING[action], reward))
    # env.render()

  print('\nTotal Moves: {}'.format(moves))
