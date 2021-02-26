# pip install gym-super-mario-bros

from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)



done = False
state = env.reset()

while not done:
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
    env.render()
    print(state)

env.close()        