from nes_py.wrappers import JoypadSpace
from nes_py.app.play_human import play_human
from nes_py.app.play_random import play_random
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT


# the different sets of actionspaces available
# can chose the standard "nes" one, which contains all (256??) of the possible actions
# See actions.py
_ACTION_SPACES = {
    'right': RIGHT_ONLY,
    'simple': SIMPLE_MOVEMENT,
    'complex': COMPLEX_MOVEMENT,
}

level_id = 'SuperMarioBros-v0'      # For the level IDs : check https://pypi.org/project/gym-super-mario-bros/#description
mode = 'computer'                      # choices=['human', 'random']
actionspace = 'nes'                 # choices=['nes', 'right', 'simple', 'complex']
steps = 500                         # Number of steps to take


env = gym_super_mario_bros.make(level_id)
#actions = SIMPLE_MOVEMENT
#env = JoypadSpace(env, actions)

# wrap the environment with an action space if specified
def main(env = env, level_id = level_id, mode = mode, actionspace = actionspace, steps = steps, _ACTION_SPACES = _ACTION_SPACES):
    if actionspace != 'nes':
        actions = _ACTION_SPACES[actionspace]
        # wrap the environment with the new action space
        env = JoypadSpace(env, actions)
    # play the environment with the given mode
    if mode == 'human':
        play_human(env)
    else:
        play_random(env, steps)

def random_agent(env):
    done = True
    for step in range(5000):
        if done:
            state = env.reset()
        state, reward, done, info = env.step(env.action_space.sample())
        env.render()

    env.close()

if __name__ == "__main__":
    main()