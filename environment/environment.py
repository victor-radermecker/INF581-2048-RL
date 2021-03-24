from enum import Enum

import numpy as np
import numpy.ma as ma
import gym

from . import logic, constants as c

class Action(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class GameEnv(gym.Env):
    NB_ACTIONS = 4

    action_space = gym.spaces.Discrete(NB_ACTIONS)
    observation_space = gym.spaces.Box(low=0., high = (2 ** 16), shape = (c.GRID_LEN, c.GRID_LEN))



    _ACTION_MAP = {Action.UP : logic.up,
                 Action.DOWN : logic.down,
                 Action.LEFT : logic.left,
                 Action.RIGHT : logic.right}

    _matrix = None

    _inactive_penalty = 0
    _inactive_penalty_function = None

    _reward_transform = lambda x : x
    _matrix_transform = lambda x : x
    
    total_score = 0

    def __init__(self, inactive_penalty=2, log_reward = True, log_matrix = True):
        """ 
        Args:
            inactive_penalty : 0 -> no inactive penalty
            1 -> constant (-1) inactive penalty
            2 -> -1, then -2, then -3
        """
        self._inactive_penalty_function = {
            0 : (lambda _ : 0),
            1 : (lambda _ : -1),
            2 : self._linear_penalty
        }[inactive_penalty]

        self._reward_transform = self._log_reward if log_reward else lambda x : x  #if log_reward True, we apply the _log_reward function
        self._matrix_transform = self._log_matrix if log_matrix else lambda x : x
        self.max_tile = 0
        self.nbr_merge = 0
        self.sum_tiles = 0

    def reset(self):
        self._matrix = logic.new_game(c.GRID_LEN)
        self._inactive_penalty = 0
        self.total_score = 0
        self.max_tile=0
        self.nbr_merge=0
        self.sum_tiles = 0
        return self._matrix

    def step(self, action: Action):

        action = Action(action)
        new_matrix, action_done, score = self._ACTION_MAP[action](self._matrix)
        self.total_score += score

        if 0 in new_matrix :
            new_matrix = logic.add_two(new_matrix)

        prev_matrix = self._matrix
        self._matrix = new_matrix

        state = logic.game_state(new_matrix)
        done = state == logic.State.LOSE

        #Counting the number of merges
        N_i = np.count_nonzero(prev_matrix)
        N_f = np.count_nonzero(self._matrix)
        self.nbr_merge = N_i - N_f + 1

        self.max_tile = np.max(new_matrix)

        info = {"observation_prev": prev_matrix, 'max_tile':self.max_tile, 'score':self.total_score}

        score = self._reward_transform(score)
        new_matrix = self._matrix_transform(new_matrix)

        if not action_done:
            return  new_matrix, score + self._inactive_penalty_function(), done, info

        self._reset_inactive_penalty()
        return new_matrix, score, done, info


    def _reset_inactive_penalty(self):
        self.inactive_penalty = 0

    def _linear_penalty(self):
        self._inactive_penalty -= 1
        return self._inactive_penalty

    @staticmethod
    def _log_reward(reward):
        return np.log2(reward) if reward != 0 else 0
    
    @staticmethod
    def _log_matrix(matrix):
        masked_matrix = ma.masked_values(matrix, 0)
        masked_matrix = np.log2(masked_matrix)
        return masked_matrix.filled(0)


            






