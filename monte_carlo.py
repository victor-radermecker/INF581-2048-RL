import numpy as np
from gym_board import GymBoard


nb_iters = 13
#depth = 50


env = GymBoard()

env.reset()

done = False
while not done:
    score_per_move = [0] * GymBoard.NB_ACTIONS
    for i in range (GymBoard.NB_ACTIONS):
        for n in range (nb_iters):
            copy_env = GymBoard(zero_invalid_move_reward=False)
            copy_env.reset()
            copy_env.matrix = env.matrix
            iter_reward = 0
            copy_next_state, copy_reward, copy_done, copy_info = copy_env.step(action=i)
            d = 0
            iter_reward += copy_reward
            while not copy_done: #and d < depth:
                r_action = np.random.randint(0,GymBoard.NB_ACTIONS)
                copy_next_state, copy_reward, copy_done, copy_info = copy_env.step(action=r_action)
                iter_reward += copy_reward
                d += 1
            score_per_move[i] += iter_reward
    next_action = np.argmax(score_per_move)

    next_state, reward, done, info = env.step(action=next_action)
    print("SCORE:", env.score, "\tSTEP:", env.n_steps_valid, "\tHIGHEST VALUE:", env.highest_value)
    print(env.matrix)

print("FINAL SCORE:", env.score, "\tSTEP:", env.n_steps_valid, "\tHIGHEST VALUE:", env.highest_value)




