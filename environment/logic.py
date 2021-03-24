#
# CS1010FC --- Programming Methodology
#
# Mission N Solutions
#
# Note that written answers are commented out to allow us to run your
# code easily while grading your problem set.
# EDITED By Tristan Brugere

from enum import Enum
import random

import numpy as np


from . import constants as c

#######
# Task 1a #
#######

# [Marking Scheme]
# Points to note:
# Matrix elements must be equal but not identical
# 1 mark for creating the correct matrix


def new_game(n):
    matrix = []
    for i in range(n):
        matrix.append([0] * n)
    matrix = np.array(matrix)
    matrix = add_two(matrix)
    matrix = add_two(matrix)
    return matrix

###########
# Task 1b #
###########

# [Marking Scheme]
# Points to note:
# Must ensure that it is created on a zero entry
# 1 mark for creating the correct loop


def add_two(mat):
    a = random.randint(0, len(mat)-1)
    b = random.randint(0, len(mat)-1)
    while mat[a, b] != 0:
        a = random.randint(0, len(mat)-1)
        b = random.randint(0, len(mat)-1)
    mat[a, b] = 2
    return mat

###########
# Task 1c #
###########

# [Marking Scheme]
# Points to note:
# Matrix elements must be equal but not identical
# 0 marks for completely wrong solutions
# 1 mark for getting only one condition correct
# 2 marks for getting two of the three conditions
# 3 marks for correct checking

class State(Enum):
    WIN = 0
    NOT_OVER = 1
    LOSE = -1


def game_state(mat):
    # check for win cell
    for elem in np.nditer(mat):
        if elem == 2048:
            return State.WIN
    
    # check for any zero entries
    for elem in np.nditer(mat):
        if elem == 0:
            return State.NOT_OVER

    # check for same cells that touch each other
    for i in range(len(mat)-1):
        # intentionally reduced to check the row on the right and below
        # more elegant to use exceptions but most likely this will be their solution -> dont agree
        for j in range(len(mat[0])-1):
            if mat[i][j] == mat[i+1][j] or mat[i][j+1] == mat[i][j]:
                return State.NOT_OVER
    for k in range(len(mat)-1):  # to check the left/right entries on the last row
        if mat[len(mat)-1][k] == mat[len(mat)-1][k+1]:
            return State.NOT_OVER
    for j in range(len(mat)-1):  # check up/down entries on last column
        if mat[j][len(mat)-1] == mat[j+1][len(mat)-1]:
            return State.NOT_OVER
    return State.LOSE

###########
# Task 2a #
###########

# [Marking Scheme]
# Points to note:
# 0 marks for completely incorrect solutions
# 1 mark for solutions that show general understanding
# 2 marks for correct solutions that work for all sizes of matrices


def reverse(mat):
    return mat[:, ::-1]

###########
# Task 2b #
###########

# [Marking Scheme]
# Points to note:
# 0 marks for completely incorrect solutions
# 1 mark for solutions that show general understanding
# 2 marks for correct solutions that work for all sizes of matrices


def transpose(mat):
    return np.transpose(mat)

##########
# Task 3 #
##########

# [Marking Scheme]
# Points to note:
# The way to do movement is compress -> merge -> compress again
# Basically if they can solve one side, and use transpose and reverse correctly they should
# be able to solve the entire thing just by flipping the matrix around
# No idea how to grade this one at the moment. I have it pegged to 8 (which gives you like,
# 2 per up/down/left/right?) But if you get one correct likely to get all correct so...
# Check the down one. Reverse/transpose if ordered wrongly will give you wrong result.


def cover_up(mat):
    new = []
    for j in range(c.GRID_LEN):
        partial_new = []
        for i in range(c.GRID_LEN):
            partial_new.append(0)
        new.append(partial_new)
    done = False
    for i in range(c.GRID_LEN):
        count = 0
        for j in range(c.GRID_LEN):
            if mat[i][j] != 0:
                new[i][count] = mat[i][j]
                if j != count:
                    done = True
                count += 1
    return np.array(new), done


def merge(mat, done):
    score = 0
    for i in range(c.GRID_LEN):
        for j in range(c.GRID_LEN-1):
            if mat[i][j] == mat[i][j+1] and mat[i][j] != 0:
                mat[i][j] *= 2
                mat[i][j+1] = 0
                done = True
                score += mat[i][j]
    return np.array(mat), done, score


def up(game):
    # print("up")
    # return matrix after shifting up
    game = transpose(game)
    game, done, score = left(game)
    game = transpose(game)
    return game, done, score


def down(game):
    # print("down")
    game = reverse(transpose(game))
    game, done, score = left(game)
    game = transpose(reverse(game))
    return game, done, score


def left(game):
    # print("left")
    # return matrix after shifting left
    game, done = cover_up(game)
    game, done, score = merge(game, done)
    game = cover_up(game)[0]
    return game, done, score


def right(game):
    # print("right")
    # return matrix after shifting right
    game = reverse(game)
    game, done, score = left(game)
    game = reverse(game)
    return game, done, score
