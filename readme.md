## Project Presentation

2048 is a simple game, with simple rules : you can move the tiles up, down, left or right, to combine them to get a 2048 tile. However, while the action-space is very limited, the state-space of this game is huge, and designing a reinforcement learning agent to beat it is a non-trivial task. This paper presents the two major investigations of our project. On the one hand, we implemented a Gym Environment based on an existing Python implementation of the game. On the second hand, we designed various agents based on different reward functions and learning algorithms, our final goal being to fully compare them and understand which one performs best.


![screenshot](environment/img/screenshot.png)


## Installation

Install required libraries :

- gym
- pytorch
- numpy
- matplotlib
- pygame
- pickle
- pprint



## Launch

The training is launched from the file pytorch-train.py
You can parametrize the agent to be trained inside this file, in the section `AGENT DEFINITION` of the code.
To launch the training, you then simply execute the python file : 

```bash
$ python pytorch-train.py
```

The corresponding weights and training plots will be save in the /checkpoints directory. You can then load these weights to evaluate the agent.

In order to evaluate an agent, you just have to run eval.py, the same way than for the training. You must parametrize the agent in the section `EVAL PARAMETERS` of the code.
You then execute the script using :

```bash
$ python eval.py
```


## Results

The directory /Results contains the training plots and agents' evaluation bar plots, for all the agents we trained and cited in our report.
