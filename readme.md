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
