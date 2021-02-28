from torch import nn
import copy


class Net2048(nn.Module):
    """
    See this paper for the network's structure.
    """

    def __init__(self, input_dim, output_dim, agent_type):
        super().__init__()
        r, d1, d2, n = input_dim  #it should be 8 x 4 x 4 x 16

        if (d1 != d2 or d1 != 4 or r != 8 or n != 16):
            raise ValueError(f"Expecting input dimensions: [8 x 4 x 4 x 16], but got: {r, d1, d2, n}")
        
        self.online = nn.Sequential(
            nn.Linear(2048, 900),
            nn.ReLU(),
            nn.Linear(900, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, output_dim),
            nn.Softmax(),
        )

        self.agent_type = agent_type

        if self.agent_type == "DDQN":
            self.target = copy.deepcopy(self.online)

            # Q_target parameters are frozen.
            for p in self.target.parameters():
                p.requires_grad = False

    def forward(self, input, model):
        if model == "online":
            return self.online(input)
        elif self.agent_type == "DDQN" and model == "target":
            return self.target(input)