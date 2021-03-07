from torch import nn
import copy
import torch.nn.functional as F
import torch

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
            nn.Linear(900, 200),
            nn.ReLU(),
            nn.Linear(200, output_dim),
            #nn.Softmax(),   #Should not use softmax as we want the q value, not sth in [0,1]
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


class Net2048_conv(nn.Module):
    """
    See this paper for the network's structure.

    For this network, we have decided not to use the symmetries to make training faster.

    Input: [4x4x16]
    (4x4x16) --> conv1_layer1 --> ((4-1+1)x(4-2+1)x128) = (4x3x128)
    (4x4x16) --> conv2_layer1 --> ((4-2+1)x(4-1+1)x128) = (3x4x128)

    (4x3x128) --> conv1_layer2 --> (4x2x128)
    (4x3x128) --> conv2_layer2 --> (3x3x128)

    (4x3x128) --> conv1_layer2 --> (3x3x128)
    (4x3x128) --> conv2_layer2 --> (2x4x128)

    """
    ### Deep Q-Learning Network
    def __init__(self, input_dim):
        super(Net2048_conv, self).__init__()

        r, d1, d2, n = input_dim  
        if (d1 != d2 or d1 != 4 or r != 1 or n != 16):
            raise ValueError(f"Expecting input dimensions: [1 x 4 x 4 x 16], but got: {r, d1, d2, n}")
        
        # Convolutional layers
        self.Conv1_Layer1 = nn.Conv2d(16, 128, kernel_size=(1,2))                
        self.Conv2_Layer1 = nn.Conv2d(16, 128, kernel_size=(2,1))
        
        self.Conv1_Layer2 = nn.Conv2d(128, 128, kernel_size=(1,2))
        self.Conv2_Layer2 = nn.Conv2d(128, 128, kernel_size=(2,1))
        
        # Linear and Relu Layers
        self.fc = nn.Sequential(         
            nn.Linear(7424, 256),
            nn.ReLU(),
            nn.Linear(256, 4)
        )
        
    def forward(self, x):
      # Forward function of the layer
        x_1 = F.relu(self.Conv1_Layer1(x)) 
        x_2 = F.relu(self.Conv2_Layer1(x))
        
        x_3 = F.relu(self.Conv1_Layer2(x_1))
        x_4 = F.relu(self.Conv2_Layer2(x_1))
        
        x_5 = F.relu(self.Conv1_Layer2(x_2))
        x_6 = F.relu(self.Conv2_Layer2(x_2))
        
        sh_a = x_1.shape
        sh_aa = x_3.shape
        sh_ab = x_4.shape
        sh_b = x_2.shape
        sh_ba = x_5.shape
        sh_bb = x_6.shape
       
        x_a = x_1.view(sh_a[0],sh_a[1]*sh_a[2]*sh_a[3])
        x_aa = x_3.view(sh_aa[0],sh_aa[1]*sh_aa[2]*sh_aa[3])
        x_ab = x_4.view(sh_ab[0],sh_ab[1]*sh_ab[2]*sh_ab[3])
        x_b = x_2.view(sh_b[0],sh_b[1]*sh_b[2]*sh_b[3])
        x_ba = x_5.view(sh_ba[0],sh_ba[1]*sh_ba[2]*sh_ba[3])
        x_bb = x_6.view(sh_bb[0],sh_bb[1]*sh_bb[2]*sh_bb[3])
        
        concat = torch.cat((x_a,x_b,x_aa,x_ab,x_ba,x_bb),dim=1)
        output = self.fc(concat)
        
        return output
