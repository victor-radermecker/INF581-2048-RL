import torch
import numpy as np
from agentNet import Net2048
from collections import deque
import random

class Agent:
    def __init__(self, state_dim, action_dim, save_dir):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.save_dir = save_dir

        self.use_cuda = False#torch.cuda.is_available()

        # 2048 DQN Network to predict the most optimal action
        self.net = Net2048(self.state_dim, self.action_dim).float()

        self.device = "cuda:0" if self.use_cuda else "cpu"
        self.net = self.net.to(self.device)

        #Training parameters
        self.exploration_rate = 0.8
        self.exploration_rate_decay = 0.9999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 20000  # no. of experiences between saving 2048Net's weights


        #Train
        self.memory = deque(maxlen=100000)
        self.batch_size = 16
        self.gamma = 0.9
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        self.loss_fn = torch.nn.MSELoss()

        #Learn
        self.burnin = 100000  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

    def act(self, state):
        """
        Given a state, choose an epsilon-greedy action and update value of step.

        Inputs:
        state(LazyFrame): A single observation of the current state, dimension is (state_dim)
        Outputs:
        action_idx (int): An integer representing which action Mario will perform
        """

        # EXPLORE
        if np.random.rand() < self.exploration_rate:
            action_idx = np.random.randint(self.action_dim) #Randomly selects an action between [0, 1, 2, 3]

        # EXPLOIT
        else:
            state = state.__array__()

            state = self.preprocess(state)
            action_values = self.net(state, model="online").reshape(-1,4)
#            action_values = action_values.reshape((-1, 4))
            action_idx = torch.argmax(action_values, axis=1).item()

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1
        return action_idx

    def preprocess(self, state):
        if torch.is_tensor(state):
            state = state.cpu().numpy()

        one_hot_mat1 = np.eye(16)[state.astype('int')]  #one hot encoding of log2(x) values

        #Horizontal flip
        one_hot_mat1f = np.flip(one_hot_mat1, 0)


        #Matrix rotations
        one_hot_mat2 = np.rot90(one_hot_mat1)
        one_hot_mat3 = np.rot90(one_hot_mat2)
        one_hot_mat4 = np.rot90(one_hot_mat3)

        one_hot_mat2f = np.rot90(one_hot_mat1f)
        one_hot_mat3f = np.rot90(one_hot_mat2f)
        one_hot_mat4f = np.rot90(one_hot_mat3f)



        #Matrix reshaping
        one_hot_mat1 = np.reshape(one_hot_mat1, (-1,4,4,16))
        one_hot_mat2 = np.reshape(one_hot_mat2, (-1,4,4,16))
        one_hot_mat3 = np.reshape(one_hot_mat3, (-1,4,4,16))
        one_hot_mat4 = np.reshape(one_hot_mat4, (-1,4,4,16))

        one_hot_mat1f = np.reshape(one_hot_mat1f, (-1,4,4,16))
        one_hot_mat2f = np.reshape(one_hot_mat2f, (-1,4,4,16))
        one_hot_mat3f = np.reshape(one_hot_mat3f, (-1,4,4,16))
        one_hot_mat4f = np.reshape(one_hot_mat4f, (-1,4,4,16))


        
        #TODO
        
        state = np.concatenate((one_hot_mat1, one_hot_mat2, one_hot_mat3, one_hot_mat4), axis=0)
        if self.use_cuda:
            state = torch.tensor(state).to(self.device)
        else:
            state = torch.tensor(state)
        state = torch.flatten(state).float()
        return state

    def preprocess_batch(self, batch_states):
        dim = self.state_dim
        new_batch = torch.tensor(np.zeros((self.batch_size, dim[0]*dim[1]*dim[2]*dim[3])))

        for i in range(batch_states.shape[0]):
            new_batch[i] = self.preprocess(batch_states[i])
        return new_batch.float().to(self.device)


    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame), next_state (LazyFrame), action (int), reward (float), done(bool))
        """
        
        state = state.__array__()
        next_state = next_state.__array__()

        if self.use_cuda:
            state = torch.tensor(state).to(self.device)
            next_state = torch.tensor(next_state).to(self.device)
            action = torch.tensor([action]).to(self.device)
            reward = torch.tensor([reward]).to(self.device)
            done = torch.tensor([done]).to(self.device)
        else:
            state = torch.tensor(state)
            next_state = torch.tensor(next_state)
            action = torch.tensor([action])
            reward = torch.tensor([reward])
            done = torch.tensor([done])

        self.memory.append((state, next_state, action, reward, done,))


    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action.squeeze(), reward.squeeze(), done.squeeze()
       

    def td_estimate(self, state, action):
        state = self.preprocess_batch(state)
        current_Q = self.net(state, model="online")[
            np.arange(0, self.batch_size), action
        ]  # Q_online(s,a)
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        next_state = self.preprocess_batch(next_state)
        next_state_Q = self.net(next_state, model="online")
        best_action = torch.argmax(next_state_Q, axis=1)
        next_Q = self.net(next_state, model="target")[
            np.arange(0, self.batch_size), best_action
        ]
        return (reward + (1 - done.float()) * self.gamma * next_Q).float()

    def update_Q_online(self, td_estimate, td_target):
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        self.net.target.load_state_dict(self.net.online.state_dict())


    def save(self):
        save_path = (
            self.save_dir / f"2048_net_{int(self.curr_step // self.save_every)}.chkpt"
        )
        torch.save(
            dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            save_path,
        )
        print(f"2048Net saved to {save_path} at step {self.curr_step}")


    def learn(self):

        if self.curr_step % self.sync_every == 0:       #synchronization time
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:       #saving time
            self.save()

        if self.curr_step < self.burnin:                #only experiencing time
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate
        td_est = self.td_estimate(state, action)

        # Get TD Target
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(td_est, td_tgt)

        return (td_est.mean().item(), loss)



