import random

import numpy as np
import torch
from torch.nn import MSELoss
from torch.optim import Adam

from ReplayMemory import Transition
from SudokuNetwork import SudokuNetwork


class SudokuAgent:
    def __init__(self, parameters, replay_buffer, device):
        self.device = device
        self.buffer = replay_buffer
        self.parameters = parameters

        self.local_policy = SudokuNetwork().to(device)
        self.target_policy = SudokuNetwork().to(device)
        self.target_policy.load_state_dict(self.local_policy.state_dict())

        self.optimizer = Adam(self.local_policy.parameters(), lr=parameters.LR)
        self.criterion = MSELoss().to(device)

        self.steps = 0

    def act(self, sudoku, loc, action_space, eps=0):
        # sudoku /= 9
        # sudoku -= 0.5

        state = torch.from_numpy(sudoku).float().unsqueeze(0).unsqueeze(0).to(self.device)
        loc = torch.from_numpy(loc).float().unsqueeze(0).to(self.device)

        self.local_policy.eval()

        with torch.no_grad():
            q_values = self.local_policy(state, loc)

        self.local_policy.train()

        if random.random() > eps:
            return torch.argmax(q_values).item()
        else:
            return action_space.sample()

    def optimize(self, update):
        if len(self.buffer) < self.parameters.BATCH_SIZE:
            return

        experiences = self.buffer.sample(self.parameters.BATCH_SIZE)

        state, actions, rewards, next_state, dones = self.unpack_samples(Transition(*zip(*experiences)))

        actions = actions.unsqueeze(1)

        self.local_policy.train()
        self.target_policy.eval()

        predicted_targets = self.local_policy(*state)
        predicted_targets = predicted_targets.gather(1, actions)

        with torch.no_grad():
            labels_next = self.target_policy(*next_state).detach().max(1)[0].unsqueeze(1)

        dones = (1 - dones).unsqueeze(1)
        target_error = rewards + (self.parameters.GAMMA * labels_next * dones)

        self.optimizer.zero_grad()
        loss = self.criterion(predicted_targets, target_error)
        loss.backward()
        self.optimizer.step()

        if update:
            self.update_policy()

    def unpack_samples(self, batch):

        state = (torch.from_numpy(np.array([item[0] for item in batch.state])).unsqueeze(1).float().to(self.device),
                 torch.from_numpy(np.array([item[1] for item in batch.state])).float().to(self.device))
        # state = torch.from_numpy(np.array(batch.state)).float().to(self.device)
        actions = torch.from_numpy(np.array(batch.action)).long().to(self.device)
        rewards = torch.from_numpy(np.array(batch.reward)).float().to(self.device)

        # next_state = torch.from_numpy(np.array(batch.next_state)).float().to(self.device)
        next_state = (torch.from_numpy(np.array([item[0] for item in batch.next_state])).unsqueeze(1).float().to(self.device),
                      torch.from_numpy(np.array([item[1] for item in batch.next_state])).float().to(self.device))
        dones = torch.from_numpy(np.array(batch.done).astype(np.uint8)).float().to(self.device)

        return state, actions, rewards, next_state, dones

    def update_policy(self):
        target_net_state_dict = self.target_policy.state_dict()
        policy_net_state_dict = self.local_policy.state_dict()

        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.parameters.TAU + target_net_state_dict[
                key] * (1 - self.parameters.TAU)

        self.target_policy.load_state_dict(target_net_state_dict)
