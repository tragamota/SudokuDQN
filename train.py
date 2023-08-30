import math

import numpy as np
import torch
from numpy import random
from torch.optim import Adam

from Environment import SudokuEnvironment
from ReplayMemory import ReplayMemory, Transition
from SudokuAgent import SudokuAgent
from preprocessing import preprocess_sudoku

BATCH_SIZE = 256
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.2
EPS_DECAY = 1000
TAU = 0.02
LR = 1e-5

EPISODES = 500
EPISODE_DUR = 2000


def select_device():
    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')

    if torch.backends.mps.is_available():
        device = torch.device('mps')

    return device


def optimize_policy(memory, policy, target, optimizer, criterion):
    if len(memory) < BATCH_SIZE:
        return 0

    transitions = memory.sample(BATCH_SIZE)

    sample_batch = Transition(*zip(*transitions))

    state_batch = sample_batch.state

    sudoku_batch = torch.cat([item[0] for item in state_batch])
    loc_batch = torch.cat([item[1] for item in state_batch])

    state_batch = (sudoku_batch, loc_batch)

    action_batch = torch.cat(sample_batch.action)
    reward_batch = torch.cat(sample_batch.reward)

    next_state_batch = sample_batch.next_state

    next_sudoku_batch = torch.cat([item[0] for item in next_state_batch])
    next_loc_batch = torch.cat([item[1] for item in next_state_batch])

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, next_sudoku_batch)), device=device, dtype=torch.bool)
    non_final_next_states = (next_sudoku_batch[non_final_mask], next_loc_batch[non_final_mask])

    state_action_values = policy(state_batch).gather(1, action_batch)
    next_state_values = torch.zeros(BATCH_SIZE, device=device)

    with torch.no_grad():
        next_state_values[non_final_mask] = target(non_final_next_states).max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy.parameters(), 100)
    optimizer.step()

    return loss.item()


def select_action(env, state, policy, step_count):
    rand = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1 * step_count / EPS_DECAY)

    if rand > eps_threshold:
        with torch.no_grad():
            return policy(state).max(1)[1].view(1, 1)

    return torch.tensor([[env.action_space.sample()]])


if __name__ == "__main__":
    device = select_device()
    train_X, test_X, train_Y, test_Y = preprocess_sudoku('./data/sudoku.csv', n=100_000)

    steps = 0

    env = SudokuEnvironment(train_X, train_Y)

    policy = SudokuAgent().to(device)
    target = SudokuAgent().to(device)
    target.load_state_dict(policy.state_dict())

    optimizer = Adam(policy.parameters(), lr=LR)
    criterion = torch.nn.SmoothL1Loss()

    replayBuffer = ReplayMemory(50000)

    for episode_number in range(EPISODES):
        loss = []
        rewards = []

        obs = env.reset()

        current_obs = torch.tensor([obs[0]], dtype=torch.float32, device=device).unsqueeze(0)
        loc = torch.tensor([obs[1]], dtype=torch.float32, device=device)

        for step_num in range(EPISODE_DUR):
            action = select_action(env, (current_obs, loc), policy, steps).to(device)

            next_obs, reward, done, info = env.step(action)
            rewards.append(reward)
            reward = torch.tensor([reward], device=device)

            new_obs = torch.tensor([next_obs[0]], dtype=torch.float32, device=device).unsqueeze(0)
            new_loc = torch.tensor([next_obs[1]], dtype=torch.float32, device=device)

            if done:
                next_obs = None

            replayBuffer.push((current_obs, loc), action, (new_obs, new_loc), reward)
            loss.append(optimize_policy(replayBuffer, policy, target, optimizer, criterion))

            current_obs = new_obs
            loc = new_loc
            steps += 1

            if done:
                break

        target_net_state_dict = target.state_dict()
        policy_net_state_dict = policy.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)

        target.load_state_dict(target_net_state_dict)

        print("=" * 20)
        print(f"Episode {episode_number}")
        print(f"Loss: {np.sum(loss) / len(loss)}, {np.sum(loss)}")
        print(f"Reward: {np.sum(rewards) / len(rewards)}, {np.sum(rewards)}")
        print(f"Min reward: {np.min(rewards)}, Max reward: {np.max(rewards)}")
        print("=" * 20)