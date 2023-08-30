from collections import deque

import gymnasium
import torch

from Environment import SudokuEnvironment
from Parameters import Parameters
from ReplayMemory import ReplayMemory
from SudokuAgent import SudokuAgent
from preprocessing import preprocess_sudoku


def select_device():
    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')

    if torch.backends.mps.is_available():
        device = torch.device('mps')

    return device


if __name__ == '__main__':
    device = select_device()
    parameters = Parameters()

    # train_X, test_X, train_Y, test_Y = preprocess_sudoku('./data/sudoku.csv', n=100_000)

    replay_buffer = ReplayMemory(25000)

    env = gymnasium.make("CartPole-v1")
    agent = SudokuAgent(parameters, replay_buffer, device)

    steps = 0

    eps = parameters.EPS_START
    score = 0
    score_windows = deque(maxlen=100)

    for episode in range(parameters.EPISODES):
        state, _ = env.reset()
        done = False
        for count in range(parameters.EPISODE_DUR):
            action = agent.act(state, env.action_space, eps)
            next_state, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                done = True

            replay_buffer.push(state, action, [reward], next_state, done)

            state = next_state
            score += reward
            steps += 1

            agent.optimize((steps + 1) % 10 == 0)

            if done:
                break

            eps = max(eps * parameters.EPS_DECAY, parameters.EPS_END)

        print(score)
        score = 0