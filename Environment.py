import random

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class SudokuEnvironment(gym.Env):
    def __init__(self, puzzles, solutions):
        self.puzzles = puzzles
        self.solutions = solutions

        self.fixed = None
        self.sudoku = None
        self.solved = None

        assert(len(self.puzzles) == len(self.solutions))

        self.observation_space = spaces.MultiDiscrete([10] * (9 * 9))
        self.action_space = spaces.Tuple((
            spaces.MultiDiscrete([9, 9]),
            spaces.Discrete(9)
        ))

    def reset(self):
        select_index = random.randint(0, len(self.puzzles))

        self.fixed = self.puzzles[select_index]
        self.sudoku = self.puzzles[select_index]
        self.solved = self.solutions[select_index]

        return self.sudoku

    def step(self, action):
        row, col, value = action

        next_observation = self.sudoku
        done = False

        if not self.is_valid_action(row, col, value):
            return next_observation, -100, done, {}

        if self.fixed[row, col] != 0:
            return next_observation, -50, done, {}

        next_observation[row, col] = value
        value_solution_dist = self.solved[row, col] - value
        reward = 20 - value_solution_dist

        if self.is_solved():
            reward = 500
            done = True

        return next_observation, reward, done, {}

    @staticmethod
    def is_valid_action(row, col, value):
        if 0 <= row < 9 and 0 <= col < 9 and 0 <= value < 9:
            return False

        return True

    def is_solved(self):
        return np.array_equal(self.sudoku, self.solved)

    def render(self, mode='human'):
        print(self.sudoku)