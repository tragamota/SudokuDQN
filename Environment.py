import random
from copy import copy

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

        self.loc = np.array([0, 0], dtype=int)
        self.select_index = random.randint(0, len(self.puzzles))

        assert (len(self.puzzles) == len(self.solutions))

        self.observation_space = spaces.Tuple((
            spaces.MultiDiscrete([10] * (9 * 9)),
            spaces.Discrete(2)
        ))
        self.action_space = spaces.Discrete(13)

    def reset(self):
        self.loc = np.array([0, 0], dtype=int)

        self.fixed = self.puzzles[self.select_index]
        self.sudoku = self.puzzles[self.select_index]
        self.solved = self.solutions[self.select_index]

        observation = copy(self.sudoku)
        observation /= 9
        observation -= 0.5

        return observation, self.loc

    def step(self, action):
        reward = 0
        next_observation = self.sudoku
        done = False

        new_loc = copy(self.loc)

        if action < 4:
            if action == 0:
                new_loc[1] += 1
            elif action == 1:
                new_loc[0] -= 1
            elif action == 2:
                new_loc[0] += 1
            else:
                new_loc[1] -= 1

            if not self.is_valid_action(new_loc):
                return (next_observation, self.loc), -0.1, done, {}
            else:
                self.loc = new_loc

                return (next_observation, self.loc), 0, done, {}

        if self.fixed[self.loc[0], self.loc[1]] != 0:
            return (next_observation, self.loc), -2, done, {}

        value = action - 3

        next_observation[self.loc[0], self.loc[1]] = value

        if self.solved[self.loc[0], self.loc[1]] == value:
            reward = 1
        else:
            reward = -1

        if self.is_solved():
            reward = 100
            done = True

        self.sudoku = next_observation

        next_observation = copy(self.sudoku)

        next_observation /= 9
        next_observation -= 0.5

        return (next_observation, self.loc), reward, done, {}

    @staticmethod
    def is_valid_action(loc):
        if 0 <= loc[0] < 9 and 0 <= loc[1] < 9:
            return True

        return False

    def is_solved(self):
        return np.array_equal(self.sudoku, self.solved)

    def render(self, mode='human'):
        print(self.sudoku)
