import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def preprocess_sudoku(file, n):
    dataset_content = pd.read_csv(file).sample(n=n)

    puzzles = dataset_content['puzzle']
    solutions = dataset_content['solution']

    processed_puzzles = []
    processed_solutions = []

    for puzzle in tqdm(puzzles):
        x = np.array([int(digit) for digit in puzzle]).reshape(9, 9)
        x = x.astype(float)

        processed_puzzles.append(x)

    for solution in tqdm(solutions):
        x = np.array([int(digit) for digit in solution]).reshape(9, 9)
        x = x.astype(int)

        processed_solutions.append(x)

    return train_test_split(processed_puzzles, processed_solutions, test_size=0.2, shuffle=True)