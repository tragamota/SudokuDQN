import torch
from torch import nn


class SudokuNetwork(nn.Module):

    def __init__(self):
        super(SudokuNetwork, self).__init__()

        self.conv_layers = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3),
                                         nn.BatchNorm2d(64),
                                         nn.ReLU(),
                                         nn.Conv2d(64, 128, kernel_size=5),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),
                                         nn.Conv2d(128, 128, kernel_size=3),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU(),
                                         nn.Conv2d(128, 128, kernel_size=1),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU())

        self.fc = nn.Sequential(nn.Linear(130, 512),
                                nn.ReLU(),
                                nn.Linear(512, 128),
                                nn.ReLU(),
                                nn.Linear(128, 13))

    def forward(self, sudoku, loc):
        x = self.conv_layers(sudoku)
        x = nn.Flatten()(x)

        x = torch.cat((x, loc), dim=1)

        return self.fc(x)
