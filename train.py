import torch


def select_device():
    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')

    if torch.backends.mps.is_available():
        device = torch.device('mps')

    return device


if __name__ == "__main__":
    device = select_device()

    
