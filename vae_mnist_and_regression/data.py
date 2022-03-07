import pathlib
import numpy as np
import torch
import torch.utils.data
from torchvision import datasets, transforms


def boolean_argument(string):
    return str(string).lower() in {"true", "yes", "1"}


def list_argument(dtype, separator=","):
    def constructor(string):
        return [dtype(x) for x in string.split(separator)]
    return constructor

def print_digit(matrix):
    rows = []
    for i in range(matrix.size(0)):
        row = ""
        for j in range(matrix.size(1)):
            row += "x" if matrix[i,j] >= 0.5 else " "
        rows.append(row)
    return "\n".join(rows)


def load_mnist(batch_size, save_to, height=28, width=28):
    """
    :param batch_size: the dataloader will create batches of this size
    :param save_to: a folder where we download the data into    
    :param height: using something other than 28 implies a Resize transformation
    :param width: using something other than 28 implies a Resize transformation
    :return: 3 data loaders
        training, validation, test
    """
    # create directory
    pathlib.Path(save_to).mkdir(parents=True, exist_ok=True)
    
    if height == width == 28:
        transform = transforms.ToTensor()    
    else:        
        transform = transforms.Compose([
            transforms.Resize((height, width)), 
            transforms.ToTensor()]
        )

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(
            datasets.MNIST(
                save_to,
                train=True, 
                download=True, 
                transform=transform),
            indices=range(55000)), 
        batch_size=batch_size,
        shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(
            datasets.MNIST(
                save_to,
                train=True, 
                download=True, 
                transform=transform),
            indices=range(55000, 60000)), 
        batch_size=batch_size,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            save_to,
            train=False, 
            download=True, 
            transform=transform),
        batch_size=batch_size
    )
    return train_loader, valid_loader, test_loader


class Batcher:
    """
    Deals with options such as
        * dynamic binarization
        * change to device
        * shape
        * one-hot encoding of digits
    """
    
    def __init__(self, data_loader, height, width, device, binarize=False, onehot=False, num_classes=10): 
        self.data_loader = data_loader
        self.height = height
        self.width = width
        self.device = device
        self.binarize = binarize
        self.num_batches = len(data_loader)
        self.onehot = onehot
        self.num_classes = num_classes
            
    def __len__(self):
        return self.num_batches
    
    def __iter__(self):
        """        
        Yields
            x: [B, H, W], y: [B]
        or
            x: [B, H, W], y: [B, 10]
        """
        for x, y in self.data_loader: 
            # x: [B, C=1, H, W], y: [B]
            # [B, H, W]
            x = x.reshape(x.size(0), self.height, self.width).to(self.device)
            if self.binarize:
                x = (x > torch.rand_like(x)).float()
            # [B]
            y = y.to(self.device)
            if self.onehot:
                # [B, 10]
                y = torch.nn.functional.one_hot(y, num_classes=self.num_classes)
            yield x, y
                

