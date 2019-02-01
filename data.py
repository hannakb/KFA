import torch
from torchvision import datasets, transforms

import os
import numpy as np

def _get_dataset_MNIST(permutation, get_train=True, batch_size=64, root='./data'):
    trans_perm = transforms.Compose([transforms.ToTensor(),
              transforms.Lambda(lambda x: x.view(-1)[permutation].view(1, 28, 28))])
    
    dataset = datasets.MNIST(root=root, train=get_train, transform=trans_perm, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def get_datasets(dataset_name="pMNIST", random_seed=42, task_number=10,
                 batch_size_train=64, batch_size_test=100):
    
    if dataset_name == "pMNIST":
        
        root = './data'
        if not os.path.exists(root):
            os.mkdir(root)

        np.random.seed(random_seed)
        permutations = [
            np.random.permutation(28 * 28) for
            _ in range(task_number)
        ]

        train_datasets = [
            _get_dataset_MNIST(p, True, batch_size_train, root) for p in permutations
        ]
        test_datasets = [
            _get_dataset_MNIST(p, False, batch_size_test, root) for p in permutations
        ]
    return train_datasets, test_datasets