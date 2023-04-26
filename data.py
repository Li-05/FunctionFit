import torch
from torch.utils.data import Dataset
import random
from util import *

class MyDataset(Dataset):
    def func(self, x):
        return fitFunc(x) + random.uniform(-1, 1)
    
    def __len__(self):
        return 1024

    def __init__(self):
        super().__init__()
        self.x = torch.linspace(-2, 2, self.__len__()).unsqueeze(1)
        self.y = self.func(self.x)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]


if __name__ == '__main__':
    data = MyDataset()
    print(data[99][0])
    print(data[99][1])