import torch
from torch import nn

class MyNet(nn.Module):
    def __init__(self) -> None:
        super(MyNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(1, 8),nn.Sigmoid(),
            nn.Linear(8, 4),nn.Sigmoid(),
            nn.Linear(4, 1)
        )
    
    def forward(self, x):
        return self.layer(x)

if __name__ == '__main__':
    x = torch.randn(32,1,dtype=torch.float32)
    net = MyNet()
    print(net(x).shape)
    print(x.shape)