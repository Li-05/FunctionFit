# FunctionFit
一元函数拟合

## loss图
![avatar](/param/loss_curve.png)

## 拟合图
![avatar](/param/fit_curve.png)

## 网络结构
```
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
```
