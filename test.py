import torch
import matplotlib.pyplot as plt
from net import *
from util import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = ".\\param\\func.pth"
net = MyNet().to(device)
net.load_state_dict(torch.load(weight_path))

net.eval()  # 将模型设为评估模式

inputs = torch.linspace(-2, 2, 1024).reshape(-1, 1).to(device)
outputs = net(inputs).cpu().detach().numpy()
truth = fitFunc(inputs.cpu())

# 绘制曲线图
plt.plot(inputs.cpu().numpy(), outputs, label='MyNet Output')
plt.plot(inputs.cpu().numpy(), truth, label='Ground Truth')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Funtion Fitten Curve")
plt.legend()  # 添加图例
plt.savefig(".\\param\\fit_curve.png")  # 保存fit曲线图
plt.show()