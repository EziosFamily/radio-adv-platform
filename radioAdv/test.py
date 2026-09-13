import time
import torch
import torch.nn as nn

# 简单卷积模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        
        # 全连接层的输入大小需要动态计算
        self.fc = None

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        
        # 计算卷积层输出形状
        x_flat = torch.flatten(x, 1)
        
        # 动态初始化全连接层
        if self.fc is None:
            self.fc = nn.Linear(x_flat.shape[1], 10).cuda()  # 根据展平后的大小初始化
            
        x = self.fc(x_flat)
        return x

# 创建模型和数据
model = SimpleCNN().cuda()
input_data = torch.randn(32, 3, 32, 32).cuda()  # batch size of 32, 3-channel 32x32 images

# 启用 CuDNN
torch.backends.cudnn.enabled = True
start_time = time.time()
for _ in range(100):
    output = model(input_data)
print(f"With CuDNN: {time.time() - start_time:.4f} seconds")

# 禁用 CuDNN
torch.backends.cudnn.enabled = False
start_time = time.time()
for _ in range(100):
    output = model(input_data)
print(f"Without CuDNN: {time.time() - start_time:.4f} seconds")
