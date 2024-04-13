import torch
from torch import nn
from model.ResUnet_d_k9 import ResUnet  # 将此处的 "ResUnet_d_k9" 替换为匹配的模型类
import torch.onnx

# 加载训练好的模型
pth_path = 'data/data_SNR075/pretrained_model/model_resnet_d_k9.pth'
# 在转换模型为 ONNX 格式时，通常会使用 CPU 进行操作. 因此在gpu上训练的模型，需要先移到cpu设备上才可以转换
device = torch.device('cpu')  # 将模型加载到 CPU 上
model = torch.load(pth_path, map_location=device)
model.eval()

# 创建示例输入
# example_input = torch.randn(1, 1, 1000).cuda()  # 请将输入形状和数据类型修改为你的模型接受的示例输入
example_input = torch.randn(1, 1, 1000).to(device)

# 导出模型为 ONNX 格式
onnx_path = 'model/onnx_FP32IR/model_resnet_d_k9.onnx'
torch.onnx.export(model, example_input, onnx_path, verbose=True)

