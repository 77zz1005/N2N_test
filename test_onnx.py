import onnx
from process_data import Method, GetData
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch
from scipy.stats import pearsonr
from model.model_500_k5 import UNet
import time
import torch.autograd.profiler as profiler
import onnxruntime
import numpy as np

# 设置参数
num_wavenumber = 1000
a = 200
snr_sum = 0
p_sum = 0
num_wavenumber = 1000
a = 200
path = ['model/onnx_FP32IR/model_resnet_d_k9.onnx_FP32IR',  # onnx模型地址
        'data/data_SNR075/test_data/clean_data.csv',  # 测试集干净数据地址
        'data/data_SNR075/test_data/noise_data.csv',  # 测试集噪声数据地址
        'data/data_SNR075/denoise_data/denoise_data/onnx_denoise_data_resnet_d_k9.csv',  # 测试集去噪后保存地址
        'data/data_SNR075/wavenumber/wave_number.csv']  # 波数地址

# 记录每次推理时间的列表
inference_times = []

# 加载所有测试数据
start_load_data = time.time()\

# str
# clean_data_list = [GetData.get_csv_data(path[1], i) for i in range(a)]
# noise_raman_data_list = [GetData.get_csv_data(path[2], i) for i in range(a)]
# 加载所有测试数据为 NumPy 数组
clean_data_np = np.array([GetData.get_csv_data(path[1], i) for i in range(a)], dtype=np.float32)
noise_raman_data_np = np.array([GetData.get_csv_data(path[2], i) for i in range(a)], dtype=np.float32)

load_data_time = time.time() - start_load_data
print(f"Load data finished, Data size:{a}, Using {load_data_time} seconds")

# 创建ONNX运行时会话
ort_session = onnxruntime.InferenceSession(path[0], providers=['CPUExecutionProvider'])
# 获取模型的输入节点名称
input_names = ort_session.get_inputs()
print("Model Input Names:", [input_.name for input_ in input_names])
'''
    with torch.no_grad():
        # 使用 ONNX Runtime 进行推理
        output = ort_session.run(None, {'input.1': noise_raman_data.reshape(1, 1, num_wavenumber)})
    中需要输入与该输出相同的节点名
'''

# 进行推理和评估
total_loss = 0
total_samples = 0
total_correlation_coefficient = 0

# 定义损失函数
loss_fn = torch.nn.MSELoss()

for i in range(a):
    # ---------------------------输出测试集结果-----------------------------------
    # 在推理前开始计时
    start_time = time.time()

    # 载入无噪声数据
    clean_data = clean_data_np[i]
    # 载入噪声数据
    noise_raman_data = noise_raman_data_np[i]
    '''
    # 对噪声数据处理，整形
    # noise_raman_data = torch.tensor(noise_raman_data, dtype=torch.float32).reshape(1, 1, num_wavenumber)

    在加载数据时已经将数据转换为 NumPy 数组.因此，在每次迭代中，您可以直接使用相应的 NumPy 数组作为输入数据，而不需要再次转换为 PyTorch 张量。
    在使用 ONNX Runtime 进行推理时，您只需将 NumPy 数组传递给模型'''

    # 生成输出结果
    with torch.no_grad():
        # 使用 ONNX Runtime 进行推理
        output = ort_session.run(None, {'input.1': noise_raman_data.reshape(1, 1, num_wavenumber)})
        '''output对象是一个列表，而不是一个NumPy数组，因此无法调用reshape方法'''
        output = np.array(output[0])  # 转换为NumPy数组
    # 保存去噪数据
    Method.save(output, path[3], 'a', num_wavenumber)

    # 推理完成
    inference_time = time.time() - start_time
    print(inference_time)
    inference_times.append(inference_time)

    # 计算损失
    output_tensor = torch.from_numpy(output[0]).view(1, 1, num_wavenumber)  # 转换为张量并调整形状
    loss = loss_fn(output_tensor, torch.tensor(clean_data, dtype=torch.float32).view(1, 1, num_wavenumber))
    total_loss += loss.item()

    # 计算信噪比
    signal_power = torch.mean(output_tensor ** 2)
    noise_power = torch.mean(
        (output_tensor - torch.tensor(clean_data, dtype=torch.float32).view(1, 1, num_wavenumber)) ** 2)
    SNR = 10 * torch.log10(signal_power / noise_power)
    snr_sum += SNR.item()

    # 计算皮尔逊相关系数
    p = pearsonr(output_tensor.view(-1), torch.tensor(clean_data, dtype=torch.float32).view(-1))[0]
    p_sum += p

    with open('data/data_SNR075/denoise_data/output/snr/onnx_snr_resnet_d_k9.csv', 'a') as file:
        file.write(f'{SNR}\n')
    with open('data/data_SNR075/denoise_data/output/p/onnx_p_resnet_d_k9.csv', 'a') as file:
        file.write(f'{p}\n')

# 计算推理速度的平均值
average_inference_time = sum(inference_times) / len(inference_times)
print(f"Average Inference Time: {average_inference_time} seconds")
print(f"Total Inference Time:{sum(inference_times)} seconds")

# 计算损失的平均值
print(f"Average Loss: {total_loss}")
average_loss = total_loss / a
print(f"Average Loss: {average_loss}")

snr_average = snr_sum / a
p_average = p_sum / a
with open('data/data_SNR075/denoise_data/output/snr/snr_resnet_d_k9.csv', 'a') as file:
    file.write(f'{snr_average}\n')
print(f"snr_average:{snr_average}")
with open('data/data_SNR075/denoise_data/output/p/p_resnet_d_k9.csv', 'a') as file:
    file.write(f'{p_average}\n')
print(f"p_average{p_average}")
file.close()
