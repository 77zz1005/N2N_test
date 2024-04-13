import onnx
from process_data import Method, GetData
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch
from scipy.stats import pearsonr
import numpy as np
import time
import onnxruntime

# 设置参数
num_wavenumber = 1000
a = 200
snr_sum = 0
p_sum = 0
path = ['model/ir/model_resnet_d_k9.xml',  # IR模型地址
        'data/data_SNR075/test_data/clean_data.csv',  # 测试集干净数据地址
        'data/data_SNR075/test_data/noise_data.csv',  # 测试集噪声数据地址
        'data/data_SNR075/denoise_data/denoise_data/ir_denoise_data_resnet_d_k9.csv']  # 测试集去噪后保存地址

# 记录每次推理时间的列表
inference_times = []

# 加载所有测试数据为 NumPy 数组
clean_data_np = np.array([GetData.get_csv_data(path[1], i) for i in range(a)], dtype=np.float32)
noise_raman_data_np = np.array([GetData.get_csv_data(path[2], i) for i in range(a)], dtype=np.float32)

# 创建ONNX运行时会话
ort_session = onnxruntime.InferenceSession(path[0], providers=['CPUExecutionProvider'])
# 获取模型的输入节点名称
input_names = ort_session.get_inputs()
print("Model Input Names:", [input_.name for input_ in input_names])

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

    # 生成输出结果
    with torch.no_grad():
        # 使用 ONNX Runtime 进行推理
        output = ort_session.run(None, {'input.1': noise_raman_data.reshape(1, 1, num_wavenumber)})
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

    with open('data/data_SNR075/denoise_data/output/snr/ir_snr_resnet_d_k9.csv', 'a') as file:
        file.write(f'{SNR}\n')
    with open('data/data_SNR075/denoise_data/output/p/ir_p_resnet_d_k9.csv', 'a') as file:
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
