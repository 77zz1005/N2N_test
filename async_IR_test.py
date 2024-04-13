"""
基于 OpenVINO™ Runtime API 实现同步推理计算程序的典型流程，主要有三步：
1.       创建Core对象；
2.       载入并编译模型；
3.       执行推理计算获得推理结果；
"""
from process_data import Method, GetData
import torch
import torch.nn as nn  # 损失函数
import time
import numpy as np
from scipy.stats import pearsonr
from openvino.inference_engine import IECore

# 设置参数
num_wavenumber = 1000
a = 200
snr_sum = 0
p_sum = 0
path = ['model/onnx_FP32IR/resnet_d_k9_IR_FP32.xml',  # IR模型地址
        'data/data_SNR075/test_data/clean_data.csv',  # 测试集干净数据地址
        'data/data_SNR075/test_data/noise_data.csv',  # 测试集噪声数据地址
        'data/data_SNR075/denoise_data/denoise_data/onnx_denoise_data_resnet_d_k9.csv',  # 测试集去噪后保存地址
        'data/data_SNR075/wavenumber/wave_number.csv']  # 波数地址
# # 设置推理环境（CPU、GPU、MYRIAD【神经棒2】）
# device = 'CPU'  # 'MYRIAD'

# Step1: Create OpenVINO Runtime Core
"""使用 IECore :OpenVINO 推理引擎的核心类之一，用于加载和执行推理模型"""
ie = IECore()  # 创建类实例
net = ie.read_network(model=path[0])  # 从 XML 文件中读取 IR 模型的网络结构

# 返回设备支持的最佳异步推理请求数量。您可以根据需要使用这个值来配置异步推理的数量
# 结果：RuntimeError: CPU plugin: Unsupported metric key: OPTIMAL_NUMBER_OF_INFER_REQUESTS
# num_request = ie.get_metric(device_name='CPU', metric_name='OPTIMAL_NUMBER_OF_INFER_REQUESTS')
# print(f"Maximal ayns device number:{num_request}")

exec_net = ie.load_network(network=net, device_name='CPU', num_requests=2)  # 使用 IECore 类的 load_network 加载 IR 模型并在 CPU 上创建推理引擎
input_blob = next(iter(net.input_info))
print(input_blob)
output_blob = next(iter(net.outputs))
print(output_blob)
"""
Log
net.input_info是一个字典，包含了模型的输入节点信息，键是输入节点的名称，值是相应的输入信息
net.outputs也是一个字典，包含了模型的输出节点信息，键是输出节点的名称，值是相应的输出信息

input_blob_names = []
output_blob_names = []
# 打印输入节点名称(以防推理时输入的节点名错误)
print("Input Blob Names:")
for input_key in net.input_info:
    input_blob_names.append(input_key)
    print(input_key)
# 打印输出节点名称
print("\nOutput Blob Names:")
for output_key in net.outputs:
    output_blob_names.append(output_key)
    print(output_key)
"""
# 使用numpy数组加载数据集
clean_data_np = np.array([GetData.get_csv_data(path[1], i) for i in range(a)], dtype=np.float32)
noise_raman_data_np = np.array([GetData.get_csv_data(path[2], i) for i in range(a)], dtype=np.float32)

# 记录推理时间
inference_times = []
# 定义损失函数
loss_fn = nn.MSELoss()
total_loss = 0  # 未初始化就会报错

# 进行推理和评估
for i in range(a):
    # 载入无噪声数据和噪声数据
    clean_data = clean_data_np[i]
    noise_raman_data = noise_raman_data_np[i]

    # 开始推理计时
    start_time = time.time()

    # 进行推理
    exec_net.start_async(request_id=0, inputs={'input.1': noise_raman_data.reshape(1, 1, num_wavenumber)})
    if exec_net.requests[0].wait(-1) == 0:
        output = exec_net.requests[0].output_blobs[output_blob]
        """
        Err: AttributeError: 'openvino.inference_engine.ie_api.InferRequest' object has no attribute 'outputs'
        解决：较新的openvino属性名变了
        链接：https://stackoverflow.com/questions/72934477/attributeerror-openvino-inference-engine-ie-api-inferrequest-object-has-no-at
        """
    else:
        print("Inference failed")

    # 记录推理时间
    inference_time = time.time() - start_time
    print(inference_time)
    inference_times.append(inference_time)

    # 将推理结果类型转为tensor
    if exec_net.requests[0].wait(-1) == 0:
        output_data = exec_net.requests[0].output_blobs[output_blob].buffer
        output_tensor = torch.from_numpy(output_data).view(1, 1, num_wavenumber)
    # 计算损失
    loss = loss_fn(output_tensor, torch.tensor(clean_data, dtype=torch.float32).view(1, 1, num_wavenumber))
    total_loss += loss
    # 计算信噪比
    signal_power = torch.mean(output_tensor ** 2)
    noise_power = torch.mean(
        (output_tensor - torch.tensor(clean_data, dtype=torch.float32).view(1, 1, num_wavenumber)) ** 2)
    SNR = 10 * torch.log10(signal_power / noise_power)
    snr_sum += SNR.item()

    # 计算皮尔逊相关系数
    p = pearsonr(output_tensor.view(-1), torch.tensor(clean_data, dtype=torch.float32).view(-1))[0]
    p_sum += p

    # 保存信噪比和皮尔逊相关系数
    with open('data/data_SNR075/denoise_data/output/snr/onnx_snr_resnet_d_k9.csv', 'a') as file:
        file.write(f'{SNR}\n')
    with open('data/data_SNR075/denoise_data/output/p/onnx_p_resnet_d_k9.csv', 'a') as file:
        file.write(f'{p}\n')

# 计算推理速度的平均值
average_inference_time = sum(inference_times) / len(inference_times)
print(f"Average Inference Time: {average_inference_time} seconds")
print(f"Total Inference Time: {sum(inference_times)} seconds")

# 计算损失的平均值
average_loss = total_loss / a
print(f"Average Loss: {average_loss}")

# 计算信噪比和皮尔逊相关系数的平均值
snr_average = snr_sum / a
p_average = p_sum / a
with open('data/data_SNR075/denoise_data/output/snr/snr_resnet_d_k9.csv', 'a') as file:
    file.write(f'{snr_average}\n')
print(f"snr_average: {snr_average}")
with open('data/data_SNR075/denoise_data/output/p/p_resnet_d_k9.csv', 'a') as file:
    file.write(f'{p_average}\n')
print(f"p_average: {p_average}")
file.close()
