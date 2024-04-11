import numpy as np
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from process_data import Method, GetData
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import torch
from scipy.stats import pearsonr
from model.model_500_k5 import UNet
import time
import torch.autograd.profiler as profiler

num_wavenumber = 1000
a = 200
snr_sum = 0
p_sum = 0
path = ['data/data_SNR075/pretrained_model/model_resnet_d_k9.pth',                  # 0 模型地址
        'data/data_SNR075/test_data/clean_data.csv',                                    # 1 测试集干净数据地址
        'data/data_SNR075/test_data/noise_data.csv',                                    # 2 测试集噪声数据地址
        'data/data_SNR075/denoise_data/denoise_data/denoise_data_resnet_d_k7_50.csv',   # 3 测试集去噪后保存地址
        'data/data_SNR075/wavenumber/wave_number.csv']                                  # 4 波数地址
# 记录每次推理时间的列表
inference_times = []
# 加载模型需要在推理循环之外
model = torch.load(path[0], map_location=torch.device('cpu'))

for i in range(a):
    # ---------------------------输出测试集结果-----------------------------------
    # 在推理前开始计时
    start_time = time.time()
    # 载入模型地址
    # 载入无噪声数据地址
    clean_data = GetData.get_csv_data(path[1], i)
    # 载入噪声数据地址
    noise_raman_data = GetData.get_csv_data(path[2], i)
    # 对噪声数据处理，整形
    noise_raman_data = torch.tensor(noise_raman_data, dtype=torch.float32).reshape(1, 1, num_wavenumber)
    # 生成输出结果
    model.eval()
    with torch.no_grad():
        output = model(noise_raman_data)
    output = np.array(output).reshape(num_wavenumber,)
    # 保存去噪数据
    Method.save(output, path[3], 'a', num_wavenumber)

    # 推理完成 计算推理时间并添加到列表中
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"The {i}th times Inference_Time: {inference_time} seconds")
    inference_times.append(inference_time)

    # 计算信噪比
    SNR = round(Method.snr(output, output - clean_data), 2)
    # 计算皮尔逊相关系数
    p = round(Method.pearson_corr(output, clean_data), 4)
    snr_sum = SNR + snr_sum
    p_sum = p + p_sum
    with open('data/data_SNR075/denoise_data/denoise_data/denoise_data_resnet_d_k7_50.csv', 'a') as file:
        file.write(f'{SNR}\n')
    # print(SNR)
    with open('data/data_SNR075/denoise_data/output/p/p_resnet_d_k9.csv', 'a') as file:
        file.write(f'{p}\n')
    # print(p)

# 计算推理速度的平均值
average_inference_time = sum(inference_times) / len(inference_times)
print(f"Average Inference Time: {average_inference_time} seconds")
print(f"Total Inference Time:{inference_times} seconds")

snr_average = snr_sum/200
p_average = p_sum/200
with open('data/data_SNR075/denoise_data/output/snr/snr_resnet_d_k9.csv', 'a') as file:
    file.write(f'{snr_average}\n')
print(f"snr_average:{snr_average}")
with open('data/data_SNR075/denoise_data/output/p/p_resnet_d_k9.csv', 'a') as file:
    file.write(f'{p_average}\n')
print(f"p_average{p_average}")
file.close()

# ---------------------------输出测试集结果-----------------------------------
#
# for i in range(a):
#     # ---------------------------输出结果绘制图像---------------------------------
#     # 获取干净数据
#     clean_data = GetData.get_csv_data(path[1], i)
#     # 获取带噪数据
#     noise_data = GetData.get_csv_data(path[2], i)
#     # 获取去噪数据区
#     denoise_data = GetData.get_csv_data(path[3], i)
#     # 获取光谱波数
#     wave_number = GetData.get_csv_data(path[4], 0)
#
#     # 绘图
#     plt.figure(figsize=(10, 5))
#     plt.plot(wave_number, noise_data, linestyle='solid', color='#CCCCCC', label='noise data')
#     plt.plot(wave_number, clean_data, linestyle='solid', color='g', label='ground truth')
#     plt.plot(wave_number, denoise_data, linestyle='dashed', color='r', label='denoise data')
#     plt.xlim(29, 318)
#     plt.xlabel('Wavenumber (cm\u207B\u00B9)', fontsize=15)
#     plt.ylabel('Intensity(a.u.)', fontsize=15)
#     # plt.text(29, -0.28, 'U-net:\nBefore denoising : SNR ={}±{}\nAfter\u00A0\u00A0denoising : SNR = {}'
#     #          .format(0.5, 0.05, SNR), fontsize=15, color='k', fontproperties='Times New Roman')
#     plt.grid(linestyle='--', color='gray', linewidth=0.5, alpha=0.5)
#     plt.legend(loc='upper right')
#     plt.rcParams['font.sans-serif'] = ['Times New Roman']
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.tight_layout()
#     # 保存路径
#     plt.savefig('data/data_SNR100/denoise_data/output/pic/ResUnet_d_k7/' + f'{i}.png')
#     plt.clf()
#     plt.close()
