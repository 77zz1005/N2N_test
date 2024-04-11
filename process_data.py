import csv
import numpy as np
from scipy.interpolate import interp1d
from skimage.metrics import structural_similarity as ssim
import scipy.io
import pandas as pd
import itertools
import random


class Method:
    def __init__(self):
        pass

    # 将数据进行归一化处理
    @staticmethod
    def l1_normalization(non_normalized_data):
        l1 = 0
        for i in non_normalized_data:
            l1 = l1 + (abs(i) ** 2)
        normalization_data = non_normalized_data/np.sqrt(l1)
        return normalization_data

    @staticmethod
    def max_normalization(non_normalized_data):
        normalization_data = (non_normalized_data-min(non_normalized_data))/(max(non_normalized_data)-min(non_normalized_data))
        return normalization_data

    @staticmethod
    def mix(data_list, concentration_list):
        length_1 = len(data_list)
        length_2 = len(concentration_list)
        composite_intensity = 0
        if length_1 == length_2:
            for i in range(length_1):
                composite_intensity = concentration_list[i]*data_list[i] + composite_intensity
        else:
            print('数据个数与浓度个数不匹配')
        return composite_intensity

    @staticmethod
    def interpolated_data(wave_number, intensity_data, num_wavenumber):
        linear_interpolator = interp1d(wave_number, intensity_data, kind='linear')
        # 生成新的X坐标，包括插值点
        wave_number_new = np.linspace(min(wave_number), max(wave_number), num_wavenumber)
        # 使用线性插值函数获取新的Y坐标
        intensity_data_new = linear_interpolator(wave_number_new)
        return wave_number_new, intensity_data_new

    # 加上高斯噪声，并生成1条数据
    @staticmethod
    def gaussian_noise_generator2(signal, target_snr, num_wavenumber):
        shape = (1, num_wavenumber)
        mean = 0.0
        while True:
            noise_data = np.empty(shape, dtype=float)
            signal_power = np.mean(signal)
            desired_noise_power = signal_power / (10 ** (target_snr / 10))
            noise_std = desired_noise_power
            # gaussian_noise = mean + noise_std_dev * np.random.randn(shape[1])
            gaussian_noise = np.random.normal(mean, noise_std, len(signal))

            noise_power = np.std(gaussian_noise)
            actual_snr = 10 * np.log10(signal_power / noise_power)
            if abs(actual_snr - target_snr) < (0.1*target_snr):
                noise_data[0, :] = signal + gaussian_noise
                break
        return noise_data

    @staticmethod
    def gaussian_noise_generator(signal, target_snr, num_wavenumber):
        shape = (1, num_wavenumber)
        mean = 0.0
        while True:
            noise_data = np.empty(shape, dtype=float)
            signal_power = np.mean(signal)
            desired_noise_power = signal_power / target_snr
            noise_std = desired_noise_power
            # gaussian_noise = mean + noise_std_dev * np.random.randn(shape[1])
            gaussian_noise = np.random.normal(mean, noise_std, len(signal))

            noise_power = np.std(gaussian_noise)
            actual_snr = signal_power / noise_power
            if abs(actual_snr - target_snr) < (0.1 * target_snr):
                noise_data[0, :] = signal + gaussian_noise
                break
        return noise_data

    @staticmethod
    def snr(signal, noise):
        signal_power = np.mean(signal)
        noise_power = np.std(noise)
        # 计算信噪比
        snr = signal_power / noise_power
        return snr

    @staticmethod
    def ssim(signal1, signal2):
        win_size = 25
        signal1 = signal1.astype(np.float32)
        signal2 = signal2.astype(np.float32)
        ssim_score = ssim(signal1, signal2, win_size=win_size)
        return ssim_score

    @staticmethod
    def pearson_corr(x, y):
        # 检查数组长度是否一致
        if len(x) != len(y):
            raise ValueError("数组长度不一致")
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        # 计算协方差和标准差
        cov_xy = np.sum((x - x_mean) * (y - y_mean))
        std_dev_x = np.sqrt(np.sum((x - x_mean)**2))
        std_dev_y = np.sqrt(np.sum((y - y_mean)**2))
        # 计算皮尔逊相关系数
        p = cov_xy / (std_dev_x * std_dev_y)
        return p

    @staticmethod
    def save(data, save_file_path, mode, num_wavenumber):
        with open(save_file_path, mode=mode, newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            if csvfile.tell() == 0:
                csv_writer.writerow(["data"]*num_wavenumber)
            data = data.reshape(1, num_wavenumber)
            csv_writer.writerows(data)
            # print(f"数据已写入到 {save_file_path} 文件中。")


# 主要用于获取csv文件，mat文件数据
class GetData:
    def __init__(self):
        pass

    # 获取csv文件的数据，返回值
    @staticmethod
    def get_csv_data(csv_file_path, i):
        row_data = pd.read_csv(csv_file_path, sep=',')
        length = len(row_data.iloc[0, ])
        # 输入数据可能是光谱数据length = 290，可能是物质组合或浓度组合 length < 10
        if length < 10:
            # 物质组合或浓度组合，切片取元素保存成list
            data = row_data.iloc[i, 0:length]
            data = data.tolist()
        else:
            # 光谱数据，切片取元素保存成numpy
            data = row_data.iloc[i, 0:length]
            data = np.array(data)
        return data

    # 获取mat文件的数据，返回波数，某物质光谱强度值
    @staticmethod
    def get_mat_data(mat_file_path, wave_number_idx, matter_list):
        # 物质的种类字典，方便将matter_name->num取对应的光谱数据
        m_dict = {'CL': 0, 'DNA': 1, 'Erg': 2, 'LPC': 3, 'OPC': 4, 'PA': 5, 'PC': 6,
                  'PE': 7, 'PS': 8, 'SPH': 9, 'cyto': 10, 'Protein': 11, 'PI': 12}
        intensity_list = []
        mat_data = scipy.io.loadmat(mat_file_path)
        # 获取wave_number，有两种wave_number，wave_dict = ['lowwn', 'highwn']
        wave_number = mat_data[wave_number_idx]
        # 得到的wave_number是（1,290），整形为（290，）方便插值、绘图
        wave_number = wave_number.reshape(290,)
        matter_data = mat_data['Plow']
        # 取若干物质对应的光谱数据，保存到list
        for i in matter_list:
            intensity_data = matter_data[m_dict[i]]
            intensity_list.append(intensity_data)
        return wave_number, intensity_list


# 主要用于生成浓度，物质组成的csv文件
# num_matter/num_parts，分别控制参与混的物质个数，生成浓度组成个数
# num_groups，生成组的数量
# save_file_path，文件存放地址 'concentration/concentration_2.csv', 'combination/combination_2.csv'
class DataGenerator:
    def __init__(self):
        pass

    @staticmethod
    def generator_combination(num_matter, num_groups, save_file_path):
        matter_dict = ['CL', 'Erg', 'LPC', 'OPC', 'PA', 'PC', 'PE', 'PS', 'SPH', 'cyto', 'Protein', 'PI']
        all_combinations = list(itertools.combinations(matter_dict, num_matter))
        with open(save_file_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Matter"] * num_matter)
            for group in range(num_groups):
                combination = all_combinations[group % len(all_combinations)]
                csv_writer.writerow(combination)
        print(f"数据已写入到 {save_file_path} 文件中。")

    @staticmethod
    def generator_concentration(num_parts, num_groups, save_file_path):
        data = []
        for _ in range(num_groups):
            group = []
            total_sum = 1
            for _ in range(num_parts - 1):
                part = round(random.uniform(0, total_sum), 2)
                total_sum = round(total_sum - part, 2)
                group.append(part)
            group.append(total_sum)
            data.append(group)
        with open(save_file_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["concentration"] * num_parts)
            csv_writer.writerows(data)
        print(f"数据已写入到 {save_file_path} 文件中。")

    @staticmethod
    def generator_combination5(num_groups, save_file_path):
        matter_dict = [['PE', 'PS'], ['PE', 'SPH'], ['PE', 'cyto'], ['PE', 'Protein'], ['PE', 'PI']]
        # matter_dict = [['LPC', 'PS'], ['PC', 'SPH'], ['PE', 'cyto'], ['PA', 'Protein'], ['OPC', 'PI']]
        with open(save_file_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Matter"] * 2)
            for group in range(num_groups):
                combination = matter_dict[group % 5]
                csv_writer.writerow(combination)
        print(f"数据已写入到 {save_file_path} 文件中。")

    @staticmethod
    def generator_combination10(num_groups, save_file_path):
        matter_dict = [['PE', 'PS'], ['PE', 'SPH'], ['PE', 'cyto'], ['PE', 'Protein'], ['PE', 'PI'],
                       ['PE', 'PC'], ['PE', 'PA'], ['PE', 'OPC'], ['PE', 'LPC'], ['PE', 'Erg']]
        with open(save_file_path, mode='w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["Matter"] * 2)
            for group in range(num_groups):
                combination = matter_dict[group % 5]
                csv_writer.writerow(combination)
        print(f"数据已写入到 {save_file_path} 文件中。")
