import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


# 自定义数据集类
class RamanDataset(Dataset):
    def __init__(self, file1, file2, wave_number):
        self.data1 = pd.read_csv(file1)  # 读取第一个CSV文件
        self.data2 = pd.read_csv(file2)  # 读取第二个CSV文件
        self.wave_number = wave_number
    def __len__(self):
        return min(len(self.data1), len(self.data2))  # 返回两个数据集中较小的数据点数量

    def __getitem__(self, idx):
        spectrum_1 = torch.tensor(self.data1.iloc[idx].values, dtype=torch.float32).view(1, self.wave_number)
        spectrum_2 = torch.tensor(self.data2.iloc[idx].values, dtype=torch.float32).view(1, self.wave_number)
        return spectrum_1, spectrum_2

if __name__ == '__main__':
    # 文件路径
    file_x = "data/data_SNR100/train_data/noise_data_1.csv"
    file_y = "data/data_SNR100/train_data/noise_data_2.csv"

    # 创建数据集实例
    dataset = RamanDataset(file_x, file_y)

    # 使用DataLoader加载数据集
    batch_size = 100  # 设置批次大小
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    i = 0
    # 现在，您可以使用dataloader迭代批次数据
    for batch in dataloader:
        spectrum_x, spectrum_y = batch
        i = i + 1
        print(spectrum_x, spectrum_y)
        # 在这里执行您的训练代码或其他操作
    print(i)

