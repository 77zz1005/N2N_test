from dataset import RamanDataset
from torch.utils.data import DataLoader


csv_list = ['data/data_SNR100/train_data/noise_data_1.csv',
            'data/data_SNR100/train_data/noise_data_2.csv',
            '',
            '']

train_dataset = RamanDataset(csv_list[0], csv_list[1])
# vali_dataset = RamanDataset(csv_list[2], csv_list[3])

batch_size = 100
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

for data in train_dataloader:
    spectrum_1, spectrum_2 = data
    spectrum_1 = spectrum_1.cuda()
    print(spectrum_1)
    print("Shape:", spectrum_1.shape)
    print('device:', spectrum_1.device)
