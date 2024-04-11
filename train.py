import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataset import RamanDataset
from model.ResUnet_d_k7 import ResUnet

torch.cuda.empty_cache()
# 定义CSV文件的路径
csv_list = ['data/data_SNR100/train_data/noise_data_1.csv',
            'data/data_SNR100/train_data/noise_data_2.csv',
            '',
            '']

train_dataset = RamanDataset(csv_list[0], csv_list[1], 1000)
# vali_dataset = RamanDataset(csv_list[2], csv_list[3])

batch_size = 100
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
# vali_dataloader = DataLoader(vali_dataset, batch_size=batch_size, shuffle=False)

train_data_size = len(train_dataset)
# vali_data_size = len(vali_dataset)
lambda_reg = 1
epoch = 60
model = ResUnet()
model.cuda()
# 创建损失函数
loss_fn = nn.MSELoss()
loss_fn.cuda()
# 优化器
learning_rate = 0.0003
parameters = model.parameters()
optimizer = torch.optim.Adam(parameters, lr=learning_rate)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# 记录训练的次数
total_train_step = 0
total_vali_step = 0
# 记录测试的次数
# total_test_step = 0
writer = SummaryWriter(log_dir='data/data_SNR100/log/log_ResUnet_d_k7_50', filename_suffix='ResUnet_d_k7_50')

for i in range(epoch):
    print("------第 {} 轮训练开始------".format(i+1))
    # 训练步骤开始
    for data in train_dataloader:
        spectrum_1, spectrum_2 = data
        spectrum_1 = spectrum_1.cuda()
        spectrum_2 = spectrum_2.cuda()
        outputs = model(spectrum_1)
        loss = loss_fn(outputs, spectrum_2)
        l2_regularization = sum(torch.norm(param) ** 2 for param in parameters)
        total_loss = loss + lambda_reg * l2_regularization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 10 == 0:
            print('train_data:训练次数：{}, Loss: {}'.format(total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)
    # scheduler.step()
    # 测试步骤开始
    # total_test_loss = 0
    # total_correlation_coefficient = 0
    # with torch.no_grad():
    #     for data10W in vali_dataloader:
    #         spectrum_1, spectrum_2 = data10W
    #         spectrum_1 = spectrum_1.cuda()
    #         spectrum_2 = spectrum_2.cuda()
    #         outputs = model(spectrum_1)
    #         loss = loss_fn(outputs, spectrum_2)
    #         # total_test_loss = total_test_loss + loss.item()
    #         outputs = outputs.cpu()
    #         outputs = outputs.numpy()
    #         spectrum_2 = spectrum_2.cpu()
    #         spectrum_2 = spectrum_2.numpy()
    #         total_vali_step = total_vali_step + 1
    #         if total_vali_step % 10 == 0:
    #             print('vali:训练次数：{}, Loss: {}'.format(total_vali_step, loss.item()))
    #             writer.add_scalar('vali_loss', loss.item(), total_vali_step)
    #         for i in range(outputs.shape[0]):
    #             sub_outputs = np.array(outputs[i, 0, :])
    #             sub_spectrum_2 = np.array(spectrum_2[i, 0, :])
    #             correlation_coefficient, _ = scipy.stats.pearsonr(sub_outputs, sub_spectrum_2)
    #             total_correlation_coefficient = total_correlation_coefficient + correlation_coefficient
    #     print("验证集上的相似度：{}".format(round(total_correlation_coefficient/vali_data_size, 4)))
    # writer.add_scalar('total_correlation_coefficient', total_correlation_coefficient/vali_data_size, total_vali_step)

save_path = 'data/data_SNR100/pretrained_model/model_ResUnet_d_k7_50.pth'
torch.save(model, save_path)
print(f'模型{save_path}已保存')
writer.close()
