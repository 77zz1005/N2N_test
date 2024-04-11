import matplotlib
matplotlib.use("TkAgg")
from process_data import Method, GetData, DataGenerator
from tqdm import tqdm

num_wavenumber = 1000
num_groups = 10000
iterations = range(num_groups)

# 生成num_groups对物质组合，num_groups对浓度组合，生成文件后注释，以免再次运行时会覆盖数据
# DataGenerator.generator_combination5(num_groups, 'data/data_SNR05/cc_data/combo.csv')
# DataGenerator.generator_concentration(2, num_groups, 'data/data_SNR05/cc_data/conc.csv')

# --------------------------------------------------train_data----------------------------------------------------------
for i in tqdm(iterations, desc="处理进度", ncols=100):
    # 获取物质组合
    combination_list = GetData.get_csv_data('data/data_SNR100/cc_data/combo.csv', i)
    # 通过物质组合返回每个物质的光谱强度数据
    wave_number, intensity_list = GetData.get_mat_data('Pure.mat', 'lowwn', combination_list)
    # 获取浓度组合
    concentration_list = GetData.get_csv_data('data/data_SNR100/cc_data/conc.csv', i)
    # 把获取每个物质的光谱强度数据与对应浓度相乘再叠加起来
    composite_intensity = Method.mix(intensity_list, concentration_list)
    # 把插值后的数据归一化
    intensity_data_new = Method.l1_normalization(composite_intensity)
    # 把叠加的光谱强度做插值，290->1000
    wave_number_new, intensity_data_new = Method.interpolated_data(wave_number, intensity_data_new, num_wavenumber)
    # 将干净数据保存
    # Method.save(wave_number_new, 'data/data_SNR075/wavenumber/wave_number.csv', 'w', num_wavenumber)
    # Method.save(intensity_data_new, 'data/data_SNR100/train_data/clean_data.csv', 'a', num_wavenumber)
    # 将干净数据加入噪声，并在保存
    noise_data = Method.gaussian_noise_generator(intensity_data_new, target_snr=1.00, num_wavenumber=num_wavenumber)
    Method.save(noise_data, 'data/data_SNR100/train_data/noise_data_2.csv', 'a', num_wavenumber)
print("任务完成！")

# ------------------------------------------------test_data-------------------------------------------------------------
# for i in tqdm(iterations, desc="处理进度", ncols=100):
#     # 获取物质组合
#     combination_list = GetData.get_csv_data('data/data_SNR100/cc_data/combo_test.csv', i)
#     # 通过物质组合返回每个物质的光谱强度数据
#     wave_number, intensity_list = GetData.get_mat_data('Pure.mat', 'lowwn', combination_list)
#     # 获取浓度组合
#     concentration_list = GetData.get_csv_data('data/data_SNR100/cc_data/conc_test.csv', i)
#     # 把获取每个物质的光谱强度数据与对应浓度相乘再叠加起来
#     composite_intensity = Method.mix(intensity_list, concentration_list)
#     # 把插值后的数据归一化
#     intensity_data_new = Method.l1_normalization(composite_intensity)
#     # 把叠加的光谱强度做插值，290->1000
#     wave_number_new, intensity_data_new = Method.interpolated_data(wave_number, intensity_data_new, num_wavenumber)
#     # 将干净数据保存
#     # Method.save(wave_number_new, 'data/data_SNR_neighbor2neighbor/wavenumber/wave_number500.csv', 'w', num_wavenumber)
#     Method.save(intensity_data_new, 'data/data_SNR100/test_data/clean_data.csv', 'a', num_wavenumber)
#     # 将干净数据加入噪声，并在保存
#     noise_data = Method.gaussian_noise_generator(intensity_data_new, target_snr=1.00, num_wavenumber=num_wavenumber)
#     Method.save(noise_data, 'data/data_SNR100/test_data/noise_data.csv', 'a', num_wavenumber)
# print("任务完成！")
