# import matplotlib
# import numpy as np
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
# from process_data import Method, GetData, DataGenerator
# import math
# # 单一物质
# substances = {'CL': 0, 'DNA': 1, 'Erg': 2, 'LPC': 3, 'OPC': 4, 'PA': 5, 'PC': 6,
#               'PE': 7, 'PS': 8, 'SPH': 9, 'cyto': 10, 'Protein': 11, 'PI': 12}
# # 获取物质的名称列表
# substances_list = list(substances.keys())
# # 列出所有可能的组合
# combinations = []
# for i in range(len(substances_list) - 2):
#     for j in range(i + 1, len(substances_list) - 1):
#         for k in range(j + 1, len(substances_list)):
#             combination = [substances_list[i], substances_list[j], substances_list[k]]
#             combinations.append(combination)
# # 打印所有可能的组合
# for idx, combo in enumerate(combinations, 1):
#     print("组合 {}: {}".format(idx, combo))
# for i in range(107, 134):
#     combination_list = GetData.get_csv_data('../data/data_SNR05/cc_data/combo_test.csv', i)
#     wave_number, intensity_list = GetData.get_mat_data('Pure.mat', 'lowwn', combination_list)
#     concentration_list = GetData.get_csv_data('../data/data_SNR05/cc_data/conc_test.csv', i)
#     composite_intensity = Method.mix(intensity_list, concentration_list)
#     intensity_data_new = Method.l1_normalization(composite_intensity)
#     plt.figure(figsize=(10, 2))
#     plt.plot(wave_number, intensity_data_new, linestyle='solid', color='#CCCCCC')
#     plt.xlim(29, 318)
#     plt.grid(linestyle='--', color='gray', linewidth=0.5, alpha=0.5)
#     plt.rcParams['font.sans-serif'] = ['Times New Roman']
#     plt.rcParams['axes.unicode_minus'] = False
#     plt.tight_layout()
#     plt.show()



