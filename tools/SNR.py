import matplotlib
import pandas as pd
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from process_data import Method, GetData

path = ['../data/data_SNR_neighbor2neighbor/denoise_data/output/snr/snr_UNet_500_k9.csv',
        '../data/data_SNR_neighbor2neighbor/denoise_data/output/p/p_UNet_500_k9.csv']

numbers = list(range(200))
snr_data = pd.read_csv(path[0], sep=',', header=None)
snr_data = snr_data.iloc[0:201, 0]
snr_data = snr_data.tolist()
SNR = snr_data[0:200]
average_SNR = round(snr_data[200], 2)

p_data = pd.read_csv(path[1], sep=',', header=None)
p_data = p_data.iloc[0:201, 0]
p_data = p_data.tolist()
Pearson = p_data[0:200]
average_Pearson = round(p_data[200], 4)

plt.figure(figsize=(10, 5))
plt.subplot(2, 1, 1)
plt.plot(numbers, SNR, 'r',  linestyle='solid', label='SNR')
plt.axhline(y=average_SNR, color='b',  linestyle='dashed', label='Average_SNR')
plt.title('SNR(k9)')
plt.grid(linestyle='--', color='gray', linewidth=0.5, alpha=0.5)
plt.legend(borderaxespad=0.1, loc='lower right')
plt.text(0, average_SNR, str(average_SNR), ha='center', va='bottom', fontsize=15, color='b')

plt.subplot(2, 1, 2)
plt.plot(numbers, Pearson, 'r', linestyle='solid', label='Pearson')
plt.axhline(y=average_Pearson, color='b',  linestyle='dashed', label='Average_pearson')
plt.title('Pearson correlation coefficient(k9)')
plt.ylim(0, 1)
plt.grid(linestyle='--', color='gray', linewidth=0.5, alpha=0.5)
plt.legend(borderaxespad=0.1, loc='lower right')
plt.text(0, average_Pearson, str(average_Pearson), ha='center', va='bottom', fontsize=15, color='b')

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.tight_layout()
plt.show()
