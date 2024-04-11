import matplotlib
import numpy as np
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

SNR_ks = [1.65, 1.93, 1.37, 1.62, 1.51, 3.64]
p = [0.8425, 0.8670, 0.8144, 0.8582, 0.8439, 0.8305]

num = np.arange(1, 7)
# 绘图
plt.figure(figsize=(10, 6))

plt.bar(num, SNR_ks, color='#99E5FF', width=0.4, label='kernal_size')
# plt.bar(num+0.2, UL, color='#66E5FF', width=0.4, label='U-net LSTM')
# plt.bar(num+0.1, SNR_075, color='#33E5FF', width=0.4, label='SNR=0.75')
# plt.bar(num+0.2, UL, color='#00E0FF', width=0.4, label='U-net LSTM')

# plt.xticks(np.arange(min(num), max(num)+1, 1))  # 从最小整数到最大整数，步长为1

for i, j in enumerate(SNR_ks):
    plt.text(i+1, j, str(j), ha='center', va='bottom', fontsize=10)
# for i, j in enumerate(UL):
    # plt.text(i+1.2, j, str(j), ha='center', va='bottom', fontsize=10)
# for i, j in enumerate(SNR_075):
#     plt.text(i+1.1, j, str(j), ha='center', va='bottom', fontsize=10)
# for i, j in enumerate(SNR_100):
#     plt.text(i+1.3, j, str(j), ha='center', va='bottom', fontsize=10)

# plt.title('Pearson correlation coefficient', fontsize=15)
plt.title('SNR', fontsize=15)
#
plt.xticks([1, 2, 3, 4, 5, 6], ['k=3', 'k=5', 'k=7', 'k=9', 'k=11', 'k=9(n2n)'])
# plt.ylim(0, 5)
# plt.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5])
# plt.ylabel('Average SNR', fontsize=15)
# plt.xlabel('Different SNR', fontsize=15)

plt.grid(axis='y', color='gray', linewidth=0.5, alpha=0.5)
plt.legend(borderaxespad=0.1)
plt.tight_layout()
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False
plt.show()

"""
蓝色系列
#99CCFF
#66CCFF
#33CCFF
#00CCFF
"""

"""
α (Alpha): \u03B1
β (Beta): \u03B2
γ (Gamma): \u03B3
δ (Delta): \u03B4
ε (Epsilon): \u03B5
ζ (Zeta): \u03B6
η (Eta): \u03B7
θ (Theta): \u03B8
ι (Iota): \u03B9
κ (Kappa): \u03BA
λ (Lambda): \u03BB
μ (Mu): \u03BC
ν (Nu): \u03BD
ξ (Xi): \u03BE
ο (Omicron): \u03BF
π (Pi): \u03C0
ρ (Rho): \u03C1
σ (Sigma): \u03C3
τ (Tau): \u03C4
υ (Upsilon): \u03C5
φ (Phi): \u03C6
χ (Chi): \u03C7
ψ (Psi): \u03C8
ω (Omega): \u03C9
"""