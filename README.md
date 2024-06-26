# 测试N2N在树莓派上的推理速度
### 配置
  cuda 11.6<br>
  onnx-1.16.0 、onnxruntime-gpu(cuda11.6对应1.14-1.13)

### TODO
·pycharm断点调试 按F8后只停留在第一个断点，不往下执行<br>
·如何修改让代码保持在一个地方，比如说

### To Check
·给的pth模型、所用训练网络是否匹配？model_resnet_d_k9.pth & from model.model_500_k5 import UNet

# Log
### Day1:
#### ·修改原始代码加载模型的位置
解释：推理过程中，加载模型的代码不能写到for循环里面，否则每次推理都会重新Load一遍<br>
效果：每次的inference时间由原来的0.7-0.8s，变为0.36-0.40s左右<br>
### Day2:
#### ·修改加载数据集的位置
区别：若内存不足，则用原来的版本。一次性加载所有数据集，加快推理速度，但占用内存资源。在推理时加载数据集降低内存占用，但更耗时
效果【注：Data size:200】：<br>
提前加载： 加载用时32.8s 平均推理用时0.175s 总推理用时35s 加载+推理1min左右
单个加载（原）：平均用时0.36-0.40s，总用时69.243s<br>
<br>
#### ·将模型.pth转为.onnx格式
Average Inference Time: 0.1133693015575409 seconds  
Total Inference Time:22.67386031150818 seconds  
Average Loss: 0.042698722067143535  
Average Loss: 0.00021349361033571769  
snr_average:12.607415153980256  
p_average0.9414271152117163  
<br>
#### ·将模型.onnx转为IR格式
用openvino测试IR模型的性能<br>

    benchmark_app -m model/onnx/resnet_d_k9_IR.xml -i data/data_SNR075/test_data/noise_data.csv -d CPU -api sync -t 15000
    //benchmark_app -m <IR模型路径.xml> -i <测试集的路径> -d <设备类型> -api <异步或同步API> -t <持续时间> -b <批处理大小>
  
### Day3:
#### ·用openvino的IECore类在CPU上进行异步推理
Average Inference Time: 0.11506508231163025 seconds  
Total Inference Time: 23.01301646232605 seconds  
Average Loss: 0.00021349360758904368  
snr_average: 12.607415227890014  
p_average: 0.941427119089166  

<br>
<br>
<br>

## ?
### ·Intel MKL 库
一套高度优化的数学函数库，专门用于提高科学、工程和金融应用程序的性能。它提供了一系列高性能的数学函数，涵盖了线性代数、傅立叶变换、随机数生成、特征值求解等领域.MKL 的优势在于其针对 Intel 处理器架构进行了高度优化，可以充分利用处理器的多核和矢量化指令集，从而实现更高的计算性能。它支持多种操作系统和开发环境，并提供了简单易用的 API 接口.<br>
在机器学习、科学计算、数据分析等领域，MKL 往往被用于加速矩阵运算、向量运算等计算密集型任务，从而提高程序的性能和效率。<br>
