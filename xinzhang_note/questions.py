# zhangxin TODO

# @PIPELINES.register_module()


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

"""
张量可以看作是"增强版"的多维数组，它:

保持了多维数组的数据组织方式
添加了严格的类型约束
提供了丰富的数学运算
支持GPU加速计算
优化了内存布局和访问效率

"""

# 创建一个与给定张量 shape 相同、但所有元素都为 0 的新张量
# torch.zeros_like()


# print("\n=== 性能对比 ===")
# size = 1000000

# # 创建大型列表和张量
# large_list = list(range(size))
# large_tensor = torch.arange(size, dtype=torch.float32)

# # 计算平方和
# start_time = time.time()
# list_squared_sum = sum(x*x for x in large_list)
# list_time = time.time() - start_time

# start_time = time.time()
# tensor_squared_sum = torch.sum(large_tensor * large_tensor)
# tensor_time_cpu = time.time() - start_time

# print(f"列表计算时间: {list_time:.6f}秒")
# print(f"张量(CPU)计算时间: {tensor_time_cpu:.6f}秒")

# # 如果有GPU，可以测试GPU加速
# if torch.cuda.is_available():
#     large_tensor_gpu = large_tensor.cuda()      # 将张量移到GPU上
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    

#     start_time = time.time()
#     tensor_squared_sum_gpu = torch.sum(large_tensor_gpu * large_tensor_gpu)
#     tensor_time_gpu = time.time() - start_time
#     print(f"张量(GPU)计算时间: {tensor_time_gpu:.6f}秒")
#     print(f"GPU加速比: {list_time/tensor_time_gpu:.2f}倍")





"""
torch.cat 函数用于将多个张量沿着指定维度拼接在一起。语法如下:
torch.cat(tensors, dim=0, *, out=None)
tensors:一个包含多个张量的列表或元组。
dim:指定拼接的维度。dim=0 表示按行,dim=1 表示按列拼接。
out:可选参数，用于指定输出张量的形状。
torch.cat 函数的返回值是一个新的张量，它包含了所有输入张量在指定维度上的拼接结果。
注意:输入张量的形状在拼接维度上必须相同，否则会抛出异常。
"""

tensor1 = torch.cat([torch.tensor([1, 2]), torch.tensor([3, 4])], dim = -1)
tensor4 = torch.cat([torch.tensor([[1, 2]]), torch.tensor([[3, 4]])], dim = 0)
tensor2 = torch.cat([torch.tensor([[1], [2]]), torch.tensor([[3], [4]])], dim = 0)
tensor3 = torch.cat([torch.tensor([[1], [2]]), torch.tensor([[3], [4]])], dim = -1)
print(tensor1, tensor4, tensor2, tensor3)
# tensor3 = torch.cat([[1, 2], [3, 4]], dim = 1)