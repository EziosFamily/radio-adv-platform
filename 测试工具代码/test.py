
import pickle
import numpy as np
from collections import OrderedDict

# 加载原始数据
with open('D:/173/数据集/RML2016.10a_dict.pkl', 'rb') as f:
    data = pickle.load(f, encoding='latin1')  # 字典，220个键，每个键对应 (1000, 2, 128) 的数组

# 计算每个键需要抽取的样本数
total_samples_needed = 5000
samples_per_key = total_samples_needed // len(data)  # 每个键至少13条
remaining_samples = total_samples_needed % len(data)  # 剩余14条需分配

# 创建新字典保存3000条数据
new_data = {}

for i, key in enumerate(data.keys()):
    # 当前键抽取的样本数（前14个键多抽1条）
    n_samples = samples_per_key + (1 if i < remaining_samples else 0)

    # 从当前键的数组中随机抽取n_samples条（避免连续抽取）
    indices = np.random.choice(data[key].shape[0], n_samples, replace=False)
    new_data[key] = data[key][indices]  # 保存抽取的样本
    if (i>=remaining_samples)&(samples_per_key==0):break

# 保存新文件
with open('D:/173/数据集/random_rml2016_subset_5000.pkl', 'wb') as f:
    pickle.dump(new_data, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"已从 {len(data)} 个键中抽取共 {sum(v.shape[0] for v in new_data.values())} 条数据，保存为 random_rml2016_subset_5000.pkl")

# with open('extracted_3000_samples.pkl', 'rb') as f:
#     extracted_data = pickle.load(f)
#     total_extracted = sum(v.shape[0] for v in extracted_data.values())
#     print(f"实际抽取条数: {total_extracted}")
#     print(f"示例键的形状: {extracted_data[next(iter(extracted_data))].shape}")  # 应类似 (23, 2, 128)

