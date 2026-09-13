import pickle
import numpy as np
import pandas as pd

src_path = 'D:/173/数据集/random_rml2016_subset_300.pkl'  # 原始数据（字典： (mod, snr) -> np.array(samples) ）
with open(src_path, 'rb') as f:
    data = pickle.load(f, encoding='latin-1')

# 目标：提取总计100条样本，保持与原数据相同的字典结构
target_total = 100
selected = {}
count = 0

# 按键的字母序/数值序稳定遍历
keys = sorted(list(data.keys()), key=lambda k: (str(k[0]), k[1]))

for key in keys:
    if count >= target_total:
        break
    arr = np.array(data[key])  # 形状一般为 (n_samples, 2, T)
    if arr.ndim == 1:
        # 如果是单条样本，扩一维
        arr = arr[None, ...]
    remaining = target_total - count
    take_n = min(len(arr), remaining)
    if take_n <= 0:
        continue
    # 取前 take_n 条
    take_arr = arr[:take_n]
    if key not in selected:
        selected[key] = []
    selected[key].append(take_arr)
    count += take_n

# 拼接每个key下收集到的片段
final_dict = {}
for k, chunks in selected.items():
    final_dict[k] = np.concatenate(chunks, axis=0)

# 保存为新的 pkl，结构与原始一致：dict[(mod, snr)] -> np.array
dst_path = 'D:/173/数据集/random_rml2016_subset_100.pkl'
with open(dst_path, 'wb') as f:
    pickle.dump(final_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

# 返回统计信息
stats = {
    'total_selected': sum(len(v) for v in final_dict.values()),
    'per_key_counts': {str(k): int(len(v)) for k, v in final_dict.items()}
}
dst_path, stats
