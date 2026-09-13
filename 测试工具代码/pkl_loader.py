import pickle

# 安全加载函数（防止恶意代码执行）
def safe_load_pkl(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"加载错误: {e}")
        return None

# 定义文件路径
file1 = 'D:/173/数据集/random_rml2016_subset_100.pkl'
file2 = 'D:/173/数据集/random_rml2016_subset7.pkl'
output_file='D:/173/数据集/random_rml2016_merged_1.pkl'

data = safe_load_pkl(file1)
if data is not None:
    # 通用查看方法
    print(file1)
    print("数据类型:", type(data))
    
    # 根据数据类型显示内容
    if isinstance(data, dict):
        print((data.keys().__len__()))
        print("\n字典键:", list(data.keys())[:], "...")  # 显示前5个键
        data_iter=iter(data)
        sample_key = next(data_iter)  # 获取第一个键
        print(len(data[sample_key]))
        print(f"\n示例值 ({sample_key}): {data[sample_key][:3]}")  # 显示前3个值
        # sample_key = next(data_iter)  # 获取第一个键
        # print(len(data[sample_key]))
        # print(f"\n示例值 ({sample_key}): {data[sample_key][:3]}")  # 显示前3个值
    
    elif isinstance(data, list):
        print("\n列表长度:", len(data))
        print("\n前3项:", data[:3])
    
    else:
        # 尝试打印所有内容（小文件适用）
        print("\n完整内容:")
        print(data)

# 合并两个pkl文件
def merge_dict_pkl_with_conflict(file1, file2, output_file, conflict_strategy='merge'):
    # 加载数据
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        dict1 = pickle.load(f1)
        dict2 = pickle.load(f2)
    
    merged_dict = dict1.copy()  # 从第一个字典开始
    
    for key, value in dict2.items():
        if key in merged_dict:
            # 处理冲突
            if conflict_strategy == 'merge':
                # 如果值都是列表，则合并列表
                if isinstance(merged_dict[key], list) and isinstance(value, list):
                    merged_dict[key].extend(value)
                # 如果值都是字典，则递归合并字典
                elif isinstance(merged_dict[key], dict) and isinstance(value, dict):
                    merged_dict[key] = merge_dicts(merged_dict[key], value)
                # 其他情况使用覆盖策略
                else:
                    merged_dict[key] = value
            elif conflict_strategy == 'overwrite':
                merged_dict[key] = value
            elif conflict_strategy == 'keep_first':
                pass  # 保留第一个字典的值
        else:
            merged_dict[key] = value
    
    # 保存结果
    with open(output_file, 'wb') as f_out:
        pickle.dump(merged_dict, f_out)
    
    return merged_dict

def merge_dicts(d1, d2):
    """递归合并字典"""
    for key, value in d2.items():
        if key in d1 and isinstance(d1[key], dict) and isinstance(value, dict):
            d1[key] = merge_dicts(d1[key], value)
        else:
            d1[key] = value
    return d1

# merge_dict_pkl_with_conflict(file1, file2, output_file, conflict_strategy='merge')
