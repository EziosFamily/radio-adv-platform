import random
import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from local_paths import resolve_pkl_stem, resolve_split_files, resolve_data_file


def load_data(dirname, prop, dataset_name):
    """[加载RML2016.10a数据集]

    Args:
        dirname ([str]]): [RML2016.10a数据集所在的绝对路径]
        prop ([float]]): [训练集所占的比例]
        expand_dims ([bool]): 是否需要拓展维度, 用户LSTM模型的输入

    Returns:
        [train_x]: [训练数据，(220000*prop,1,2,128)，若不拓展维度，输出为(220000*prop,2,128) 以下同理]
        [train_y]: [训练标签，(220000*prop,1,)]
        [test_x]: [测试数据，(220000*(1-prop),1,2,128)]
        [test_y]: [测试标签，(220000*(1-prop),1,)]
        [snrs]: [数据中所包含的所有的SNR,list[int]]
        [mods]: [数据中所包含的所有的调制类型,list[str]] 其index对应与标签
    """
    dirname, dataset_name = resolve_pkl_stem(dirname, dataset_name)
    f = open(os.path.join(dirname, dataset_name + '.pkl'), 'rb')
    rml_data = pickle.load(f, encoding = 'latin1')
    snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], rml_data.keys())))), [1,0])
    datas = []
    labels = []
    for mod in mods:
        for snr in snrs:
            if rml_data[(mod, snr)].shape[0] == 0:
                    continue
            datas.append(rml_data[(mod, snr)])
            for i in range(rml_data[(mod, snr)].shape[0]):
                labels.append((mod, snr))
    datas = np.vstack(datas)
    datas = np.expand_dims(datas, axis = 1)

    np.random.seed(2016)
    n_examples = datas.shape[0]
    n_train = int(n_examples * 0.5)
    train_idx = np.random.choice(range(0, n_examples),size = n_train,replace = False)
    test_idx = list(set(range(0,n_examples)) - set(train_idx))
    train_x = datas[train_idx]
    test_x = datas[test_idx]
    train_y = list(map(lambda x: mods.index(labels[x][0]), train_idx))
    train_snr = list(map(lambda x: labels[x][1], train_idx))
    test_y = list(map(lambda x: mods.index(labels[x][0]), test_idx))
    test_snr = list(map(lambda x: labels[x][1], test_idx))
    return train_x, train_y, train_snr, test_x, test_y, test_snr, snrs, mods

def data_save(dirname, prop, dataset_name):
    """[加载RML2016.10a数据集]
    将数据存储到文件中
    训练数据train_x, train_y, train_snr 存储到文件'train_data.p'中，加载方式如下：
        train_x, train_y, train_snr = pickle.load(open(os.path.join(dirname, 'train_data.p'), 'rb'))
        train_x : 训练数据，(220000*prop,1,2,128)，若不拓展维度，输出为(220000*prop,2,128) 以下同理
        train_y : 训练标签，(220000*prop,1,)
        train_snr : 训练数据所对应的snr , (220000*prop,1)
    测试数据test_x, test_y, test_snr 存储到文件'test_data.p'中，加载方式如下：
        test_x, test_y, test_snr = pickle.load(open(os.path.join(dirname, 'test_data.p'), 'rb'))
        test_x : 测试数据，(220000*(1-prop),1,2,128)
        test_y : 测试标签，(220000*(1-prop),1,)
        test_snr : 测试数据所对应的snr , (220000*(1-prop),1,)
    各类超参数snrs(SNR种类)，mods(调制方式种类) 存储到文件'augments.p'中，加载方式如下：
        snrs, mods = pickle.load(open(os.path.join(dirname, 'augments.p'), 'rb'))
        snrs : 数据中所包含的所有的SNR,list[int]
        mods : 数据中所包含的所有的调制类型,list[str] 其index对应与标签

    Args:
        dirname ([str]]): [RML2016.10a数据集所在的绝对路径]
        prop ([float]]): [训练集所占的比例]

    Returns:

    """
    dirname, dataset_name = resolve_pkl_stem(dirname, dataset_name)
    train_x, train_y, train_snr, test_x, test_y, test_snr, snrs, mods = load_data(dirname, prop, dataset_name)
    pickle.dump([train_x, train_y, train_snr], open(os.path.join(dirname, dataset_name + '_train_data.p'), 'wb'))
    pickle.dump([test_x, test_y, test_snr], open(os.path.join(dirname, dataset_name + '_test_data.p'), 'wb'))
    pickle.dump([snrs, mods], open(os.path.join(dirname, dataset_name + '_augments.p'), 'wb'))



class Rml2016_10aTrainSet(Dataset):
    def __init__(self,dirname, prop, dataset_name=None):
        """[加载RML2016.10a训练集]

        Args:
            dirname ([str]): [RML2016.10a文件的绝对路径]]
            prop ([float]): [训练集所占的比例]
        """
        folder, stem, train_path, test_path, aug_path, need_generate = resolve_split_files(dirname, dataset_name)
        if need_generate:
            data_save(folder, prop, stem)
            print('Data genreated Successfully')
            folder, stem, train_path, test_path, aug_path, need_generate = resolve_split_files(folder, stem)
        
        train_x, train_y, train_snr = pickle.load(open(train_path, 'rb'))
        snrs, mods = pickle.load(open(aug_path, 'rb'))
        self.data = train_x
        self.labels = train_y
        self.train_snr = train_snr
        self.snrs = snrs
        self.mods = mods
        
    def __getitem__(self,idx):
        return (self.data[idx],self.labels[idx], self.train_snr[idx])
    
    def __len__(self):
        return len(self.data)

    def get_snr_and_mod(self):
        return self.snrs, self.mods


class Rml2016_10aTestSet(Dataset):
    def __init__(self,dirname, prop, dataset_name=None):
        """[加载RML2016.10a训练集]

        Args:
            dirname ([str]): [RML2016.10a文件的绝对路径]]
            prop ([float]): [训练集所占的比例]
        """        
        folder, stem, train_path, test_path, aug_path, need_generate = resolve_split_files(dirname, dataset_name)
        if need_generate:
            data_save(folder, prop, stem)
            print('Data genreated Successfully')
            folder, stem, train_path, test_path, aug_path, need_generate = resolve_split_files(folder, stem)

        test_x, test_y, test_snr = pickle.load(open(test_path, 'rb'))
        snrs, mods = pickle.load(open(aug_path, 'rb'))

        self.data = test_x
        self.labels = test_y
        self.data_snr = test_snr
        self.snrs = snrs
        self.mods = mods
        
    def __getitem__(self,idx):
        return (self.data[idx], self.labels[idx], self.data_snr[idx])
    
    def __len__(self):
        return len(self.data)

    def get_snr_and_mod(self):
        return self.snrs, self.mods


class Rml2016_10aAttackSet(Dataset):
    def __init__(self,dirname, prop, dataset_name=None):
        """[加载RML2016.10a攻击集，在各种模型上分类都正确，snr>=0]

        Args:
            dirname ([str]): [RML2016.10a文件的绝对路径]]
            prop ([float]): [训练集所占的比例]
        """        
        folder, stem, train_path, test_path, aug_path, need_generate = resolve_split_files(dirname, dataset_name)
        self.dataset_name = stem

        if need_generate:
            data_save(folder, prop, stem)
            print('Data genreated Successfully')
            folder, stem, train_path, test_path, aug_path, need_generate = resolve_split_files(folder, stem)
        attack_x, attack_y, attack_snr = pickle.load(open(test_path, 'rb'))
        snrs, mods = pickle.load(open(aug_path, 'rb'))

        self.data = attack_x
        self.labels = attack_y
        self.data_snr = attack_snr
        self.snrs = snrs

        # snr_0 = snrs.index(0)
        # snr_index = (self.data_snr == 10) | (self.data_snr == 0)
        # snr_index = (self.data_snr == 10)
        # self.data = attack_x[snr_index]
        # self.labels = attack_y[snr_index]
        # self.data_snr = attack_snr[snr_index]
        # self.snrs = np.array([10])

        # label_index = self.labels >= 8 
        # self.data = attack_x[label_index]
        # self.labels = attack_y[label_index]
        # self.data_snr = attack_snr[label_index]
        
        self.mods = mods
        
    def __getitem__(self,idx):
        return (self.data[idx], self.labels[idx], self.data_snr[idx])
    
    def __len__(self):
        return len(self.data)

    def get_snr_and_mod(self):
        return self.snrs, self.mods


def load_rml_pkl_records(file_name):
    """Load every IQ sample from an RML-style pkl, in a stable order."""
    resolved = resolve_data_file(file_name)
    with open(resolved, 'rb') as f:
        raw = pickle.load(f, encoding='latin-1')
    mods = sorted(set(key[0] for key in raw.keys()))
    snrs = sorted(set(key[1] for key in raw.keys()))
    records = []
    for mod, snr in sorted(raw.keys(), key=lambda item: (str(item[0]), item[1])):
        block = raw[(mod, snr)]
        for iq in block:
            records.append({
                'type': mod,
                'snr': snr,
                'iq_data': np.asarray(iq),
                'label': mods.index(mod),
            })
    return records, mods, snrs, resolved


class RmlPklSignalSet(Dataset):
    """Use the selected pkl as-is so the UI count matches the dataset file."""

    def __init__(self, file_name):
        records, mods, snrs, resolved = load_rml_pkl_records(file_name)
        xs = []
        for item in records:
            x = np.asarray(item['iq_data'], dtype=np.float32)
            if x.ndim == 2:
                x = np.expand_dims(x, 0)
            xs.append(x)
        self.data = np.stack(xs, axis=0) if xs else np.zeros((0, 1, 2, 128), dtype=np.float32)
        self.labels = np.array([item['label'] for item in records], dtype=np.int64)
        self.data_snr = np.array([item['snr'] for item in records])
        self.mods = mods
        self.snrs = snrs
        self.records = records
        self.source_path = resolved

    def __getitem__(self, idx):
        return (self.data[idx], self.labels[idx], self.data_snr[idx])

    def __len__(self):
        return len(self.data)

    def get_snr_and_mod(self):
        return self.snrs, self.mods


class Rml2016_10aAdvSampleSet(Dataset):
    def __init__(self, adv_sample, targets, x_snrs, dataset_name):
        """[用于训练过程中生成的对抗样本构建数据集]

        Args:
            dirname ([str]): [RML2016.10a文件的绝对路径]]
            prop ([float]): [训练集所占的比例]
        """        
        self.data = adv_sample
        self.labels = targets
        self.data_snr = x_snrs
        
    def __getitem__(self,idx):
        return (self.data[idx], self.labels[idx], self.data_snr[idx])
    
    def __len__(self):
        return len(self.data)