import random
import pickle
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset



def load_data_rml2018(dirname, prop):
    """[加载RML2018数据集]

    Args:
        dirname ([str]]): [RML2016.10a数据集所在的绝对路径]
        prop ([float]]): [训练集所占的比例]
        expand_dims ([bool]): 是否需要拓展维度, 用户LSTM模型的输入

    Returns:
        [train_x]: [训练数据，(2555904*prop,1,2,1024)，若不拓展维度，输出为(2555904*prop,2,1024) 以下同理]
        [train_y]: [训练标签，(2555904*prop,1,)]
        [test_x]: [测试数据，(2555904*(1-prop),1,2,1024)]
        [test_y]: [测试标签，(2555904*(1-prop),1,)]
        [snrs]: [数据中所包含的所有的SNR,list[int], 包括-20dB~30dB]
        [mods]: [数据中所包含的所有的调制类型,list[str]，包括0~23] 其index对应与标签
        classes = ['32PSK','16APSK', '32QAM', 'FM', 'GMSK', '32APSK', 'OQPSK', '8ASK', 'BPSK', '8PSK', 'AM-SSB-SC', '4ASK', '16PSK', '64APSK', '128QAM', '128APSK', 'AM-DSB-SC', 'AM-SSB-WC', '64QAM', 'QPSK', '256QAM', 'AM-DSB-WC', 'OOK', '16QAM']

    """
    hdf5_file = h5py.File(os.path.join(dirname, "GOLD_XYZ_OSC.0001_1024.hdf5"),  'r')
    datas = np.array(hdf5_file['X'])
    modulation_onehot = hdf5_file['Y']
    snrs_data = np.squeeze(np.array(hdf5_file['Z']))
    mods_data = np.argmax(modulation_onehot, axis=1)

    snrs = list(set(snrs_data))
    snrs.sort()
    mods = list(set(mods_data))

    datas = np.expand_dims(datas, axis = 1)
    datas = datas.transpose(0,1,3,2)

    np.random.seed(2018)
    n_examples = datas.shape[0]
    n_train = int(n_examples * prop)
    train_idx = np.random.choice(range(0, n_examples),size = n_train,replace = False)
    test_idx = list(set(range(0,n_examples)) - set(train_idx))
    train_x = datas[train_idx]
    train_y = mods_data[train_idx]
    train_snr = snrs_data[train_idx]
    test_x = datas[test_idx]
    test_y = mods_data[test_idx]
    test_snr = snrs_data[test_idx]
    return train_x, train_y, train_snr, test_x, test_y, test_snr, snrs, mods

def data_save_rml2018(dirname, prop):
    """[加载RML2018数据集]
    将数据存储到文件中
    训练数据train_x, train_y, train_snr 存储到文件'train_data.p'中，加载方式如下：
        train_x, train_y, train_snr = pickle.load(open(os.path.join(dirname, 'train_data.p'), 'rb'))
        train_x : 训练数据，(2555904*prop,1,2,128)，若不拓展维度，输出为(2555904*prop,2,128) 以下同理
        train_y : 训练标签，(2555904*prop,1,)
        train_snr : 训练数据所对应的snr , (2555904*prop,1)
    测试数据test_x, test_y, test_snr 存储到文件'test_data.p'中，加载方式如下：
        test_x, test_y, test_snr = pickle.load(open(os.path.join(dirname, 'test_data.p'), 'rb'))
        test_x : 测试数据，(2555904*(1-prop),1,2,128)
        test_y : 测试标签，(2555904*(1-prop),1,)
        test_snr : 测试数据所对应的snr , (2555904*(1-prop),1,)
    各类超参数snrs(SNR种类)，mods(调制方式种类) 存储到文件'augments.p'中，加载方式如下：
        snrs, mods = pickle.load(open(os.path.join(dirname, 'augments.p'), 'rb'))
        snrs : 数据中所包含的所有的SNR,list[int]
        mods : 数据中所包含的所有的调制类型,list[str] 其index对应与标签

    Args:
        dirname ([str]]): [RML2016.10a数据集所在的绝对路径]
        prop ([float]]): [训练集所占的比例]

    Returns:

    """
    train_x, train_y, train_snr, test_x, test_y, test_snr, snrs, mods = load_data_rml2018(dirname, prop)
    pickle.dump([train_x, train_y, train_snr], open(os.path.join(dirname, 'train_data_rml2018.p'), 'wb'),protocol = 4)
    pickle.dump([test_x, test_y, test_snr], open(os.path.join(dirname, 'test_data_rml2018.p'), 'wb'),protocol = 4)
    pickle.dump([snrs, mods], open(os.path.join(dirname, 'augments_rml2018.p'), 'wb'),protocol = 4)



class Rml2018_01aTrainSet(Dataset):
    def __init__(self,dirname, prop):
        """[加载Rml2018.01a训练集]

        Args:
            dirname ([str]): [RML2016.10a文件的绝对路径]]
            prop ([float]): [训练集所占的比例]
        """
        if not os.path.exists(os.path.join(dirname, 'train_data_rml2018.p')):
            data_save_rml2018(dirname, prop)     
            print('Data genreated Successfully')
        
        train_x, train_y, train_snr = pickle.load(open(os.path.join(dirname, 'train_data_rml2018.p'), 'rb'))
        snrs, mods = pickle.load(open(os.path.join(dirname, 'augments_rml2018.p'), 'rb'))
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


class Rml2018_01aTestSet(Dataset):
    def __init__(self,dirname, prop):
        """[加载RML2016.10a训练集]

        Args:
            dirname ([str]): [RML2016.10a文件的绝对路径]]
            prop ([float]): [训练集所占的比例]
        """        

        test_x, test_y, test_snr = pickle.load(open(os.path.join(dirname, 'test_data_rml2018.p'), 'rb'))
        snrs, mods = pickle.load(open(os.path.join(dirname, 'augments_rml2018.p'), 'rb'))

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


class Rml2018_01aAttackSet(Dataset):
    def __init__(self,dirname, prop):
        """[加载RML2016.10a攻击集，在各种模型上分类都正确，snr>=0]

        Args:
            dirname ([str]): [RML2016.10a文件的绝对路径]]
            prop ([float]): [训练集所占的比例]
        """        

        attack_x, attack_y, attack_snr = pickle.load(open(os.path.join(dirname, 'attack_data_rml2018.p'), 'rb'))
        snrs, mods = pickle.load(open(os.path.join(dirname, 'augments_rml2018.p'), 'rb'))

        self.data = attack_x
        self.labels = attack_y
        self.data_snr = attack_snr
        self.snrs = snrs
        # snr_0 = snrs.index(0)
        # # snr_index = (self.data_snr == 10) | (self.data_snr == 0)
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


class Rml2018_01aAdvSampleSet(Dataset):
    def __init__(self, adv_sample, targets, x_snrs):
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