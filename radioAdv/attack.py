'''
Author: your name
Date: 2020-12-07 15:03:08
LastEditTime: 2021-02-09 15:32:51
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /radioAdv/attack.py
'''
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import adv_method
import data_loader
import model_loader
from model_loader import *
from user import *
from config import *
from utils import *
from local_paths import dataloader_workers, build_dataset_kwargs, resolve_model_path
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import json

import argparse
import yaml
import torch
torch.backends.cudnn.enabled=True

def main():
    ######################################加载配置
    setup_seed(2020)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                        default=None,
                        help="The config parameter yaml file to be changed",
                        type=str)

    parser.add_argument('--vGPU',
                        nargs='+',
                        type=int,
                        default=None,
                        help="Specify which GPUs to use.")

    args = parser.parse_args()


    config = Config()

    if args.config:
        assert os.path.exists(args.config), "There's no '" + args.config + "' file."
        with open(args.config, "r") as load_f:
            config_parameter = yaml.load(load_f, Loader=yaml.SafeLoader)
            config.load_parameter(config_parameter)

    if args.vGPU:
        config.GPU['device_id'] = args.vGPU
        
    Log = config.log_output()

    ######################################加载数据集
    dataset_name = config.CONFIG['dataset_name']
    attack_set = getattr(data_loader, dataset_name + 'AttackSet')(**build_dataset_kwargs(getattr(config, dataset_name)))
    # test_set = getattr(data_loader, dataset_name + 'TestSet')(**getattr(config, dataset_name))
    # train_set = getattr(data_loader, dataset_name + 'TrainSet')(**getattr(config, dataset_name))
    snrs, mods = attack_set.get_snr_and_mod()  
    # snrs, mods = test_set.get_snr_and_mod()
    print(mods)


    ######################################加载数据加载器
    attack_loader = DataLoader(attack_set, batch_size = 1024, shuffle = False, num_workers = dataloader_workers())
    # test_loader = DataLoader(test_set, batch_size = 32, shuffle = True, num_workers = 1)
    # train_loader = DataLoader(train_set, batch_size = 1024, shuffle = True, num_workers = 1)

    ######################################加载模型
    model_name = config.CONFIG['model_name']
    model_kwargs = dict(getattr(config, model_name))
    if 'filepath' in model_kwargs:
        model_kwargs['filepath'] = resolve_model_path(model_kwargs['filepath'], model_name)
    model = getattr(model_loader, 'load' + model_name)(**model_kwargs)
    model.eval()
    ######################################加载损失函数
    criterion_name = config.CONFIG['criterion_name']
    criterion = getattr(nn, criterion_name)()

    ## 加载metric
    metrics = [getattr(module_metric, metric) for metric in config.CONFIG['metrics']]

    ######################################加载攻击方式
    attack_name = config.CONFIG['attack_name']
    attack_parameter = getattr(config, attack_name)
    attack_method = getattr(adv_method, attack_name)(model, criterion, **config.GPU, **attack_parameter)


    ######################################加载攻击训练器
    attacker = Attacker(model, metrics, criterion, config, attack_method, snrs, mods)


    #####################################开始攻击

    ###############################攻击一张图片
    # x,y = TrainSet.__getitem__(7000)
    # x_adv,pertubation,nowLabel = attacker.attack_one_img(x,y)
    # print(y,nowLabel)

    ###############################攻击整个数据集
    # attack_log = attacker.attack_set(test_loader)
    # Log['attack_log'] = attack_log
    log, real_sample, adv_sample, acc, predicts, targets, all_change_ratios, x_snrs, mean_change_ratio = attacker.start_attack(attack_loader)
    Log.update(log)
    # 假设 real_sample 是形状为 (84, 1, 2, 128) 的张量
    print(len(real_sample))
    first_sample = real_sample[0]  # 提取第一个样本, 形状为 (1, 2, 128)
    first_sample_2x128 = first_sample.squeeze(0)  # 去掉第一个维度，得到形状为 (2, 128)

    # print(first_sample_2x128)  # 输出 (2, 128)
    print(adv_sample.shape)
    print(predicts.shape)
    print(all_change_ratios.shape)
    print(mean_change_ratio)

    # 构造 JSON 数据
    data = []
    for i in range(len(predicts)):
        entry = {
            "type": predicts[i].item(),  # 将 (84, 1) 的张量转为标量
            "iq_data": adv_sample[i].squeeze(0).tolist(),  # 将 (1, 2, 128) 的张量去掉第一个维度，并转为列表
            "change_ratio": all_change_ratios[i].item()  # 将 (84,) 的标量转为 Python 数值
        }
        data.append(entry)

    # 转换为 JSON 格式
    json_data = json.dumps({"data": data}, indent=4)

    # 打印 JSON 数据
    # print(json_data)

    ####################################log保存
    filename = os.path.join(config.Checkpoint['log_dir'], config.Checkpoint['log_filename'])
    f = open(filename,'w')

    for key, value in Log.items():
        print('    {:15s}: {}'.format(str(key), value))
        log = {}
        log[key] = value
        log_write(f, log)
    # dirname = '/home/yuzhen/wireless/RML2016.10a'
    # attacker.adversarial_training_dataset_generating(train_loader, dirname)


    # # plt.switch_backend('agg')
    # plt.imshow(x[0])
    # plt.imshow(x_adv[0])
    # plt.show()
    # plt.imshow(x_adv[0])

    # attacker.generate_data(test_loader)
    # dirname = '/home/yuzhen/wireless/RML2016.10a'
    # attack_x, attack_y, attack_snr = pickle.load(open(os.path.join(dirname, 'attack_data.p'), 'rb'))
    # print(len(attack_x))

    # attacker.pick_data(test_loader)

if __name__ == "__main__":
    main()