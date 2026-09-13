from flask import Flask, request, jsonify
import os

# 引入现有的模块和库
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import adv_method
import data_loader
import model_loader
from model_loader import *
from user import *
from config import *
from utils import *
import matplotlib.pyplot as plt
import os
import json
import argparse
import yaml
from flask_cors import CORS, cross_origin
from train_detector import *
import logging


logging.basicConfig(filename='./error.log', level=logging.DEBUG)
app = Flask(__name__)
CORS(app)  # 允许所有跨域请求

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 攻击核心逻辑，独立为函数
def run_attack(config_file, attack_name=None, model_name="VTCNN2", dataset_name=None):
    setup_seed(2020)

    config = Config()
    print(config_file)
    if config_file:
        assert os.path.exists(config_file), "There's no '" + config_file + "' file."
        with open(config_file, "r") as load_f:
            config_parameter = yaml.load(load_f, Loader=yaml.SafeLoader)
            config.load_parameter(config_parameter)
    

    # 动态修改攻击方法、模型和数据集
    if attack_name:
        config.CONFIG['attack_name'] = attack_name
    if model_name:
        config.CONFIG['model_name'] = model_name
    if dataset_name:
        config.CONFIG['dataset_name'] = dataset_name

    Log = config.log_output()

    # 加载数据集
    dataset_name = config.CONFIG['dataset_name']
    print(1)
    attack_set = getattr(data_loader, 'Rml2016_10aAttackSet')(**getattr(config, 'Rml2016_10a'), dataset_name = dataset_name)
    print(2)

    snrs, mods = attack_set.get_snr_and_mod()
    # 加载数据加载器
    attack_loader = DataLoader(attack_set, batch_size=32, shuffle=False, num_workers=4)

    # 加载模型
    model_name = config.CONFIG['model_name']
    model = getattr(model_loader, 'load' + model_name)(**getattr(config, model_name))
    model.eval()

    # 加载损失函数
    criterion_name = config.CONFIG['criterion_name']
    criterion = getattr(nn, criterion_name)()
    print(3)
    # 加载metric
    metrics = [getattr(module_metric, metric) for metric in config.CONFIG['metrics']]

    # 加载攻击方式
    attack_name = config.CONFIG['attack_name']
    print(attack_name)
    attack_parameter = getattr(config, attack_name)
    print(attack_parameter)
    attack_method = getattr(adv_method, attack_name)(model, criterion, **config.GPU, **attack_parameter)
    print(attack_method)

    # 加载攻击训练器
    attacker = Attacker(model, metrics, criterion, config, attack_method, snrs, mods)
    print('done')
    # 开始攻击
    log, real_sample, adv_sample, acc, predicts, targets, all_change_ratios, x_snrs, mean_change_ratio = attacker.start_attack(attack_loader)
    Log.update(log)
    print('log')


    return mods, predicts, adv_sample, all_change_ratios, acc, mean_change_ratio

# Flask API 路由
@app.route('/attack', methods=['GET'])
@cross_origin()
def attack():
    # 从 URL 参数中获取数据
    config_file = request.args.get('config', None)
    attack_name = request.args.get('attack_name', None)
    model_name = request.args.get('model_name', None)
    dataset_name = request.args.get('dataset_name', None)
    print(attack_name)

    
    # 执行攻击，并获取结果
    try:
        mods, predicts, adv_sample, all_change_ratios, acc, mean_change_ratio = run_attack(config_file, attack_name, model_name, dataset_name)
        data = []
        for i in range(len(predicts)):
            entry = {
                "type": mods[predicts[i].item()],  # 将 (84, 1) 的张量转为标量
                "iq_data": adv_sample[i].squeeze(0).tolist(),  # 将 (1, 2, 128) 的张量去掉第一个维度，并转为列表
                "change_ratio": all_change_ratios[i].item()  # 将 (84,) 的标量转为 Python 数值
            }
            data.append(entry)
        info = {
            "successInfo": f"攻击成功率{1 - float(acc)}",  # 转换为标准的 Python float
            "transformInfo": f"样本平均变化率: {float(mean_change_ratio)}"  # 转换为标准的 Python float
        }
        # 转换为 JSON 格式

        return jsonify({"code": "00000", "attackedRml2016Data": data, "info": info})
    except Exception as e:
        return jsonify({"code": "10000", "message": str(e)}), 500

@app.route('/load_data', methods=['GET'])
def load_data():
    # 从 GET 请求中获取文件名参数
    file_name = request.args.get('file')
    
    # 检查文件是否存在
    if not os.path.exists(file_name):
        return jsonify({"error": "File not found"}), 404

    try:
        # 加载指定的 pkl 文件
        with open(file_name, 'rb') as f:
            data = pickle.load(f, encoding="latin-1")
        
        # 准备返回的 JSON 数据
        response_data = []
        for key, value in data.items():
            mod_type, snr = key  # 调制类型和 SNR
            iq_data_list = value.tolist()  # IQ 数据转换为列表格式
            
            # 遍历 iq_data 列表，将每个 iq_data 对应一个条目
            for iq_data in iq_data_list:
                response_data.append({
                    'type': mod_type,
                    'snr': snr,
                    'iq_data': iq_data  # 单个 IQ 数据
                })

        # # 返回 JSON 响应
        # for item in response_data:
        #     print("type: ", item['type'], ", snr: ", item['snr'], "iq_data: ", item['iq_data'])

        return jsonify({
            "code": "00000",  # 成功响应的状态码
            "data": response_data
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/detect', methods=['GET'])
def detect_adversarial_samples():
    # 获取 POST 请求中的 JSON 参数
    dataset_name = request.args.get('selectedSignals', None)
    print(dataset_name)

    # 检查是否传入了 dataset_name 参数
    if not dataset_name:
        return jsonify({'error': 'Missing dataset_name parameter'}), 400

    # 调用 run_adversarial_detection_pipeline 函数执行检测
    dataset_dir = '/mnt/home/lzh/project/radioAdv/data_loader/dataset/RML2016.10a'
    prop = 0
    batch_size = 1024
    lr = 0.001
    epochs = 10
    pretrained_model_path = '/mnt/home/lzh/project/radioAdv/model/VTCNN2/VTCNN2_Epoch190.pkl'
    detector_model_path = '/mnt/home/lzh/project/radioAdv/model/model_epoch_10.pth'

    detector, accuracy, test_adv_samples, predictions, actuals = run_adversarial_detection_pipeline(
        dataset_dir, prop, batch_size, lr, epochs, 
        pretrained_model_path=pretrained_model_path, 
        detector_model_path=detector_model_path, 
        dataset_name=dataset_name
    )

    # 生成响应数据，将 test_adv_samples 和 test_predictions 对应起来
    response_data = []
    for i in range(len(test_adv_samples)):

        response_data.append({
            'predict_type': int(predictions[i]),
            'actual_type': int(actuals[i]),
            'iq_data': test_adv_samples[i].tolist(),  # 单个 IQ 数据转换为列表
        })

    return jsonify({
        'code': '00000',
        'accuracy': accuracy,
        'data': response_data
    })
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
