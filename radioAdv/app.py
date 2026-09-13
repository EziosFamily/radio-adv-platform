import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
import pickle

# 引入现有的模块和库
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import adv_method
import data_loader
import model_loader
from model_loader import *
from user import *
from config import *
from utils import *
from local_paths import (
    dataloader_workers,
    build_dataset_kwargs,
    resolve_model_path,
    resolve_data_file,
    resolve_config_file,
    PROJECT_DATASET_DIR,
    LOCAL_DATASET_DIR,
)
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
import yaml
from flask_cors import CORS, cross_origin


app = Flask(__name__)
app.debug = True
CORS(app)  # 允许所有跨域请求

ATTACK_ALIASES = {
    'FGSM': 'FGSM',
    'PGD': 'PGD',
    'BIM': 'BIM',
    'MI_FGSM': 'MI_FGSM',
    'MIFGSM': 'MI_FGSM',
    'NI_FGSM': 'NI_FGSM',
    'NIFGSM': 'NI_FGSM',
    'DEEPFOOL': 'DeepFool',
    'CW': 'CW',
    'NAM': 'NAM',
    'MPDSM': 'MPDSM',
    'TSMIFGSM': 'TSMIFGSM',
    'PIM_FGSM': 'PIM_FGSM',
    'PIM_PGD': 'PIM_PGD',
    'PIM_NAM': 'PIM_NAM',
    'JSMA': 'JSMA',
    'JAMMING': 'Jamming',
}

MODEL_ALIASES = {
    'RNN': 'Based_LSTM',
    'CNN': 'VTCNN2',
    'LSTM': 'Based_LSTM',
    'BASED_LSTM': 'Based_LSTM',
    'VTCNN2': 'VTCNN2',
}


def normalize_attack_name(name):
    if not name:
        return 'FGSM'
    key = str(name).upper().replace('-', '_')
    return ATTACK_ALIASES.get(key, name)


def normalize_model_name(name):
    if not name:
        return 'VTCNN2'
    return MODEL_ALIASES.get(str(name).upper(), name)


def _limit_dataset(dataset, max_samples):
    if max_samples is None or max_samples == '':
        return dataset
    max_samples = int(max_samples)
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    return Subset(dataset, list(range(max_samples)))


def _signal_dataset(dataset_name, max_samples=0):
    if dataset_name:
        raw_set = data_loader.RmlPklSignalSet(dataset_name)
    else:
        config = Config()
        raw_set = data_loader.Rml2016_10aAttackSet(**build_dataset_kwargs(getattr(config, 'Rml2016_10a'), dataset_name))
    return _limit_dataset(raw_set, max_samples), raw_set.get_snr_and_mod()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# 攻击核心逻辑，独立为函数
def run_attack(config_file, attack_name=None, model_name="VTCNN2", dataset_name=None, max_samples=0):
    setup_seed(2020)

    config = Config()
    attack_name = normalize_attack_name(attack_name)
    model_name = normalize_model_name(model_name)

    if config_file:
        try:
            config_file = resolve_config_file(config_file)
        except FileNotFoundError:
            print("Config file not found, fallback to defaults:", config_file)
            config_file = None
    if config_file:
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
    config.Switch_Method['method'] = 'White_Attack'

    Log = config.log_output()

    # 加载与前端同一份 pkl 的全部样本，保证信号点数等于数据集条数
    attack_set, (snrs, mods) = _signal_dataset(dataset_name, max_samples)

    # 加载数据加载器
    attack_loader = DataLoader(attack_set, batch_size=min(64, len(attack_set)), shuffle=False, num_workers=dataloader_workers())

    # 加载模型
    model_name = config.CONFIG['model_name']
    model_kwargs = dict(getattr(config, model_name))
    if 'filepath' in model_kwargs:
        model_kwargs['filepath'] = resolve_model_path(model_kwargs['filepath'], model_name)
    model = getattr(model_loader, 'load' + model_name)(**model_kwargs)
    model.eval()

    # 加载损失函数
    criterion_name = config.CONFIG['criterion_name']
    criterion = getattr(nn, criterion_name)()

    # 加载metric
    metrics = [getattr(module_metric, metric) for metric in config.CONFIG['metrics']]

    # 加载攻击方式
    attack_name = config.CONFIG['attack_name']
    attack_parameter = getattr(config, attack_name)
    attack_method = getattr(adv_method, attack_name)(model, criterion, **config.GPU, **attack_parameter)

    # 加载攻击训练器
    attacker = Attacker(model, metrics, criterion, config, attack_method, snrs, mods)

    # 开始攻击
    log, real_sample, adv_sample, acc, predicts, targets, all_change_ratios, x_snrs, mean_change_ratio = attacker.start_attack(attack_loader)
    Log.update(log)


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
    # 始终攻击所选数据集的全部样本，条数必须和 pkl 一致
    try:
        mods, predicts, adv_sample, all_change_ratios, acc, mean_change_ratio = run_attack(
            config_file, attack_name, model_name, dataset_name, 0
        )
        data = []
        for i in range(len(predicts)):
            entry = {
                "type": mods[predicts[i].item()],  # 将 (84, 1) 的张量转为标量
                "iq_data": adv_sample[i].squeeze(0).tolist(),  # 将 (1, 2, 128) 的张量去掉第一个维度，并转为列表
                "change_ratio": all_change_ratios[i].item()  # 将 (84,) 的标量转为 Python 数值
            }
            data.append(entry)
        success_rate = 1 - float(acc)
        total_count = len(predicts)
        success_count = int(round(success_rate * total_count))
        info = {
            "successInfo": success_rate,
            "successCount": success_count,
            "totalCount": total_count,
            "successText": "攻击成功率{:.4f}".format(success_rate),
            "transformInfo": float(mean_change_ratio),
        }
        # 转换为 JSON 格式

        return jsonify({"code": "00000", "attackedRml2016Data": data, "info": info})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"code": "10000", "message": str(e)}), 500

@app.route('/load_data', methods=['GET'])
def load_data():
    # 从 GET 请求中获取文件名参数
    file_name = request.args.get('file')
    
    try:
        resolved = resolve_data_file(file_name)
    except FileNotFoundError:
        return jsonify({"error": "File not found", "file": file_name}), 404

    try:
        records, _mods, _snrs, _resolved = data_loader.load_rml_pkl_records(file_name)
        total = len(records)
        include_iq = str(request.args.get('include_iq', '1')).lower() not in ('0', 'false', 'no')
        response_data = []
        for item in records:
            entry = {
                'type': item['type'],
                'snr': int(item['snr']) if not isinstance(item['snr'], str) else item['snr'],
            }
            if include_iq:
                entry['iq_data'] = np.asarray(item['iq_data']).tolist()
            response_data.append(entry)

        return jsonify({
            "code": "00000",  # 成功响应的状态码
            "data": response_data,
            "total": total,
            "returned": len(response_data),
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/getAll2016FileNames', methods=['GET'])
def get_all_2016_file_names():
    names = []
    for folder in (PROJECT_DATASET_DIR, LOCAL_DATASET_DIR):
        if not os.path.isdir(folder):
            continue
        for filename in sorted(os.listdir(folder)):
            if filename.lower().endswith('.pkl'):
                names.append(filename)
    return jsonify({"code": "00000", "data": names})


@app.route('/dataset_counts', methods=['GET'])
def dataset_counts():
    counts = {}
    file_name = request.args.get('file')
    names = [file_name] if file_name else []
    if not names:
        for folder in (PROJECT_DATASET_DIR, LOCAL_DATASET_DIR):
            if not os.path.isdir(folder):
                continue
            for filename in sorted(os.listdir(folder)):
                if filename.lower().endswith('.pkl') and '10a_dict' not in filename.lower():
                    names.append(filename)
    for name in names:
        try:
            records, _mods, _snrs, _resolved = data_loader.load_rml_pkl_records(name)
            counts[os.path.basename(name)] = len(records)
        except Exception:
            continue
    return jsonify({"code": "00000", "data": counts})


@app.route('/sample_iq', methods=['GET'])
def sample_iq():
    file_name = request.args.get('file')
    try:
        index = int(request.args.get('index', 0))
        records, _mods, _snrs, _resolved = data_loader.load_rml_pkl_records(file_name)
        if index < 0 or index >= len(records):
            return jsonify({"code": "10000", "message": "index out of range", "total": len(records)}), 400
        item = records[index]
        return jsonify({
            "code": "00000",
            "index": index,
            "total": len(records),
            "type": item['type'],
            "snr": int(item['snr']) if not isinstance(item['snr'], str) else item['snr'],
            "iq_data": np.asarray(item['iq_data']).tolist(),
        })
    except FileNotFoundError:
        return jsonify({"code": "10000", "message": "File not found", "file": file_name}), 404
    except Exception as e:
        return jsonify({"code": "10000", "message": str(e)}), 500


@app.route('/getAllAttackMethods', methods=['GET'])
def get_all_attack_methods():
    methods = [
        'FGSM', 'PGD', 'BIM', 'MI_FGSM', 'NAM',
        'MPDSM', 'TSMIFGSM', 'PIM_FGSM', 'PIM_PGD', 'CW'
    ]
    return jsonify({"code": "00000", "data": methods})


@app.route('/getAllDetectMethods', methods=['GET'])
def get_all_detect_methods():
    methods = [
        'FGSM', 'PGD', 'BIM', 'MI_FGSM', 'NAM',
        'MPDSM', 'TSMIFGSM', 'PIM_FGSM', 'PIM_PGD', 'CW'
    ]
    return jsonify({"code": "00000", "data": methods})


def _load_detect_teacher(device):
    config = Config()
    teacher = model_loader.loadVTCNN2(**{
        'filepath': resolve_model_path(getattr(config, 'VTCNN2').get('filepath'), 'VTCNN2')
    })
    return teacher.to(device).eval()


def _build_detect_attacker(model, criterion, config, attack_name):
    attack_name = normalize_attack_name(attack_name)
    params = dict(getattr(config, attack_name, {}) or {})
    attacker = getattr(adv_method, attack_name)(model, criterion, **config.GPU, **params)
    return attacker, attack_name


def run_detect(dataset_name, model_name=None, attack_name='FGSM', max_samples=0):
    setup_seed(2020)
    model_name = normalize_model_name(model_name)
    config = Config()
    dataset, (_snrs, _mods) = _signal_dataset(dataset_name, max_samples)
    loader = DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=False, num_workers=dataloader_workers())

    model = getattr(model_loader, 'load' + model_name)(**{
        'filepath': resolve_model_path(getattr(config, model_name).get('filepath'), model_name)
    })
    model.eval()
    criterion = nn.CrossEntropyLoss()
    attacker, attack_name = _build_detect_attacker(model, criterion, config, attack_name)
    device = next(model.parameters()).device
    teacher = _load_detect_teacher(device)

    clean_list, adv_list, pred_clean_list, pred_adv_list = [], [], [], []
    for x, y, _snr in loader:
        x = x.to(device).float()
        y = y.to(device).long()
        x_adv, _, _, _ = attacker.attack(x, y)
        x_adv = torch.tensor(x_adv, device=device, dtype=torch.float32)
        with torch.no_grad():
            pred_clean_list.append((teacher(x).argmax(1) != y).long().cpu())
            pred_adv_list.append((teacher(x_adv).argmax(1) != y).long().cpu())
        clean_list.append(x.detach().cpu())
        adv_list.append(x_adv.detach().cpu())

    clean = torch.cat(clean_list, dim=0)
    adv = torch.cat(adv_list, dim=0)
    pred_clean = torch.cat(pred_clean_list)
    pred_adv = torch.cat(pred_adv_list)
    actual_clean = torch.zeros_like(pred_clean)
    actual_adv = torch.ones_like(pred_adv)
    correct = (pred_clean == actual_clean).sum() + (pred_adv == actual_adv).sum()
    accuracy = float(correct) / float(len(pred_clean) + len(pred_adv))

    response_data = []
    for i in range(clean.shape[0]):
        iq = clean[i].detach().cpu().numpy()
        if iq.ndim == 2:
            iq = np.expand_dims(iq, 0)
        response_data.append({
            'predict_type': int(pred_clean[i]),
            'actual_type': 0,
            'iq_data': iq.tolist(),
        })
    for i in range(adv.shape[0]):
        iq = adv[i].detach().cpu().numpy()
        if iq.ndim == 2:
            iq = np.expand_dims(iq, 0)
        response_data.append({
            'predict_type': int(pred_adv[i]),
            'actual_type': 1,
            'iq_data': iq.tolist(),
        })
    return accuracy, response_data, attack_name


@app.route('/detect', methods=['GET'])
def detect_adversarial_samples():
    dataset_name = request.args.get('selectedSignals') or request.args.get('dataset_name')
    model_name = request.args.get('selectedModel') or request.args.get('model_name')
    detect_name = request.args.get('detect_name') or request.args.get('selectedMethod') or 'FGSM'
    if not dataset_name:
        return jsonify({'code': '10000', 'message': 'Missing selectedSignals'}), 400
    try:
        accuracy, response_data, method_name = run_detect(dataset_name, model_name, detect_name, 0)
        dataset_count = len(response_data) // 2 if len(response_data) >= 2 else len(response_data)
        return jsonify({
            'code': '00000',
            'accuracy': accuracy,
            'data': response_data,
            'dataset_count': dataset_count,
            'method': method_name,
            'model': model_name,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'code': '10000', 'message': str(e)}), 500


@app.route('/vulnerability', methods=['GET'])
def vulnerability():
    try:
        mods, predicts, adv_sample, all_change_ratios, acc, mean_change_ratio = run_attack(
            request.args.get('config'),
            request.args.get('attack_name'),
            request.args.get('model_name'),
            request.args.get('dataset_name') or request.args.get('signal_name'),
            request.args.get('max_samples', 32),
        )
        data = []
        for i in range(len(predicts)):
            data.append({
                "type": mods[predicts[i].item()],
                "iq_data": adv_sample[i].squeeze(0).tolist(),
                "change_ratio": all_change_ratios[i].item(),
            })
        success_rate = 1 - float(acc)
        return jsonify({
            "code": "00000",
            "attackedRml2016Data": data,
            "info": {
                "successInfo": success_rate,
                "transformInfo": float(mean_change_ratio),
            }
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"code": "10000", "message": str(e)}), 500


def _demo_graph(name):
    size = 18 if 'medium' in str(name) else 10
    nodes = [{'id': str(i)} for i in range(size)]
    edges = []
    for i in range(size):
        edges.append({'source': str(i), 'target': str((i + 1) % size)})
        if i % 3 == 0:
            edges.append({'source': str(i), 'target': str((i + 3) % size)})
    return {'nodes': nodes, 'edges': edges}


@app.route('/getAllFileNames', methods=['GET'])
def get_all_file_names():
    return jsonify({"code": "00000", "data": ['demo_net_small', 'demo_net_medium']})


@app.route('/graph', methods=['GET'])
def graph():
    file_name = request.args.get('fileName', 'demo_net_small')
    return jsonify({"code": "00000", "data": _demo_graph(file_name)})


@app.route('/getSolution', methods=['POST'])
def get_solution():
    payload = request.get_json(silent=True) or {}
    names = payload.get('data_test_name') or ['demo_net_small']
    name = names[0] if isinstance(names, list) else names
    graph_data = _demo_graph(name)
    degrees = {node['id']: 0 for node in graph_data['nodes']}
    for edge in graph_data['edges']:
        degrees[edge['source']] = degrees.get(edge['source'], 0) + 1
        degrees[edge['target']] = degrees.get(edge['target'], 0) + 1
    ranked = sorted(degrees.items(), key=lambda item: item[1], reverse=True)
    solution = [int(node_id) for node_id, _ in ranked[: max(3, len(ranked) // 5)]]
    return jsonify({"code": "00000", "data": {"solution": solution}})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
