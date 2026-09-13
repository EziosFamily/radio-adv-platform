"""Quick Windows sanity check: dataset, weights, one attack step, one train step."""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

import adv_method
import data_loader
import model_loader
from local_paths import (
    build_dataset_kwargs,
    resolve_model_path,
    dataloader_workers,
    LOCAL_DATASET_DIR,
    LOCAL_MODEL_DIR,
)


def main():
    print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('gpu', torch.cuda.get_device_name(0))

    print('dataset dir', LOCAL_DATASET_DIR, 'exists', os.path.isdir(LOCAL_DATASET_DIR))
    print('model dir', LOCAL_MODEL_DIR, 'exists', os.path.isdir(LOCAL_MODEL_DIR))

    ds_kwargs = build_dataset_kwargs({'dirname': LOCAL_DATASET_DIR, 'prop': 0.5})
    attack_set = data_loader.Rml2016_10aAttackSet(**ds_kwargs)
    print('attack set size', len(attack_set), 'mods', attack_set.get_snr_and_mod()[1])

    subset = Subset(attack_set, list(range(min(8, len(attack_set)))))
    loader = DataLoader(subset, batch_size=8, shuffle=False, num_workers=dataloader_workers())
    x, y, snr = next(iter(loader))
    print('batch', tuple(x.shape), y[:4].tolist())

    model = model_loader.loadVTCNN2(resolve_model_path(None, 'VTCNN2'))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).eval()
    criterion = nn.CrossEntropyLoss()

    attacker = adv_method.FGSM(model, criterion, use_gpu=device.type == 'cuda', device_id=[0], eps=0.002)
    x = x.to(device).float()
    y = y.to(device).long()
    x_adv, pertub, logits, pred = attacker.attack(x, y)
    print('fgsm adv', x_adv.shape, 'pred', pred[:4], 'true', y[:4].cpu().numpy())

    model.train()
    logits_n = model(x)
    x_adv_t = torch.tensor(x_adv, device=device, dtype=torch.float32)
    logits_a = model(x_adv_t)
    loss = criterion(logits_n, y) + criterion(logits_a, y)
    loss.backward()
    print('train step loss', float(loss.detach().cpu()))
    print('SMOKE_OK')


if __name__ == '__main__':
    main()
