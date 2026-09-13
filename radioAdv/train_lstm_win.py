"""Train a Based_LSTM classifier so the frontend RNN option can run attacks."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import data_loader
from local_paths import LOCAL_MODEL_DIR, build_dataset_kwargs, dataloader_workers
from model_loader.based_lstm import Based_LSTM


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = 'random_rml2016_subset_2000'
    epochs = 12
    batch_size = 32
    lr = 1e-3

    kwargs = build_dataset_kwargs(
        {'dirname': './data_loader/dataset/RML2016.10a', 'prop': 0.5},
        dataset_name,
    )
    train_set = data_loader.Rml2016_10aTrainSet(**kwargs)
    test_set = data_loader.Rml2016_10aTestSet(**kwargs)
    workers = dataloader_workers()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=workers)

    model = Based_LSTM(output_dim=11).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    print('device={} train={} test={}'.format(device, len(train_set), len(test_set)))
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        correct = 0
        total = 0
        for x, y, _snr in train_loader:
            x = x.to(device).float()
            y = y.to(device).long()
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * x.size(0)
            correct += int((logits.argmax(1) == y).sum().item())
            total += int(x.size(0))
        train_acc = correct / max(total, 1)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y, _snr in test_loader:
                x = x.to(device).float()
                y = y.to(device).long()
                pred = model(x).argmax(1)
                correct += int((pred == y).sum().item())
                total += int(x.size(0))
        test_acc = correct / max(total, 1)
        print('epoch {}/{} loss={:.4f} train_acc={:.4f} test_acc={:.4f}'.format(
            epoch, epochs, running / max(len(train_set), 1), train_acc, test_acc
        ))

    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    out_path = os.path.join(LOCAL_MODEL_DIR, 'Based_LSTM_Epoch12.pkl')
    torch.save({'state_dict': model.state_dict(), 'epoch': epochs, 'test_acc': test_acc}, out_path)
    print('saved', out_path, 'test_acc', test_acc)


if __name__ == '__main__':
    main()
