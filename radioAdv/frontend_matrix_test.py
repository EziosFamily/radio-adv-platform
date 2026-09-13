"""Exercise every frontend dataset / attack / model combination."""
import os
import sys
import time
import json
import traceback

os.environ.setdefault('MPLBACKEND', 'Agg')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from local_paths import resolve_data_file
from app import run_attack, normalize_attack_name, normalize_model_name
from user.attacker import Attacker

Attacker.plot_snr_figure = lambda *args, **kwargs: None
Attacker.plot_confusion_matrix_figure = lambda *args, **kwargs: None


FRONTEND_DATASETS = [
    'radio1.pkl',
    'radio2.pkl',
    'radio3.pkl',
    'random_rml2016_subset_300.pkl',
    'random_rml2016_subset_500.pkl',
    'random_rml2016_subset_1000.pkl',
    'random_rml2016_subset_2000.pkl',
    'random_rml2016_subset_3000.pkl',
    'random_rml2016_subset_5000.pkl',
]

FRONTEND_ATTACKS = [
    'FGSM', 'PGD', 'BIM', 'MI_FGSM', 'NAM',
    'MPDSM', 'TSMIFGSM', 'PIM_FGSM', 'PIM_PGD', 'CW',
]

FRONTEND_MODELS = ['CNN', 'RNN']


def max_samples_for(attack):
    if attack == 'CW':
        return 1
    if attack in ('PIM_PGD', 'PIM_FGSM'):
        return 2
    return 4


def test_load(dataset):
    resolved = resolve_data_file(dataset)
    if not os.path.isfile(resolved):
        raise FileNotFoundError(resolved)
    return resolved


def test_attack(dataset, attack, model):
    samples = max_samples_for(attack)
    mods, predicts, adv_sample, all_change_ratios, acc, mean_change_ratio = run_attack(
        None, attack, normalize_model_name(model), dataset, samples
    )
    total = len(predicts)
    success_rate = 1.0 - float(acc)
    success_count = int(round(success_rate * total))
    return {
        'ok': True,
        'total': total,
        'success_count': success_count,
        'success_rate': round(success_rate, 4),
        'mean_change': round(float(mean_change_ratio), 6),
        'n_adv': int(adv_sample.shape[0]) if hasattr(adv_sample, 'shape') else len(adv_sample),
        'samples': samples,
    }


def main():
    results = []
    print('=== load_data ===')
    for dataset in FRONTEND_DATASETS:
        t0 = time.time()
        row = {'kind': 'load', 'dataset': dataset}
        try:
            path = test_load(dataset)
            row.update(ok=True, path=path, seconds=round(time.time() - t0, 3))
            print('OK  load {:<40} {}'.format(dataset, path))
        except Exception as e:
            row.update(ok=False, error=str(e), seconds=round(time.time() - t0, 3))
            print('FAIL load', dataset, e)
        results.append(row)

    print('\n=== attack matrix ===')
    for dataset in FRONTEND_DATASETS:
        for attack in FRONTEND_ATTACKS:
            for model in FRONTEND_MODELS:
                t0 = time.time()
                row = {
                    'kind': 'attack',
                    'dataset': dataset,
                    'attack': attack,
                    'model': model,
                    'backend_attack': normalize_attack_name(attack),
                    'backend_model': normalize_model_name(model),
                }
                try:
                    stats = test_attack(dataset, attack, model)
                    row.update(stats)
                    row['seconds'] = round(time.time() - t0, 3)
                    print('OK  {model:<3} {attack:<10} {dataset:<36} '
                          'succ={success_count}/{total} rate={success_rate:.4f} '
                          'chg={mean_change:.6f} {seconds:.1f}s'.format(**row))
                except Exception as e:
                    row.update(ok=False, error=str(e), seconds=round(time.time() - t0, 3))
                    row['trace'] = traceback.format_exc()
                    print('FAIL {model} {attack} {dataset}: {e}'.format(
                        model=model, attack=attack, dataset=dataset, e=e
                    ))
                results.append(row)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend_matrix_results.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    loads = [r for r in results if r['kind'] == 'load']
    attacks = [r for r in results if r['kind'] == 'attack']
    print('\n=== summary ===')
    print('load  {}/{}'.format(sum(1 for r in loads if r.get('ok')), len(loads)))
    print('attack {}/{}'.format(sum(1 for r in attacks if r.get('ok')), len(attacks)))
    fails = [r for r in results if not r.get('ok')]
    for r in fails:
        print('FAIL', r.get('kind'), r.get('dataset'), r.get('attack'), r.get('model'), r.get('error'))
    print('wrote', out)


if __name__ == '__main__':
    main()
