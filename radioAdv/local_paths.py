"""Windows-local path and runtime helpers for radioAdv."""
import os
import sys

RADIOADV_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(RADIOADV_ROOT)

LOCAL_DATASET_DIR = os.path.join(RADIOADV_ROOT, 'data_loader', 'dataset', 'RML2016.10a')
PROJECT_DATASET_DIR = os.path.join(PROJECT_ROOT, '数据集')
LOCAL_MODEL_DIR = os.path.join(RADIOADV_ROOT, 'model')

LINUX_PATH_ALIASES = (
    '/mnt/home/lzh/project/radioAdv',
    '/home/yuzhen/wireless/RML2016.10a',
    '/home/yuzhen/wireless',
)

MODEL_FILE_ALIASES = {
    'VTCNN2': 'VTCNN2_Epoch190.pkl',
    'Based_LSTM': 'Based_LSTM_Epoch12.pkl',
    'Based_GRU': 'VTCNN2_Epoch190.pkl',
    'Based_VGG': 'VTCNN2_Epoch190.pkl',
    'Based_ResNet': 'VTCNN2_Epoch190.pkl',
    'CLDNN': 'VTCNN2_Epoch190.pkl',
    'VTCNN2_18': 'VTCNN2_Epoch190.pkl',
    'Based_GRU_18': 'VTCNN2_Epoch190.pkl',
}

CLASS_PREFIXES = {'Rml2016_10a', 'Rml2018_01a', 'Mnist', 'Cifar10'}


def dataloader_workers(preferred=4):
    if sys.platform.startswith('win'):
        return 0
    return preferred


def strip_pkl_suffix(name):
    if not name:
        return None
    name = os.path.basename(str(name).replace('\\', '/'))
    if name.lower().endswith('.pkl'):
        name = name[:-4]
    if name in CLASS_PREFIXES:
        return None
    return name


def _iter_dataset_dirs(dirname=None):
    seen = set()
    for folder in (dirname, LOCAL_DATASET_DIR, PROJECT_DATASET_DIR):
        if not folder:
            continue
        folder = os.path.abspath(folder)
        if folder in seen:
            continue
        seen.add(folder)
        if os.path.isdir(folder):
            yield folder


def resolve_existing_path(path):
    if not path:
        return path
    path = os.path.expandvars(str(path))
    if os.path.exists(path):
        return os.path.abspath(path)

    normalized = path.replace('\\', '/')
    for alias in LINUX_PATH_ALIASES:
        if normalized.startswith(alias):
            rest = normalized[len(alias):].lstrip('/')
            mapped = os.path.join(RADIOADV_ROOT, *rest.split('/')) if rest else RADIOADV_ROOT
            if os.path.exists(mapped):
                return os.path.abspath(mapped)
            if alias.endswith('RML2016.10a') and rest:
                mapped = os.path.join(LOCAL_DATASET_DIR, *rest.split('/'))
                if os.path.exists(mapped):
                    return os.path.abspath(mapped)

    basename = os.path.basename(normalized)
    for folder in (LOCAL_DATASET_DIR, PROJECT_DATASET_DIR, LOCAL_MODEL_DIR, RADIOADV_ROOT):
        candidate = os.path.join(folder, basename)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return path


def resolve_dataset_dir(dirname=None):
    mapped = resolve_existing_path(dirname) if dirname else None
    if mapped and os.path.isdir(mapped):
        return mapped
    if os.path.isdir(LOCAL_DATASET_DIR):
        return LOCAL_DATASET_DIR
    if os.path.isdir(PROJECT_DATASET_DIR):
        return PROJECT_DATASET_DIR
    raise FileNotFoundError(
        'Cannot find RML2016 dataset directory. Expected: {}'.format(LOCAL_DATASET_DIR)
    )


def resolve_data_file(file_name):
    if not file_name:
        raise FileNotFoundError('Empty dataset file path')

    candidates = [file_name]
    if not str(file_name).lower().endswith('.pkl'):
        candidates.append(str(file_name) + '.pkl')
    else:
        candidates.append(str(file_name)[:-4])

    for candidate in candidates:
        mapped = resolve_existing_path(candidate)
        if mapped and os.path.isfile(mapped):
            return mapped

    basename = os.path.basename(str(file_name).replace('\\', '/'))
    if not basename.lower().endswith('.pkl'):
        basename += '.pkl'
    for folder in _iter_dataset_dirs():
        candidate = os.path.join(folder, basename)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError('Dataset file not found: {}'.format(file_name))


def resolve_pkl_stem(dirname=None, dataset_name=None):
    stem = strip_pkl_suffix(dataset_name)
    search_dirs = list(_iter_dataset_dirs(resolve_dataset_dir(dirname) if dirname else None))

    if stem:
        for folder in search_dirs:
            if os.path.isfile(os.path.join(folder, stem + '.pkl')):
                return folder, stem

    for folder in search_dirs:
        if os.path.isfile(os.path.join(folder, 'RML2016.10a_dict.pkl')):
            return folder, 'RML2016.10a_dict'
        if os.path.isfile(os.path.join(folder, 'train_data.p')):
            return folder, 'RML2016.10a_dict'

    raise FileNotFoundError(
        'Cannot find a RML2016 pkl under {} (dataset_name={})'.format(dirname, dataset_name)
    )


def resolve_split_files(dirname=None, dataset_name=None):
    """Locate train/test/augment split files, or the source pkl used to generate them."""
    folder, stem = resolve_pkl_stem(dirname, dataset_name)
    prefixed = (
        os.path.join(folder, stem + '_train_data.p'),
        os.path.join(folder, stem + '_test_data.p'),
        os.path.join(folder, stem + '_augments.p'),
    )
    generic = (
        os.path.join(folder, 'train_data.p'),
        os.path.join(folder, 'test_data.p'),
        os.path.join(folder, 'augments.p'),
    )

    use_generic = stem in (None, 'RML2016.10a_dict', 'Rml2016_10a')
    if use_generic and all(os.path.isfile(p) for p in generic):
        return folder, stem, generic[0], generic[1], generic[2], False
    if all(os.path.isfile(p) for p in prefixed):
        return folder, stem, prefixed[0], prefixed[1], prefixed[2], False

    pkl_path = os.path.join(folder, stem + '.pkl')
    if os.path.isfile(pkl_path):
        return folder, stem, prefixed[0], prefixed[1], prefixed[2], True

    raise FileNotFoundError('No split files or source pkl for {} in {}'.format(stem, folder))


def resolve_config_file(config_file):
    if not config_file:
        return None
    mapped = resolve_existing_path(config_file)
    if mapped and os.path.isfile(mapped):
        return mapped

    basename = os.path.basename(str(config_file).replace('\\', '/'))
    search_roots = (
        os.path.join(RADIOADV_ROOT, 'parameters'),
        os.path.join(PROJECT_ROOT, 'yaml'),
        RADIOADV_ROOT,
    )
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        direct = os.path.join(root, basename)
        if os.path.isfile(direct):
            return direct
        for dirpath, _, filenames in os.walk(root):
            if basename in filenames:
                return os.path.join(dirpath, basename)
    raise FileNotFoundError('Config file not found: {}'.format(config_file))


def resolve_model_path(filepath, model_name=None):
    if filepath:
        mapped = resolve_existing_path(filepath)
        if mapped and os.path.isfile(mapped):
            return mapped
        basename = os.path.basename(str(filepath).replace('\\', '/'))
        candidate = os.path.join(LOCAL_MODEL_DIR, basename)
        if os.path.isfile(candidate):
            return candidate

    alias = MODEL_FILE_ALIASES.get(model_name)
    if alias:
        candidate = os.path.join(LOCAL_MODEL_DIR, alias)
        if os.path.isfile(candidate):
            return candidate

    default = os.path.join(LOCAL_MODEL_DIR, 'VTCNN2_Epoch190.pkl')
    if os.path.isfile(default):
        return default

    raise FileNotFoundError(
        'Cannot find model weights for {} (filepath={})'.format(model_name, filepath)
    )


def build_dataset_kwargs(cfg_dict, dataset_name=None):
    kwargs = dict(cfg_dict or {})
    kwargs['dirname'] = resolve_dataset_dir(kwargs.get('dirname'))
    extra = strip_pkl_suffix(dataset_name) if dataset_name else strip_pkl_suffix(kwargs.get('dataset_name'))
    kwargs['dataset_name'] = extra
    return kwargs


def merge_config_dict(orig, incoming):
    merged = dict(orig)
    merged.update(incoming)
    return merged
