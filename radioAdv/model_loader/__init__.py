from model_loader.vtcnn2 import *
from model_loader.based_vgg import *
from model_loader.based_resnet import *
from model_loader.based_gru import *
from model_loader.based_lstm import *
from model_loader.cldnn import *

from model_loader.vtcnn2_18 import *
from model_loader.based_gru_18 import *

from local_paths import resolve_model_path

def _wrap_loader(loader, model_name):
    def wrapped(filepath, **kwargs):
        return loader(resolve_model_path(filepath, model_name), **kwargs)
    wrapped.__name__ = loader.__name__
    return wrapped

loadVTCNN2 = _wrap_loader(loadVTCNN2, 'VTCNN2')
loadBased_LSTM = _wrap_loader(loadBased_LSTM, 'Based_LSTM')
loadBased_GRU = _wrap_loader(loadBased_GRU, 'Based_GRU')
loadBased_VGG = _wrap_loader(loadBased_VGG, 'Based_VGG')
loadBased_ResNet = _wrap_loader(loadBased_ResNet, 'Based_ResNet')
loadCLDNN = _wrap_loader(loadCLDNN, 'CLDNN')
loadVTCNN2_18 = _wrap_loader(loadVTCNN2_18, 'VTCNN2_18')
loadBased_GRU_18 = _wrap_loader(loadBased_GRU_18, 'Based_GRU_18')