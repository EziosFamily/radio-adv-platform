import torch
import torch.nn as nn
import torch.optim as optim
from art.estimators.classification import PyTorchClassifier
from data_loader import *
from model_loader import *
from torch.utils.data import DataLoader
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, DeepFool, CarliniL2Method
from art.defences.detector.evasion import BinaryInputDetector
from art.estimators.classification import PyTorchClassifier

# 定义一个简单的 k-WTA CNN 二分类探测器模型
class KWTACNNDetectorModel(nn.Module):
    def __init__(self, input_channels=1, num_classes=2, k=5):
        super(KWTACNNDetectorModel, self).__init__()
        self.k = k  # 每个卷积层保留的最大激活数
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        
        # 全连接层
        self.fc1 = nn.Linear(32768, 128)  # 假设输入为 (batch_size, 1, 2, 128)
        self.fc2 = nn.Linear(128, num_classes)  # 输出2类（正常样本和对抗样本）

    def forward(self, x):
        # x: (batch_size, 1, 2, 128)
        # print(f"Input shape: {x.shape}")
        
        # 使用卷积层、k-WTA 和池化层
        x = self.pool(self.k_wta(self.conv1(x)))  # (batch_size, 32, 1, 64)
        # print(f"Shape after conv1 and k-WTA: {x.shape}")
        
        x = self.pool(self.k_wta(self.conv2(x)))  # (batch_size, 64, 1, 32)
        # print(f"Shape after conv2 and k-WTA: {x.shape}")
        
        x = self.pool(self.k_wta(self.conv3(x)))  # (batch_size, 128, 1, 16)
        # print(f"Shape after conv3 and k-WTA: {x.shape}")
        
        # 展平特征
        x = x.view(x.size(0), -1)  # (batch_size, 128 * 16 * 2)
        # print(f"Shape after flatten: {x.shape}")
        
        # 全连接层
        x = F.relu(self.fc1(x))  # (batch_size, 128)
        x = self.fc2(x)  # (batch_size, num_classes)
        
        return x
    
    def k_wta(self, x):
        """应用 k-WTA 机制，只保留前 k 个激活值，其余置零"""
        # x: (batch_size, num_channels, height, width)
        with torch.no_grad():
            # 计算每个通道的激活值阈值，第 k 大的值
            threshold = x.view(x.size(0), x.size(1), -1).topk(self.k, dim=2).values[:, :, -1]
            threshold = threshold.view(x.size(0), x.size(1), 1, 1)  # 调整维度以便比较

        # 保留大于阈值的激活值，小于阈值的置零
        x = torch.where(x >= threshold, x, torch.zeros_like(x))
        return x

# 定义一个简单的二分类探测器模型
class LSTMDetectorModel(nn.Module):
    def __init__(self, input_size=2, hidden_size=128, num_layers=2, num_classes=2):
        super(LSTMDetectorModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)  # 输出2类（正常样本和对抗样本）

    def forward(self, x):
        # x: (batch_size, 1, 2, 128)
        print(f"Input shape before squeeze: {x.shape}")
        
        # 移除多余的维度 1，变成 (batch_size, 2, 128)
        x = x.squeeze(1)  # 变为 (batch_size, 2, 128)
        print(f"Input shape after squeeze: {x.shape}")
        
        # 交换维度，变成 (batch_size, 128, 2) 以匹配 LSTM 输入
        x = x.permute(0, 2, 1)  # 变为 (batch_size, 128, 2)
        print(f"Input shape after permute: {x.shape}")

        # 初始化 LSTM 隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 初始化为零
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播 LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, sequence_length, hidden_size)
        
        # 取最后一个时间步的输出 (batch_size, hidden_size)
        out = out[:, -1, :]
        
        # 全连接层，输出 (batch_size, num_classes)
        out = self.fc(out)
        
        return out
    


def load_rml_datasets(dataset_dir, prop, batch_size, dataset_name):
    """
    加载 RML2016.10a 数据集并返回 DataLoader 对象。

    Args:
        dataset_dir (str): 数据集路径。
        prop (float): 训练集所占的比例。
        batch_size (int): DataLoader 的批量大小。

    Returns:
        train_loader: PyTorch DataLoader 对象，用于加载训练数据。
        test_loader: PyTorch DataLoader 对象，用于加载测试数据。
        snrs: 数据中的 SNR 值。
        mods: 数据中的调制类型。
    """
    # 加载训练集和测试集
    if prop != 0:
        train_dataset = Rml2016_10aTrainSet(dirname=dataset_dir, prop=prop, dataset_name=dataset_name)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = Rml2016_10aTestSet(dirname=dataset_dir, prop=1- prop, dataset_name=dataset_name)

    # print(f"Total training samples: {len(train_dataset)}")
    print(f"Total testing samples: {len(test_dataset)}")


    # 使用 DataLoader 进行数据加载
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    # 获取 SNR 和调制类型"""  """
    
    if prop != 0:
        snrs, mods = train_dataset.get_snr_and_mod()
        return train_loader, test_loader, snrs, mods
    else:
        snrs, mods = test_dataset.get_snr_and_mod()
        return None, test_loader, snrs, mods



def generate_adversarial_samples(classifier, x_train, y_train, attacks=['FGSM']):
    """
    根据不同的攻击方法生成对抗样本。

    Args:
        classifier (PyTorchClassifier): ART 封装的 PyTorch 分类器。
        x_train (torch.Tensor): 原始训练数据。
        attacks (list): 要应用的攻击方法名称。

    Returns:
        adv_samples (dict): 包含不同攻击方法生成的对抗样本的字典。
    """
    adv_samples = {}
    
    # 将 PyTorch Tensor 转换为 NumPy 数组
    x_train_np = x_train.cpu().numpy()
    
    
    if 'FGSM' in attacks:
        fgsm_attack = FastGradientMethod(estimator=classifier, eps=0.0018)
        adv_samples['FGSM'] = fgsm_attack.generate(x=x_train_np)
        original_preds = classifier.predict(x_train.cpu().numpy())
        adversarial_preds = classifier.predict(adv_samples['PGD'])  # 或其他攻击方法生成的样本
        # 将 logits 转换为预测标签（类别）
        original_preds_labels = np.argmax(original_preds, axis=1)  # 每行取最大值的索引
        adversarial_preds_fgsm_labels = np.argmax(adversarial_preds, axis=1)
        successful_pred = np.sum(y_train.cpu().numpy() == original_preds_labels)
        successful_pred_rate = successful_pred / len(original_preds_labels)
         # 计算成功干扰的比例
        successful_attacks = np.sum(original_preds_labels != adversarial_preds_fgsm_labels)
        attack_success_rate = successful_attacks / len(adversarial_preds_fgsm_labels)

        # 打印输出
        # print(f"Original true labels: {y_train.cpu().numpy()}")
        # print(f"Original predictions labels: {original_preds_labels}")
        # print(f"predictions rate: {successful_pred_rate * 100:.2f}%")
        # print(f"Adversarial FGSM predictions labels: {adversarial_preds_fgsm_labels}")
        # print(f"Number of successful attacks: {successful_attacks}")
        # print(f"FGSM attack success rate: {attack_success_rate * 100:.2f}%")

        
        
        
    
    if 'PGD' in attacks:
        pgd_attack = ProjectedGradientDescent(estimator=classifier, eps=0.002, eps_step=0.005, max_iter=8)
        adv_samples['PGD'] = pgd_attack.generate(x=x_train_np)
        original_preds = classifier.predict(x_train.cpu().numpy())
        adversarial_preds = classifier.predict(adv_samples['PGD'])  # 或其他攻击方法生成的样本
        # 将 logits 转换为预测标签（类别）
        original_preds_labels = np.argmax(original_preds, axis=1)  # 每行取最大值的索引
        adversarial_preds_fgsm_labels = np.argmax(adversarial_preds, axis=1)
        successful_pred = np.sum(y_train.cpu().numpy() == original_preds_labels)
        successful_pred_rate = successful_pred / len(original_preds_labels)
         # 计算成功干扰的比例
        successful_attacks = np.sum(original_preds_labels != adversarial_preds_fgsm_labels)
        attack_success_rate = successful_attacks / len(adversarial_preds_fgsm_labels)

        # 打印输出
        # print(f"Original true labels: {y_train.cpu().numpy()}")
        # print(f"Original predictions labels: {original_preds_labels}")
        # print(f"predictions rate: {successful_pred_rate * 100:.2f}%")
        # print(f"Adversarial PGD predictions labels: {adversarial_preds_fgsm_labels}")
        # print(f"Number of successful attacks: {successful_attacks}")
        # print(f"PGD attack success rate: {attack_success_rate * 100:.2f}%")
            
        
    
    if 'DeepFool' in attacks:
        deepfool_attack = DeepFool(classifier=classifier)
        adv_samples['DeepFool'] = deepfool_attack.generate(x=x_train_np)
        original_preds = classifier.predict(x_train.cpu().numpy())
        adversarial_preds = classifier.predict(adv_samples['DeepFool'])  # 或其他攻击方法生成的样本
        # 将 logits 转换为预测标签（类别）
        original_preds_labels = np.argmax(original_preds, axis=1)  # 每行取最大值的索引
        adversarial_preds_fgsm_labels = np.argmax(adversarial_preds, axis=1)

        successful_pred = np.sum(y_train.cpu().numpy() == original_preds_labels)
        successful_pred_rate = successful_pred / len(original_preds_labels)
         # 计算成功干扰的比例
        successful_attacks = np.sum(original_preds_labels != adversarial_preds_fgsm_labels)
        attack_success_rate = successful_attacks / len(adversarial_preds_fgsm_labels)

        # 打印输出
        # print(f"Original true labels: {y_train.cpu().numpy()}")
        # print(f"Original predictions labels: {original_preds_labels}")
        # print(f"predictions rate: {successful_pred_rate * 100:.2f}%")
        # print(f"Adversarial DeepFool predictions labels: {adversarial_preds_fgsm_labels}")
        # print(f"Number of successful attacks: {successful_attacks}")
        # print(f"DeepFool attack success rate: {attack_success_rate * 100:.2f}%")
        
        
    
    if 'CW' in attacks:
        cw_attack = CarliniL2Method(classifier=classifier, confidence=0.1)
        adv_samples['CW'] = cw_attack.generate(x=x_train_np)
        original_preds = classifier.predict(x_train.cpu().numpy())
        adversarial_preds = classifier.predict(adv_samples['CW'])  # 或其他攻击方法生成的样本
        # 将 logits 转换为预测标签（类别）
        original_preds_labels = np.argmax(original_preds, axis=1)  # 每行取最大值的索引
        adversarial_preds_fgsm_labels = np.argmax(adversarial_preds, axis=1)

        successful_pred = np.sum(y_train.cpu().numpy() == original_preds_labels)
        successful_pred_rate = successful_pred / len(original_preds_labels)
         # 计算成功干扰的比例
        successful_attacks = np.sum(original_preds_labels != adversarial_preds_fgsm_labels)
        attack_success_rate = successful_attacks / len(adversarial_preds_fgsm_labels)

        # 打印输出
        # print(f"Original true labels: {y_train.cpu().numpy()}")
        # print(f"Original predictions labels: {original_preds_labels}")
        # print(f"predictions rate: {successful_pred_rate * 100:.2f}%")
        # print(f"Adversarial CW predictions labels: {adversarial_preds_fgsm_labels}")
        # print(f"Number of successful attacks: {successful_attacks}")
        # print(f"CW attack success rate: {attack_success_rate * 100:.2f}%")
        
        
    
    
    return adv_samples

    

def train_detector(classifier, x_train, y_train, adv_samples):
    """
    训练 BinaryInputDetector 来检测对抗样本。

    Args:
        classifier (PyTorchClassifier): ART 封装的 PyTorch 分类器。
        x_train (torch.Tensor): 原始训练数据。
        y_train (torch.Tensor): 原始训练标签。
        adv_samples (dict): 包含不同攻击方法生成的对抗样本的字典。

    Returns:
        detector (BinaryInputDetector): 训练好的对抗样本检测器。
    """
    # 将不同的对抗样本和正常样本组合在一起
    print(x_train.device)

    x_train_combined = np.vstack([x_train.cpu().numpy()] + list(adv_samples.values()))
    
    # 创建标签：0 表示正常样本，1 表示对抗样本
    y_train_combined = np.hstack([np.zeros(len(x_train))] +
                                 [np.ones(len(adv_samples[attack])) for attack in adv_samples])
    
   
    # 前向传播，计算损失
    predictions = classifier.model(torch.tensor(x_train_combined, dtype=torch.float32).to(x_train.device))
    
    loss = classifier.loss(predictions, torch.tensor(y_train_combined, dtype=torch.long).to(x_train.device))
    # print(f"Training loss: {loss:.4f}")

    # 创建并训练 BinaryInputDetector，传入探测器模型
    
    classifier.fit(x_train_combined, y_train_combined)
    
    return classifier


def test_detector(detector, classifier, x_test, y_test):
    """
    测试检测器在测试数据上的性能，并打印每次的检测结果。

    Args:
        detector (BinaryInputDetector): 训练好的对抗样本检测器。
        classifier (PyTorchClassifier): ART 封装的 PyTorch 分类器。
        x_test (torch.Tensor): 测试数据。
        y_test (torch.Tensor): 测试标签。

    Returns:
        accuracy (float): 对抗样本检测的准确率。
    """
    # 将 x_test 转换为 NumPy 数组
    x_test_np = x_test.cpu().numpy()
    
    # 生成测试集上的对抗样本
    fgsm_attack = FastGradientMethod(estimator=classifier, eps=0.00136)
    x_test_adv_fgsm = fgsm_attack.generate(x=x_test_np)
    
    # 将测试数据和对抗样本组合
    x_test_combined = np.vstack([x_test_np, x_test_adv_fgsm])
    y_test_combined = np.hstack([np.zeros(len(x_test)), np.ones(len(x_test_adv_fgsm))])


    

    # 进行检测
    detections = detector.predict(x_test_combined)
    # loss = detector.loss(detections, y_test_combined)
    detections_bool = np.argmax(detections, axis=1)

    
    # 打印测试的 loss

    # 打印检测结果的类型和内容
    # print(f"Detections type: {type(detections)}")
    # print(f"Detections (tuple): {detections}")

    # 提取布尔检测结果 (True / False)，忽略 'predictions' 字典部分

    # 确保 detections 和 y_test_combined 是一维数组
    detections_bool = np.squeeze(detections_bool)
    y_test_combined = np.squeeze(y_test_combined)
    y_test_combined = y_test_combined.astype(bool)


    # 打印每个样本的检测结果和实际标签
    print("Detection Results (Predicted vs Actual):")
    for i in range(len(detections_bool)):
        print(f"Sample {i}: Predicted = {detections_bool[i]}, Actual = {y_test_combined[i]}")

    # 调试输出形状
    # print(f"detections_bool shape after squeeze: {detections_bool.shape}")
    # print(f"y_test_combined shape: {y_test_combined.shape}")

    # 计算检测器的准确率
    accuracy = np.mean(detections_bool == y_test_combined)
    print(f"Detection accuracy: {accuracy * 100:.2f}%")

    return accuracy, x_test_combined, detections_bool, y_test_combined





import torch

def run_adversarial_detection_pipeline(dataset_dir, prop, batch_size=1024, lr=0.001, epochs=100, save_interval=5, log_file="/mnt/home/lzh/project/radioAdv/log/detector_training_log.txt", pretrained_model_path=None, detector_model_path = None, dataset_name = None, model_type=None):
    """
    运行完整的对抗样本检测流程，并每5轮保存模型，记录准确率到日志文件。

    Args:
        dataset_dir (str): 数据集路径。
        prop (float): 训练集所占比例。
        batch_size (int): 批量大小。
        lr (float): 学习率。
        epochs (int): 训练的轮数。
        save_interval (int): 每隔多少轮保存一次模型。
        log_file (str): 用于保存日志的文件路径。

    Returns:
        detector (BinaryInputDetector): 训练好的检测器。
        accuracy (float): 检测器的检测准确率。
    """
    # 1. 加载数据集
    train_loader, test_loader, snrs, mods = load_rml_datasets(dataset_dir, prop, batch_size, dataset_name=dataset_name)
    
    # for x_train, y_train, snr_train in train_loader:
    #     print(f"Batch size: {x_train.size(0)}")  # 输出当前 batch 的大小
    
    if pretrained_model_path:
        # 加载预训练模型
        model = load_pretrained_model(pretrained_model_path)
        # print(f"Loaded pretrained model from {pretrained_model_path}")
    else:
        # 否则训练新模型
        model = VTCNN2(11)  # 定义新模型
        print("Training a new model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    
    # 3. 创建 ART PyTorch 分类器
    classifier = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=optimizer,
        input_shape=(2, 128, 128),
        nb_classes=11
    )
    if detector_model_path:
        if model_type == 'RNN':
            detector_model = load_RNN_detector_model(detector_model_path)
        else:
            detector_model = load_CNN_detector_model(detector_model_path)
    else:
        detector_model = KWTACNNDetectorModel(1)

    # 使用 ART PyTorchClassifier 封装探测器模型
    
    detector_classifier = PyTorchClassifier(
        model=detector_model,
        loss=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(detector_model.parameters(), lr=0.001),
        input_shape=(2 * 128 * 128,),  # 输入维度与模型的输入一致
        nb_classes=2  # 二分类问题
    )

    test_adv_samples = None  # 用于保存测试集生成的对抗样本
    test_predictions = None  # 用于保存测试集的识别结果
        
    # 打开日志文件
    with open(log_file, "w") as log:
        log.write("Epoch,Accuracy\n")  # 写入表头
        if prop == 0:
            x_test, y_test, snr_test = next(iter(test_loader))
            accuracy, test_adv_samples, predictions, actuals = test_detector(detector_classifier, classifier, x_test, y_test)
                
            return detector_classifier, accuracy,test_adv_samples, predictions, actuals

        # 4. 训练对抗样本检测器
        for epoch in range(epochs):
            # print(f"Epoch {epoch + 1}/{epochs}")

            # 训练一个 epoch
            for x_train, y_train, snr_train in train_loader:
                x_train, y_train = x_train.to(device), y_train.to(device)

                # 生成对抗样本
                adv_samples = generate_adversarial_samples(classifier, x_train, y_train, attacks=['FGSM', "PGD"])

                # 训练对抗样本检测器
                detector_classifier = train_detector(detector_classifier, x_train, y_train, adv_samples)

            # 每个 epoch 之后测试对抗样本检测器
            x_test, y_test, snr_test = next(iter(test_loader))
            accuracy = test_detector(detector_classifier, classifier, x_test, y_test)

            # 记录准确率到日志文件
            log.write(f"{epoch + 1},{accuracy:.4f}\n")

            if epoch == epochs - 1:
                test_adv_samples = x_test
                test_predictions = detector_classifier.predict(test_adv_samples)
                detections_bool = np.argmax(test_predictions, axis=1)


            # 每 5 轮保存一次模型
            if (epoch + 1) % save_interval == 0:
                model_path = f"/mnt/home/lzh/project/radioAdv/model/model_epoch_{epoch + 1}.pth"
                torch.save(detector_classifier.model.state_dict(), model_path)
                # print(f"Model saved at epoch {epoch + 1} to {model_path}")
                print(model_path)


            print(f"Epoch {epoch + 1}/{epochs} - Adversarial sample detection accuracy: {accuracy * 100:.2f}%")

    return detector_classifier, accuracy, test_adv_samples, detections_bool, actuals

def load_pretrained_model(model_path):
    model = VTCNN2(11)  # 定义模型结构
    checkpoint = torch.load(model_path, map_location='cpu')
    print(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()  # 设置模型为评估模式
    return model

def load_CNN_detector_model(model_path):
    model = KWTACNNDetectorModel(1)
    print(model_path)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()  # 设置模型为评估模式
    return model

def load_RNN_detector_model(model_path):
    model = Based_LSTM(2)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()  # 设置模型为评估模式
    return model


def main():
    dataset_dir = '/mnt/home/lzh/project/radioAdv/data_loader/dataset/RML2016.10a'
    prop = 0  # 定义训练集占总数据的比例
    batch_size = 1024
    lr = 0.001
    epochs = 10
    pretrained_model_path = '/mnt/home/lzh/project/radioAdv/model/VTCNN2/VTCNN2_Epoch190.pkl'
    model_type = 'CNN'
    if model_type == 'RNN':
        detector_model_path = '/mnt/home/lzh/project/radioAdv/model/model2_epoch_10.pth'
    else:
        detector_model_path = '/mnt/home/lzh/project/radioAdv/model/CNNmodel3_epoch_10.pth'
    dataset_name = 'radio1.pkl'

    detector, accuracy, test_adv_samples, test_predictions, actuals = run_adversarial_detection_pipeline(dataset_dir, prop, batch_size, lr, epochs, pretrained_model_path=pretrained_model_path, detector_model_path = detector_model_path, dataset_name = dataset_name, model_type=model_type)
    print(accuracy)
    print(test_adv_samples)
    print(test_predictions)
if __name__ == "__main__":
    main()
