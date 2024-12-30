"""
@FileName: MBNN.py
@Time : 2024/10/31 10:35
@Author : ZhaoLei
"""
"""
@FileName: MBNN_with_custom_file_logger.py
@Time : 2024/10/24
@Author : ZhaoLei
"""
import torch
import torch.nn as nn
import torch.optim as optim
from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.models import DynamicModule
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics, cpu_usage_metrics
from avalanche.training.templates import SupervisedTemplate
from sklearn.metrics.pairwise import cosine_similarity
import torchvision
from avalanche.logging import TensorboardLogger, TextLogger, InteractiveLogger


# 1. 定义基本的特征提取网络
class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.features = nn.Sequential(*list(pretrained_model.children())[:-1])

    def forward(self, x):
        return self.features(x).view(x.size(0), -1)

# 2. 定义动态分类器模块
class DynamicClassifier(DynamicModule):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

# 3. 创建自定义的持续学习策略
class DynamicArchitectureStrategy:
    def __init__(self, model, criterion, optimizer, train_mb_size, train_epochs,
                 eval_mb_size, device, similarity_threshold=0.8):
        self.base_strategy = SupervisedTemplate(model, optimizer, criterion, train_mb_size, train_epochs, eval_mb_size,
                                                device)
        self.similarity_threshold = similarity_threshold
        self.classifiers = []
        self.feature_vectors = []
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def _compute_similarity(self, new_feature):
        similarities = [cosine_similarity(new_feature.view(1, -1), f.view(1, -1)).item() for f in self.feature_vectors]
        return similarities

    def train(self, experience):
        self.base_strategy.train(experience)

        for mbatch in self.base_strategy.dataloader:
            x, y = mbatch[0], mbatch[1]
            features = self.model.features(x.to(self.device))
            similarity = self._compute_similarity(features)

            if max(similarity, default=0) < self.similarity_threshold:
                new_classifier = DynamicClassifier(features.size(1), 100)
                self.classifiers.append(new_classifier.to(self.device))
                self.feature_vectors.append(features.mean(dim=0))
                logits = new_classifier(features)
            else:
                best_idx = similarity.index(max(similarity))
                classifier = self.classifiers[best_idx]
                logits = classifier(features)

            loss = self.criterion(logits, y.to(self.device))
            loss.backward()
            self.optimizer.step()

    def eval(self, test_stream):
        return self.base_strategy.eval(test_stream)

# 4. 数据集与实验设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
benchmark = SplitCIFAR100(n_experiences=10)
model = FeatureExtractor(torchvision.models.resnet18(pretrained=True)).to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 5. 日志记录
interactive_logger = InteractiveLogger()
loggers = [interactive_logger]

# 评估插件，用于记录和可视化多种指标
eval_plugin = EvaluationPlugin(
    accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
    forgetting_metrics(experience=True, stream=True),
    cpu_usage_metrics(minibatch=True, epoch=True, experience=True),
    loggers=[loggers]  # 将日志记录器添加到评估插件中
)

# 创建策略并包含评估插件
strategy = DynamicArchitectureStrategy(
    model, criterion, optimizer,
    train_mb_size=32, train_epochs=10, eval_mb_size=32,
    device=device, similarity_threshold=0.7
)

# 6. 开始训练和评估
for experience in benchmark.train_stream:
    print("开始训练 ", experience.current_experience)
    strategy.train(experience)
    print("结束训练", experience.current_experience)
    metrics = strategy.eval(benchmark.test_stream)


