import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from transformers import BertTokenizer
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel
from sklearn.model_selection import train_test_split
from torchvision.models import ResNet50_Weights
import torch.nn.functional as F


# 数据路径
data_path = '/Users/hushaoyan/Documents/实验五数据/data'
train_file = '/Users/hushaoyan/Documents/实验五数据/train.txt'
test_file = '/Users/hushaoyan/Documents/实验五数据/test_without_label.txt'

# 图像预处理
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 文本预处理
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def process_text(text):
    return tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors='pt')


# 自定义数据集
class MultiModalDataset(Dataset):
    def __init__(self, data_path, txt_file, transform=None):
        self.data_path = data_path
        self.txt_file = txt_file
        self.transform = transform
        self.data = []

        # 使用 ISO-8859-1 编码读取文件，跳过第一行
        with open(txt_file, 'r', encoding='ISO-8859-1') as f:
            lines = f.readlines()[1:]  # 跳过第一行
            for line in lines:
                line = line.strip()
                if not line or len(line.split(','))!= 2:
                    continue
                guid, label = line.split(',')
                self.data.append((guid, label))  # 保持label为字符串类型

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        guid, label = self.data[idx]

        # 加载文本
        text_file = os.path.join(self.data_path, f'{guid}.txt')

        # 使用 ISO-8859-1 编码读取文本文件
        with open(text_file, 'r', encoding='ISO-8859-1') as f:
            text = f.read().strip()

        # 加载图像
        image_file = os.path.join(self.data_path, f'{guid}.jpg')
        image = Image.open(image_file).convert('RGB')

        # 图像预处理
        if self.transform:
            image = self.transform(image)

        # 文本预处理
        text_input = process_text(text)

        return text_input, image, label


def collate_fn(batch):
    text_inputs, image_inputs, labels = zip(*batch)

    # 先提取出所有的input_ids和attention_mask
    input_ids = [text['input_ids'].squeeze(0) for text in text_inputs]
    attention_mask = [text['attention_mask'].squeeze(0) for text in text_inputs]

    # 使用 pad_sequence 填充，使得所有文本的长度一致
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)

    # 将标签映射为数字
    label_map = {'positive': 0, 'neutral': 1, 'negative': 2, 'null': -1}  # 增加对 null 的映射
    labels = torch.tensor([label_map[label] for label in labels])

    # 确保图像输入的尺寸一致
    image_size = image_inputs[0].size(1)  # 假设所有图像大小相同
    for image in image_inputs:
        if image.size(1)!= image_size or image.size(2)!= image_size:
            # 对不同尺寸的图像进行填充或者裁剪
            print(f"Image with different size detected: {image.size()}")
            # 这里可以根据需要进行处理（填充、裁剪等）
            image = F.interpolate(image.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)
    
    # 将图像输入堆叠成一个批次
    image_inputs = torch.stack(image_inputs, dim=0)

    return {'input_ids': input_ids, 'attention_mask': attention_mask}, image_inputs, labels


class MultiModalModel(nn.Module):
    def __init__(self, text_embedding_dim=768, image_embedding_dim=512, num_classes=3):
        super(MultiModalModel, self).__init__()

        # 文本模型
        self.text_model = BertModel.from_pretrained('bert-base-uncased')

        # 图像模型
        self.image_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # 修改全连接层的输出维度为 image_embedding_dim (512)
        self.image_model.fc = nn.Linear(self.image_model.fc.in_features, image_embedding_dim)

        # 融合层
        self.fc1 = nn.Linear(text_embedding_dim + image_embedding_dim, 512)  # 修改这里：1280 -> 512
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, text_input, image_input, mode='both'):
        # 文本输入
        text_output = self.text_model(
            input_ids=text_input['input_ids'],
            attention_mask=text_input['attention_mask']
        ).pooler_output  # 获取文本的池化输出，这是一个 Tensor，不是元组

        # 图像输入
        image_output = self.image_model(image_input)

        # 根据模式选择不同的输入
        if mode == 'text':
            # 只使用文本，扩展维度以匹配 fc1 的输入维度
            combined = torch.cat((text_output, torch.zeros((text_output.size(0), self.fc1.in_features - text_output.size(1))).to(text_output.device)), dim=1)
        elif mode == 'image':
            # 只使用图像，扩展维度以匹配 fc1 的输入维度
            combined = torch.cat((torch.zeros((image_output.size(0), self.fc1.in_features - image_output.size(1))).to(image_output.device), image_output), dim=1)
        else:
            # 使用多模态
            combined = torch.cat((text_output, image_output), dim=1)  # 将text_output和image_output合并

        # 融合
        x = torch.relu(self.fc1(combined))  # 使用合并后的维度作为输入
        output = self.fc2(x)

        return output


# 加载数据集
train_dataset = MultiModalDataset(data_path, train_file, transform=image_transform)

# 划分验证集
train_indices, val_indices = train_test_split(range(len(train_dataset)), test_size=0.1, random_state=42)
train_subset = torch.utils.data.Subset(train_dataset, train_indices)
val_subset = torch.utils.data.Subset(train_dataset, val_indices)

# 在创建 DataLoader 时，使用这个自定义的 collate_fn
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiModalModel().to(device)

# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)  # 使用AdamW优化器

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

# 训练模型
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for text_input, image_input, labels in train_loader:
        optimizer.zero_grad()

        # 前向传播
        text_input = {key: value.to(device) for key, value in text_input.items()}  # 不使用 squeeze
        image_input = image_input.to(device)
        labels = labels.clone().detach().to(device)

        outputs = model(text_input, image_input)

        # 计算损失
        loss = criterion(outputs, labels)
        total_loss += loss.item()

        # 反向传播
        loss.backward()
        optimizer.step()

    scheduler.step(total_loss)  # 更新学习率
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader)}")


# 评估函数
def evaluate(model, val_loader, mode='both'):
    model.eval()
    val_loss = 0
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for text_input, image_input, labels in val_loader:
            text_input = {key: value.to(device) for key, value in text_input.items()}
            image_input = image_input.to(device)
            labels = labels.clone().detach().to(device)

            # 选择使用的模式
            outputs = model(text_input, image_input, mode=mode)

            # 计算损失
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

    accuracy = correct_predictions / total_predictions
    return val_loss / len(val_loader), accuracy


# 评估多模态模型
val_loss, val_accuracy = evaluate(model, val_loader, mode='both')
print(f"Multi-modal model - Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# 评估仅文本模型
val_loss, val_accuracy = evaluate(model, val_loader, mode='text')
print(f"Text-only model - Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")

# 评估仅图像模型
val_loss, val_accuracy = evaluate(model, val_loader, mode='image')
print(f"Image-only model - Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}")


def predict(model, test_loader):
    model.eval()
    predictions = []
    with torch.no_grad():
        for text_input, image_input, _ in test_loader:
            text_input = {key: value.to(device) for key, value in text_input.items()}
            image_input = image_input.to(device)

            outputs = model(text_input, image_input)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    return predictions


# 加载测试集
test_dataset = MultiModalDataset(data_path, test_file, transform=image_transform)
# 在创建 test_loader 时，使用自定义的 collate_fn
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)  

# 预测结果
predictions = predict(model, test_loader)

# 直接修改test_without_label.txt文件的null标签
with open(test_file, 'r', encoding='ISO-8859-1') as f:
    lines = f.readlines()

with open('/Users/hushaoyan/Documents/当代人工智能实验五/test_predictions.txt', 'w', encoding='ISO-8859-1') as f:
    f.write(lines[0])  # 保留第一行（表头）
    for idx, pred in enumerate(predictions):
        # 将null标签替换为预测的标签
        guid = lines[idx+1].split(',')[0]
        f.write(f"{guid},{['positive', 'neutral', 'negative'][pred]}\n")