import torch
import torch.nn as nn
import torch.optim as optim
import torch_directml
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import DigitClassifier

# RX 570 장치 설정
device = torch_directml.device()

# 이미지 변환 설정 (이미지를 텐서로 바꾸고 정규화)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 데이터셋 다운로드 (최초 1회)
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

model = DigitClassifier().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("RX 570으로 손글씨 학습을 시작합니다...")

for epoch in range(3): # 3회 반복 학습
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1} 완료 - 평균 오차: {running_loss/len(train_loader):.4f}")

# 저장할 파일 이름
MODEL_PATH = "mnist_model.pth"

# 모델의 state_dict(가중치 데이터)를 저장
torch.save(model.state_dict(), MODEL_PATH)
print(f"모델 저장 완료: {MODEL_PATH}")
print("학습 완료!")