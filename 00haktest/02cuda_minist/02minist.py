import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 1. 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# 2. 데이터 불러오기 (배치 사이즈를 32로 낮게 설정하여 메모리 관리)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

# 3. 간단한 모델 정의
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).to(device)

# 4. 손실함수 및 최적화 도구
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 한 번의 학습 루프 실행 테스트
model.train()
for i, (images, labels) in enumerate(train_loader):
    images, labels = images.to(device), labels.to(device)
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    if i % 100 == 0:
        print(f"Step {i}, Loss: {loss.item():.4f}")
    if i == 1000: break # 테스트용으로 일찍 종료

print("GTX 960에서 학습 테스트 성공!")