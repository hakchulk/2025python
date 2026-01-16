import torch
import torch_directml
from model import DigitClassifier # 동일하게 모델 구조를 가져옴
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch_directml.device()

# 1. 빈 모델 객체 생성 후 가중치 입히기
model = DigitClassifier().to(device)
model.load_state_dict(torch.load("mnist_model.pth.data", map_location=device))
model.eval()

# ... 이후 테스트 및 시각화 코드 실행 ...
print("모델 불러오기 완료!")

# 4. 테스트 데이터로 확인
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=True)

images, labels = next(iter(test_loader))
images, labels = images.to(device), labels.to(device)

outputs = model(images)
_, predicted = torch.max(outputs, 1)

# 결과 시각화
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i in range(5):
    img = images[i].cpu().squeeze()
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"AI: {predicted[i].item()}\n(Real: {labels[i].item()})")
    axes[i].axis('off')

plt.show()

correct = 0
total = 0

# 평가 모드에서는 기울기(gradient) 계산이 필요 없으므로 성능과 메모리를 위해 비활성화
with torch.no_grad():
    for images, labels in test_loader:
        # 데이터를 DirectML 장치로 이동
        images, labels = images.to(device), labels.to(device)
        
        # 모델 예측
        outputs = model(images)
        
        # 가장 높은 확률을 가진 인덱스가 예측한 숫자
        _, predicted = torch.max(outputs.data, 1)
        
        # 전체 개수와 맞힌 개수 업데이트
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"테스트 데이터셋 전체 크기: {total}개")
print(f"정확하게 맞힌 개수: {correct}개")
print(f"최종 정답률(Accuracy): {accuracy:.2f}%")