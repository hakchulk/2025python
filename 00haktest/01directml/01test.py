import torch
import torch_directml

# 1. DirectML 장치 설정 (RX 570 할당)
device = torch_directml.device()
print(f"사용 중인 장치: {device}")

# 2. 간단한 텐서 연산 테스트
x = torch.ones(3, 3).to(device)
y = torch.ones(3, 3).to(device)
z = x + y

print(z) # 결과가 출력되면 RX 570으로 계산된 것입니다.