import torch
import torch.nn as nn
import time

# 간단한 CNN 모델 정의
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"사용 중인 장치: {device}")

# 모델, 입력, 반복 횟수 정의
model = SimpleCNN().to(device)
input_tensor = torch.randn(64, 3, 224, 224).to(device)
iterations = 100

# 워밍업
with torch.no_grad():
    for _ in range(10):
        _ = model(input_tensor)

# 측정
start_time = time.time()
with torch.no_grad():
    for _ in range(iterations):
        _ = model(input_tensor)
end_time = time.time()

# 결과 출력
total_time = end_time - start_time
average_time = total_time / iterations
print(f"총 수행 시간: {total_time:.4f}초")
print(f"평균 처리 시간 (1 배치): {average_time:.4f}초")
