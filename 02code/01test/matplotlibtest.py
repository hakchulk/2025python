import matplotlib.pyplot as plt

# 데이터 설정 (x: 속도 지수, y: 보안 지수)
# 지수가 높을수록 각각 빠르고 강력함을 의미
algorithms = ['AES-128', 'AES-256', 'ChaCha20', '3DES', 'SHA-256']
speed_index = [100, 75, 90, 15, 60]     # 높을수록 빠름
security_index = [70, 100, 85, 30, 90]  # 높을수록 안전

plt.figure(figsize=(10, 6))
plt.scatter(speed_index, security_index, s=200, c='skyblue', edgecolors='blue')

# 각 점에 알고리즘 이름 표시
for i, txt in enumerate(algorithms):
    plt.annotate(txt, (speed_index[i]+2, security_index[i]), fontsize=12)

plt.title('Encryption Algorithms: Speed vs Security', fontsize=15)
plt.xlabel('Speed Index (Higher = Faster)', fontsize=12)
plt.ylabel('Security Index (Higher = Stronger)', fontsize=12)
plt.axvline(50, color='gray', linestyle='--', alpha=0.5) # 기준선
plt.grid(True, alpha=0.3)
plt.show()