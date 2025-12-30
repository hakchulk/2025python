import hashlib

def generate_sha256(text):
    # 문자열을 바이트(bytes) 형태로 인코딩하여 입력
    data = text.encode('utf-8')
    
    # SHA-256 해시 객체 생성 및 업데이트
    sha256_hash = hashlib.sha256(data).hexdigest()
    
    return sha256_hash

input_str = "Hello Python"
print(f"입력: {input_str}")
print(f"해시 결과: {generate_sha256(input_str)}")